"""Fused CUDA kernels for Metamath proof verification.

Replaces the PyTorch tensor-op Phase 3 GPU pipeline with 3 custom CUDA
kernels that eliminate ALL intermediate memory allocations:

  Kernel 1 — push_nodes:       write push expressions + compute hash
  Kernel 2 — execute_assertion: stream-substitute, check ehyps, write conclusion
  Kernel 3 — final_check:      compare conclusions, write pass/fail

Each assertion thread reads expr_buffer directly by index — no gather copies,
no scatter outputs, no ehyp replication tensors. Memory usage = expr_buffer +
tracking arrays + assertion table. Zero transient allocations.

JIT-compiled via torch.utils.cpp_extension.load_inline() on first use.
Falls back gracefully if compilation fails (caller uses torch path).
"""

from __future__ import annotations

import sys

import numpy as np
import torch

# ══════════════════════════════════════════════════════════════════════
#  CUDA Source
# ══════════════════════════════════════════════════════════════════════

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// ─── Constants & types ───────────────────────────────────────────────
static constexpr long long HASH_BASE = 1000000007LL;
using tok_t = unsigned short;  // uint16 token type — halves expr_buffer memory

// ─── Failure reason codes (stored in node_fail_code, int8_t) ─────────
// 0: no failure
// 1: propagated — an input node already failed
// 2: ehyp mismatch — essential hypothesis substitution did not match
// 3: conclusion overflow — substituted conclusion exceeded allocated capacity
static constexpr int8_t FAIL_NONE        = 0;
static constexpr int8_t FAIL_INPUT       = 1;
static constexpr int8_t FAIL_EHYP        = 2;
static constexpr int8_t FAIL_OVERFLOW    = 3;

// ─── Helper: stream-substitute a pattern and compare against a target ─
// Returns true if the substituted pattern matches the target exactly.
// Never materializes the substituted output — compares token-by-token.
//
// Pattern tokens that match a variable ID get replaced by the variable's
// substitution (expr_buffer[input_global_idx, 1:length]).  Non-variable
// tokens are identity (output = input token, length 1).
//
// Parameters all come from global memory; the function does sequential
// reads which are L2-cache-friendly for the typical access pattern.
__device__ bool stream_substitute_and_compare(
    // Assertion table row for this node's assertion
    const tok_t* __restrict__ pattern,       // [P_max] row in tbl_pattern_toks
    int pat_len,
    const tok_t* __restrict__ fhyp_var_ids,  // [max_fhyps] row in tbl
    int n_fhyps,
    // Substitution source: packed 1D expr_buffer
    const tok_t* __restrict__ expr_buffer,   // [total_expr_tokens] packed
    const int* __restrict__ expr_lengths,  // [total_nodes]
    const long long* __restrict__ expr_offsets,  // [total_nodes+1]
    const int* __restrict__ fhyp_input_global, // [n_fhyps] global indices of $f inputs
    // Target to compare against
    const tok_t* __restrict__ target_row,    // packed row in expr_buffer
    int target_len
) {
    int out_pos = 0;

    for (int p = 0; p < pat_len; p++) {
        tok_t tok = pattern[p];

        // Check if tok is a variable
        int matched_f = -1;
        for (int f = 0; f < n_fhyps; f++) {
            if (tok == fhyp_var_ids[f]) {
                matched_f = f;
                break;
            }
        }

        if (matched_f >= 0) {
            // Variable: substitute with expr_buffer[input_step][1:]
            int input_gi = fhyp_input_global[matched_f];
            int input_len = expr_lengths[input_gi];
            int sub_len = input_len - 1;  // skip type code at position 0
            if (sub_len < 0) sub_len = 0;

            const tok_t* sub_row = expr_buffer + expr_offsets[input_gi] + 1;
            for (int s = 0; s < sub_len; s++) {
                if (out_pos >= target_len) return false;
                if (sub_row[s] != target_row[out_pos]) return false;
                out_pos++;
            }
        } else {
            // Constant: identity
            if (out_pos >= target_len) return false;
            if (tok != target_row[out_pos]) return false;
            out_pos++;
        }
    }

    return (out_pos == target_len);
}


// ─── Helper: stream-substitute a pattern and WRITE to expr_buffer ────
// Also computes the polynomial rolling hash as tokens are written.
// Returns the output length.
__device__ int stream_substitute_and_write(
    const tok_t* __restrict__ pattern,
    int pat_len,
    const tok_t* __restrict__ fhyp_var_ids,
    int n_fhyps,
    const tok_t* __restrict__ expr_buffer,
    const int* __restrict__ expr_lengths,
    const long long* __restrict__ expr_offsets,
    const int* __restrict__ fhyp_input_global,
    // Output destination
    tok_t* __restrict__ output_row,          // packed row to write
    long long* __restrict__ out_hash,      // single hash value to write
    int out_capacity                       // pre-computed capacity of output_row
) {
    int out_pos = 0;
    long long h = 0;

    for (int p = 0; p < pat_len; p++) {
        tok_t tok = pattern[p];

        int matched_f = -1;
        for (int f = 0; f < n_fhyps; f++) {
            if (tok == fhyp_var_ids[f]) {
                matched_f = f;
                break;
            }
        }

        if (matched_f >= 0) {
            int input_gi = fhyp_input_global[matched_f];
            int input_len = expr_lengths[input_gi];
            int sub_len = input_len - 1;
            if (sub_len < 0) sub_len = 0;

            const tok_t* sub_row = expr_buffer + expr_offsets[input_gi] + 1;
            for (int s = 0; s < sub_len; s++) {
                tok_t val = sub_row[s];
                h = h * HASH_BASE + (long long)val;
                if (out_pos < out_capacity) {
                    output_row[out_pos] = val;
                }
                out_pos++;
            }
        } else {
            h = h * HASH_BASE + (long long)tok;
            if (out_pos < out_capacity) {
                output_row[out_pos] = tok;
            }
            out_pos++;
        }
    }

    *out_hash = h;
    return out_pos;
}


// ═════════════════════════════════════════════════════════════════════
//  Kernel 1: Push Nodes
// ═════════════════════════════════════════════════════════════════════
// One thread per push node. Copies expression into expr_buffer and
// computes polynomial hash.

__global__ void push_nodes_kernel(
    // Push data
    const int* __restrict__ push_global_indices,  // [num_push]
    const tok_t* __restrict__ push_expressions,     // [num_push, push_width]
    const int* __restrict__ push_expr_lengths,    // [num_push]
    int push_width,
    int num_push,
    // Output buffers (packed 1D)
    tok_t* __restrict__ expr_buffer,                // [total_expr_tokens] packed
    int* __restrict__ expr_lengths_buf,           // [total_nodes]
    long long* __restrict__ expr_hashes,          // [total_nodes]
    const long long* __restrict__ expr_offsets    // [total_nodes+1]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_push) return;

    int gi = push_global_indices[tid];
    int len = push_expr_lengths[tid];
    const tok_t* src = push_expressions + (long long)tid * push_width;
    tok_t* dst = expr_buffer + expr_offsets[gi];
    int capacity = (int)(expr_offsets[gi + 1] - expr_offsets[gi]);

    // Copy expression
    long long h = 0;
    int copy_len = len < capacity ? len : capacity;
    for (int i = 0; i < copy_len; i++) {
        dst[i] = src[i];
        h = h * HASH_BASE + (long long)src[i];
    }
    // Hash includes tokens beyond capacity if any (defensive)
    for (int i = copy_len; i < len; i++) {
        h = h * HASH_BASE + (long long)src[i];
    }
    // Zero remaining capacity
    for (int i = copy_len; i < capacity; i++) {
        dst[i] = 0;
    }

    expr_lengths_buf[gi] = len;
    expr_hashes[gi] = h;
}


// ═════════════════════════════════════════════════════════════════════
//  Kernel 2: Execute Assertion Nodes
// ═════════════════════════════════════════════════════════════════════
// One thread per assertion node in a sublevel chunk.
// Reads inputs directly from expr_buffer by index (no gather copy).
// Stream-substitutes to check ehyps and write conclusion.

__global__ void execute_assertion_kernel(
    int B,                                          // number of nodes in this launch
    int batch_offset,                               // offset into batch arrays
    // Per-node batch arrays (on device, contiguous)
    const int* __restrict__ assertion_idx,          // [total_batch, ] int32
    const int* __restrict__ input_global_indices,   // [total_batch, max_inputs] int32
    const int* __restrict__ input_counts,           // [total_batch, ] int32
    const int* __restrict__ fhyp_input_positions,   // [total_batch, max_fhyps_batch] int32
    const int* __restrict__ ehyp_input_positions,   // [total_batch, max_ehyps_batch] int32
    const int* __restrict__ output_global_indices,  // [total_batch, ] int32
    int max_inputs,
    int max_fhyps_batch,                            // width of fhyp_input_positions
    int max_ehyps_batch,                            // width of ehyp_input_positions
    // Assertion table (on device, shared across all levels)
    const tok_t* __restrict__ tbl_pattern_toks,       // [A, P_max]
    const int* __restrict__ tbl_pattern_lengths,    // [A]
    const tok_t* __restrict__ tbl_fhyp_var_ids,       // [A, tbl_max_fhyps]
    const int* __restrict__ tbl_fhyp_count,         // [A]
    const tok_t* __restrict__ tbl_ehyp_patterns,      // [A, tbl_max_ehyps, E_max]
    const int* __restrict__ tbl_ehyp_pattern_lengths, // [A, tbl_max_ehyps]
    const int* __restrict__ tbl_ehyp_count,         // [A]
    int P_max,
    int tbl_max_fhyps,
    int tbl_max_ehyps,
    int E_max,
    // Global expression buffer (packed 1D)
    tok_t* __restrict__ expr_buffer,                  // [total_expr_tokens] packed
    int* __restrict__ expr_lengths_buf,             // [total_nodes]
    long long* __restrict__ expr_hashes,            // [total_nodes]
    int8_t* __restrict__ node_fail_code,            // [total_nodes] — FAIL_* constants
    const long long* __restrict__ expr_offsets      // [total_nodes+1]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B) return;

    int b = batch_offset + tid;  // index into batch arrays

    int a_idx = assertion_idx[b];
    int n_inputs = input_counts[b];
    int out_gi = output_global_indices[b];

    // ── (a) Propagate worst input failure code ────────────────────────
    // Carry the root cause code (max severity) from all inputs instead of
    // collapsing to FAIL_INPUT. This preserves FAIL_EHYP/FAIL_OVERFLOW across
    // nodes that have no ehyps of their own (e.g. ax-maj wrapping a failed ax-mp).
    int8_t input_fail_code = FAIL_NONE;
    for (int k = 0; k < n_inputs; k++) {
        int in_gi = input_global_indices[(long long)b * max_inputs + k];
        if (in_gi >= 0) {
            int8_t c = node_fail_code[in_gi];
            if (c > input_fail_code) input_fail_code = c;
        }
    }

    // ── (b) Gather $f hyp input global indices ──────────────────────
    // For each floating hyp, find which input slot it maps to, then
    // look up the global index of that input.
    int n_fhyps = tbl_fhyp_count[a_idx];
    // Stack-allocate space for fhyp global indices (max ~20 in practice)
    // Using a fixed-size array since CUDA doesn't do dynamic stack alloc.
    int fhyp_gi[64];  // 64 is way more than any real assertion needs
    int actual_fhyps = n_fhyps < 64 ? n_fhyps : 64;

    for (int f = 0; f < actual_fhyps; f++) {
        int pos = fhyp_input_positions[(long long)b * max_fhyps_batch + f];
        // Clamp position to valid range
        if (pos < 0) pos = 0;
        if (pos >= max_inputs) pos = max_inputs - 1;
        int in_gi = input_global_indices[(long long)b * max_inputs + pos];
        fhyp_gi[f] = (in_gi >= 0) ? in_gi : 0;
    }

    // Pointer to this assertion's fhyp var IDs in the table
    const tok_t* my_var_ids = tbl_fhyp_var_ids + (long long)a_idx * tbl_max_fhyps;

    // ── (c) Check essential hypotheses ──────────────────────────────
    int n_ehyps = tbl_ehyp_count[a_idx];
    bool ehyps_ok = true;

    for (int e = 0; e < n_ehyps && ehyps_ok; e++) {
        // Get ehyp pattern from table
        const tok_t* ehyp_pat = tbl_ehyp_patterns +
            ((long long)a_idx * tbl_max_ehyps + e) * E_max;
        int ehyp_pat_len = tbl_ehyp_pattern_lengths[
            (long long)a_idx * tbl_max_ehyps + e];

        // Get target: the input expression at the ehyp's input position
        int ehyp_pos = 0;
        if (e < max_ehyps_batch) {
            ehyp_pos = ehyp_input_positions[(long long)b * max_ehyps_batch + e];
        }
        if (ehyp_pos < 0) ehyp_pos = 0;
        if (ehyp_pos >= max_inputs) ehyp_pos = max_inputs - 1;
        int target_gi = input_global_indices[(long long)b * max_inputs + ehyp_pos];
        if (target_gi < 0) target_gi = 0;

        int target_len = expr_lengths_buf[target_gi];
        const tok_t* target_row = expr_buffer + expr_offsets[target_gi];

        bool match = stream_substitute_and_compare(
            ehyp_pat, ehyp_pat_len,
            my_var_ids, actual_fhyps,
            expr_buffer, expr_lengths_buf, expr_offsets,
            fhyp_gi,
            target_row, target_len
        );

        if (!match) ehyps_ok = false;
    }

    // ── (d) Compute and write conclusion ────────────────────────────
    const tok_t* my_pattern = tbl_pattern_toks + (long long)a_idx * P_max;
    int my_pat_len = tbl_pattern_lengths[a_idx];

    tok_t* out_row = expr_buffer + expr_offsets[out_gi];
    int out_capacity = (int)(expr_offsets[out_gi + 1] - expr_offsets[out_gi]);
    // Zero the output row first
    for (int i = 0; i < out_capacity; i++) {
        out_row[i] = 0;
    }

    long long out_hash = 0;
    int out_len = stream_substitute_and_write(
        my_pattern, my_pat_len,
        my_var_ids, actual_fhyps,
        expr_buffer, expr_lengths_buf, expr_offsets,
        fhyp_gi,
        out_row, &out_hash, out_capacity
    );

    expr_lengths_buf[out_gi] = out_len;
    expr_hashes[out_gi] = out_hash;

    // Write failure code: local failures take priority, then propagated root cause.
    // Priority: overflow > ehyp_mismatch > propagated-from-input > none.
    int8_t fail_code = input_fail_code;  // root cause from upstream (may be FAIL_NONE)
    if (!ehyps_ok)           fail_code = FAIL_EHYP;
    if (out_len > out_capacity) fail_code = FAIL_OVERFLOW;
    node_fail_code[out_gi] = fail_code;
}


// ═════════════════════════════════════════════════════════════════════
//  Kernel 3: Final Check
// ═════════════════════════════════════════════════════════════════════
// One thread per proof. Compares the final expression against the
// expected conclusion. Uses token comparison if it fits in expr_buffer,
// hash comparison otherwise.

__global__ void final_check_kernel(
    int num_proofs,
    const int* __restrict__ final_node_indices,       // [num_proofs]
    const tok_t* __restrict__ expected_conclusions,      // [num_proofs, max_concl_stored]
    const int* __restrict__ conclusion_lengths,        // [num_proofs]
    const long long* __restrict__ expected_hashes,     // [num_proofs]
    int max_concl_stored,
    // Global buffers (packed 1D)
    const tok_t* __restrict__ expr_buffer,              // [total_expr_tokens] packed
    const int* __restrict__ expr_lengths_buf,         // [total_nodes]
    const long long* __restrict__ expr_hashes,        // [total_nodes]
    const int8_t* __restrict__ node_fail_code,        // [total_nodes] — FAIL_* constants
    const long long* __restrict__ expr_offsets,        // [total_nodes+1]
    // Output
    bool* __restrict__ proof_passed                   // [num_proofs]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_proofs) return;

    int final_gi = final_node_indices[tid];
    int final_len = expr_lengths_buf[final_gi];
    long long final_hash = expr_hashes[final_gi];
    int8_t fail_code = node_fail_code[final_gi];

    int expected_len = conclusion_lengths[tid];
    long long expected_hash = expected_hashes[tid];

    // Any kernel-level failure or length mismatch → proof fails
    if (fail_code != FAIL_NONE || final_len != expected_len) {
        proof_passed[tid] = false;
        return;
    }

    // With packed buffer the expression is always fully stored.
    // Token comparison is possible when expected_conclusions wasn't truncated.
    bool fits = (final_len <= max_concl_stored);
    if (fits) {
        const tok_t* final_row = expr_buffer + expr_offsets[final_gi];
        const tok_t* expected_row = expected_conclusions + (long long)tid * max_concl_stored;

        for (int i = 0; i < final_len; i++) {
            if (final_row[i] != expected_row[i]) {
                proof_passed[tid] = false;
                return;
            }
        }
        proof_passed[tid] = true;
    } else {
        // Hash comparison (expected conclusion was truncated at max_concl_stored)
        proof_passed[tid] = (final_hash == expected_hash);
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Launcher functions (must be in .cu file — they use <<<>>> syntax)
// ══════════════════════════════════════════════════════════════════════

void push_nodes_launch(
    torch::Tensor push_global_indices,
    torch::Tensor push_expressions,
    torch::Tensor push_expr_lengths,
    int push_width, int num_push,
    torch::Tensor expr_buffer,
    torch::Tensor expr_lengths_buf,
    torch::Tensor expr_hashes,
    torch::Tensor expr_offsets
) {
    if (num_push == 0) return;
    int threads = 256;
    int blocks = (num_push + threads - 1) / threads;
    push_nodes_kernel<<<blocks, threads>>>(
        push_global_indices.data_ptr<int>(),
        reinterpret_cast<const tok_t*>(push_expressions.data_ptr<int16_t>()),
        push_expr_lengths.data_ptr<int>(),
        push_width, num_push,
        reinterpret_cast<tok_t*>(expr_buffer.data_ptr<int16_t>()),
        expr_lengths_buf.data_ptr<int>(),
        reinterpret_cast<long long*>(expr_hashes.data_ptr<int64_t>()),
        reinterpret_cast<const long long*>(expr_offsets.data_ptr<int64_t>())
    );
}

void execute_assertion_launch(
    int B, int batch_offset,
    torch::Tensor assertion_idx,
    torch::Tensor input_global_indices,
    torch::Tensor input_counts,
    torch::Tensor fhyp_input_positions,
    torch::Tensor ehyp_input_positions,
    torch::Tensor output_global_indices,
    int max_inputs, int max_fhyps_batch, int max_ehyps_batch,
    torch::Tensor tbl_pattern_toks,
    torch::Tensor tbl_pattern_lengths,
    torch::Tensor tbl_fhyp_var_ids,
    torch::Tensor tbl_fhyp_count,
    torch::Tensor tbl_ehyp_patterns,
    torch::Tensor tbl_ehyp_pattern_lengths,
    torch::Tensor tbl_ehyp_count,
    int P_max, int tbl_max_fhyps, int tbl_max_ehyps, int E_max,
    torch::Tensor expr_buffer,
    torch::Tensor expr_lengths_buf,
    torch::Tensor expr_hashes,
    torch::Tensor node_fail_code,
    torch::Tensor expr_offsets
) {
    if (B == 0) return;
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    execute_assertion_kernel<<<blocks, threads>>>(
        B, batch_offset,
        assertion_idx.data_ptr<int>(),
        input_global_indices.data_ptr<int>(),
        input_counts.data_ptr<int>(),
        fhyp_input_positions.data_ptr<int>(),
        ehyp_input_positions.data_ptr<int>(),
        output_global_indices.data_ptr<int>(),
        max_inputs, max_fhyps_batch, max_ehyps_batch,
        reinterpret_cast<const tok_t*>(tbl_pattern_toks.data_ptr<int16_t>()),
        tbl_pattern_lengths.data_ptr<int>(),
        reinterpret_cast<const tok_t*>(tbl_fhyp_var_ids.data_ptr<int16_t>()),
        tbl_fhyp_count.data_ptr<int>(),
        reinterpret_cast<const tok_t*>(tbl_ehyp_patterns.data_ptr<int16_t>()),
        tbl_ehyp_pattern_lengths.data_ptr<int>(),
        tbl_ehyp_count.data_ptr<int>(),
        P_max, tbl_max_fhyps, tbl_max_ehyps, E_max,
        reinterpret_cast<tok_t*>(expr_buffer.data_ptr<int16_t>()),
        expr_lengths_buf.data_ptr<int>(),
        reinterpret_cast<long long*>(expr_hashes.data_ptr<int64_t>()),
        node_fail_code.data_ptr<int8_t>(),
        reinterpret_cast<const long long*>(expr_offsets.data_ptr<int64_t>())
    );
}

void final_check_launch(
    int num_proofs,
    torch::Tensor final_node_indices,
    torch::Tensor expected_conclusions,
    torch::Tensor conclusion_lengths,
    torch::Tensor expected_hashes,
    int max_concl_stored,
    torch::Tensor expr_buffer,
    torch::Tensor expr_lengths_buf,
    torch::Tensor expr_hashes,
    torch::Tensor node_fail_code,
    torch::Tensor expr_offsets,
    torch::Tensor proof_passed
) {
    if (num_proofs == 0) return;
    int threads = 256;
    int blocks = (num_proofs + threads - 1) / threads;
    final_check_kernel<<<blocks, threads>>>(
        num_proofs,
        final_node_indices.data_ptr<int>(),
        reinterpret_cast<const tok_t*>(expected_conclusions.data_ptr<int16_t>()),
        conclusion_lengths.data_ptr<int>(),
        reinterpret_cast<const long long*>(expected_hashes.data_ptr<int64_t>()),
        max_concl_stored,
        reinterpret_cast<const tok_t*>(expr_buffer.data_ptr<int16_t>()),
        expr_lengths_buf.data_ptr<int>(),
        reinterpret_cast<const long long*>(expr_hashes.data_ptr<int64_t>()),
        node_fail_code.data_ptr<int8_t>(),
        reinterpret_cast<const long long*>(expr_offsets.data_ptr<int64_t>()),
        proof_passed.data_ptr<bool>()
    );
}

// --- Kernel 4: compute_assertion_table_stats ----------------------------
// One thread per assertion. Scans the pattern token row and for each
// token within the valid length either:
//   matches one of the n_fhyps variable IDs -> increments var_occ[i, f]
//   matches nothing                         -> increments const_count[i]
//
// Replaces the CPU numpy broadcast [A, P, F] which allocates a ~1.3 GB
// intermediate bool tensor for set.mm (42k x 796 x 40). This kernel uses
// O(1) registers per thread and no intermediate allocation.
__global__ void compute_assertion_table_stats_kernel(
    const tok_t* __restrict__ pattern_toks,    // [A, P_max] row-major
    const int*   __restrict__ pattern_lengths, // [A]
    const tok_t* __restrict__ fhyp_var_ids,    // [A, F_max] row-major
    const int*   __restrict__ fhyp_count,      // [A]
    int A, int P_max, int F_max,
    int* __restrict__ const_count,             // [A] output
    int* __restrict__ var_occ                  // [A, F_max] output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A) return;

    int pl = pattern_lengths[i];
    int nf = fhyp_count[i];
    int cc = 0;

    const tok_t* pat_row = pattern_toks + (long long)i * P_max;
    const tok_t* var_row = fhyp_var_ids + (long long)i * F_max;
    int*         occ_row = var_occ      + (long long)i * F_max;

    for (int p = 0; p < pl; p++) {
        tok_t tok = pat_row[p];
        bool matched = false;
        for (int f = 0; f < nf; f++) {
            if (tok == var_row[f]) {
                occ_row[f]++;
                matched = true;
                break;
            }
        }
        if (!matched) cc++;
    }
    const_count[i] = cc;
}

void compute_assertion_table_stats_launch(
    torch::Tensor pattern_toks,    // [A, P_max] int16
    torch::Tensor pattern_lengths, // [A] int32
    torch::Tensor fhyp_var_ids,    // [A, F_max] int16
    torch::Tensor fhyp_count,      // [A] int32
    torch::Tensor const_count,     // [A] int32 -- output, pre-zeroed
    torch::Tensor var_occ          // [A, F_max] int32 -- output, pre-zeroed
) {
    int A     = pattern_toks.size(0);
    int P_max = pattern_toks.size(1);
    int F_max = fhyp_var_ids.size(1);
    if (A == 0) return;
    int threads = 256;
    int blocks  = (A + threads - 1) / threads;
    compute_assertion_table_stats_kernel<<<blocks, threads>>>(
        reinterpret_cast<const tok_t*>(pattern_toks.data_ptr<int16_t>()),
        pattern_lengths.data_ptr<int>(),
        reinterpret_cast<const tok_t*>(fhyp_var_ids.data_ptr<int16_t>()),
        fhyp_count.data_ptr<int>(),
        A, P_max, F_max,
        const_count.data_ptr<int>(),
        var_occ.data_ptr<int>()
    );
}
"""

_CPP_SOURCE = r"""
#include <torch/extension.h>

// Forward declarations — implementations are in the .cu file
void compute_assertion_table_stats_launch(
    torch::Tensor pattern_toks,
    torch::Tensor pattern_lengths,
    torch::Tensor fhyp_var_ids,
    torch::Tensor fhyp_count,
    torch::Tensor const_count,
    torch::Tensor var_occ
);

void push_nodes_launch(
    torch::Tensor push_global_indices,
    torch::Tensor push_expressions,
    torch::Tensor push_expr_lengths,
    int push_width, int num_push,
    torch::Tensor expr_buffer,
    torch::Tensor expr_lengths_buf,
    torch::Tensor expr_hashes,
    torch::Tensor expr_offsets
);

void execute_assertion_launch(
    int B, int batch_offset,
    torch::Tensor assertion_idx,
    torch::Tensor input_global_indices,
    torch::Tensor input_counts,
    torch::Tensor fhyp_input_positions,
    torch::Tensor ehyp_input_positions,
    torch::Tensor output_global_indices,
    int max_inputs, int max_fhyps_batch, int max_ehyps_batch,
    torch::Tensor tbl_pattern_toks,
    torch::Tensor tbl_pattern_lengths,
    torch::Tensor tbl_fhyp_var_ids,
    torch::Tensor tbl_fhyp_count,
    torch::Tensor tbl_ehyp_patterns,
    torch::Tensor tbl_ehyp_pattern_lengths,
    torch::Tensor tbl_ehyp_count,
    int P_max, int tbl_max_fhyps, int tbl_max_ehyps, int E_max,
    torch::Tensor expr_buffer,
    torch::Tensor expr_lengths_buf,
    torch::Tensor expr_hashes,
    torch::Tensor node_fail_code,
    torch::Tensor expr_offsets
);

void final_check_launch(
    int num_proofs,
    torch::Tensor final_node_indices,
    torch::Tensor expected_conclusions,
    torch::Tensor conclusion_lengths,
    torch::Tensor expected_hashes,
    int max_concl_stored,
    torch::Tensor expr_buffer,
    torch::Tensor expr_lengths_buf,
    torch::Tensor expr_hashes,
    torch::Tensor node_fail_code,
    torch::Tensor expr_offsets,
    torch::Tensor proof_passed
);
"""

# ══════════════════════════════════════════════════════════════════════
#  JIT Compilation
# ══════════════════════════════════════════════════════════════════════

_compiled_module = None
_compilation_attempted = False


def _try_compile():
    """Attempt to JIT-compile the CUDA kernels. Returns module or None."""
    global _compiled_module, _compilation_attempted
    if _compilation_attempted:
        return _compiled_module
    _compilation_attempted = True

    if not torch.cuda.is_available():
        return None

    try:
        from torch.utils.cpp_extension import load_inline
        _compiled_module = load_inline(
            name="mmgpu_cuda_kernels_v7",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=[
                "compute_assertion_table_stats_launch",
                "push_nodes_launch",
                "execute_assertion_launch",
                "final_check_launch",
            ],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
        return _compiled_module
    except Exception as e:
        print(f"[CUDA kernels] JIT compilation failed: {e}", file=sys.stderr)
        print("[CUDA kernels] Falling back to PyTorch tensor path", file=sys.stderr)
        return None


def is_available() -> bool:
    """Check if CUDA kernels compiled successfully."""
    return _try_compile() is not None


def get_module():
    """Get the compiled CUDA module, or None if unavailable."""
    return _try_compile()


# ══════════════════════════════════════════════════════════════════════
#  Python API — thin wrappers that handle numpy→torch conversion
# ══════════════════════════════════════════════════════════════════════


def cuda_compute_assertion_table_stats(
    pattern_toks: np.ndarray,    # [A, P_max] int16
    pattern_lengths: np.ndarray, # [A] int32
    fhyp_var_ids: np.ndarray,    # [A, F_max] int16
    fhyp_count: np.ndarray,      # [A] int32
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute const_count [A] and var_occ [A, F_max] on CUDA.

    Returns numpy arrays.  Falls back to the numpy broadcast path if CUDA
    kernels are unavailable (no GPU, or compilation failed).
    """
    mod = _try_compile()
    if mod is None:
        return None, None  # caller handles fallback

    A     = pattern_toks.shape[0]
    F_max = fhyp_var_ids.shape[1]

    dev = device
    pt  = torch.from_numpy(pattern_toks).to(dev)
    pl  = torch.from_numpy(pattern_lengths).to(dev)
    vi  = torch.from_numpy(fhyp_var_ids).to(dev)
    fc  = torch.from_numpy(fhyp_count).to(dev)
    cc  = torch.zeros(A,        dtype=torch.int32, device=dev)
    vo  = torch.zeros((A, F_max), dtype=torch.int32, device=dev)

    mod.compute_assertion_table_stats_launch(pt, pl, vi, fc, cc, vo)
    torch.cuda.synchronize(dev)

    return cc.cpu().numpy(), vo.cpu().numpy()


def cuda_push_nodes(
    push_global_indices: np.ndarray,   # [num_push] int32
    push_expressions: np.ndarray,      # [num_push, push_width] int16
    push_expr_lengths: np.ndarray,     # [num_push] int32
    expr_buffer: torch.Tensor,         # [total_expr_tokens] int16 on device (packed 1D)
    expr_lengths: torch.Tensor,        # [total_nodes] int32 on device
    expr_hashes: torch.Tensor,         # [total_nodes] int64 on device
    expr_offsets: torch.Tensor,        # [total_nodes+1] int64 on device
    device: torch.device,
) -> None:
    """Launch push_nodes_kernel in chunks to avoid OOM on wide push arrays."""
    mod = get_module()
    assert mod is not None

    num_push = len(push_global_indices)
    if num_push == 0:
        return

    push_width = push_expressions.shape[1]

    # Chunk size: keep GPU upload < ~512 MB per chunk
    # Each row = push_width * 2 bytes (int16)
    bytes_per_row = push_width * 2
    chunk_rows = max(1, (512 * 1024 * 1024) // bytes_per_row)

    for start in range(0, num_push, chunk_rows):
        end = min(start + chunk_rows, num_push)
        n = end - start

        gi_t = torch.from_numpy(np.ascontiguousarray(push_global_indices[start:end])).to(device)
        ex_t = torch.from_numpy(np.ascontiguousarray(push_expressions[start:end])).to(device)
        el_t = torch.from_numpy(np.ascontiguousarray(push_expr_lengths[start:end])).to(device)

        mod.push_nodes_launch(
            gi_t, ex_t, el_t,
            push_width, n,
            expr_buffer, expr_lengths, expr_hashes,
            expr_offsets,
        )

        del gi_t, ex_t, el_t


def cuda_execute_level(
    batch_assertion_idx: np.ndarray,         # [B] int32
    batch_input_global_indices: np.ndarray,  # [B, max_inputs] int32
    batch_input_counts: np.ndarray,          # [B] int32
    batch_fhyp_input_positions: np.ndarray,  # [B, max_fhyps_batch] int32
    batch_ehyp_input_positions: np.ndarray,  # [B, max_ehyps_batch] int32
    batch_output_global_indices: np.ndarray, # [B] int32
    sublevel_ranges: list[tuple[int, int]],
    # Assertion table (already on device)
    tbl_pattern_toks: torch.Tensor,
    tbl_pattern_lengths: torch.Tensor,
    tbl_fhyp_var_ids: torch.Tensor,
    tbl_fhyp_count: torch.Tensor,
    tbl_ehyp_patterns: torch.Tensor,
    tbl_ehyp_pattern_lengths: torch.Tensor,
    tbl_ehyp_count: torch.Tensor,
    # Global buffers (on device, packed 1D)
    expr_buffer: torch.Tensor,
    expr_lengths: torch.Tensor,
    expr_hashes: torch.Tensor,
    node_fail_code: torch.Tensor,   # int8 — FAIL_* codes
    expr_offsets: torch.Tensor,
    device: torch.device,
) -> None:
    """Launch execute_assertion_kernel for a batch, respecting sublevel ordering."""
    mod = get_module()
    assert mod is not None

    B_total = len(batch_assertion_idx)
    if B_total == 0:
        return

    max_inputs = batch_input_global_indices.shape[1]
    max_fhyps_batch = batch_fhyp_input_positions.shape[1]
    max_ehyps_batch = batch_ehyp_input_positions.shape[1]

    P_max = tbl_pattern_toks.shape[1]
    tbl_max_fhyps = tbl_fhyp_var_ids.shape[1]
    tbl_max_ehyps = tbl_ehyp_patterns.shape[1]
    E_max = tbl_ehyp_patterns.shape[2]

    # Pad batch fhyp/ehyp position arrays to match table widths if needed
    if max_fhyps_batch < tbl_max_fhyps:
        batch_fhyp_input_positions = np.pad(
            batch_fhyp_input_positions,
            ((0, 0), (0, tbl_max_fhyps - max_fhyps_batch)),
        )
        max_fhyps_batch = tbl_max_fhyps
    if max_ehyps_batch < tbl_max_ehyps:
        batch_ehyp_input_positions = np.pad(
            batch_ehyp_input_positions,
            ((0, 0), (0, tbl_max_ehyps - max_ehyps_batch)),
        )
        max_ehyps_batch = tbl_max_ehyps

    # Upload batch arrays ONCE (small: B × max_inputs ints, etc.)
    # np.ascontiguousarray ensures torch.from_numpy won't fail on sliced arrays
    asrt_t = torch.from_numpy(np.ascontiguousarray(batch_assertion_idx)).to(device)
    in_gi_t = torch.from_numpy(np.ascontiguousarray(batch_input_global_indices)).to(device)
    in_cnt_t = torch.from_numpy(np.ascontiguousarray(batch_input_counts)).to(device)
    fhyp_pos_t = torch.from_numpy(np.ascontiguousarray(batch_fhyp_input_positions)).to(device)
    ehyp_pos_t = torch.from_numpy(np.ascontiguousarray(batch_ehyp_input_positions)).to(device)
    out_gi_t = torch.from_numpy(np.ascontiguousarray(batch_output_global_indices)).to(device)

    # Launch one kernel per sublevel range (topological correctness)
    for sl_start, sl_end in sublevel_ranges:
        sl_size = sl_end - sl_start
        if sl_size == 0:
            continue

        mod.execute_assertion_launch(
            sl_size, sl_start,
            asrt_t, in_gi_t, in_cnt_t,
            fhyp_pos_t, ehyp_pos_t, out_gi_t,
            max_inputs, max_fhyps_batch, max_ehyps_batch,
            tbl_pattern_toks, tbl_pattern_lengths,
            tbl_fhyp_var_ids, tbl_fhyp_count,
            tbl_ehyp_patterns, tbl_ehyp_pattern_lengths, tbl_ehyp_count,
            P_max, tbl_max_fhyps, tbl_max_ehyps, E_max,
            expr_buffer, expr_lengths, expr_hashes, node_fail_code,
            expr_offsets,
        )
        # Sync between sublevels to ensure writes are visible
        torch.cuda.synchronize(device)


def cuda_final_check(
    final_node_indices: np.ndarray,        # [num_proofs] int32
    expected_conclusions: np.ndarray,       # [num_proofs, max_concl_stored] int16
    conclusion_lengths: np.ndarray,         # [num_proofs] int32
    expected_hashes: np.ndarray,            # [num_proofs] int64
    expr_buffer: torch.Tensor,
    expr_lengths: torch.Tensor,
    expr_hashes: torch.Tensor,
    node_fail_code: torch.Tensor,           # int8 — FAIL_* codes
    expr_offsets: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Launch final_check_kernel. Returns [num_proofs] bool numpy array."""
    mod = get_module()
    assert mod is not None

    num_proofs = len(final_node_indices)
    if num_proofs == 0:
        return np.ones(0, dtype=np.bool_)

    max_concl_stored = expected_conclusions.shape[1]

    # Upload final check data
    fi_t = torch.from_numpy(final_node_indices).to(device)
    ec_t = torch.from_numpy(expected_conclusions).to(device)
    cl_t = torch.from_numpy(conclusion_lengths).to(device)
    eh_t = torch.from_numpy(np.ascontiguousarray(expected_hashes)).to(device)

    # Output tensor
    proof_passed = torch.zeros(num_proofs, dtype=torch.bool, device=device)

    mod.final_check_launch(
        num_proofs,
        fi_t, ec_t, cl_t, eh_t,
        max_concl_stored,
        expr_buffer, expr_lengths, expr_hashes, node_fail_code,
        expr_offsets,
        proof_passed,
    )

    torch.cuda.synchronize(device)
    return proof_passed.cpu().numpy()
