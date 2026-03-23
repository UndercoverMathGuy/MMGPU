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

// ─── Constants ───────────────────────────────────────────────────────
static constexpr long long HASH_BASE = 1000000007LL;

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
    const int* __restrict__ pattern,       // [P_max] row in tbl_pattern_toks
    int pat_len,
    const int* __restrict__ fhyp_var_ids,  // [max_fhyps] row in tbl
    int n_fhyps,
    // Substitution source: expr_buffer rows for each $f input
    const int* __restrict__ expr_buffer,   // [total_nodes, max_expr_len]
    const int* __restrict__ expr_lengths,  // [total_nodes]
    int max_expr_len,
    const int* __restrict__ fhyp_input_global, // [n_fhyps] global indices of $f inputs
    // Target to compare against
    const int* __restrict__ target_row,    // [max_expr_len] row in expr_buffer
    int target_len
) {
    int out_pos = 0;

    for (int p = 0; p < pat_len; p++) {
        int tok = pattern[p];

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

            const int* sub_row = expr_buffer + (long long)input_gi * max_expr_len + 1;
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
    const int* __restrict__ pattern,
    int pat_len,
    const int* __restrict__ fhyp_var_ids,
    int n_fhyps,
    const int* __restrict__ expr_buffer,
    const int* __restrict__ expr_lengths,
    int max_expr_len,
    const int* __restrict__ fhyp_input_global,
    // Output destination
    int* __restrict__ output_row,          // [max_expr_len] row to write
    long long* __restrict__ out_hash       // single hash value to write
) {
    int out_pos = 0;
    long long h = 0;

    for (int p = 0; p < pat_len; p++) {
        int tok = pattern[p];

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

            const int* sub_row = expr_buffer + (long long)input_gi * max_expr_len + 1;
            for (int s = 0; s < sub_len; s++) {
                int val = sub_row[s];
                h = h * HASH_BASE + (long long)val;
                if (out_pos < max_expr_len) {
                    output_row[out_pos] = val;
                }
                out_pos++;
            }
        } else {
            h = h * HASH_BASE + (long long)tok;
            if (out_pos < max_expr_len) {
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
    const int* __restrict__ push_expressions,     // [num_push, push_width]
    const int* __restrict__ push_expr_lengths,    // [num_push]
    int push_width,
    int num_push,
    // Output buffers
    int* __restrict__ expr_buffer,                // [total_nodes, max_expr_len]
    int* __restrict__ expr_lengths_buf,           // [total_nodes]
    long long* __restrict__ expr_hashes,          // [total_nodes]
    int max_expr_len
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_push) return;

    int gi = push_global_indices[tid];
    int len = push_expr_lengths[tid];
    const int* src = push_expressions + (long long)tid * push_width;
    int* dst = expr_buffer + (long long)gi * max_expr_len;

    // Copy expression
    long long h = 0;
    int copy_len = len < max_expr_len ? len : max_expr_len;
    for (int i = 0; i < copy_len; i++) {
        dst[i] = src[i];
        h = h * HASH_BASE + (long long)src[i];
    }
    // Hash includes tokens beyond max_expr_len if any (for correctness)
    for (int i = copy_len; i < len; i++) {
        h = h * HASH_BASE + (long long)src[i];
    }
    // Zero remaining
    for (int i = copy_len; i < max_expr_len; i++) {
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
    const int* __restrict__ tbl_pattern_toks,       // [A, P_max]
    const int* __restrict__ tbl_pattern_lengths,    // [A]
    const int* __restrict__ tbl_fhyp_var_ids,       // [A, tbl_max_fhyps]
    const int* __restrict__ tbl_fhyp_count,         // [A]
    const int* __restrict__ tbl_ehyp_patterns,      // [A, tbl_max_ehyps, E_max]
    const int* __restrict__ tbl_ehyp_pattern_lengths, // [A, tbl_max_ehyps]
    const int* __restrict__ tbl_ehyp_count,         // [A]
    int P_max,
    int tbl_max_fhyps,
    int tbl_max_ehyps,
    int E_max,
    // Global expression buffer
    int* __restrict__ expr_buffer,                  // [total_nodes, max_expr_len]
    int* __restrict__ expr_lengths_buf,             // [total_nodes]
    long long* __restrict__ expr_hashes,            // [total_nodes]
    bool* __restrict__ node_failed,                 // [total_nodes]
    int max_expr_len
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B) return;

    int b = batch_offset + tid;  // index into batch arrays

    int a_idx = assertion_idx[b];
    int n_inputs = input_counts[b];
    int out_gi = output_global_indices[b];

    // ── (a) Check if any input failed ────────────────────────────────
    bool any_failed = false;
    for (int k = 0; k < n_inputs; k++) {
        int in_gi = input_global_indices[(long long)b * max_inputs + k];
        if (in_gi >= 0 && node_failed[in_gi]) {
            any_failed = true;
            break;
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
    const int* my_var_ids = tbl_fhyp_var_ids + (long long)a_idx * tbl_max_fhyps;

    // ── (c) Check essential hypotheses ──────────────────────────────
    int n_ehyps = tbl_ehyp_count[a_idx];
    bool ehyps_ok = true;

    for (int e = 0; e < n_ehyps && ehyps_ok; e++) {
        // Get ehyp pattern from table
        const int* ehyp_pat = tbl_ehyp_patterns +
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
        const int* target_row = expr_buffer + (long long)target_gi * max_expr_len;

        bool match = stream_substitute_and_compare(
            ehyp_pat, ehyp_pat_len,
            my_var_ids, actual_fhyps,
            expr_buffer, expr_lengths_buf, max_expr_len,
            fhyp_gi,
            target_row, target_len
        );

        if (!match) ehyps_ok = false;
    }

    // ── (d) Compute and write conclusion ────────────────────────────
    const int* my_pattern = tbl_pattern_toks + (long long)a_idx * P_max;
    int my_pat_len = tbl_pattern_lengths[a_idx];

    int* out_row = expr_buffer + (long long)out_gi * max_expr_len;
    // Zero the output row first
    for (int i = 0; i < max_expr_len; i++) {
        out_row[i] = 0;
    }

    long long out_hash = 0;
    int out_len = stream_substitute_and_write(
        my_pattern, my_pat_len,
        my_var_ids, actual_fhyps,
        expr_buffer, expr_lengths_buf, max_expr_len,
        fhyp_gi,
        out_row, &out_hash
    );

    expr_lengths_buf[out_gi] = out_len;
    expr_hashes[out_gi] = out_hash;
    node_failed[out_gi] = !ehyps_ok || any_failed;
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
    const int* __restrict__ expected_conclusions,      // [num_proofs, max_concl_stored]
    const int* __restrict__ conclusion_lengths,        // [num_proofs]
    const long long* __restrict__ expected_hashes,     // [num_proofs]
    int max_concl_stored,
    // Global buffers
    const int* __restrict__ expr_buffer,              // [total_nodes, max_expr_len]
    const int* __restrict__ expr_lengths_buf,         // [total_nodes]
    const long long* __restrict__ expr_hashes,        // [total_nodes]
    const bool* __restrict__ node_failed,             // [total_nodes]
    int max_expr_len,
    // Output
    bool* __restrict__ proof_passed                   // [num_proofs]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_proofs) return;

    int final_gi = final_node_indices[tid];
    int final_len = expr_lengths_buf[final_gi];
    long long final_hash = expr_hashes[final_gi];
    bool failed = node_failed[final_gi];

    int expected_len = conclusion_lengths[tid];
    long long expected_hash = expected_hashes[tid];

    // Length must match
    if (final_len != expected_len || failed) {
        proof_passed[tid] = false;
        return;
    }

    // If expression fits in buffer: token comparison (exact)
    bool fits = (final_len <= max_expr_len);
    if (fits) {
        const int* final_row = expr_buffer + (long long)final_gi * max_expr_len;
        const int* expected_row = expected_conclusions + (long long)tid * max_concl_stored;
        int cmp_len = final_len < max_concl_stored ? final_len : max_concl_stored;

        for (int i = 0; i < cmp_len; i++) {
            if (final_row[i] != expected_row[i]) {
                proof_passed[tid] = false;
                return;
            }
        }
        // If final_len > max_concl_stored, remaining tokens can't be checked
        // via token comparison — fall through to hash
        if (final_len > max_concl_stored) {
            fits = false;
        }
    }

    if (!fits) {
        // Hash comparison
        proof_passed[tid] = (final_hash == expected_hash);
        return;
    }

    proof_passed[tid] = true;
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
    int max_expr_len
) {
    if (num_push == 0) return;
    int threads = 256;
    int blocks = (num_push + threads - 1) / threads;
    push_nodes_kernel<<<blocks, threads>>>(
        push_global_indices.data_ptr<int>(),
        push_expressions.data_ptr<int>(),
        push_expr_lengths.data_ptr<int>(),
        push_width, num_push,
        expr_buffer.data_ptr<int>(),
        expr_lengths_buf.data_ptr<int>(),
        expr_hashes.data_ptr<long long>(),
        max_expr_len
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
    torch::Tensor node_failed,
    int max_expr_len
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
        tbl_pattern_toks.data_ptr<int>(),
        tbl_pattern_lengths.data_ptr<int>(),
        tbl_fhyp_var_ids.data_ptr<int>(),
        tbl_fhyp_count.data_ptr<int>(),
        tbl_ehyp_patterns.data_ptr<int>(),
        tbl_ehyp_pattern_lengths.data_ptr<int>(),
        tbl_ehyp_count.data_ptr<int>(),
        P_max, tbl_max_fhyps, tbl_max_ehyps, E_max,
        expr_buffer.data_ptr<int>(),
        expr_lengths_buf.data_ptr<int>(),
        expr_hashes.data_ptr<long long>(),
        node_failed.data_ptr<bool>(),
        max_expr_len
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
    torch::Tensor node_failed,
    int max_expr_len,
    torch::Tensor proof_passed
) {
    if (num_proofs == 0) return;
    int threads = 256;
    int blocks = (num_proofs + threads - 1) / threads;
    final_check_kernel<<<blocks, threads>>>(
        num_proofs,
        final_node_indices.data_ptr<int>(),
        expected_conclusions.data_ptr<int>(),
        conclusion_lengths.data_ptr<int>(),
        expected_hashes.data_ptr<long long>(),
        max_concl_stored,
        expr_buffer.data_ptr<int>(),
        expr_lengths_buf.data_ptr<int>(),
        expr_hashes.data_ptr<long long>(),
        node_failed.data_ptr<bool>(),
        max_expr_len,
        proof_passed.data_ptr<bool>()
    );
}
"""

_CPP_SOURCE = r"""
#include <torch/extension.h>

// Forward declarations — implementations are in the .cu file
void push_nodes_launch(
    torch::Tensor push_global_indices,
    torch::Tensor push_expressions,
    torch::Tensor push_expr_lengths,
    int push_width, int num_push,
    torch::Tensor expr_buffer,
    torch::Tensor expr_lengths_buf,
    torch::Tensor expr_hashes,
    int max_expr_len
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
    torch::Tensor node_failed,
    int max_expr_len
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
    torch::Tensor node_failed,
    int max_expr_len,
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
            name="mmgpu_cuda_kernels_v2",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=[
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


def cuda_push_nodes(
    push_global_indices: np.ndarray,   # [num_push] int32
    push_expressions: np.ndarray,      # [num_push, push_width] int32
    push_expr_lengths: np.ndarray,     # [num_push] int32
    expr_buffer: torch.Tensor,         # [total_nodes, max_expr_len] int32 on device
    expr_lengths: torch.Tensor,        # [total_nodes] int32 on device
    expr_hashes: torch.Tensor,         # [total_nodes] int64 on device
    device: torch.device,
) -> None:
    """Launch push_nodes_kernel."""
    mod = get_module()
    assert mod is not None

    num_push = len(push_global_indices)
    if num_push == 0:
        return

    push_width = push_expressions.shape[1]
    max_expr_len = expr_buffer.shape[1]

    # Upload push data (small, one-time per pipeline run)
    gi_t = torch.from_numpy(push_global_indices).to(device)
    ex_t = torch.from_numpy(push_expressions).to(device)
    el_t = torch.from_numpy(push_expr_lengths).to(device)

    mod.push_nodes_launch(
        gi_t, ex_t, el_t,
        push_width, num_push,
        expr_buffer, expr_lengths, expr_hashes,
        max_expr_len,
    )


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
    # Global buffers (on device)
    expr_buffer: torch.Tensor,
    expr_lengths: torch.Tensor,
    expr_hashes: torch.Tensor,
    node_failed: torch.Tensor,
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
    max_expr_len = expr_buffer.shape[1]

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
            expr_buffer, expr_lengths, expr_hashes, node_failed,
            max_expr_len,
        )
        # Sync between sublevels to ensure writes are visible
        torch.cuda.synchronize(device)


def cuda_final_check(
    final_node_indices: np.ndarray,        # [num_proofs] int32
    expected_conclusions: np.ndarray,       # [num_proofs, max_concl_stored] int32
    conclusion_lengths: np.ndarray,         # [num_proofs] int32
    expected_hashes: np.ndarray,            # [num_proofs] int64
    expr_buffer: torch.Tensor,
    expr_lengths: torch.Tensor,
    expr_hashes: torch.Tensor,
    node_failed: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Launch final_check_kernel. Returns [num_proofs] bool numpy array."""
    mod = get_module()
    assert mod is not None

    num_proofs = len(final_node_indices)
    if num_proofs == 0:
        return np.ones(0, dtype=np.bool_)

    max_expr_len = expr_buffer.shape[1]
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
        expr_buffer, expr_lengths, expr_hashes, node_failed,
        max_expr_len,
        proof_passed,
    )

    torch.cuda.synchronize(device)
    return proof_passed.cpu().numpy()
