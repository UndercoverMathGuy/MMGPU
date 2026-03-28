"""True GPU-accelerated Metamath proof verification — no CPU replay.

The CPU only does cheap O(n) parsing of proof label sequences into dependency
graphs. The GPU computes substitutions, applies them, checks essential
hypotheses, and determines proof validity.

Architecture: Topological Level Batching
  Phase 1 — GRAPH CONSTRUCTION (CPU): walk label sequences with virtual stack
             of step indices, produce dependency DAG with topological levels
  Phase 2 — LEVEL PACKING (CPU → GPU tensors): group all nodes by level,
             pack into padded numpy arrays
  Phase 3 — GPU EXECUTION (level by level): maintain expr_buffer on GPU,
             push nodes write known expressions, assertion nodes gather inputs,
             build sub_tables, check essential hyps, compute conclusions
  Phase 4 — $d POST-CHECK (CPU): check disjoint variable constraints
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import torch
import numba
from numba import njit, prange

from tensormm.parser import ParsedDatabase
from tensormm.tokenizer import Tokenizer
from tensormm import cuda_kernels as _cuda_mod

# ══════════════════════════════════════════════════════════════════════
#  Phase 1 — Graph Construction (CPU, O(n))
# ══════════════════════════════════════════════════════════════════════


# Node type constants (stored in ProofGraph.node_types as int8)
NODE_PUSH_F   = np.int8(0)
NODE_PUSH_E   = np.int8(1)
NODE_ASSERTION = np.int8(2)


@dataclass(slots=True)
class ProofGraph:
    """Dependency graph for a single theorem's proof — stored as flat arrays.

    All per-node data is in numpy arrays indexed by step index (0..num_nodes-1).
    input_steps is stored in CSR format: inputs for node i are
        input_data[ input_offsets[i] : input_offsets[i+1] ]

    push_expr_data / push_expr_offsets store the raw symbol strings for push
    nodes (push_node_indices[j] gives the step_idx of the j-th push node).
    Encoding to token IDs happens once in pack_levels via the shared _enc cache.
    """
    theorem_label: str
    expected_conclusion: list[str]
    max_level: int
    max_push_expr_len: int      # max expression length among push nodes

    # Per-node arrays — length num_nodes
    num_nodes: int
    node_types: np.ndarray      # int8  [N]  NODE_PUSH_F / NODE_PUSH_E / NODE_ASSERTION
    node_levels: np.ndarray     # int32 [N]
    node_label_ids: np.ndarray  # int32 [N]  index into label_id_map (built alongside)

    # CSR input_steps: inputs for node i = input_data[input_offsets[i]:input_offsets[i+1]]
    input_offsets: np.ndarray   # int32 [N+1]
    input_data: np.ndarray      # int32 [total_inputs]

    # Push node expressions (raw strings, encoded later)
    push_node_indices: np.ndarray   # int32 [num_push]  — step_idx of each push node
    push_expr_strings: list[list[str]]  # [num_push] raw symbol lists

    # Optional pre-encoded token IDs (set by Rust path to skip _enc() in pack_levels).
    # If not None, pack_levels uses these directly instead of calling _enc(push_expr_strings).
    push_enc_flat: np.ndarray | None     # int16 [total_push_tokens]  (CSR data)
    push_enc_offsets: np.ndarray | None  # int32 [num_push+1]        (CSR offsets)

    # Per-graph label_id_map: label string → int id (0-based within graph's label space)
    # Only assertion nodes use this; push nodes' label_ids are unused in Phase 2.
    label_id_map: dict[str, int]    # label → id (shared vocab built during graph walk)


def _build_label_info(parsed: ParsedDatabase) -> dict[str, tuple[str, object]]:
    """Build the label→info lookup ONCE for the whole database."""
    label_info: dict[str, tuple[str, object]] = {}
    for lbl, fh in parsed.floating_hyps.items():
        label_info[lbl] = ("$f", fh)
    for lbl, eh in parsed.essential_hyps.items():
        label_info[lbl] = ("$e", eh)
    for lbl, a in parsed.assertions.items():
        st = "$a" if a.type == "axiom" else "$p"
        label_info[lbl] = (st, a)
    return label_info


def build_proof_graph(
    parsed: ParsedDatabase,
    theorem_label: str,
    label_info: dict[str, tuple[str, object]],
) -> ProofGraph | str:
    """Build a dependency graph for a theorem's proof.

    Walks the label sequence with a virtual stack of step indices (not
    expressions). Returns ProofGraph backed by flat numpy arrays (no ProofNode
    objects), or an error string on failure.
    """
    if theorem_label not in parsed.assertions:
        return f"Label '{theorem_label}' not found"
    assertion = parsed.assertions[theorem_label]
    if assertion.type != "theorem":
        return f"'{theorem_label}' is not a theorem"

    # Accumulate per-node data as flat Python lists (converted to numpy at end)
    _node_types: list[int] = []
    _node_levels: list[int] = []
    _node_label_ids: list[int] = []
    _input_offsets: list[int] = [0]   # CSR row pointers
    _input_data: list[int] = []       # CSR column data (step indices)
    _push_node_indices: list[int] = []
    _push_expr_strings: list[list[str]] = []

    # Map assertion labels → compact int id (0-based, grows as seen)
    label_id_map: dict[str, int] = {}

    virtual_stack: list[int] = []
    step_counter = 0
    max_level = 0
    max_push_expr_len = 0
    step_levels: list[int] = []

    def _process_label(label: str) -> str | None:
        nonlocal step_counter, max_level, max_push_expr_len
        if label not in label_info:
            return f"Unknown label: {label}"

        stmt_type, data = label_info[label]

        if stmt_type == "$f":
            expr = [data.type_code, data.variable]
            if len(expr) > max_push_expr_len:
                max_push_expr_len = len(expr)
            _node_types.append(int(NODE_PUSH_F))
            _node_levels.append(0)
            _node_label_ids.append(0)   # unused for push nodes
            _input_offsets.append(_input_offsets[-1])  # no inputs
            _push_node_indices.append(step_counter)
            _push_expr_strings.append(expr)
            step_levels.append(0)
            virtual_stack.append(step_counter)
            step_counter += 1

        elif stmt_type == "$e":
            expr = list(data.expression)
            if len(expr) > max_push_expr_len:
                max_push_expr_len = len(expr)
            _node_types.append(int(NODE_PUSH_E))
            _node_levels.append(0)
            _node_label_ids.append(0)
            _input_offsets.append(_input_offsets[-1])
            _push_node_indices.append(step_counter)
            _push_expr_strings.append(expr)
            step_levels.append(0)
            virtual_stack.append(step_counter)
            step_counter += 1

        elif stmt_type in ("$a", "$p"):
            a = data
            n_f = len(a.floating_hyps)
            n_e = len(a.essential_hyps)
            npop = n_f + n_e
            if len(virtual_stack) < npop:
                return (
                    f"Stack underflow at {label}: need {npop}, "
                    f"have {len(virtual_stack)}"
                )
            input_steps = virtual_stack[len(virtual_stack) - npop:]
            del virtual_stack[len(virtual_stack) - npop:]

            level = 0
            for si in input_steps:
                lv1 = step_levels[si] + 1
                if lv1 > level:
                    level = lv1
            if level > max_level:
                max_level = level

            # Assign compact label id
            if label not in label_id_map:
                label_id_map[label] = len(label_id_map)
            lid = label_id_map[label]

            _node_types.append(int(NODE_ASSERTION))
            _node_levels.append(level)
            _node_label_ids.append(lid)
            _input_data.extend(input_steps)
            _input_offsets.append(_input_offsets[-1] + len(input_steps))
            step_levels.append(level)
            virtual_stack.append(step_counter)
            step_counter += 1

        return None

    try:
        if assertion.compressed_proof is not None:
            cp = assertion.compressed_proof
            plabels = cp.labels
            label_end = len(plabels)
            saved_indices: list[int] = []

            for proof_int in cp.proof_ints:
                if proof_int == -1:
                    if not virtual_stack:
                        return f"Z save on empty stack in {theorem_label}"
                    saved_indices.append(virtual_stack[-1])
                elif proof_int < label_end:
                    err = _process_label(plabels[proof_int])
                    if err:
                        return err
                else:
                    si = proof_int - label_end
                    if si >= len(saved_indices):
                        return (
                            f"Saved index {si} out of range "
                            f"(only {len(saved_indices)} saved) in {theorem_label}"
                        )
                    virtual_stack.append(saved_indices[si])

        elif assertion.proof is not None:
            for step_label in assertion.proof:
                err = _process_label(step_label)
                if err:
                    return err
        else:
            return f"Theorem '{theorem_label}' has no proof"

    except Exception as e:
        return f"Error in {theorem_label}: {e}"

    if len(virtual_stack) != 1:
        return (
            f"Stack has {len(virtual_stack)} entries at end of proof "
            f"for {theorem_label}, expected 1"
        )

    N = step_counter
    return ProofGraph(
        theorem_label=theorem_label,
        expected_conclusion=assertion.expression,
        max_level=max_level,
        max_push_expr_len=max_push_expr_len,
        num_nodes=N,
        node_types=np.array(_node_types, dtype=np.int8),
        node_levels=np.array(_node_levels, dtype=np.int32),
        node_label_ids=np.array(_node_label_ids, dtype=np.int32),
        input_offsets=np.array(_input_offsets, dtype=np.int32),
        input_data=np.array(_input_data, dtype=np.int32) if _input_data else np.empty(0, dtype=np.int32),
        push_node_indices=np.array(_push_node_indices, dtype=np.int32),
        push_expr_strings=_push_expr_strings,
        push_enc_flat=None,
        push_enc_offsets=None,
        label_id_map=label_id_map,
    )


# ── Rust-accelerated parallel graph construction ──────────────────

try:
    import mmgpu_rs as _mmgpu_rs
    _HAVE_RUST = True
except ImportError:
    _mmgpu_rs = None  # type: ignore
    _HAVE_RUST = False


def _serialize_db_for_rust(
    parsed: ParsedDatabase,
    tokenizer: "Tokenizer | None",
    theorem_labels: list[str],
) -> tuple:
    """Serialise ParsedDatabase into flat arrays for the Rust extension.

    Returns tuple of byte-strings / scalars matching the build_graphs() signature.
    """
    # ── Symbol table ──────────────────────────────────────────────────
    # Assign integer IDs to every symbol string encountered in the database.
    sym_to_id: dict[str, int] = {}

    def sid(s: str) -> int:
        v = sym_to_id.get(s)
        if v is None:
            v = len(sym_to_id)
            sym_to_id[s] = v
        return v

    # ── Label table ───────────────────────────────────────────────────
    # Enumerate every label in a stable order.
    # label_type: 0=$f, 1=$e, 2=$a or $p
    all_labels: list[str] = []
    label_to_lid: dict[str, int] = {}

    for lbl in parsed.floating_hyps:
        label_to_lid[lbl] = len(all_labels)
        all_labels.append(lbl)
    for lbl in parsed.essential_hyps:
        label_to_lid[lbl] = len(all_labels)
        all_labels.append(lbl)
    for lbl in parsed.assertions:
        label_to_lid[lbl] = len(all_labels)
        all_labels.append(lbl)

    L = len(all_labels)
    lt_arr      = bytearray(L)          # label_types  uint8
    lf_tc_arr   = np.full(L, -1, dtype=np.int32)  # $f type_code symbol id
    lf_var_arr  = np.full(L, -1, dtype=np.int32)  # $f variable symbol id
    le_off_arr  = np.zeros(L + 1, dtype=np.int32) # $e expr CSR offsets
    la_nf_arr   = np.zeros(L, dtype=np.int32)     # $a/$p n_f
    la_ne_arr   = np.zeros(L, dtype=np.int32)     # $a/$p n_e
    le_data: list[int] = []

    for lbl, lid in label_to_lid.items():
        if lbl in parsed.floating_hyps:
            fh = parsed.floating_hyps[lbl]
            lt_arr[lid] = 0
            lf_tc_arr[lid] = sid(fh.type_code)
            lf_var_arr[lid] = sid(fh.variable)
        elif lbl in parsed.essential_hyps:
            eh = parsed.essential_hyps[lbl]
            lt_arr[lid] = 1
            enc = [sid(s) for s in eh.expression]
            le_data.extend(enc)
            le_off_arr[lid + 1] = le_off_arr[lid] + len(enc)
        else:
            a = parsed.assertions[lbl]
            lt_arr[lid] = 2
            la_nf_arr[lid] = len(a.floating_hyps)
            la_ne_arr[lid] = len(a.essential_hyps)

    # Fix up $e CSR offsets: only $e rows were filled in; carry forward others.
    # A simpler approach: just recompute cumsum.
    # le_off_arr[0] = 0, and we only set le_off_arr[lid+1] for $e labels.
    # The others stay 0, which is wrong. Recompute as proper cumsum:
    le_data_arr = np.array(le_data, dtype=np.int32)
    # Rebuild le_off_arr correctly from lengths
    le_len_arr = np.zeros(L, dtype=np.int32)
    for lbl, lid in label_to_lid.items():
        if lbl in parsed.essential_hyps:
            eh = parsed.essential_hyps[lbl]
            le_len_arr[lid] = len(eh.expression)
    le_off_arr2 = np.zeros(L + 1, dtype=np.int32)
    np.cumsum(le_len_arr, out=le_off_arr2[1:])
    le_off_arr = le_off_arr2

    # ── Theorem proofs ────────────────────────────────────────────────
    # For each theorem (in order of theorem_labels), serialise its proof.
    T = len(theorem_labels)
    thm_proof_offsets  = np.zeros(T + 1, dtype=np.int64)
    thm_plabel_offsets = np.zeros(T + 1, dtype=np.int32)
    thm_expr_offsets   = np.zeros(T + 1, dtype=np.int32)
    thm_proof_data: list[int] = []
    thm_plabel_data: list[int] = []
    thm_expr_data: list[int] = []

    for ti, thm_lbl in enumerate(theorem_labels):
        a = parsed.assertions[thm_lbl]
        # expected conclusion
        expr_enc = [sid(s) for s in a.expression]
        thm_expr_data.extend(expr_enc)
        thm_expr_offsets[ti + 1] = thm_expr_offsets[ti] + len(expr_enc)

        if a.compressed_proof is not None:
            cp = a.compressed_proof
            plabels_lids = [label_to_lid[lbl] for lbl in cp.labels]
            thm_plabel_data.extend(plabels_lids)
            thm_plabel_offsets[ti + 1] = thm_plabel_offsets[ti] + len(plabels_lids)
            thm_proof_data.extend(cp.proof_ints)
            thm_proof_offsets[ti + 1] = thm_proof_offsets[ti] + len(cp.proof_ints)
        elif a.proof is not None:
            # Uncompressed: convert to "label_end + 0" format used by Rust.
            # We encode as: proof_int = label_to_lid[step_label], plabels = []
            # The Rust code checks proof_int < label_end; with label_end=0,
            # all proof_ints are >= 0 = label_end, which would be treated as
            # saved refs. So for uncompressed proofs we encode differently:
            # put all step labels as plabels (one per step), proof_ints = [0,1,2,...]
            for step_lbl in a.proof:
                thm_plabel_data.append(label_to_lid[step_lbl])
                thm_proof_data.append(len(thm_plabel_data) - 1 - int(thm_plabel_offsets[ti]))
            thm_plabel_offsets[ti + 1] = thm_plabel_offsets[ti] + len(a.proof)
            thm_proof_offsets[ti + 1]  = thm_proof_offsets[ti]  + len(a.proof)
        else:
            thm_plabel_offsets[ti + 1] = thm_plabel_offsets[ti]
            thm_proof_offsets[ti + 1]  = thm_proof_offsets[ti]

    thm_proof_arr  = np.array(thm_proof_data,  dtype=np.int32)
    thm_plabel_arr = np.array(thm_plabel_data, dtype=np.int32)
    thm_expr_arr   = np.array(thm_expr_data,   dtype=np.int32)

    # Build sym_id → token_id mapping if tokenizer is provided.
    # This allows push_expr_data (symbol IDs) to be converted directly to
    # token IDs in deserialization, bypassing the slow string roundtrip.
    sym_id_to_tok: np.ndarray | None = None
    if tokenizer is not None:
        N_sym = len(sym_to_id)
        sym_id_to_tok = np.empty(N_sym, dtype=np.int16)
        for sym, sid_val in sym_to_id.items():
            sym_id_to_tok[sid_val] = np.int16(tokenizer.encode_symbol(sym))

    return (
        sym_to_id,
        label_to_lid,
        sym_id_to_tok,
        bytes(lt_arr),
        lf_tc_arr.tobytes(),
        lf_var_arr.tobytes(),
        le_off_arr.tobytes(),
        le_data_arr.tobytes(),
        la_nf_arr.tobytes(),
        la_ne_arr.tobytes(),
        T,
        thm_proof_offsets.tobytes(),
        thm_proof_arr.tobytes(),
        thm_plabel_offsets.tobytes(),
        thm_plabel_arr.tobytes(),
        thm_expr_offsets.tobytes(),
        thm_expr_arr.tobytes(),
    )


def _rust_results_to_proof_graphs(
    rust_results: list,
    theorem_labels: list[str],
    parsed: ParsedDatabase,
    sym_to_id: dict[str, int],
    label_to_lid: dict[str, int],
    sym_id_to_tok: "np.ndarray | None" = None,
) -> tuple[list["ProofGraph"], list[tuple[str, str]]]:
    """Convert raw Rust output into ProofGraph objects.

    Batched deserialization: instead of 13 frombuffer calls per graph (611k
    total for set.mm), we concatenate each field's bytes across all theorems
    in one pass, do a single frombuffer+split per field, then reconstruct.
    """
    lid_to_label: dict[int, str] = {v: k for k, v in label_to_lid.items()}
    id_to_sym: dict[int, str] | None = (
        None if sym_id_to_tok is not None
        else {v: k for k, v in sym_to_id.items()}
    )

    # Partition results into successes and errors; track which theorem each maps to
    success_indices: list[int] = []   # theorem index for each successful result
    success_results: list[tuple] = []
    errors: list[tuple[str, str]] = []

    for ti, (thm_lbl, res) in enumerate(zip(theorem_labels, rust_results)):
        if isinstance(res, str):
            errors.append((thm_lbl, res))
        else:
            success_indices.append(ti)
            success_results.append(res)

    if not success_results:
        return [], errors

    # ── Batch-decode all per-theorem scalar fields ────────────────────
    max_levels        = [int(r[0]) for r in success_results]
    max_push_exprlens = [int(r[1]) for r in success_results]
    num_nodes_list    = [int(r[2]) for r in success_results]

    # ── Batch-decode variable-length byte arrays ──────────────────────
    # For each field, concatenate all theorems' bytes, frombuffer once,
    # then split using per-theorem lengths.
    def _batch_decode_i8(field_idx: int) -> list[np.ndarray]:
        cat = b"".join(r[field_idx] for r in success_results)
        arr = np.frombuffer(cat, dtype=np.int8)
        sizes = [len(r[field_idx]) for r in success_results]
        return np.split(arr, np.cumsum(sizes[:-1]))

    def _batch_decode_i32(field_idx: int) -> list[np.ndarray]:
        cat = b"".join(r[field_idx] for r in success_results)
        arr = np.frombuffer(cat, dtype=np.int32)
        sizes = [len(r[field_idx]) // 4 for r in success_results]
        return np.split(arr, np.cumsum(sizes[:-1]))

    node_types_list     = _batch_decode_i8(3)
    node_levels_list    = _batch_decode_i32(4)
    node_label_ids_list = _batch_decode_i32(5)
    input_offsets_list  = _batch_decode_i32(6)
    input_data_list     = _batch_decode_i32(7)
    push_node_idx_list  = _batch_decode_i32(8)
    push_expr_data_list = _batch_decode_i32(9)
    push_expr_off_list  = _batch_decode_i32(10)
    lm_keys_list        = _batch_decode_i32(11)
    lm_vals_list        = _batch_decode_i32(12)

    # ── Convert push_expr_data to token IDs (one global op) ───────────
    if sym_id_to_tok is not None:
        all_push_expr_data = np.concatenate(push_expr_data_list) if push_expr_data_list else np.empty(0, dtype=np.int32)
        all_enc_flat       = sym_id_to_tok[all_push_expr_data]    # int16, one call
        # Split back using per-graph sizes
        ped_sizes = [len(a) for a in push_expr_data_list]
        enc_flat_list: list[np.ndarray] = np.split(all_enc_flat, np.cumsum(ped_sizes[:-1]))
    else:
        enc_flat_list = [None] * len(success_results)  # type: ignore

    # ── Build ProofGraph objects ──────────────────────────────────────
    graphs: list[ProofGraph] = []

    for j, ti in enumerate(success_indices):
        thm_lbl = theorem_labels[ti]
        a = parsed.assertions[thm_lbl]

        node_types      = node_types_list[j].copy()
        node_levels     = node_levels_list[j].copy()
        node_label_ids  = node_label_ids_list[j].copy()
        input_offsets   = input_offsets_list[j].copy()
        input_data_raw  = input_data_list[j].copy()
        push_node_idx   = push_node_idx_list[j].copy()
        push_expr_off   = push_expr_off_list[j].copy()
        lm_keys         = lm_keys_list[j]
        lm_vals         = lm_vals_list[j]

        if sym_id_to_tok is not None:
            g_enc_flat = enc_flat_list[j].copy()
            g_enc_off  = push_expr_off
            push_expr_strings_g: list[list[str]] = []
        else:
            assert id_to_sym is not None
            ped = push_expr_data_list[j]
            peo = push_expr_off
            push_expr_strings_g = []
            for jj in range(len(push_node_idx)):
                s, e = int(peo[jj]), int(peo[jj + 1])
                push_expr_strings_g.append([id_to_sym[int(x)] for x in ped[s:e]])
            g_enc_flat = None
            g_enc_off  = None

        # Rebuild label_id_map
        label_id_map: dict[str, int] = {}
        for gk, lv in zip(lm_keys.tolist(), lm_vals.tolist()):
            label_id_map[lid_to_label[gk]] = lv

        graphs.append(ProofGraph(
            theorem_label=thm_lbl,
            expected_conclusion=a.expression,
            max_level=max_levels[j],
            max_push_expr_len=max_push_exprlens[j],
            num_nodes=num_nodes_list[j],
            node_types=node_types,
            node_levels=node_levels,
            node_label_ids=node_label_ids,
            input_offsets=input_offsets,
            input_data=input_data_raw if len(input_data_raw) > 0
                       else np.empty(0, dtype=np.int32),
            push_node_indices=push_node_idx,
            push_expr_strings=push_expr_strings_g,
            push_enc_flat=g_enc_flat,
            push_enc_offsets=g_enc_off,
            label_id_map=label_id_map,
        ))

    return graphs, errors


def build_all_proof_graphs_rs(
    parsed: ParsedDatabase,
    theorem_labels: list[str],
    tokenizer: "Tokenizer | None" = None,
    verbose: bool = False,
) -> tuple[list["ProofGraph"], list[tuple[str, str]]]:
    """Rust-accelerated proof graph construction (rayon parallel, no subprocess).

    If tokenizer is provided, push expression token IDs are pre-computed and
    stored in ProofGraph.push_enc_flat/push_enc_offsets, bypassing the _enc()
    call in pack_levels (significant speedup for set.mm scale).

    Falls back to the Python implementation if mmgpu_rs is not available.
    """
    if not _HAVE_RUST:
        return build_all_proof_graphs(parsed, theorem_labels, verbose=verbose)

    if not theorem_labels:
        return [], []

    if verbose:
        print(f"  Graph construction (Rust): serialising {len(theorem_labels):,} theorems...",
              flush=True)

    t0 = time.perf_counter() if verbose else 0.0
    serialised = _serialize_db_for_rust(parsed, tokenizer, theorem_labels)
    (sym_to_id, label_to_lid, sym_id_to_tok,
     lt_b, lf_tc_b, lf_var_b, le_off_b, le_data_b, la_nf_b, la_ne_b,
     T,
     proof_off_b, proof_data_b, plabel_off_b, plabel_data_b,
     expr_off_b, expr_data_b) = serialised

    if verbose:
        print(f"  Graph construction (Rust): serialised in {time.perf_counter()-t0:.2f}s, "
              f"calling Rust kernel...", flush=True)

    t1 = time.perf_counter() if verbose else 0.0
    rust_results = _mmgpu_rs.build_graphs(
        lt_b, lf_tc_b, lf_var_b, le_off_b, le_data_b, la_nf_b, la_ne_b,
        T,
        proof_off_b, proof_data_b, plabel_off_b, plabel_data_b,
        expr_off_b, expr_data_b,
    )

    if verbose:
        print(f"  Graph construction (Rust): Rust kernel done in "
              f"{time.perf_counter()-t1:.2f}s, deserialising...", flush=True)

    graphs, errors = _rust_results_to_proof_graphs(
        rust_results, theorem_labels, parsed, sym_to_id, label_to_lid, sym_id_to_tok
    )

    if verbose:
        print(
            f"  Graph construction (Rust): {len(graphs):,} graphs, "
            f"{len(errors):,} errors, total {time.perf_counter()-t0:.2f}s",
            flush=True,
        )
    return graphs, errors


# ── Parallel graph construction ────────────────────────────────────

_GRAPH_WORKER_PARSED: ParsedDatabase | None = None
_GRAPH_WORKER_LABEL_INFO: dict[str, tuple[str, object]] | None = None
_MAX_GRAPH_WORKERS = 512  # no artificial cap — use all available cores


def _init_graph_worker(parsed: ParsedDatabase) -> None:
    global _GRAPH_WORKER_PARSED, _GRAPH_WORKER_LABEL_INFO
    _GRAPH_WORKER_PARSED = parsed
    _GRAPH_WORKER_LABEL_INFO = _build_label_info(parsed)


def _build_graphs_chunk(labels: list[str]) -> list[tuple[str, ProofGraph | str]]:
    parsed = _GRAPH_WORKER_PARSED
    label_info = _GRAPH_WORKER_LABEL_INFO
    assert parsed is not None and label_info is not None
    return [(lbl, build_proof_graph(parsed, lbl, label_info)) for lbl in labels]


def build_all_proof_graphs(
    parsed: ParsedDatabase,
    theorem_labels: list[str],
    max_workers: int | None = None,
    verbose: bool = False,
) -> tuple[list[ProofGraph], list[tuple[str, str]]]:
    """Build proof graphs for all theorems in parallel.

    Returns:
        (graphs, errors) where errors is a list of (label, reason) pairs.
    """
    if not theorem_labels:
        return [], []

    workers = min(max_workers or os.cpu_count() or 1, _MAX_GRAPH_WORKERS)

    # Smaller chunks so workers drain incrementally and RAM doesn't spike
    # from all workers holding a full chunk in memory simultaneously.
    # Target ~4 chunks per worker so the pool stays saturated.
    chunk_size = max(1, len(theorem_labels) // (workers * 4))
    chunks = [
        theorem_labels[i: i + chunk_size]
        for i in range(0, len(theorem_labels), chunk_size)
    ]

    global _GRAPH_WORKER_PARSED, _GRAPH_WORKER_LABEL_INFO
    if sys.platform == "linux":
        # Build label_info once on main process; workers inherit via fork
        # (copy-on-write — no per-worker rebuild cost).
        if verbose:
            print(f"  Graph construction: building label_info...", flush=True)
        _GRAPH_WORKER_PARSED = parsed
        _GRAPH_WORKER_LABEL_INFO = _build_label_info(parsed)
        ctx = multiprocessing.get_context("fork")
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    else:
        # spawn/forkserver: each worker calls _init_graph_worker once,
        # which builds label_info inside the worker (no serialisation of it).
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_graph_worker,
            initargs=(parsed,),
        )

    if verbose:
        print(
            f"  Graph construction: {len(theorem_labels):,} theorems, "
            f"{workers} workers, {len(chunks)} chunks of ~{chunk_size}",
            flush=True,
        )

    graphs: list[ProofGraph] = []
    errors: list[tuple[str, str]] = []  # (label, reason)
    ordered: list[list[tuple[str, ProofGraph | str]]] = [None] * len(chunks)  # type: ignore
    done = 0

    with pool as executor:
        future_to_idx = {
            executor.submit(_build_graphs_chunk, chunk): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            ordered[idx] = future.result()
            done += len(ordered[idx])
            if verbose and done % max(1, len(theorem_labels) // 10) < chunk_size:
                pct = 100 * done / len(theorem_labels)
                print(f"  Graph construction: {done:,}/{len(theorem_labels):,} ({pct:.0f}%)", flush=True)

    for chunk_results in ordered:
        for lbl, result in chunk_results:
            if isinstance(result, str):
                errors.append((lbl, result))
            else:
                graphs.append(result)

    return graphs, errors


# ══════════════════════════════════════════════════════════════════════
#  Phase 2 — Level Packing (CPU → GPU tensors)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class AssertionTable:
    """Deduplicated lookup table of all unique assertions used in a plan.

    Instead of repeating pattern/ehyp data once per proof node (which
    creates multi-GB arrays when B is large), each node stores a single
    integer index into this table.  For set.mm, ~37k unique assertions
    replace ~10M repeated rows.

    Shapes:
        pattern_toks:        [A, P_max]           int32
        pattern_lengths:     [A]                  int32
        fhyp_var_ids:        [A, max_fhyps]       int32
        fhyp_count:          [A]                  int32
        ehyp_patterns:       [A, max_ehyps, E]    int32
        ehyp_pattern_lengths:[A, max_ehyps]       int32
        ehyp_count:          [A]                  int32
    where A = number of unique assertions.
    """
    assertion_labels: list[str]           # [A] label for each row
    label_to_idx: dict[str, int]          # label -> row index

    pattern_toks: np.ndarray              # [A, P_max]
    pattern_lengths: np.ndarray           # [A]

    fhyp_var_ids: np.ndarray              # [A, max_fhyps]
    fhyp_count: np.ndarray               # [A]

    ehyp_patterns: np.ndarray            # [A, max_ehyps, E_max]
    ehyp_pattern_lengths: np.ndarray     # [A, max_ehyps]
    ehyp_count: np.ndarray               # [A]


@dataclass
class AssertionLevelBatch:
    """Packed tensor data for assertion nodes at one or more topological levels.

    Pattern/ehyp data is NOT stored here — instead each node carries an
    `assertion_idx` into the shared AssertionTable.  This reduces memory
    from O(B × pat_len) to O(B) for the per-batch arrays.
    """
    level: int
    max_level: int
    count: int  # B

    node_levels: np.ndarray              # [B] int32

    # Which assertion is being applied (for $d post-check / error reporting)
    assertion_labels: list[str]
    theorem_labels: list[str]

    # Index into AssertionTable for each node
    assertion_idx: np.ndarray            # [B] int32

    # Per-node input mapping
    input_global_indices: np.ndarray     # [B, max_inputs] int32 — -1 for padding
    input_counts: np.ndarray             # [B] int32

    # Floating hyp input positions (which input slot holds each $f value)
    fhyp_input_positions: np.ndarray     # [B, max_fhyps] int32

    # Essential hyp input positions (which input slot holds each $e value)
    ehyp_input_positions: np.ndarray     # [B, max_ehyps] int32

    # Output: where to write in expr_buffer
    output_global_indices: np.ndarray    # [B] int32

    # Pre-computed sub-level ranges for merged batches.
    # List of (start, end) row slices — one per unique topological level.
    # Computed during merge so _execute_level doesn't need np.unique/np.where.
    sublevel_ranges: list[tuple[int, int]] | None = None


@dataclass
class GlobalPlan:
    """Complete execution plan for GPU verification."""
    total_nodes: int
    max_expr_len: int
    num_proofs: int

    # Shared assertion lookup table (deduplicated)
    assertion_table: AssertionTable

    # Level 0 push data
    push_global_indices: np.ndarray    # [num_push] int32
    push_expressions: np.ndarray       # [num_push, push_width] int32
    push_expr_lengths: np.ndarray      # [num_push] int32

    # Assertion levels (sorted by level)
    assertion_batches: list[AssertionLevelBatch]

    # Final check data
    final_node_indices: np.ndarray        # [num_proofs] int32
    expected_conclusions: np.ndarray      # [num_proofs, max_concl_len] int32
    conclusion_lengths: np.ndarray        # [num_proofs] int32
    expected_conclusion_hashes: np.ndarray  # [num_proofs] int64 — polynomial hash

    # For $d post-check: per-proof theorem label
    proof_theorem_labels: list[str]

    # Tokenizer for decoding error messages
    vocab_size: int

    # Per-proof start offset in the global buffer: global_idx of proof pi's
    # first node = graph_offsets[pi].  Length num_proofs+1 (last entry =
    # total_nodes) so that proof pi's node range is [offsets[pi], offsets[pi+1]).
    graph_offsets: np.ndarray  # [num_proofs+1] int64

    # Pre-computed exact expression lengths for every node (packed buffer).
    # Enables a compact 1D expr_buffer instead of padded 2D [N, max_expr_len].
    # None when unavailable (torch fallback still uses padded layout).
    node_expr_lengths: np.ndarray | None = None  # [total_nodes] int32
    total_expr_tokens: int = 0  # sum(node_expr_lengths) — for budget sizing


# ── Numba kernels for Phase 2 array filling ──────────────────────────────────
# These are compiled once (cached by Numba) and run at C speed with no Python
# overhead or GIL contention. All inputs/outputs are flat numpy arrays.

@njit(parallel=True, cache=True)
def _nb_fill_push_expressions(
    push_enc_flat: np.ndarray,      # [total_push_tokens] int16
    push_enc_offsets: np.ndarray,   # [num_push+1] int32
    out_expressions: np.ndarray,    # [num_push, max_push_width] int16 — pre-zeroed
) -> None:
    """Copy each push node's encoded tokens into its row of out_expressions."""
    num_push = len(push_enc_offsets) - 1
    for i in prange(num_push):
        start = push_enc_offsets[i]
        end   = push_enc_offsets[i + 1]
        for k in range(end - start):
            out_expressions[i, k] = push_enc_flat[start + k]


@njit(parallel=True, cache=True)
def _nb_build_flat_push_enc(
    push_uid_arr: np.ndarray,          # [num_push] int32  — unique-expr id per push node
    push_enc_offsets: np.ndarray,      # [num_push+1] int32 — per-node start in flat output
    uid_enc_flat: np.ndarray,          # [total_unique_tokens] int16 — all unique encs concatenated
    uid_enc_offsets: np.ndarray,       # [num_unique+1] int32 — CSR offsets into uid_enc_flat
    out_flat: np.ndarray,              # [total_push_tokens] int16 — output
) -> None:
    """Fill the push flat token buffer in parallel — one thread per push node."""
    num_push = len(push_uid_arr)
    for i in prange(num_push):
        uid   = push_uid_arr[i]
        src_s = uid_enc_offsets[uid]
        src_e = uid_enc_offsets[uid + 1]
        dst_s = push_enc_offsets[i]
        for k in range(src_e - src_s):
            out_flat[dst_s + k] = uid_enc_flat[src_s + k]


@njit(parallel=True, cache=True)
def _nb_pack_assertion_level(
    # Per-node inputs for this level's B nodes
    node_global_indices: np.ndarray,      # [B] int32 — global buffer index for each node
    node_graph_pis: np.ndarray,           # [B] int32 — unused (kept for API compat)
    node_global_assertion_idxs: np.ndarray,  # [B] int32 — index into AssertionTable
    node_input_offsets: np.ndarray,       # [B+1] int32 — CSR offsets into node_input_data
    node_input_data: np.ndarray,          # [total_inputs] int32 — already global indices
    node_n_f: np.ndarray,                 # [B] int32
    node_n_e: np.ndarray,                 # [B] int32
    graph_offsets: np.ndarray,            # [num_graphs+1] int64 — unused (kept for API compat)
    max_inputs: int,
    max_fhyps: int,
    max_ehyps: int,
    # outputs
    out_assertion_idx: np.ndarray,        # [B] int32
    out_input_global_indices: np.ndarray, # [B, max_inputs] int32  (pre-filled with -1)
    out_input_counts: np.ndarray,         # [B] int32
    out_fhyp_input_positions: np.ndarray, # [B, max_fhyps] int32
    out_ehyp_input_positions: np.ndarray, # [B, max_ehyps] int32
    out_output_global_indices: np.ndarray,# [B] int32
) -> None:
    B = len(node_global_indices)
    for b in prange(B):
        out_assertion_idx[b] = node_global_assertion_idxs[b]
        out_output_global_indices[b] = node_global_indices[b]

        inp_start = node_input_offsets[b]
        inp_end   = node_input_offsets[b + 1]
        n_inputs  = inp_end - inp_start
        out_input_counts[b] = n_inputs
        for k in range(n_inputs):
            out_input_global_indices[b, k] = node_input_data[inp_start + k]

        nf = node_n_f[b]
        ne = node_n_e[b]
        for f in range(nf):
            out_fhyp_input_positions[b, f] = f
        for e in range(ne):
            out_ehyp_input_positions[b, e] = nf + e


@njit(cache=True)
def _nb_compute_expr_lengths_batch(
    output_global: np.ndarray,     # [B] int32
    assertion_idxs: np.ndarray,    # [B] int32
    input_global: np.ndarray,      # [B, max_inputs] int32
    fhyp_positions: np.ndarray,    # [B, max_fhyps] int32
    tbl_fhyp_count: np.ndarray,   # [A] int32 — shared, indexed by assertion row
    tbl_const_count: np.ndarray,   # [A] int32
    tbl_var_occ: np.ndarray,       # [A, max_f] int32
    node_expr_lengths: np.ndarray, # [total_nodes] int32 — read prev levels, write this level
) -> None:
    """Compute expression lengths for one level's assertion nodes.

    Called once per level (~300 calls total).  The f-loop runs in compiled
    Numba, eliminating the ~48k numpy temporary allocations that the old
    vectorised approach required.
    """
    B = len(output_global)
    max_in = input_global.shape[1]
    max_f  = tbl_var_occ.shape[1]
    for b in range(B):
        a_idx = assertion_idxs[b]
        nf = tbl_fhyp_count[a_idx]
        out_len = tbl_const_count[a_idx]
        for f in range(nf):
            if f >= max_f:
                break
            fhyp_pos = fhyp_positions[b, f]
            if fhyp_pos < 0:
                fhyp_pos = 0
            if fhyp_pos >= max_in:
                fhyp_pos = max_in - 1
            input_gi = input_global[b, fhyp_pos]
            if input_gi >= 0:
                sub_len = node_expr_lengths[input_gi] - 1
                if sub_len < 0:
                    sub_len = 0
                out_len += tbl_var_occ[a_idx, f] * sub_len
        node_expr_lengths[output_global[b]] = out_len


@njit(parallel=True, cache=True)
def _nb_gather_csr(
    positions: np.ndarray,          # [B] int32 — global node indices for this level
    global_inp_offsets: np.ndarray, # [total_nodes+1] int32
    global_inp_data: np.ndarray,    # [total_inputs] int32 — global input indices
    local_inp_offsets: np.ndarray,  # [B+1] int32 — pre-computed local CSR offsets
    local_inp_data: np.ndarray,     # [total_local_inputs] int32 — output
) -> None:
    """Gather input data for B nodes from global CSR into a local CSR."""
    B = len(positions)
    for b in prange(B):
        g_start = global_inp_offsets[positions[b]]
        g_end   = global_inp_offsets[positions[b] + 1]
        l_start = local_inp_offsets[b]
        for k in range(g_end - g_start):
            local_inp_data[l_start + k] = global_inp_data[g_start + k]


def pack_levels(
    graphs: list[ProofGraph],
    parsed: ParsedDatabase,
    tokenizer: Tokenizer,
    verbose: bool = False,
) -> GlobalPlan:
    """Pack all proof graphs into level-indexed GPU-ready tensors."""
    if verbose:
        print(f"  Phase 2: building label_info ({len(parsed.assertions)} assertions)...", flush=True)
    label_info = _build_label_info(parsed)
    if verbose:
        print(f"  Phase 2: label_info done, assigning global indices...", flush=True)

    # ── Assign global buffer indices ─────────────────────────────────
    # Use a simple offset array instead of a dict.  Nodes within each graph
    # have contiguous step indices (0, 1, 2, …), so:
    #     global_idx = graph_offsets[pi] + node.step_idx
    # This replaces a Python dict with ~200 bytes/entry overhead with a
    # compact numpy int64 array (8 bytes/entry).
    graph_offsets = np.empty(len(graphs) + 1, dtype=np.int64)
    total_nodes = 0
    for pi, g in enumerate(graphs):
        graph_offsets[pi] = total_nodes
        total_nodes += g.num_nodes
    graph_offsets[len(graphs)] = total_nodes
    if verbose:
        print(f"  Phase 2: {total_nodes:,} total nodes, computing max_expr_len...", flush=True)

    # ── Pre-encode every unique expression once ───────────────────────
    # For set.mm, ax-mp appears in ~30k proofs. Without caching,
    # tokenizer.encode_expression(ax-mp.expression) is called 30k times.
    # With caching it's called once.
    _enc_cache: dict[tuple[str, ...], list[int]] = {}

    def _enc(expr: list[str]) -> list[int]:
        key = tuple(expr)
        if key not in _enc_cache:
            _enc_cache[key] = tokenizer.encode_expression(expr)
        return _enc_cache[key]

    # ── Compute max_expr_len for GPU expr_buffer ─────────────────────
    # Must cover all INTERMEDIATE expressions (assertion outputs used as inputs
    # to subsequent steps). If truncated, downstream substitutions are corrupted.
    # Also include expected conclusion lengths so the final written value fits
    # for token comparison — EXCEPT theorems whose conclusion exceeds the cap
    # (e.g. quartfull at 11548 tokens), which fall back to hash comparison.
    #
    # Cap at 1024: the actual max intermediate expression in set.mm is 796
    # tokens (mulsasslem1/2). Giving 28% headroom keeps expr_buffer at
    # 6M × 1024 × 4 = ~24 GB on set.mm, vs 98 GB at 4096 or 278 GB at 11548.
    # Theorems with conclusions > 1024 use rolling-hash comparison instead.
    #
    # Phase 1 already computes max_push_expr_len per graph, so we reuse that
    # instead of re-scanning every node (O(total_nodes) → O(num_graphs)).
    _EXPR_BUF_CAP = 1024
    max_expr_len = 512
    for g in graphs:
        if g.max_push_expr_len > max_expr_len:
            max_expr_len = g.max_push_expr_len
        if len(g.expected_conclusion) > max_expr_len:
            max_expr_len = len(g.expected_conclusion)
    max_expr_len = min(max_expr_len, _EXPR_BUF_CAP)

    # ── Build AssertionTable: one row per unique assertion label ─────
    # Collect all unique assertion labels actually used across all graphs.
    if verbose:
        print(f"  Phase 2: building assertion table...", flush=True)
    # Collect unique assertion labels via each graph's label_id_map (built in Phase 1)
    used_labels: list[str] = []
    seen_labels: set[str] = set()
    for g in graphs:
        for lbl in g.label_id_map:
            if lbl not in seen_labels:
                seen_labels.add(lbl)
                used_labels.append(lbl)

    A = len(used_labels)
    del seen_labels
    label_to_idx: dict[str, int] = {lbl: i for i, lbl in enumerate(used_labels)}

    # Single-pass: compute table dimensions AND pre-encode all data.
    # Avoids iterating used_labels twice (old code had separate dimension
    # scan + data fill loops, causing redundant dict lookups and encoding).
    global_max_pat_len  = 1
    global_max_ehyp_len = 1
    global_max_fhyps    = 1
    global_max_ehyps    = 1
    # Cache assertion data and encodings for second step (array fill).
    _table_cache: list[tuple] = [None] * A  # type: ignore[list-item]
    for i, lbl in enumerate(used_labels):
        a = label_info[lbl][1]
        pat_enc = _enc(a.expression)
        n_f = len(a.floating_hyps)
        n_e = len(a.essential_hyps)
        fhyp_var_ids_list = [
            tokenizer.encode_symbol(parsed.floating_hyps[flbl].variable)
            for flbl in a.floating_hyps
        ]
        ehyp_encs = [_enc(parsed.essential_hyps[elbl].expression) for elbl in a.essential_hyps]
        _table_cache[i] = (pat_enc, n_f, fhyp_var_ids_list, n_e, ehyp_encs)
        if len(pat_enc) > global_max_pat_len:
            global_max_pat_len = len(pat_enc)
        if n_f > global_max_fhyps:
            global_max_fhyps = n_f
        if n_e > global_max_ehyps:
            global_max_ehyps = n_e
        for enc in ehyp_encs:
            if len(enc) > global_max_ehyp_len:
                global_max_ehyp_len = len(enc)

    if verbose:
        print(f"  Phase 2: {A} unique assertions, max_pat={global_max_pat_len}, "
              f"max_ehyp={global_max_ehyp_len}, max_fhyps={global_max_fhyps}, "
              f"max_ehyps={global_max_ehyps}", flush=True)

    # Allocate and fill table arrays in one go using cached data.
    tbl_pattern_toks          = np.zeros((A, global_max_pat_len),                     dtype=np.int16)
    tbl_pattern_lengths       = np.zeros(A,                                            dtype=np.int32)
    tbl_fhyp_var_ids          = np.zeros((A, global_max_fhyps),                       dtype=np.int16)
    tbl_fhyp_count            = np.zeros(A,                                            dtype=np.int32)
    tbl_ehyp_patterns         = np.zeros((A, global_max_ehyps, global_max_ehyp_len),  dtype=np.int16)
    tbl_ehyp_pattern_lengths  = np.zeros((A, global_max_ehyps),                       dtype=np.int32)
    tbl_ehyp_count            = np.zeros(A,                                            dtype=np.int32)

    if verbose:
        sz_gb = (tbl_pattern_toks.nbytes + tbl_ehyp_patterns.nbytes) / 1e9
        print(f"  Phase 2: assertion table size: {sz_gb:.2f} GB", flush=True)

    for i, (pat_enc, n_f, fhyp_var_ids_list, n_e, ehyp_encs) in enumerate(_table_cache):
        pl = len(pat_enc)
        tbl_pattern_lengths[i] = pl
        tbl_pattern_toks[i, :pl] = pat_enc
        tbl_fhyp_count[i] = n_f
        if n_f:
            tbl_fhyp_var_ids[i, :n_f] = fhyp_var_ids_list
        tbl_ehyp_count[i] = n_e
        for e_idx, enc in enumerate(ehyp_encs):
            el = len(enc)
            tbl_ehyp_pattern_lengths[i, e_idx] = el
            tbl_ehyp_patterns[i, e_idx, :el] = enc
    del _table_cache

    assertion_table = AssertionTable(
        assertion_labels=used_labels,
        label_to_idx=label_to_idx,
        pattern_toks=tbl_pattern_toks,
        pattern_lengths=tbl_pattern_lengths,
        fhyp_var_ids=tbl_fhyp_var_ids,
        fhyp_count=tbl_fhyp_count,
        ehyp_patterns=tbl_ehyp_patterns,
        ehyp_pattern_lengths=tbl_ehyp_pattern_lengths,
        ehyp_count=tbl_ehyp_count,
    )
    # Free raw table arrays — now owned by assertion_table
    del tbl_pattern_toks, tbl_pattern_lengths
    del tbl_fhyp_var_ids, tbl_fhyp_count
    del tbl_ehyp_patterns, tbl_ehyp_pattern_lengths, tbl_ehyp_count

    # ── Build global per-node arrays (pure numpy concatenation, no Python loops) ──
    # Concatenate all graphs' arrays into single global arrays indexed by global_idx.
    # This replaces per-node Python scatter loops with numpy bulk operations.

    # Build global assertion_idx mapping: local_label_id → global assertion_table idx
    # for each graph.  Build as flat arrays + offsets so the per-graph dict loop
    # is replaced by a single numpy scatter.
    #
    # Flatten all (local_id, global_id) pairs across all graphs into two arrays,
    # then use numpy fancy-index assignment to fill per-graph lookup arrays at once.
    _ltg_flat_local:  list[np.ndarray] = []
    _ltg_flat_global: list[np.ndarray] = []
    _ltg_graph_off:   list[int] = []   # cumulative local-id offset per graph
    _ltg_sizes:       list[int] = []
    for g in graphs:
        lmap = g.label_id_map
        L = len(lmap)
        _ltg_sizes.append(L)
        if L:
            lbls_arr   = np.array([label_to_idx[lbl]  for lbl in lmap], dtype=np.int32)
            lids_arr   = np.array([lid for lid in lmap.values()],        dtype=np.int32)
            _ltg_flat_global.append(lbls_arr)
            _ltg_flat_local.append(lids_arr)
        else:
            _ltg_flat_global.append(np.empty(0, dtype=np.int32))
            _ltg_flat_local.append(np.empty(0, dtype=np.int32))

    # Build per-graph lookup arrays from the flat data (one numpy op per graph,
    # no Python loop over individual label entries).
    _local_to_global_maps: list[np.ndarray] = []
    for i, L in enumerate(_ltg_sizes):
        ltg = np.empty(max(L, 1), dtype=np.int32)
        if L:
            ltg[_ltg_flat_local[i]] = _ltg_flat_global[i]
        _local_to_global_maps.append(ltg)
    del _ltg_flat_local, _ltg_flat_global, _ltg_graph_off, _ltg_sizes

    # Concatenate node_types, node_levels across all graphs → [total_nodes]
    all_node_types  = np.concatenate([g.node_types  for g in graphs])  # int8
    all_node_levels = np.concatenate([g.node_levels for g in graphs])  # int32

    # Build global assertion_idx array [total_nodes] — only meaningful for assertion nodes
    # node_label_ids[i] is a local id; map through local_to_global per graph.
    all_assert_idxs = np.zeros(total_nodes, dtype=np.int32)
    for pi, g in enumerate(graphs):
        goff = int(graph_offsets[pi])
        mask = g.node_types == NODE_ASSERTION
        if mask.any():
            local_ids = g.node_label_ids[mask]
            global_ids = _local_to_global_maps[pi][local_ids]
            # global positions of assertion nodes in this graph
            node_positions = np.where(mask)[0] + goff
            all_assert_idxs[node_positions] = global_ids
    del _local_to_global_maps

    # Build global CSR input arrays — adjust step indices to global space
    # input_offsets[i] → input_offsets[i+1] gives inputs for global node i
    # We need to shift each graph's input_data by graph_offsets[pi].
    all_inp_off_parts: list[np.ndarray] = []
    all_inp_dat_parts: list[np.ndarray] = []
    running = 0
    for pi, g in enumerate(graphs):
        goff = int(graph_offsets[pi])
        # Offsets: shift by running total (skip last entry — it becomes first of next)
        off = g.input_offsets[:-1] + running
        all_inp_off_parts.append(off)
        running += int(g.input_offsets[-1])
        # Data: shift step indices to global space
        if len(g.input_data) > 0:
            all_inp_dat_parts.append(g.input_data + goff)
    # Final sentinel
    all_input_offsets = np.empty(total_nodes + 1, dtype=np.int32)
    all_input_offsets[:total_nodes] = np.concatenate(all_inp_off_parts) if all_inp_off_parts else np.empty(0, dtype=np.int32)
    all_input_offsets[total_nodes] = running
    all_input_data = np.concatenate(all_inp_dat_parts) if all_inp_dat_parts else np.empty(0, dtype=np.int32)
    del all_inp_off_parts, all_inp_dat_parts

    # Build n_f / n_e per global node (from AssertionTable for assertion nodes)
    _tbl_nf = assertion_table.fhyp_count  # [A] int32
    _tbl_ne = assertion_table.ehyp_count  # [A] int32
    all_nf = np.zeros(total_nodes, dtype=np.int32)
    all_ne = np.zeros(total_nodes, dtype=np.int32)
    assert_mask_global = (all_node_types == NODE_ASSERTION)
    assert_positions   = np.where(assert_mask_global)[0]
    all_nf[assert_positions] = _tbl_nf[all_assert_idxs[assert_positions]]
    all_ne[assert_positions] = _tbl_ne[all_assert_idxs[assert_positions]]

    # Build graph_pi per global node [total_nodes] int32
    all_graph_pis = np.empty(total_nodes, dtype=np.int32)
    for pi, g in enumerate(graphs):
        goff = int(graph_offsets[pi])
        all_graph_pis[goff: goff + g.num_nodes] = pi

    # ── Push nodes: encode expressions and build flat CSR ────────────
    # Fast path: if all graphs have pre-encoded token IDs (from the Rust path
    # with tokenizer), just concatenate them — no _enc() calls needed.
    _have_pre_enc = all(g.push_enc_flat is not None for g in graphs if len(g.push_node_indices) > 0)

    push_step_global: list[int] = []
    for pi, g in enumerate(graphs):
        goff = int(graph_offsets[pi])
        pni  = g.push_node_indices
        if len(pni) == 0:
            continue
        push_step_global.extend((pni + goff).tolist())

    if _have_pre_enc:
        # All push encodings are already token IDs stored in each ProofGraph.
        # Concatenate flat arrays and adjust offsets — no Python encoding loop.
        enc_parts: list[np.ndarray] = []
        off_parts: list[np.ndarray] = []
        running_offset = np.int32(0)
        for g in graphs:
            if len(g.push_node_indices) == 0:
                continue
            enc_parts.append(g.push_enc_flat)
            # Adjust offsets so they are globally consistent
            off_parts.append(g.push_enc_offsets[:-1] + running_offset)
            running_offset += np.int32(len(g.push_enc_flat))
        # Final sentinel
        _flat_push_enc = np.concatenate(enc_parts) if enc_parts else np.empty(0, dtype=np.int16)
        if off_parts:
            push_enc_offsets_arr = np.concatenate(
                off_parts + [np.array([running_offset], dtype=np.int32)]
            )
        else:
            push_enc_offsets_arr = np.zeros(1, dtype=np.int32)
    else:
        # Slow path: collect raw string expressions and encode via _enc().
        all_push_exprs: list[list[str]] = []
        for g in graphs:
            if len(g.push_node_indices) == 0:
                continue
            all_push_exprs.extend(g.push_expr_strings)

        # Deduplicate: encode each unique expression once
        unique_push_exprs: dict[tuple, int] = {}
        push_unique_ids: list[int] = []
        for expr in all_push_exprs:
            key = tuple(expr)
            if key not in unique_push_exprs:
                unique_push_exprs[key] = len(unique_push_exprs)
            push_unique_ids.append(unique_push_exprs[key])

        unique_encs: list[list[int]] = [None] * len(unique_push_exprs)  # type: ignore
        for key, uid in unique_push_exprs.items():
            unique_encs[uid] = _enc(list(key))

        uid_enc_lengths   = np.array([len(e) for e in unique_encs], dtype=np.int32)
        uid_enc_offsets_k = np.empty(len(unique_encs) + 1, dtype=np.int32)
        uid_enc_offsets_k[0] = 0
        np.cumsum(uid_enc_lengths, out=uid_enc_offsets_k[1:])

        push_uid_arr     = np.array(push_unique_ids, dtype=np.int32)
        per_node_lengths = uid_enc_lengths[push_uid_arr]

        push_enc_offsets_arr = np.empty(len(per_node_lengths) + 1, dtype=np.int32)
        push_enc_offsets_arr[0] = 0
        np.cumsum(per_node_lengths, out=push_enc_offsets_arr[1:])
        total_push_toks = int(push_enc_offsets_arr[-1])

        uid_enc_flat = np.empty(int(uid_enc_offsets_k[-1]), dtype=np.int16)
        for uid, enc in enumerate(unique_encs):
            s = int(uid_enc_offsets_k[uid])
            e = int(uid_enc_offsets_k[uid + 1])
            uid_enc_flat[s:e] = enc

        _flat_push_enc = np.empty(total_push_toks, dtype=np.int16)
        if total_push_toks > 0:
            _nb_build_flat_push_enc(
                push_uid_arr, push_enc_offsets_arr,
                uid_enc_flat, uid_enc_offsets_k,
                _flat_push_enc,
            )
        del unique_push_exprs, push_unique_ids, unique_encs
        del push_uid_arr, per_node_lengths, uid_enc_flat, uid_enc_offsets_k, uid_enc_lengths

    # ── Level bucketing: pure numpy, no Python per-node loop ─────────
    # Sort assertion nodes by level using argsort; split into contiguous level groups.
    assert_levels  = all_node_levels[assert_positions]  # [num_assertions]
    sort_order     = np.argsort(assert_levels, kind="stable")
    sorted_assert_positions = assert_positions[sort_order]
    sorted_levels_arr       = assert_levels[sort_order]

    unique_levels, level_counts = np.unique(sorted_levels_arr, return_counts=True)
    sorted_levels = unique_levels.tolist()

    # Split sorted arrays into per-level slices (no copying — just index math)
    level_ends   = np.cumsum(level_counts)
    level_starts = np.concatenate([[0], level_ends[:-1]])

    # Build per-level views using the sorted global positions
    _lvl_positions:      dict[int, np.ndarray] = {}
    _lvl_assert_idxs_np: dict[int, np.ndarray] = {}
    _lvl_nf_np:          dict[int, np.ndarray] = {}
    _lvl_ne_np:          dict[int, np.ndarray] = {}
    _lvl_graph_pis_np:   dict[int, np.ndarray] = {}
    _lvl_labels:         dict[int, list] = {}
    _lvl_theorem_labels: dict[int, list] = {}

    # Convert to numpy object arrays for O(1) fancy-indexed string lookup
    _used_labels_arr = np.empty(len(used_labels), dtype=object)
    for _i, _s in enumerate(used_labels):
        _used_labels_arr[_i] = _s
    _graph_theorem_arr = np.empty(len(graphs), dtype=object)
    for _i, _g in enumerate(graphs):
        _graph_theorem_arr[_i] = _g.theorem_label

    for k, lvl in enumerate(sorted_levels):
        s, e = int(level_starts[k]), int(level_ends[k])
        pos = sorted_assert_positions[s:e]
        _lvl_positions[lvl]      = pos
        _lvl_assert_idxs_np[lvl] = all_assert_idxs[pos]
        _lvl_nf_np[lvl]          = all_nf[pos]
        _lvl_ne_np[lvl]          = all_ne[pos]
        _lvl_graph_pis_np[lvl]   = all_graph_pis[pos]
        # String labels — use numpy fancy indexing instead of Python list comp
        gidxs = _lvl_assert_idxs_np[lvl]
        pis   = _lvl_graph_pis_np[lvl]
        _lvl_labels[lvl]         = _used_labels_arr[gidxs].tolist()
        _lvl_theorem_labels[lvl] = _graph_theorem_arr[pis].tolist()

    del _used_labels_arr, _graph_theorem_arr

    del all_node_types, assert_mask_global, assert_positions
    del sorted_assert_positions, sorted_levels_arr, all_assert_idxs
    del all_nf, all_ne, all_graph_pis

    # ── Pack push nodes ───────────────────────────────────────────────
    num_push = len(push_step_global)
    max_push_width = max((g.max_push_expr_len for g in graphs), default=1)
    if verbose:
        print(f"  Phase 2: packing {num_push:,} push nodes, {len(sorted_levels)} levels, max_expr_len={max_expr_len}...", flush=True)

    push_global_indices = np.array(push_step_global, dtype=np.int32)
    push_expr_lengths   = np.diff(push_enc_offsets_arr).astype(np.int32)
    push_expressions    = np.zeros((num_push, max_push_width), dtype=np.int16)
    if num_push > 0:
        # Fill push_expressions: scatter each encoded token list into its row.
        # Use the Numba kernel for the parallel row-fill (main cost is memory bandwidth).
        _nb_fill_push_expressions(
            _flat_push_enc,
            push_enc_offsets_arr,
            push_expressions,
        )
    del push_step_global, _flat_push_enc, push_enc_offsets_arr

    # ── Pack assertion levels via Numba kernel ────────────────────────

    def _pack_one_level(lvl: int) -> AssertionLevelBatch:
        pos    = _lvl_positions[lvl]       # global node indices [B]
        aidx   = _lvl_assert_idxs_np[lvl]  # [B] int32
        nf_arr = _lvl_nf_np[lvl]           # [B] int32
        ne_arr = _lvl_ne_np[lvl]           # [B] int32
        pi_arr = _lvl_graph_pis_np[lvl]    # [B] int32
        B = len(pos)

        max_inputs = int((nf_arr + ne_arr).max()) if B > 0 else 1
        max_fhyps  = int(nf_arr.max()) if B > 0 else 1
        max_ehyps  = int(ne_arr.max()) if B > 0 else 1
        max_inputs = max(max_inputs, 1)
        max_fhyps  = max(max_fhyps, 1)
        max_ehyps  = max(max_ehyps, 1)

        # Build local CSR via Numba gather kernel (parallel over B nodes)
        inp_counts = (all_input_offsets[pos + 1] - all_input_offsets[pos]).astype(np.int32)
        local_inp_offsets = np.empty(B + 1, dtype=np.int32)
        local_inp_offsets[0] = 0
        np.cumsum(inp_counts, out=local_inp_offsets[1:])
        total_inp = int(local_inp_offsets[-1])
        local_inp_data = np.empty(total_inp, dtype=np.int32)
        if total_inp > 0:
            _nb_gather_csr(
                pos.astype(np.int32),
                all_input_offsets,
                all_input_data,
                local_inp_offsets,
                local_inp_data,
            )

        out_assertion_idx  = np.empty(B, dtype=np.int32)
        out_input_global   = np.full((B, max_inputs), -1, dtype=np.int32)
        out_input_counts   = np.empty(B, dtype=np.int32)
        out_fhyp_positions = np.zeros((B, max_fhyps), dtype=np.int32)
        out_ehyp_positions = np.zeros((B, max_ehyps), dtype=np.int32)
        out_output_global  = np.empty(B, dtype=np.int32)

        _nb_pack_assertion_level(
            pos.astype(np.int32),   # step indices are already global — kernel uses them as-is
            pi_arr, aidx,
            local_inp_offsets, local_inp_data,
            nf_arr, ne_arr,
            graph_offsets,
            max_inputs, max_fhyps, max_ehyps,
            out_assertion_idx, out_input_global, out_input_counts,
            out_fhyp_positions, out_ehyp_positions, out_output_global,
        )

        return AssertionLevelBatch(
            level=lvl,
            max_level=lvl,
            count=B,
            node_levels=np.full(B, lvl, dtype=np.int32),
            assertion_labels=_lvl_labels[lvl],
            theorem_labels=_lvl_theorem_labels[lvl],
            assertion_idx=out_assertion_idx,
            input_global_indices=out_input_global,
            input_counts=out_input_counts,
            fhyp_input_positions=out_fhyp_positions,
            ehyp_input_positions=out_ehyp_positions,
            output_global_indices=out_output_global,
            sublevel_ranges=[(0, B)],
        )

    if verbose:
        print(f"  Phase 2: packing {len(sorted_levels)} levels...", flush=True)
    assertion_batches = [_pack_one_level(lvl) for lvl in sorted_levels]
    del _lvl_positions, _lvl_assert_idxs_np, _lvl_nf_np, _lvl_ne_np
    del _lvl_graph_pis_np, _lvl_labels, _lvl_theorem_labels
    del all_input_offsets, all_input_data, all_node_levels

    # ── Pre-compute exact expression lengths (packed buffer) ─────────
    # The substitution output length depends only on the assertion pattern
    # structure and input expression lengths — NOT on token values.
    # This lets us pre-allocate a tight 1D packed buffer on the GPU.
    #
    # For each assertion, precompute:
    #   constant_count: number of non-variable tokens in pattern
    #   var_occurrences[f]: how many times fhyp f's variable appears
    # Then: output_len = constant_count + sum_f(var_occ[f] * (input_len[f] - 1))
    tbl = assertion_table
    _A = len(tbl.assertion_labels)
    _max_f = tbl.fhyp_var_ids.shape[1] if _A > 0 else 0
    _tbl_const_count = np.zeros(_A, dtype=np.int32)
    _tbl_var_occ = np.zeros((_A, _max_f), dtype=np.int32)
    if _A > 0 and _max_f > 0:
        # Try CUDA kernel first (one thread per assertion, no intermediate alloc).
        # Falls back to numpy broadcast if CUDA is unavailable.
        _cuda_cc, _cuda_vo = _cuda_mod.cuda_compute_assertion_table_stats(
            tbl.pattern_toks,
            tbl.pattern_lengths,
            tbl.fhyp_var_ids,
            tbl.fhyp_count,
            device=torch.device("cuda", 0) if torch.cuda.is_available() else None,
        ) if torch.cuda.is_available() else (None, None)

        if _cuda_cc is not None:
            _tbl_const_count = _cuda_cc
            _tbl_var_occ     = _cuda_vo
        else:
            # CPU fallback: iterate over each fhyp slot (max 40) separately.
            # This avoids the [A, P, F] 3D broadcast which allocates ~1.3 GB.
            # Instead each iteration is [A, P] (~33 MB) — ~40x less peak memory.
            pat_toks  = tbl.pattern_toks.astype(np.int32)   # [A, P]
            var_ids   = tbl.fhyp_var_ids.astype(np.int32)   # [A, F]
            P = pat_toks.shape[1]
            pos_valid = np.arange(P, dtype=np.int32)[None, :] < tbl.pattern_lengths[:, None]
            is_var_any = np.zeros((len(tbl.assertion_labels), P), dtype=bool)
            for f in range(_max_f):
                fhyp_active = (f < tbl.fhyp_count)          # [A] bool
                vid = var_ids[:, f]                          # [A] int32
                matches_f = (pat_toks == vid[:, None]) & pos_valid  # [A, P]
                _tbl_var_occ[:, f] = np.where(
                    fhyp_active,
                    matches_f.sum(axis=1),
                    0,
                ).astype(np.int32)
                is_var_any |= (matches_f & fhyp_active[:, None])
            _tbl_const_count = (pos_valid & ~is_var_any).sum(axis=1).astype(np.int32)
            del pat_toks, var_ids, pos_valid, is_var_any
    elif _A > 0:
        _tbl_const_count = tbl.pattern_lengths.astype(np.int32).copy()

    node_expr_lengths = np.zeros(total_nodes, dtype=np.int32)

    # Push nodes: length is known directly
    node_expr_lengths[push_global_indices] = push_expr_lengths

    # Assertion nodes: one Numba call per level (~300 calls, ~30ms overhead).
    # Each call handles the entire f-loop in compiled code — no Python loop,
    # no numpy temporary allocations.  Levels are processed sequentially to
    # respect data dependencies (level L reads inputs from level L-1).
    _tbl_fhyp_count = tbl.fhyp_count  # [A] — shared across all calls
    for batch in assertion_batches:
        if batch.count > 0:
            _nb_compute_expr_lengths_batch(
                batch.output_global_indices,
                batch.assertion_idx,
                batch.input_global_indices,
                batch.fhyp_input_positions,
                _tbl_fhyp_count,
                _tbl_const_count,
                _tbl_var_occ,
                node_expr_lengths,
            )

    total_expr_tokens = int(node_expr_lengths.sum())
    del _tbl_const_count, _tbl_var_occ

    if verbose:
        avg_len = total_expr_tokens / max(total_nodes, 1)
        padded = total_nodes * max_expr_len
        ratio = total_expr_tokens / max(padded, 1) * 100
        print(f"  Phase 2: packed buffer: {total_expr_tokens:,} tokens "
              f"(avg {avg_len:.1f}/node, {ratio:.1f}% of padded {padded:,})",
              flush=True)

    # ── Final check data ─────────────────────────────────────────────
    # Cap the stored conclusion width to max_expr_len: conclusions longer
    # than the expr_buffer use rolling-hash comparison (not token comparison),
    # so storing their full token sequences wastes memory. For set.mm this
    # avoids a num_proofs × 11548 array (1.3 GB) in favour of
    # num_proofs × 1024 (120 MB).
    num_proofs = len(graphs)
    max_concl_stored = max_expr_len
    final_node_indices         = np.zeros(num_proofs,                    dtype=np.int32)
    expected_conclusions       = np.zeros((num_proofs, max_concl_stored), dtype=np.int16)
    conclusion_lengths         = np.zeros(num_proofs,                    dtype=np.int32)
    expected_conclusion_hashes = np.zeros(num_proofs,                    dtype=np.int64)
    proof_theorem_labels: list[str] = []

    for pi, g in enumerate(graphs):
        proof_theorem_labels.append(g.theorem_label)
        final_node_indices[pi] = int(graph_offsets[pi]) + (g.num_nodes - 1)
        enc = np.array(_enc(g.expected_conclusion), dtype=np.int16)
        conclusion_lengths[pi] = len(enc)
        store_len = min(len(enc), max_concl_stored)
        expected_conclusions[pi, :store_len] = enc[:store_len]
        expected_conclusion_hashes[pi] = _poly_hash_np(enc)
    # Free encoding cache and label_info — no longer needed after this point
    del _enc_cache, label_info

    return GlobalPlan(
        total_nodes=total_nodes,
        max_expr_len=max_expr_len,
        num_proofs=num_proofs,
        assertion_table=assertion_table,
        push_global_indices=push_global_indices,
        push_expressions=push_expressions,
        push_expr_lengths=push_expr_lengths,
        assertion_batches=assertion_batches,
        final_node_indices=final_node_indices,
        expected_conclusions=expected_conclusions,
        conclusion_lengths=conclusion_lengths,
        expected_conclusion_hashes=expected_conclusion_hashes,
        proof_theorem_labels=proof_theorem_labels,
        vocab_size=tokenizer.vocab_size(),
        graph_offsets=graph_offsets,
        node_expr_lengths=node_expr_lengths,
        total_expr_tokens=total_expr_tokens,
    )


# ══════════════════════════════════════════════════════════════════════
#  Phase 3 — GPU Execution (level by level)
# ══════════════════════════════════════════════════════════════════════

# Polynomial rolling hash base (large prime, fits int64 without overflow issues
# when multiplied by token ids up to ~100k and added).
_HASH_BASE = np.int64(1_000_000_007)


_HASH_MASK = (1 << 63) - 1  # keep within signed int64 range


def _poly_hash_np(tokens: np.ndarray) -> np.int64:
    """Compute polynomial rolling hash of a 1-D token array on CPU.

    Tokens may be int16 (wrapping values > 32767 to negative). We interpret
    them as unsigned (matching CUDA's ``unsigned short``) by masking with
    0xFFFF before adding to the hash.

    Uses Python int arithmetic (arbitrary precision) then masks to int64 so
    the result matches the GPU int64 wrap-around behaviour exactly.
    """
    base = int(_HASH_BASE)
    is_i16 = tokens.dtype == np.int16
    h = 0
    for t in tokens:
        v = int(t) & 0xFFFF if is_i16 else int(t)
        h = (h * base + v) & 0xFFFFFFFFFFFFFFFF
    # Reinterpret as signed int64
    if h >= (1 << 63):
        h -= (1 << 64)
    return np.int64(h)


def _poly_hash_gpu(
    tokens: torch.Tensor,       # [B, L] int16 — possibly padded
    lengths: torch.Tensor,      # [B] int32
    device: torch.device,
) -> torch.Tensor:              # [B] int64
    """Compute polynomial rolling hash of variable-length token rows on GPU.

    Tokens are stored as int16 (wrapping values > 32767 to negative).
    We mask with 0xFFFF to recover unsigned values, matching the CUDA
    kernel's ``unsigned short`` interpretation.
    """
    B, L = tokens.shape
    # Only iterate up to the actual longest row — skip trailing padding.
    # For set.mm with max_expr_len=1024 but typical lengths ~100-200,
    # this saves 4-5x iterations on average.
    actual_L = min(L, int(lengths.max().item())) if B > 0 else 0
    h = torch.zeros(B, dtype=torch.int64, device=device)
    if actual_L == 0:
        return h
    base = torch.tensor(_HASH_BASE, dtype=torch.int64, device=device)
    # Interpret as unsigned uint16 via mask, then promote to int64
    tokens_long = (tokens[:, :actual_L].int() & 0xFFFF).long()
    valid = torch.arange(actual_L, device=device).unsqueeze(0) < lengths.long().unsqueeze(1)
    for i in range(actual_L):
        h = torch.where(valid[:, i], h * base + tokens_long[:, i], h)
    return h


def _apply_substitution_compact(
    patterns: torch.Tensor,         # [B, P_max] int32
    pattern_lengths: torch.Tensor,  # [B] int32
    var_ids: torch.Tensor,          # [B, max_fhyps] int64 — variable token IDs
    var_sub_values: torch.Tensor,   # [B, max_fhyps, S_max] int32 — replacement tokens
    var_sub_lengths: torch.Tensor,  # [B, max_fhyps] int32 — replacement lengths
    var_valid: torch.Tensor,        # [B, max_fhyps] bool — which fhyps are active
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply substitution to patterns using compact variable representation.

    Instead of a full [B, V, S_max] substitution table (which for set.mm
    allocates ~2-6 GB per chunk with V≈30k), this uses only the substituted
    variable entries [B, max_fhyps, S_max] (~40 MB with max_fhyps≈20).

    Non-variable tokens are identity-mapped: output = input token, length 1.
    Variable tokens are looked up in the compact table via broadcasting match.

    Returns:
        output: [B, max_output_len] int16 on device
        output_lengths: [B] int32 on device
    """
    B = patterns.shape[0]
    P_max = patterns.shape[1]
    max_fhyps = var_ids.shape[1]
    S_max = var_sub_values.shape[2]

    if B == 0:
        return (torch.empty(0, 0, dtype=torch.int16, device=device),
                torch.empty(0, dtype=torch.int32, device=device))

    # STEP 1: Match pattern tokens against variable IDs.
    # Avoid materializing [B, P_max, max_fhyps] (can be 16+ GB when B=cB*E).
    # Instead loop over max_fhyps (≤20) and accumulate [B, P_max] tensors.
    pats_long = patterns.long()  # [B, P_max]
    has_var = torch.zeros(B, P_max, dtype=torch.bool, device=device)
    var_idx = torch.zeros(B, P_max, dtype=torch.long, device=device)
    for f in range(max_fhyps):
        if not var_valid[:, f].any():
            continue
        match_f = (pats_long == var_ids[:, f].unsqueeze(1)) & var_valid[:, f].unsqueeze(1)  # [B, P_max]
        # First matching fhyp wins (same semantics as argmax on first True)
        new_match = match_f & ~has_var
        var_idx = torch.where(new_match, torch.full_like(var_idx, f), var_idx)
        has_var = has_var | match_f

    # STEP 2: Compute per-position replacement lengths.
    b_range_P = torch.arange(B, device=device).unsqueeze(1).expand(B, P_max)
    var_rep_len = var_sub_lengths[b_range_P, var_idx]  # [B, P_max]

    pos_range = torch.arange(P_max, device=device).unsqueeze(0)
    pat_valid = pos_range < pattern_lengths.unsqueeze(1)

    # Variables: their substitution length. Constants: 1. Padding: 0.
    token_lengths = torch.where(has_var & pat_valid, var_rep_len,
                                pat_valid.int())

    # STEP 3: PREFIX SUM
    offsets = torch.cumsum(token_lengths, dim=1) - token_lengths
    total_lengths = token_lengths.sum(dim=1)
    max_output_len = int(total_lengths.max().item()) if B > 0 else 0

    if max_output_len == 0:
        return (torch.zeros(B, 1, dtype=torch.int16, device=device),
                torch.zeros(B, dtype=torch.int32, device=device))

    # STEP 4: SCATTER — fully vectorized (no Python position loop).
    # Old code looped `for p in range(actual_P_max)` which for patterns
    # of length ~200 meant 200 Python iterations × 3-4 GPU kernel launches
    # each = ~600-800 kernel launches. Now replaced with two vectorized
    # scatter operations (constants + variables).
    output = torch.zeros(B, max_output_len, dtype=torch.int16, device=device)
    step_base = torch.arange(B, device=device, dtype=torch.long) * max_output_len
    output_flat = output.view(-1)

    actual_P_max = int(pattern_lengths.max().item())
    # Trim to actual width
    pat_valid_a = pat_valid[:, :actual_P_max]
    has_var_a = has_var[:, :actual_P_max]
    offsets_a = offsets[:, :actual_P_max]
    patterns_a = patterns[:, :actual_P_max]
    var_idx_a = var_idx[:, :actual_P_max]
    token_lengths_a = token_lengths[:, :actual_P_max]

    # (a) Constants: one token per position — single vectorized scatter.
    const_mask = ~has_var_a & pat_valid_a  # [B, actual_P_max]
    if const_mask.any():
        const_flat_idx = (step_base.unsqueeze(1) + offsets_a.long())[const_mask]
        output_flat[const_flat_idx] = patterns_a[const_mask]

    # (b) Variables: multi-token scatter — single vectorized operation.
    var_mask = has_var_a & pat_valid_a  # [B, actual_P_max]
    if var_mask.any():
        var_b, var_p = torch.where(var_mask)  # [num_vars] each
        var_match_f = var_idx_a[var_b, var_p]  # which fhyp
        var_lens = token_lengths_a[var_b, var_p]  # replacement lengths
        var_offs = offsets_a[var_b, var_p]  # output offsets

        var_max_len = int(var_lens.max().item())
        if var_max_len > 0:
            s_range_v = torch.arange(var_max_len, device=device)
            s_valid = s_range_v.unsqueeze(0) < var_lens.unsqueeze(1)  # [num_vars, var_max_len]
            flat_var_idx = (step_base[var_b].unsqueeze(1)
                           + var_offs.unsqueeze(1).long()
                           + s_range_v.long().unsqueeze(0))
            flat_var_vals = var_sub_values[var_b, var_match_f, :var_max_len]

            s_valid_flat = s_valid.reshape(-1)
            if s_valid_flat.any():
                output_flat[flat_var_idx.reshape(-1)[s_valid_flat]] = \
                    flat_var_vals.reshape(-1)[s_valid_flat]

    return output, total_lengths


def _verify_substitution_result(
    output: torch.Tensor,           # [B, max_out_len] int32
    output_lengths: torch.Tensor,   # [B] int32
    target: torch.Tensor,           # [B, T_max] int32
    target_lengths: torch.Tensor,   # [B] int32
    device: torch.device,
) -> torch.Tensor:
    """Compare substitution output against target. Returns [B] bool."""
    B = output.shape[0]
    if B == 0:
        return torch.empty(0, dtype=torch.bool, device=device)

    max_out = output.shape[1]
    T_max = target.shape[1]
    compare_dim = max(max_out, T_max)

    length_match = (output_lengths == target_lengths)

    if max_out < compare_dim:
        output = torch.nn.functional.pad(output, (0, compare_dim - max_out))
    if T_max < compare_dim:
        target = torch.nn.functional.pad(target, (0, compare_dim - T_max))

    positions = torch.arange(compare_dim, device=device).unsqueeze(0)
    valid_mask = positions < output_lengths.unsqueeze(1)
    masked_eq = (output == target) | ~valid_mask
    content_match = masked_eq.all(dim=1)

    return length_match & content_match


def _merge_sparse_levels(
    batches: list[AssertionLevelBatch],
    min_batch_size: int = 512,
) -> list[AssertionLevelBatch]:
    """Merge consecutive sparse level batches to improve GPU utilisation.

    When consecutive levels each have fewer than `min_batch_size` nodes, they
    are concatenated into a single `AssertionLevelBatch`.  The merged batch
    carries a `node_levels` array so `_execute_level` can still process
    sub-groups in strict topological order (required for correctness: a level-L
    output must be written to `expr_buffer` before a level-(L+1) node reads it).

    Large batches (count >= min_batch_size) are left untouched.
    """
    if not batches:
        return batches

    merged: list[AssertionLevelBatch] = []
    pending: list[AssertionLevelBatch] = []
    pending_count = 0

    def _flush_pending() -> None:
        nonlocal pending, pending_count
        if not pending:
            return
        if len(pending) == 1:
            merged.append(pending[0])
        else:
            m_max_inputs = max(b.input_global_indices.shape[1] for b in pending)
            m_max_fhyps  = max(b.fhyp_input_positions.shape[1] for b in pending)
            m_max_ehyps  = max(b.ehyp_input_positions.shape[1] for b in pending)
            B_total = sum(b.count for b in pending)

            # Pre-allocate final arrays and fill in one pass instead of
            # creating padded copies per batch then concatenating.
            m_input_global  = np.full((B_total, m_max_inputs), -1, dtype=np.int32)
            m_node_levels   = np.empty(B_total, dtype=np.int32)
            m_assertion_idx = np.empty(B_total, dtype=np.int32)
            m_input_counts  = np.empty(B_total, dtype=np.int32)
            m_fhyp_pos      = np.zeros((B_total, m_max_fhyps), dtype=np.int32)
            m_ehyp_pos      = np.zeros((B_total, m_max_ehyps), dtype=np.int32)
            m_output_global = np.empty(B_total, dtype=np.int32)
            m_assertion_labels: list[str] = []
            m_theorem_labels: list[str] = []

            # Pre-compute sublevel_ranges during merge so _execute_level
            # doesn't need np.unique + np.where at runtime.
            m_sublevel_ranges: list[tuple[int, int]] = []
            row = 0
            for b in pending:
                nr = b.count
                r = slice(row, row + nr)
                m_sublevel_ranges.append((row, row + nr))
                m_node_levels[r]   = b.node_levels
                m_assertion_idx[r] = b.assertion_idx
                m_input_counts[r]  = b.input_counts
                m_output_global[r] = b.output_global_indices
                w_in = b.input_global_indices.shape[1]
                m_input_global[r, :w_in] = b.input_global_indices
                w_f = b.fhyp_input_positions.shape[1]
                m_fhyp_pos[r, :w_f] = b.fhyp_input_positions
                w_e = b.ehyp_input_positions.shape[1]
                m_ehyp_pos[r, :w_e] = b.ehyp_input_positions
                m_assertion_labels.extend(b.assertion_labels)
                m_theorem_labels.extend(b.theorem_labels)
                row += nr

            merged.append(AssertionLevelBatch(
                level=pending[0].level,
                max_level=pending[-1].max_level,
                count=B_total,
                node_levels=m_node_levels,
                assertion_labels=m_assertion_labels,
                theorem_labels=m_theorem_labels,
                assertion_idx=m_assertion_idx,
                input_global_indices=m_input_global,
                input_counts=m_input_counts,
                fhyp_input_positions=m_fhyp_pos,
                ehyp_input_positions=m_ehyp_pos,
                output_global_indices=m_output_global,
                sublevel_ranges=m_sublevel_ranges,
            ))
        pending = []
        pending_count = 0

    for batch in batches:
        if batch.count >= min_batch_size:
            # Large batch — flush any pending small ones first, then emit as-is
            _flush_pending()
            merged.append(batch)
        else:
            pending.append(batch)
            pending_count += batch.count
            if pending_count >= min_batch_size:
                _flush_pending()

    _flush_pending()
    return merged


def _execute_level(
    batch: AssertionLevelBatch,
    expr_buffer: torch.Tensor,      # [total_nodes, max_expr_len] int32 on device
    expr_lengths: torch.Tensor,     # [total_nodes] int32 on device
    expr_hashes: torch.Tensor,      # [total_nodes] int64 on device
    node_failed: torch.Tensor,      # [total_nodes] bool on device
    vocab_size: int,
    device: torch.device,
    # Assertion table tensors (already on device, shared across all levels)
    tbl_pattern_toks: torch.Tensor,          # [A, P]
    tbl_pattern_lengths: torch.Tensor,       # [A]
    tbl_fhyp_var_ids: torch.Tensor,          # [A, max_fhyps]
    tbl_fhyp_count: torch.Tensor,            # [A]
    tbl_ehyp_patterns: torch.Tensor,         # [A, max_ehyps, E]
    tbl_ehyp_pattern_lengths: torch.Tensor,  # [A, max_ehyps]
    tbl_ehyp_count: torch.Tensor,            # [A]
    max_chunk_B: int = 10_000,
) -> None:
    """Execute assertion nodes for one (possibly coalesced) level group.

    Pattern and ehyp data are gathered from the shared assertion table
    (one row per unique assertion, already on device) rather than being
    stored redundantly per node.  Per-batch arrays are only the small
    node-specific fields: assertion_idx, input_global_indices,
    fhyp_input_positions, ehyp_input_positions, output_global_indices.

    Stores a polynomial rolling hash of each conclusion in expr_hashes so
    that expressions longer than max_expr_len can still be compared correctly
    at the final check step without ever allocating a buffer for them.
    """
    B = batch.count
    if B == 0:
        return

    # ── Upload slim per-batch arrays ─────────────────────────────────
    use_pinned = device.type == "cuda"
    def _to(arr: np.ndarray, long: bool = False) -> torch.Tensor:
        t = torch.from_numpy(arr)
        if use_pinned:
            t = t.pin_memory()
        t = t.to(device, non_blocking=use_pinned)
        return t.long() if long else t

    asrt_idx_t     = _to(batch.assertion_idx, long=True)           # [B]
    input_global_t = _to(batch.input_global_indices, long=True)   # [B, max_inputs]
    input_counts_t = _to(batch.input_counts)                      # [B]
    output_global_t = _to(batch.output_global_indices, long=True) # [B]

    # Position arrays may be narrower than the table's max_fhyps/max_ehyps
    # (e.g. when batches with few hyps are merged). Pad with 0 so that
    # out-of-range positions are masked away by fhyp_valid/ehyp_valid.
    tbl_max_fhyps = tbl_fhyp_var_ids.shape[1]
    tbl_max_ehyps = tbl_ehyp_patterns.shape[1]
    fhyp_pos_np = batch.fhyp_input_positions
    ehyp_pos_np = batch.ehyp_input_positions
    if fhyp_pos_np.shape[1] < tbl_max_fhyps:
        fhyp_pos_np = np.pad(fhyp_pos_np, ((0, 0), (0, tbl_max_fhyps - fhyp_pos_np.shape[1])))
    if ehyp_pos_np.shape[1] < tbl_max_ehyps:
        ehyp_pos_np = np.pad(ehyp_pos_np, ((0, 0), (0, tbl_max_ehyps - ehyp_pos_np.shape[1])))
    fhyp_pos_t = _to(fhyp_pos_np, long=True)  # [B, tbl_max_fhyps]
    ehyp_pos_t = _to(ehyp_pos_np, long=True)  # [B, tbl_max_ehyps]

    # ── Gather per-node assertion data from table ─────────────────────
    # [B, P], [B], [B, max_fhyps], [B], [B, max_ehyps, E], [B, max_ehyps], [B]
    pattern_toks_t    = tbl_pattern_toks[asrt_idx_t]
    pattern_lengths_t = tbl_pattern_lengths[asrt_idx_t]
    fhyp_var_t        = tbl_fhyp_var_ids[asrt_idx_t]
    fhyp_count_t      = tbl_fhyp_count[asrt_idx_t]
    ehyp_pats_t       = tbl_ehyp_patterns[asrt_idx_t]
    ehyp_pat_lens_t   = tbl_ehyp_pattern_lengths[asrt_idx_t]
    ehyp_count_t      = tbl_ehyp_count[asrt_idx_t]

    max_inputs   = input_global_t.shape[1]
    # fhyp/ehyp dims must come from the table-gathered tensors (not the batch
    # position arrays) because merged batches may have fewer position columns
    # than the table's max — using batch dims would cause shape mismatches.
    max_fhyps    = tbl_fhyp_var_ids.shape[1]
    max_ehyps    = tbl_ehyp_patterns.shape[1]
    max_expr_len = expr_buffer.shape[1]

    # ── Sub-level ranges for merged batches ───────────────────────────
    # Use pre-computed ranges (set during pack or merge) to avoid
    # numpy.unique + numpy.where overhead at execution time.
    if batch.sublevel_ranges is not None:
        sublevel_ranges = batch.sublevel_ranges
    elif batch.level == batch.max_level:
        sublevel_ranges: list[tuple[int, int]] = [(0, B)]
    else:
        node_levels_np = batch.node_levels
        unique_lvls = np.unique(node_levels_np)
        sublevel_ranges = []
        for lv in unique_lvls:
            idxs = np.where(node_levels_np == lv)[0]
            sublevel_ranges.append((int(idxs[0]), int(idxs[-1]) + 1))

    # Pre-clamp input indices once for the entire batch instead of per-chunk.
    # Input indices use -1 as padding sentinel; clamping to 0 makes gathers safe.
    safe_input_global_t = input_global_t.clamp(min=0)
    # Pre-clamp fhyp/ehyp positions
    safe_fhyp_pos_t = fhyp_pos_t.clamp(min=0, max=max_inputs - 1)
    safe_ehyp_pos_t = ehyp_pos_t.clamp(min=0, max=max_inputs - 1)

    for sl_start, sl_end in sublevel_ranges:
        sl_size = sl_end - sl_start

        for chunk_offset in range(0, sl_size, max_chunk_B):
            chunk_start = sl_start + chunk_offset
            chunk_end   = min(chunk_start + max_chunk_B, sl_end)
            cB = chunk_end - chunk_start
            cs = slice(chunk_start, chunk_end)

            c_pat         = pattern_toks_t[cs]
            c_pat_len     = pattern_lengths_t[cs]
            c_in_count    = input_counts_t[cs]
            c_fhyp_var    = fhyp_var_t[cs]
            c_fhyp_count  = fhyp_count_t[cs]
            c_ehyp_pats   = ehyp_pats_t[cs]
            c_ehyp_pat_lens = ehyp_pat_lens_t[cs]
            c_ehyp_count  = ehyp_count_t[cs]
            c_out_idx     = output_global_t[cs]

            # ── (a) Gather inputs ─────────────────────────────────────
            safe_in_idx   = safe_input_global_t[cs]
            inputs        = expr_buffer[safe_in_idx]       # [cB, max_inputs, max_expr_len]
            input_lens    = expr_lengths[safe_in_idx]      # [cB, max_inputs]
            input_failed  = node_failed[safe_in_idx]       # [cB, max_inputs]
            in_valid      = torch.arange(max_inputs, device=device) < c_in_count.unsqueeze(1)
            any_input_failed = (input_failed & in_valid).any(dim=1)

            # ── (b) Build compact substitution ────────────────────────
            # Instead of a full [cB, V, S_max] table (V≈30k for set.mm
            # → multi-GB allocation), build only the variable entries:
            #   var_sub_values:  [cB, max_fhyps, S_max]
            #   var_sub_lengths: [cB, max_fhyps]
            # Memory: cB × max_fhyps × S_max × 4 ≈ 40 MB vs multi-GB.
            fhyp_valid    = torch.arange(max_fhyps, device=device) < c_fhyp_count.unsqueeze(1)
            safe_fhyp_pos = safe_fhyp_pos_t[cs]
            batch_range   = torch.arange(cB, device=device).unsqueeze(1).expand(cB, max_fhyps)
            fhyp_input_exprs = inputs[batch_range, safe_fhyp_pos]   # [cB, max_fhyps, max_expr_len]
            fhyp_input_lens  = input_lens[batch_range, safe_fhyp_pos]

            var_sub_values = fhyp_input_exprs[:, :, 1:]
            var_sub_lengths = (fhyp_input_lens - 1).clamp(min=0)
            var_sub_lengths = var_sub_lengths.clamp(max=var_sub_values.shape[2]).int()
            S_max          = max(int(var_sub_lengths.max().item()), 1) if cB > 0 else 1
            var_sub_values = var_sub_values[:, :, :S_max]

            # ── (c) Check essential hypotheses ────────────────────────
            # Batch ALL ehyps into a single substitution call instead of
            # looping `for e in range(max_ehyps_val)`. This eliminates
            # max_ehyps separate calls to _apply_substitution_compact
            # (typically 2-10 calls), each with Python + GPU kernel overhead.
            max_ehyps_val = int(c_ehyp_count.max().item()) if cB > 0 else 0
            if max_ehyps_val == 0:
                all_ehyps_ok = torch.ones(cB, dtype=torch.bool, device=device)
            else:
                E = max_ehyps_val
                arange_cB = torch.arange(cB, device=device)

                # Stack all ehyps: [cB, E, ehyp_P] → [cB*E, ehyp_P]
                ehyp_pats_flat = c_ehyp_pats[:, :E, :].reshape(cB * E, -1)
                ehyp_plens_flat = c_ehyp_pat_lens[:, :E].reshape(cB * E)

                # Replicate var data for all ehyps: [cB, F, S] → [cB*E, F, S]
                var_ids_rep = c_fhyp_var.unsqueeze(1).expand(cB, E, max_fhyps).reshape(cB * E, max_fhyps)
                var_vals_rep = var_sub_values.unsqueeze(1).expand(cB, E, max_fhyps, S_max).reshape(cB * E, max_fhyps, S_max)
                var_lens_rep = var_sub_lengths.unsqueeze(1).expand(cB, E, max_fhyps).reshape(cB * E, max_fhyps)
                fhyp_valid_rep = fhyp_valid.unsqueeze(1).expand(cB, E, max_fhyps).reshape(cB * E, max_fhyps)

                # Single batched substitution call
                ehyp_out_flat, ehyp_out_len_flat = _apply_substitution_compact(
                    ehyp_pats_flat, ehyp_plens_flat,
                    var_ids_rep, var_vals_rep, var_lens_rep, fhyp_valid_rep,
                    device,
                )

                # Gather targets for all ehyps at once
                safe_ep = safe_ehyp_pos_t[cs][:, :E]  # [cB, E]
                b_range_E = arange_cB.unsqueeze(1).expand(cB, E)
                ehyp_targets = inputs[b_range_E, safe_ep]  # [cB, E, max_expr_len]
                ehyp_target_lens = input_lens[b_range_E, safe_ep]  # [cB, E]
                ehyp_targets_flat = ehyp_targets.reshape(cB * E, -1)
                ehyp_target_lens_flat = ehyp_target_lens.reshape(cB * E)

                # Single batched verification
                ehyp_match_flat = _verify_substitution_result(
                    ehyp_out_flat, ehyp_out_len_flat,
                    ehyp_targets_flat, ehyp_target_lens_flat,
                    device,
                )

                # Reshape and mask invalid ehyps
                ehyp_results = ehyp_match_flat.reshape(cB, E)
                ehyp_valid_mask = torch.arange(E, device=device) < c_ehyp_count.unsqueeze(1)
                all_ehyps_ok = (ehyp_results | ~ehyp_valid_mask).all(dim=1)

            # ── (d) Compute conclusion ────────────────────────────────
            concl_output, concl_out_len = _apply_substitution_compact(
                c_pat, c_pat_len,
                c_fhyp_var, var_sub_values, var_sub_lengths, fhyp_valid,
                device,
            )
            max_out = concl_output.shape[1]

            write_len = min(max_out, max_expr_len)
            expr_buffer[c_out_idx] = 0
            expr_buffer[c_out_idx, :write_len] = concl_output[:, :write_len]
            expr_lengths[c_out_idx] = concl_out_len.int()

            # Compute rolling hash of the full output (not truncated) so that
            # expressions longer than max_expr_len can still be compared correctly.
            expr_hashes[c_out_idx] = _poly_hash_gpu(concl_output, concl_out_len, device)

            node_failed[c_out_idx] = ~all_ehyps_ok | any_input_failed

    return


def _run_gpu_pipeline_cuda(
    plan: GlobalPlan,
    device: torch.device,
    max_expr_len: int,
    verbose: bool = False,
) -> tuple[np.ndarray, bool, int]:
    """CUDA kernel path — packed 1D expr_buffer, zero intermediate allocations.

    Uses pre-computed node_expr_lengths to build a compact 1D packed buffer
    instead of the padded 2D [total_nodes, max_expr_len] layout.  Each node
    gets exactly the capacity it needs (from the CPU pre-pass), so there is
    never any intermediate truncation.

    Returns:
        (per_proof_passed, had_intermediate_truncation, max_intermediate_len)
    """
    total_nodes = plan.total_nodes

    if total_nodes == 0:
        return np.ones(plan.num_proofs, dtype=np.bool_), False, 0

    # ── Compute offsets from pre-pass lengths ─────────────────────
    assert plan.node_expr_lengths is not None, "packed buffer requires node_expr_lengths"
    offsets_np = np.empty(total_nodes + 1, dtype=np.int64)
    offsets_np[0] = 0
    np.cumsum(plan.node_expr_lengths, out=offsets_np[1:])
    total_tokens = int(offsets_np[total_nodes])

    # ── Allocate GPU buffers ──────────────────────────────────────
    # expr_buffer is now 1D packed: each node's tokens are at
    # expr_buffer[offsets[gi] : offsets[gi+1]]
    expr_buffer  = torch.empty(max(total_tokens, 1), dtype=torch.int16, device=device)
    expr_lengths = torch.zeros(total_nodes, dtype=torch.int32, device=device)
    expr_hashes  = torch.zeros(total_nodes, dtype=torch.int64, device=device)
    # int8 failure codes: 0=ok, 1=input_propagated, 2=ehyp_mismatch, 3=conclusion_overflow
    node_fail_code = torch.zeros(total_nodes, dtype=torch.int8, device=device)
    expr_offsets = torch.from_numpy(offsets_np).to(device)

    if verbose:
        packed_mb = total_tokens * 2 / 1024**2
        padded_mb = total_nodes * max_expr_len * 2 / 1024**2
        print(f"    Packed buffer: {packed_mb:.1f} MB "
              f"(vs {padded_mb:.1f} MB padded, "
              f"{packed_mb / max(padded_mb, 0.001) * 100:.1f}%)", flush=True)

    # ── Level 0: push nodes via CUDA kernel ───────────────────────
    if len(plan.push_global_indices) > 0:
        _cuda_mod.cuda_push_nodes(
            plan.push_global_indices,
            plan.push_expressions,
            plan.push_expr_lengths,
            expr_buffer, expr_lengths, expr_hashes,
            expr_offsets,
            device,
        )
        torch.cuda.synchronize(device)

    if verbose:
        print(f"    Level 0: {len(plan.push_global_indices)} push nodes written (CUDA)")

    # ── Upload assertion table once ───────────────────────────────
    tbl = plan.assertion_table
    tbl_pattern_toks_t         = torch.from_numpy(tbl.pattern_toks).to(device)
    tbl_pattern_lengths_t      = torch.from_numpy(tbl.pattern_lengths).to(device)
    tbl_fhyp_var_ids_t         = torch.from_numpy(tbl.fhyp_var_ids).to(device)
    tbl_fhyp_count_t           = torch.from_numpy(tbl.fhyp_count).to(device)
    tbl_ehyp_patterns_t        = torch.from_numpy(tbl.ehyp_patterns).to(device)
    tbl_ehyp_pattern_lengths_t = torch.from_numpy(tbl.ehyp_pattern_lengths).to(device)
    tbl_ehyp_count_t           = torch.from_numpy(tbl.ehyp_count).to(device)
    torch.cuda.synchronize(device)

    if verbose:
        print(f"    Assertion table uploaded: {len(tbl.assertion_labels)} unique assertions (CUDA)", flush=True)

    # ── Levels 1..max: assertion nodes via CUDA kernel ────────────
    effective_batches = _merge_sparse_levels(plan.assertion_batches)
    if verbose:
        orig_n = len(plan.assertion_batches)
        merged_n = len(effective_batches)
        if merged_n < orig_n:
            print(f"    Level coalescing: {orig_n} → {merged_n} batches", flush=True)

    for batch in effective_batches:
        t_lvl = time.perf_counter()

        # Compute sublevel ranges
        if batch.sublevel_ranges is not None:
            sublevel_ranges = batch.sublevel_ranges
        elif batch.level == batch.max_level:
            sublevel_ranges = [(0, batch.count)]
        else:
            node_levels_np = batch.node_levels
            unique_lvls = np.unique(node_levels_np)
            sublevel_ranges = []
            for lv in unique_lvls:
                idxs = np.where(node_levels_np == lv)[0]
                sublevel_ranges.append((int(idxs[0]), int(idxs[-1]) + 1))

        _cuda_mod.cuda_execute_level(
            batch.assertion_idx,
            batch.input_global_indices,
            batch.input_counts,
            batch.fhyp_input_positions,
            batch.ehyp_input_positions,
            batch.output_global_indices,
            sublevel_ranges,
            tbl_pattern_toks_t, tbl_pattern_lengths_t,
            tbl_fhyp_var_ids_t, tbl_fhyp_count_t,
            tbl_ehyp_patterns_t, tbl_ehyp_pattern_lengths_t, tbl_ehyp_count_t,
            expr_buffer, expr_lengths, expr_hashes, node_fail_code,
            expr_offsets,
            device,
        )

        if verbose:
            dt = time.perf_counter() - t_lvl
            lvl_str = (
                f"{batch.level}" if batch.level == batch.max_level
                else f"{batch.level}-{batch.max_level}"
            )
            print(f"    Level {lvl_str}: {batch.count} nodes in {dt:.3f}s (CUDA)", flush=True)

    # ── No intermediate truncation with packed buffer ─────────────
    # Each node has exactly the capacity it needs from the CPU pre-pass.
    had_intermediate_truncation = False
    max_intermediate = 0

    # ── Final check via CUDA kernel ──────────────────────────────────
    proof_passed_np = _cuda_mod.cuda_final_check(
        plan.final_node_indices,
        plan.expected_conclusions,
        plan.conclusion_lengths,
        plan.expected_conclusion_hashes,
        expr_buffer, expr_lengths, expr_hashes, node_fail_code,
        expr_offsets,
        device,
    )

    # Read per-proof fail codes from the final nodes before freeing GPU memory
    final_idx_t = torch.from_numpy(plan.final_node_indices.astype(np.int64)).to(device)
    per_proof_fail_codes = node_fail_code[final_idx_t].cpu().numpy()  # [num_proofs] int8

    return proof_passed_np, had_intermediate_truncation, max_intermediate, per_proof_fail_codes


def _run_gpu_pipeline(
    plan: GlobalPlan,
    device: torch.device,
    max_expr_len: int,
    verbose: bool = False,
) -> tuple[np.ndarray, bool, int, np.ndarray]:
    """Single run of the GPU pipeline with a given max_expr_len.

    Dispatches to custom CUDA kernels when available (zero intermediate
    memory), otherwise falls back to the PyTorch tensor-op path.

    Intermediate expressions that exceed max_expr_len are stored truncated —
    this corrupts downstream substitutions, so the caller must retry with a
    larger buffer if this happens.

    Final-step expressions that exceed max_expr_len are compared via rolling
    hash (expr_hashes), so no retry is needed for those.

    Returns:
        (per_proof_passed, had_intermediate_truncation, max_intermediate_len,
         per_proof_fail_codes)  — fail codes are int8 FAIL_* values; all zeros
         on the PyTorch fallback path (no per-kernel error codes available).
    """
    # ── Dispatch to CUDA kernel path if available ─────────────────
    if device.type == "cuda" and _cuda_mod.is_available():
        return _run_gpu_pipeline_cuda(plan, device, max_expr_len, verbose=verbose)
    total_nodes = plan.total_nodes
    V = plan.vocab_size

    if total_nodes == 0:
        return np.ones(plan.num_proofs, dtype=np.bool_)

    # Allocate expr_buffer and tracking tensors on GPU.
    # expr_buffer uses empty (not zeros): every slot is fully overwritten
    # by either push-node writes (level 0) or assertion-node writes
    # (which zero + write via expr_buffer[c_out_idx] = 0 then assign).
    # For set.mm, this saves zeroing ~24 GB of GPU memory.
    expr_buffer  = torch.empty(total_nodes, max_expr_len, dtype=torch.int16, device=device)
    expr_lengths = torch.zeros(total_nodes, dtype=torch.int32, device=device)
    expr_hashes  = torch.zeros(total_nodes, dtype=torch.int64, device=device)
    node_failed  = torch.zeros(total_nodes, dtype=torch.bool,  device=device)

    # ── Level 0: write push node expressions ─────────────────────
    use_pinned = device.type == "cuda"
    if len(plan.push_global_indices) > 0:
        push_idx = torch.from_numpy(plan.push_global_indices).long().to(device)

        push_exprs_np = plan.push_expressions
        pw = push_exprs_np.shape[1]
        if pw > max_expr_len:
            push_exprs_np = push_exprs_np[:, :max_expr_len]

        if use_pinned:
            push_exprs_t = torch.from_numpy(push_exprs_np).pin_memory().to(device, non_blocking=True)
            push_lens    = torch.from_numpy(plan.push_expr_lengths).pin_memory().to(device, non_blocking=True)
        else:
            push_exprs_t = torch.from_numpy(push_exprs_np).to(device)
            push_lens    = torch.from_numpy(plan.push_expr_lengths).to(device)

        write_w = push_exprs_t.shape[1]
        expr_buffer[push_idx, :write_w] = push_exprs_t
        expr_lengths[push_idx] = push_lens.int()
        # Push expressions are short (≤ max_expr_len), so hash == token comparison;
        # compute it anyway so expr_hashes is consistent for all nodes.
        expr_hashes[push_idx] = _poly_hash_gpu(
            expr_buffer[push_idx, :write_w], push_lens.int(), device
        )

    if verbose:
        print(f"    Level 0: {len(plan.push_global_indices)} push nodes written")

    # ── Upload assertion table once — shared across all levels ───────
    tbl = plan.assertion_table
    use_pinned = device.type == "cuda"
    def _tbl_to(arr: np.ndarray, long: bool = False) -> torch.Tensor:
        t = torch.from_numpy(arr)
        if use_pinned:
            t = t.pin_memory()
        t = t.to(device, non_blocking=use_pinned)
        return t.long() if long else t

    tbl_pattern_toks_t         = _tbl_to(tbl.pattern_toks)
    tbl_pattern_lengths_t      = _tbl_to(tbl.pattern_lengths)
    tbl_fhyp_var_ids_t         = _tbl_to(tbl.fhyp_var_ids, long=True)
    tbl_fhyp_count_t           = _tbl_to(tbl.fhyp_count)
    tbl_ehyp_patterns_t        = _tbl_to(tbl.ehyp_patterns)
    tbl_ehyp_pattern_lengths_t = _tbl_to(tbl.ehyp_pattern_lengths)
    tbl_ehyp_count_t           = _tbl_to(tbl.ehyp_count)
    if use_pinned:
        torch.cuda.synchronize(device)  # ensure table is resident before compute

    if verbose:
        print(f"    Assertion table uploaded: {len(tbl.assertion_labels)} unique assertions", flush=True)

    # ── Levels 1..max: assertion nodes ───────────────────────────
    effective_batches = _merge_sparse_levels(plan.assertion_batches)
    if verbose:
        orig_n = len(plan.assertion_batches)
        merged_n = len(effective_batches)
        if merged_n < orig_n:
            print(f"    Level coalescing: {orig_n} → {merged_n} batches", flush=True)

    for batch in effective_batches:
        t_lvl = time.perf_counter()
        _execute_level(
            batch, expr_buffer, expr_lengths, expr_hashes, node_failed, V, device,
            tbl_pattern_toks_t, tbl_pattern_lengths_t,
            tbl_fhyp_var_ids_t, tbl_fhyp_count_t,
            tbl_ehyp_patterns_t, tbl_ehyp_pattern_lengths_t, tbl_ehyp_count_t,
        )
        if verbose:
            dt = time.perf_counter() - t_lvl
            lvl_str = (
                f"{batch.level}" if batch.level == batch.max_level
                else f"{batch.level}-{batch.max_level}"
            )
            print(f"    Level {lvl_str}: {batch.count} nodes in {dt:.3f}s", flush=True)

    # ── Check for intermediate truncation ────────────────────────────
    # If any non-final node has expr_lengths > max_expr_len, its stored value
    # is truncated, which will corrupt downstream substitutions. The caller
    # must retry with a larger buffer.
    all_lens = expr_lengths  # [total_nodes]
    # Build a mask of final nodes to exclude them from intermediate check
    final_mask = torch.zeros(plan.total_nodes, dtype=torch.bool, device=device)
    final_idx_t = torch.from_numpy(plan.final_node_indices).long().to(device)
    final_mask[final_idx_t] = True
    intermediate_lens = all_lens.masked_fill(final_mask, 0)
    max_intermediate = int(intermediate_lens.max().item())
    had_intermediate_truncation = max_intermediate > max_expr_len

    # ── Final check: compare last node expression to expected conclusion ──
    final_lens      = expr_lengths[final_idx_t]       # [num_proofs] int32
    final_hashes    = expr_hashes[final_idx_t]        # [num_proofs] int64
    final_node_fail = node_failed[final_idx_t]        # [num_proofs] bool

    if use_pinned:
        expected_lens   = torch.from_numpy(plan.conclusion_lengths).pin_memory().to(device, non_blocking=False)
        expected_hashes = torch.from_numpy(plan.expected_conclusion_hashes).pin_memory().to(device, non_blocking=False)
        expected        = torch.from_numpy(plan.expected_conclusions).pin_memory().to(device, non_blocking=False)
    else:
        expected_lens   = torch.from_numpy(plan.conclusion_lengths).to(device)
        expected_hashes = torch.from_numpy(plan.expected_conclusion_hashes).to(device)
        expected        = torch.from_numpy(plan.expected_conclusions).to(device)

    length_match = (final_lens == expected_lens)

    # For final expressions that fit in expr_buffer: compare tokens (exact).
    # For final expressions longer than max_expr_len: compare rolling hashes.
    fits_in_buffer = final_lens <= max_expr_len

    # Compare final expressions without padding allocation.
    # Use the narrower width and only compare within valid positions.
    final_exprs = expr_buffer[final_idx_t]  # [num_proofs, max_expr_len]
    w1 = final_exprs.shape[1]
    w2 = expected.shape[1]
    compare_dim = min(w1, w2)
    # Lengths that exceed compare_dim can't match via tokens — they'll
    # use hash comparison via fits_in_buffer check below.
    positions   = torch.arange(compare_dim, device=device).unsqueeze(0)
    valid_mask  = positions < final_lens.unsqueeze(1)
    masked_eq   = (final_exprs[:, :compare_dim] == expected[:, :compare_dim]) | ~valid_mask
    token_match = masked_eq.all(dim=1)
    # Any proof with length > compare_dim can't be token-compared correctly,
    # so force it to hash comparison path.
    if compare_dim < max(w1, w2):
        token_match = token_match & (final_lens <= compare_dim)

    hash_match = (final_hashes == expected_hashes)
    content_match = torch.where(fits_in_buffer, token_match, hash_match)

    proof_passed = length_match & content_match & ~final_node_fail
    # PyTorch path has no per-kernel fail codes — return zeros (no info)
    no_codes = np.zeros(plan.num_proofs, dtype=np.int8)
    return proof_passed.cpu().numpy(), had_intermediate_truncation, max_intermediate, no_codes

# ── CUDA warmup ────────────────────────────────────────────────────────
_CUDA_WARMED_UP: set[str] = set()


def warmup_cuda(device: torch.device) -> None:
    """Pre-compile every CUDA kernel shape used in the verification pipeline.

    On CUDA, PyTorch JIT-compiles kernels on first use.  Without warmup,
    the first few levels of a real run incur compilation latency that
    distorts timing and can make early large batches look slow.

    Also triggers JIT compilation of custom CUDA kernels (if available).

    This runs tiny synthetic tensors through every distinct operation used
    in _execute_level and _run_gpu_pipeline so all kernels are compiled and
    resident before the timed verification begins.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    if device.type != "cuda":
        return
    key = device.__str__()
    if key in _CUDA_WARMED_UP:
        return

    # JIT-compile custom CUDA kernels (first call triggers compilation)
    if _cuda_mod.is_available():
        print(f"  Custom CUDA kernels: compiled and ready", flush=True)
    else:
        print(f"  Custom CUDA kernels: unavailable, using PyTorch fallback", flush=True)

    B, S, P, F = 8, 4, 8, 2  # tiny synthetic shapes

    # Allocate
    expr_buf  = torch.zeros(B * 2, P, dtype=torch.int16, device=device)
    expr_lens = torch.zeros(B * 2, dtype=torch.int32, device=device)
    failed    = torch.zeros(B * 2, dtype=torch.bool, device=device)

    idx       = torch.zeros(B, dtype=torch.long, device=device)
    in_idx    = torch.zeros(B, 2, dtype=torch.long, device=device)
    in_count  = torch.ones(B, dtype=torch.int32, device=device)

    pat       = torch.ones(B, P, dtype=torch.int16, device=device)
    pat_len   = torch.full((B,), P, dtype=torch.int32, device=device)

    fhyp_var  = torch.zeros(B, F, dtype=torch.long, device=device)
    fhyp_cnt  = torch.ones(B, dtype=torch.int32, device=device)
    fhyp_valid = torch.arange(F, device=device) < fhyp_cnt.unsqueeze(1)

    var_sub_vals = torch.zeros(B, F, S, dtype=torch.int16, device=device)
    var_sub_lens = torch.ones(B, F, dtype=torch.int32, device=device)

    # Kernels used in _execute_level ─────────────────────────────────
    # (a) gather from expr_buffer
    _ = expr_buf[in_idx.clamp(min=0)]
    _ = expr_lens[in_idx.clamp(min=0)]
    _ = failed[in_idx.clamp(min=0)]
    _ = (torch.arange(2, device=device) < in_count.unsqueeze(1))

    # (b) compact substitution — broadcasting match + scatter
    # Exercise _apply_substitution_compact kernels
    out_buf, tot_len = _apply_substitution_compact(
        pat, pat_len, fhyp_var, var_sub_vals, var_sub_lens, fhyp_valid, device,
    )
    max_out = out_buf.shape[1] if out_buf.shape[1] > 0 else 1

    # (c) _verify_substitution_result kernels
    positions = torch.arange(max_out, device=device).unsqueeze(0)
    vm   = positions < tot_len.unsqueeze(1)
    _    = ((out_buf == out_buf) | ~vm).all(dim=1)
    _    = (tot_len == tot_len)

    # (d) expr_buffer write-back
    expr_buf[idx] = 0
    w = min(max_out, P)
    expr_buf[idx, :w] = out_buf[:, :w]
    expr_lens[idx] = tot_len.int()
    failed[idx]    = ~torch.ones(B, dtype=torch.bool, device=device)

    torch.cuda.synchronize(device)
    _CUDA_WARMED_UP.add(key)


_NUMBA_WARMED_UP = False


def warmup_numba() -> None:
    """Pre-compile all Numba JIT kernels used in Phase 2 (pack_levels).

    Numba compiles on first call with each signature.  Without warmup the
    first pack_levels invocation silently eats several seconds of LLVM
    compilation.  Subsequent calls are no-ops (kernels are cached on disk
    via cache=True anyway, but this guarantees they're resident in-process).

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _NUMBA_WARMED_UP
    if _NUMBA_WARMED_UP:
        return

    dummy_enc  = np.zeros(2, dtype=np.int16)
    dummy_off  = np.array([0, 2], dtype=np.int32)
    dummy_expr = np.zeros((1, 2), dtype=np.int16)
    _nb_fill_push_expressions(dummy_enc, dummy_off, dummy_expr)

    dummy_uid_arr   = np.zeros(1, dtype=np.int32)
    dummy_node_off  = np.array([0, 2], dtype=np.int32)
    dummy_uid_flat  = np.zeros(2, dtype=np.int16)
    dummy_uid_off   = np.array([0, 2], dtype=np.int32)
    dummy_out_flat  = np.zeros(2, dtype=np.int16)
    _nb_build_flat_push_enc(dummy_uid_arr, dummy_node_off, dummy_uid_flat, dummy_uid_off, dummy_out_flat)

    dummy_pos   = np.zeros(1, dtype=np.int32)
    dummy_goff  = np.array([0, 1], dtype=np.int32)
    dummy_gdat  = np.zeros(0, dtype=np.int32)
    dummy_loff  = np.array([0], dtype=np.int32)
    dummy_ldat  = np.zeros(0, dtype=np.int32)
    _nb_gather_csr(dummy_pos, dummy_goff, dummy_gdat, dummy_loff, dummy_ldat)

    B = 1
    dummy_graph_off = np.array([0, 1], dtype=np.int64)
    out_aidx  = np.zeros(B, dtype=np.int32)
    out_ig    = np.full((B, 1), -1, dtype=np.int32)
    out_ic    = np.zeros(B, dtype=np.int32)
    out_fp    = np.zeros((B, 1), dtype=np.int32)
    out_ep    = np.zeros((B, 1), dtype=np.int32)
    out_og    = np.zeros(B, dtype=np.int32)
    _nb_pack_assertion_level(
        np.zeros(B, dtype=np.int32),
        np.zeros(B, dtype=np.int32),
        np.zeros(B, dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        np.zeros(B, dtype=np.int32),
        np.zeros(B, dtype=np.int32),
        dummy_graph_off,
        1, 1, 1,
        out_aidx, out_ig, out_ic, out_fp, out_ep, out_og,
    )

    # Warmup _nb_compute_expr_lengths_batch (per-level expr length kernel)
    _nb_compute_expr_lengths_batch(
        np.zeros(1, dtype=np.int32),         # output_global
        np.zeros(1, dtype=np.int32),         # assertion_idxs
        np.zeros((1, 1), dtype=np.int32),    # input_global
        np.zeros((1, 1), dtype=np.int32),    # fhyp_positions
        np.zeros(1, dtype=np.int32),         # tbl_fhyp_count
        np.zeros(1, dtype=np.int32),         # tbl_const_count
        np.zeros((1, 1), dtype=np.int32),    # tbl_var_occ
        np.zeros(1, dtype=np.int32),         # node_expr_lengths
    )

    _NUMBA_WARMED_UP = True


_MAX_EXPR_LEN_CAP = 16384  # absolute safety cap


def _split_plan(plan: GlobalPlan, split: int) -> tuple["GlobalPlan", "GlobalPlan"]:
    """Split a GlobalPlan into two halves by proof index.

    Proof indices [0, split) go to plan_a, [split, num_proofs) go to plan_b.
    Global buffer indices are re-based so each half starts at 0.

    The AssertionTable is shared (read-only) — both halves reference the same object.
    AssertionLevelBatch rows are filtered by which half their output_global_index falls in.
    """
    N = plan.num_proofs
    assert 0 < split < N

    # ── Per-proof data splits ─────────────────────────────────────────
    def _split_arr(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return a[:split], a[split:]

    final_a, final_b         = _split_arr(plan.final_node_indices)
    concl_a, concl_b         = _split_arr(plan.expected_conclusions)
    clen_a, clen_b           = _split_arr(plan.conclusion_lengths)
    chash_a, chash_b         = _split_arr(plan.expected_conclusion_hashes)
    labels_a                 = plan.proof_theorem_labels[:split]
    labels_b                 = plan.proof_theorem_labels[split:]

    labels_a_set = set(labels_a)

    # graph_offsets[split] is the exact first global buffer index of proof
    # `split` — trivially correct since pack_levels stores it directly.
    offset_split = int(plan.graph_offsets[split])

    # ── Push data splits ─────────────────────────────────────────────
    push_mask_a = plan.push_global_indices < offset_split
    push_mask_b = ~push_mask_a

    push_gi_a = plan.push_global_indices[push_mask_a]
    push_gi_b = plan.push_global_indices[push_mask_b] - offset_split  # re-base
    push_ex_a = plan.push_expressions[push_mask_a]
    push_ex_b = plan.push_expressions[push_mask_b]
    push_el_a = plan.push_expr_lengths[push_mask_a]
    push_el_b = plan.push_expr_lengths[push_mask_b]

    # ── Re-base per-proof final indices ──────────────────────────────
    final_a_rebased = final_a.copy()  # already in [0, offset_split)
    final_b_rebased = (final_b - offset_split).astype(np.int32)

    # ── Split assertion batches ───────────────────────────────────────
    def _split_batch(batch: AssertionLevelBatch) -> tuple[AssertionLevelBatch | None,
                                                          AssertionLevelBatch | None]:
        mask_a = np.array([lbl in labels_a_set for lbl in batch.theorem_labels], dtype=bool)
        mask_b = ~mask_a

        def _make_half(mask: np.ndarray, rebase: int) -> AssertionLevelBatch | None:
            if not mask.any():
                return None
            idx = np.where(mask)[0]
            new_out = batch.output_global_indices[mask] - rebase
            new_in  = batch.input_global_indices[mask].copy()
            # Remap non-sentinel input indices
            valid   = new_in >= 0
            new_in[valid] = new_in[valid] - rebase
            return AssertionLevelBatch(
                level=batch.level,
                max_level=batch.max_level,
                count=int(mask.sum()),
                node_levels=batch.node_levels[mask],
                assertion_labels=[batch.assertion_labels[i] for i in idx],
                theorem_labels=[batch.theorem_labels[i] for i in idx],
                assertion_idx=batch.assertion_idx[mask],
                input_global_indices=new_in.astype(np.int32),
                input_counts=batch.input_counts[mask],
                fhyp_input_positions=batch.fhyp_input_positions[mask],
                ehyp_input_positions=batch.ehyp_input_positions[mask],
                output_global_indices=new_out.astype(np.int32),
                sublevel_ranges=None,  # recomputed at runtime
            )

        return _make_half(mask_a, 0), _make_half(mask_b, offset_split)

    batches_a: list[AssertionLevelBatch] = []
    batches_b: list[AssertionLevelBatch] = []
    for batch in plan.assertion_batches:
        ba, bb = _split_batch(batch)
        if ba is not None:
            batches_a.append(ba)
        if bb is not None:
            batches_b.append(bb)

    total_nodes_a = offset_split
    total_nodes_b = plan.total_nodes - offset_split

    # graph_offsets for each half: slice + rebase
    go_a = plan.graph_offsets[:split + 1].copy()          # [0..split] entries, already 0-based
    go_b = plan.graph_offsets[split:].copy() - offset_split  # rebase to 0
    go_b = go_b.astype(np.int64)

    # node_expr_lengths: split by node range
    nel_a: np.ndarray | None = None
    nel_b: np.ndarray | None = None
    tet_a = 0
    tet_b = 0
    if plan.node_expr_lengths is not None:
        nel_a = plan.node_expr_lengths[:offset_split].copy()
        nel_b = plan.node_expr_lengths[offset_split:].copy()
        tet_a = int(nel_a.sum())
        tet_b = int(nel_b.sum())

    plan_a = GlobalPlan(
        total_nodes=total_nodes_a,
        max_expr_len=plan.max_expr_len,
        num_proofs=split,
        assertion_table=plan.assertion_table,  # shared, read-only
        push_global_indices=push_gi_a,
        push_expressions=push_ex_a,
        push_expr_lengths=push_el_a,
        assertion_batches=batches_a,
        final_node_indices=final_a_rebased,
        expected_conclusions=concl_a,
        conclusion_lengths=clen_a,
        expected_conclusion_hashes=chash_a,
        proof_theorem_labels=labels_a,
        vocab_size=plan.vocab_size,
        graph_offsets=go_a,
        node_expr_lengths=nel_a,
        total_expr_tokens=tet_a,
    )
    plan_b = GlobalPlan(
        total_nodes=total_nodes_b,
        max_expr_len=plan.max_expr_len,
        num_proofs=N - split,
        assertion_table=plan.assertion_table,  # shared, read-only
        push_global_indices=push_gi_b,
        push_expressions=push_ex_b,
        push_expr_lengths=push_el_b,
        assertion_batches=batches_b,
        final_node_indices=final_b_rebased,
        expected_conclusions=concl_b,
        conclusion_lengths=clen_b,
        expected_conclusion_hashes=chash_b,
        proof_theorem_labels=labels_b,
        vocab_size=plan.vocab_size,
        graph_offsets=go_b,
        node_expr_lengths=nel_b,
        total_expr_tokens=tet_b,
    )
    return plan_a, plan_b


def verify_proofs_gpu(
    plan: GlobalPlan,
    device: torch.device,
    verbose: bool = False,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Execute the full GPU verification pipeline.

    Retries with a larger expr_buffer only when INTERMEDIATE nodes are
    truncated (which corrupts downstream substitutions). Final-step
    expressions that exceed the buffer are compared via rolling hash
    (expr_hashes), so no retry is needed for those — this handles
    quartfull's 11548-token conclusion without a 278 GB allocation.

    Returns:
        (per_proof_passed: np.ndarray[bool], gpu_time: float,
         per_proof_fail_codes: np.ndarray[int8])
    """
    t0 = time.perf_counter()
    max_expr_len = plan.max_expr_len

    while max_expr_len <= _MAX_EXPR_LEN_CAP:
        result, had_truncation, needed, fail_codes = _run_gpu_pipeline(
            plan, device, max_expr_len, verbose=verbose,
        )
        if not had_truncation:
            break
        new_len = min(max(needed + 64, max_expr_len * 2), _MAX_EXPR_LEN_CAP)
        if verbose:
            print(
                f"    Intermediate truncation: max intermediate {needed} > "
                f"buffer {max_expr_len}. Retrying with {new_len}...", flush=True
            )
        if new_len <= max_expr_len:
            break
        max_expr_len = new_len

    gpu_time = time.perf_counter() - t0
    return result, gpu_time, fail_codes


# ══════════════════════════════════════════════════════════════════════
#  Phase 4 — $d Post-Check
# ══════════════════════════════════════════════════════════════════════


def _serialize_dv_for_rust(
    parsed: ParsedDatabase,
    theorem_labels: list[str],
) -> tuple:
    """Serialise ParsedDatabase into flat byte arrays for check_dv_all.

    Builds a fresh sym_to_id / label_to_lid in one pass (same stable ordering
    as _serialize_db_for_rust so label IDs are consistent) then adds the
    DV-specific fields that check_dv_all needs.
    """
    sym_to_id: dict[str, int] = {}

    def sid(s: str) -> int:
        v = sym_to_id.get(s)
        if v is None:
            v = len(sym_to_id)
            sym_to_id[s] = v
        return v

    # ── Label table ───────────────────────────────────────────────────
    all_labels: list[str] = []
    label_to_lid: dict[str, int] = {}
    for lbl in parsed.floating_hyps:
        label_to_lid[lbl] = len(all_labels); all_labels.append(lbl)
    for lbl in parsed.essential_hyps:
        label_to_lid[lbl] = len(all_labels); all_labels.append(lbl)
    for lbl in parsed.assertions:
        label_to_lid[lbl] = len(all_labels); all_labels.append(lbl)

    L = len(all_labels)
    lt_arr     = bytearray(L)
    lf_tc_arr  = np.full(L, -1, dtype=np.int32)
    lf_var_arr = np.full(L, -1, dtype=np.int32)

    le_len_arr = np.zeros(L, dtype=np.int32)
    le_data: list[int] = []

    la_ne_arr = np.zeros(L, dtype=np.int32)

    # DV-specific arrays
    la_fhyp_var_len = np.zeros(L, dtype=np.int32)
    la_fhyp_var_data: list[int] = []
    la_expr_len = np.zeros(L, dtype=np.int32)
    la_expr_data: list[int] = []
    la_dv_len = np.zeros(L, dtype=np.int32)   # in i32 elements (pairs × 2)
    la_dv_data: list[int] = []

    for lbl, lid in label_to_lid.items():
        if lbl in parsed.floating_hyps:
            fh = parsed.floating_hyps[lbl]
            lt_arr[lid] = 0
            lf_tc_arr[lid]  = sid(fh.type_code)
            lf_var_arr[lid] = sid(fh.variable)
        elif lbl in parsed.essential_hyps:
            eh = parsed.essential_hyps[lbl]
            lt_arr[lid] = 1
            enc = [sid(s) for s in eh.expression]
            le_data.extend(enc)
            le_len_arr[lid] = len(enc)
        else:
            a = parsed.assertions[lbl]
            lt_arr[lid] = 2
            la_ne_arr[lid] = len(a.essential_hyps)
            # fhyp vars in order
            fv = [sid(parsed.floating_hyps[flbl].variable) for flbl in a.floating_hyps]
            la_fhyp_var_data.extend(fv)
            la_fhyp_var_len[lid] = len(fv)
            # conclusion expression
            expr_enc = [sid(s) for s in a.expression]
            la_expr_data.extend(expr_enc)
            la_expr_len[lid] = len(expr_enc)
            # mandatory DV pairs (interleaved x y, already canonical min/max from parser)
            for x, y in a.disjoint_vars:
                la_dv_data.extend([sid(x), sid(y)])
            la_dv_len[lid] = len(a.disjoint_vars) * 2

    # Build CSR offset arrays from length arrays
    def _make_csr(lengths: np.ndarray) -> np.ndarray:
        off = np.empty(L + 1, dtype=np.int32)
        off[0] = 0
        np.cumsum(lengths, out=off[1:])
        return off

    le_off_arr          = _make_csr(le_len_arr)
    la_fhyp_var_off_arr = _make_csr(la_fhyp_var_len)
    la_expr_off_arr     = _make_csr(la_expr_len)
    la_dv_off_arr       = _make_csr(la_dv_len)

    # ── Theorem proofs ────────────────────────────────────────────────
    T = len(theorem_labels)
    thm_proof_offsets  = np.zeros(T + 1, dtype=np.int64)
    thm_plabel_offsets = np.zeros(T + 1, dtype=np.int32)
    thm_proof_data: list[int] = []
    thm_plabel_data: list[int] = []

    # Active DV per theorem (canonical pairs interleaved min max)
    thm_active_dv_offsets = np.zeros(T + 1, dtype=np.int32)
    thm_active_dv_data: list[int] = []

    for ti, thm_lbl in enumerate(theorem_labels):
        a = parsed.assertions[thm_lbl]
        if a.compressed_proof is not None:
            cp = a.compressed_proof
            plabels_lids = [label_to_lid[lbl] for lbl in cp.labels]
            thm_plabel_data.extend(plabels_lids)
            thm_plabel_offsets[ti + 1] = thm_plabel_offsets[ti] + len(plabels_lids)
            thm_proof_data.extend(cp.proof_ints)
            thm_proof_offsets[ti + 1] = thm_proof_offsets[ti] + len(cp.proof_ints)
        elif a.proof is not None:
            for step_lbl in a.proof:
                thm_plabel_data.append(label_to_lid[step_lbl])
                thm_proof_data.append(len(thm_plabel_data) - 1 - int(thm_plabel_offsets[ti]))
            thm_plabel_offsets[ti + 1] = thm_plabel_offsets[ti] + len(a.proof)
            thm_proof_offsets[ti + 1]  = thm_proof_offsets[ti]  + len(a.proof)
        else:
            thm_plabel_offsets[ti + 1] = thm_plabel_offsets[ti]
            thm_proof_offsets[ti + 1]  = thm_proof_offsets[ti]

        # active DV pairs for this theorem
        pairs = a.all_disjoint_vars
        for x, y in pairs:
            thm_active_dv_data.extend([sid(x), sid(y)])
        thm_active_dv_offsets[ti + 1] = thm_active_dv_offsets[ti] + len(pairs) * 2

    # is_variable: bool per symbol (1 = variable, 0 = constant)
    N_sym = len(sym_to_id)
    is_variable_arr = bytearray(N_sym)
    for sym, sym_id in sym_to_id.items():
        if sym in parsed.variables:
            is_variable_arr[sym_id] = 1

    return (
        T,
        bytes(lt_arr),
        lf_tc_arr.tobytes(),
        lf_var_arr.tobytes(),
        le_off_arr.tobytes(),
        np.array(le_data, dtype=np.int32).tobytes(),
        la_ne_arr.tobytes(),
        la_fhyp_var_off_arr.tobytes(),
        np.array(la_fhyp_var_data, dtype=np.int32).tobytes(),
        la_expr_off_arr.tobytes(),
        np.array(la_expr_data, dtype=np.int32).tobytes(),
        la_dv_off_arr.tobytes(),
        np.array(la_dv_data, dtype=np.int32).tobytes(),
        thm_proof_offsets.tobytes(),
        np.array(thm_proof_data, dtype=np.int32).tobytes(),
        thm_plabel_offsets.tobytes(),
        np.array(thm_plabel_data, dtype=np.int32).tobytes(),
        thm_active_dv_offsets.tobytes(),
        np.array(thm_active_dv_data, dtype=np.int32).tobytes(),
        bytes(is_variable_arr),
    )


def _vars_in_expr(expr: list[str], variables: set[str]) -> set[str]:
    """Return the set of variables appearing in expr."""
    return {tok for tok in expr if tok in variables}


def _apply_subst(expr: list[str], subst: dict[str, list[str]]) -> list[str]:
    """Apply a string-level substitution to an expression."""
    out: list[str] = []
    for tok in expr:
        if tok in subst:
            out.extend(subst[tok])
        else:
            out.append(tok)
    return out


def _check_dv_one(parsed: ParsedDatabase, theorem_label: str) -> str | None:
    """Replay a proof and check all $d constraints.

    Walks the proof stack exactly like a standard Metamath verifier but only
    checks disjoint variable conditions — the GPU has already validated the
    substitution arithmetic.

    Returns None if all $d constraints are satisfied, or a reason string on failure.

    The Metamath $d rule: when applying an assertion with $d x y, for every
    variable v in subst(x) and every variable w in subst(y), the pair (v, w)
    must appear in the active $d constraints of the theorem being proved
    (assertion.all_disjoint_vars). Additionally v != w is required.
    """
    assertion = parsed.assertions[theorem_label]
    variables = parsed.variables
    # Parser stores pairs as (min, max) canonical tuples — reuse that directly.
    active_dv: set[tuple[str, str]] = set(assertion.all_disjoint_vars)

    def _info(lbl: str):
        if lbl in parsed.floating_hyps:
            return ("$f", parsed.floating_hyps[lbl])
        if lbl in parsed.essential_hyps:
            return ("$e", parsed.essential_hyps[lbl])
        if lbl in parsed.assertions:
            a = parsed.assertions[lbl]
            return ("$a" if a.type == "axiom" else "$p", a)
        return None

    stack: list[list[str]] = []

    def _step(lbl: str) -> str | None:
        """Returns None on success, or an error reason string on failure."""
        info = _info(lbl)
        if info is None:
            return f"unknown label {lbl!r}"
        kind, data = info
        if kind == "$f":
            stack.append([data.type_code, data.variable])
            return None
        if kind == "$e":
            stack.append(list(data.expression))
            return None
        # Assertion step: pop hyps, build substitution, check $d, push conclusion
        a = data
        n_pop = len(a.floating_hyps) + len(a.essential_hyps)
        if len(stack) < n_pop:
            return f"stack underflow applying {lbl}: need {n_pop}, have {len(stack)}"
        sp = len(stack) - n_pop
        subst: dict[str, list[str]] = {}
        for flbl in a.floating_hyps:
            fh = parsed.floating_hyps[flbl]
            entry = stack[sp]
            subst[fh.variable] = entry[1:]  # strip type code
            sp += 1
        # $d check: for each mandatory $d x y on the applied assertion,
        # every variable in subst(x) must be disjoint (in active_dv) from
        # every variable in subst(y), and they must be distinct variables.
        for x, y in a.disjoint_vars:
            sx = _vars_in_expr(subst.get(x, [x]), variables)
            sy = _vars_in_expr(subst.get(y, [y]), variables)
            for v in sx:
                for w in sy:
                    if v == w:
                        return (
                            f"$d violation in {lbl}: ${x} and ${y} both map to "
                            f"variable {v!r} (must be distinct)"
                        )
                    pair = (min(v, w), max(v, w))
                    if pair not in active_dv:
                        return (
                            f"$d violation in {lbl}: ${x}→{v!r} and ${y}→{w!r} "
                            f"are not disjoint in {theorem_label}"
                        )
        del stack[len(stack) - n_pop:]
        stack.append(_apply_subst(a.expression, subst))
        return None

    try:
        if assertion.compressed_proof is not None:
            cp = assertion.compressed_proof
            label_end = len(cp.labels)
            saved: list[list[str]] = []
            for pi in cp.proof_ints:
                if pi == -1:
                    if not stack:
                        return "Z-save on empty stack"
                    saved.append(list(stack[-1]))
                elif pi < label_end:
                    err = _step(cp.labels[pi])
                    if err:
                        return err
                else:
                    si = pi - label_end
                    if si >= len(saved):
                        return f"backref {si} out of range ({len(saved)} saved)"
                    stack.append(list(saved[si]))
        elif assertion.proof is not None:
            for lbl in assertion.proof:
                err = _step(lbl)
                if err:
                    return err
        else:
            return None  # axiom — no proof to check
    except Exception as e:
        return f"exception: {e}"

    return None



_DV_WORKER_PARSED: ParsedDatabase | None = None


def _init_dv_worker(parsed: ParsedDatabase) -> None:
    global _DV_WORKER_PARSED
    _DV_WORKER_PARSED = parsed


def _check_dv_chunk(labels: list[str]) -> dict[str, str | None]:
    """Returns {label: reason} where reason is None on success."""
    assert _DV_WORKER_PARSED is not None
    return {lbl: _check_dv_one(_DV_WORKER_PARSED, lbl) for lbl in labels}


def _check_dv_constraints(
    parsed: ParsedDatabase,
    graphs: list[ProofGraph],
    proof_passed: np.ndarray,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, str]]:
    """Check $d constraints for proofs that passed GPU verification.

    Uses the Rust extension (rayon parallel, same process, no spawn overhead)
    when available.  Falls back to the spawn-based process pool otherwise.

    Returns:
        (updated proof_passed array, {label: failure_reason} for each $d failure)
    """
    result = proof_passed.copy()
    dv_failures: dict[str, str] = {}
    passing_indices = [pi for pi, g in enumerate(graphs) if result[pi]]
    labels_to_check = [graphs[pi].theorem_label for pi in passing_indices]
    if not labels_to_check:
        return result, dv_failures

    if _HAVE_RUST:
        # ── Fast path: Rust/rayon, no subprocess overhead ──────────────
        ser = _serialize_dv_for_rust(parsed, labels_to_check)
        (T,
         lt_b, lf_tc_b, lf_var_b, le_off_b, le_data_b,
         la_ne_b,
         la_fhyp_var_off_b, la_fhyp_var_data_b,
         la_expr_off_b, la_expr_data_b,
         la_dv_off_b, la_dv_data_b,
         proof_off_b, proof_data_b, plabel_off_b, plabel_data_b,
         active_dv_off_b, active_dv_data_b,
         is_variable_b) = ser

        raw_results: list = _mmgpu_rs.check_dv_all(
            lt_b, lf_tc_b, lf_var_b, le_off_b, le_data_b,
            la_ne_b,
            la_fhyp_var_off_b, la_fhyp_var_data_b,
            la_expr_off_b, la_expr_data_b,
            la_dv_off_b, la_dv_data_b,
            T,
            proof_off_b, proof_data_b, plabel_off_b, plabel_data_b,
            active_dv_off_b, active_dv_data_b,
            is_variable_b,
        )

        label_to_pi = {g.theorem_label: pi for pi, g in enumerate(graphs)}
        for lbl, reason in zip(labels_to_check, raw_results):
            if reason is not None:
                pi = label_to_pi[lbl]
                result[pi] = False
                dv_failures[lbl] = f"dv:{reason}"
                if verbose:
                    print(f"    $d failure [{lbl}]: {reason}", flush=True)

    else:
        # ── Slow fallback: spawn-based process pool ─────────────────────
        workers = min(os.cpu_count() or 1, 32)
        chunk_size = max(1, len(labels_to_check) // (workers * 4))
        chunks = [
            labels_to_check[i: i + chunk_size]
            for i in range(0, len(labels_to_check), chunk_size)
        ]
        ctx = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_dv_worker,
            initargs=(parsed,),
        )
        dv_results: dict[str, str | None] = {}
        with pool as executor:
            futures = {executor.submit(_check_dv_chunk, chunk): chunk for chunk in chunks}
            for future in as_completed(futures):
                dv_results.update(future.result())

        label_to_pi = {g.theorem_label: pi for pi, g in enumerate(graphs)}
        for lbl, reason in dv_results.items():
            if reason is not None:
                result[label_to_pi[lbl]] = False
                dv_failures[lbl] = reason
                if verbose:
                    print(f"    $d failure [{lbl}]: {reason}", flush=True)

    return result, dv_failures


# ══════════════════════════════════════════════════════════════════════
#  Failure-reason helpers
# ══════════════════════════════════════════════════════════════════════

# Maps int8 node_fail_code values (from CUDA kernel) to human-readable strings.
_FAIL_CODE_NAMES: dict[int, str] = {
    0: "ok",
    1: "input_failed",    # propagated from a failed dependency
    2: "ehyp_mismatch",  # essential hypothesis substitution did not match
    3: "conclusion_overflow",  # substituted conclusion exceeded allocated capacity
}


def fail_code_name(code: int) -> str:
    """Return a human-readable name for a CUDA node_fail_code value."""
    return _FAIL_CODE_NAMES.get(code, f"unknown({code})")


def get_fail_reasons(
    node_fail_code: torch.Tensor,   # [total_nodes] int8 on device
    final_node_indices: np.ndarray, # [num_proofs] int32
    proof_labels: list[str],
) -> dict[str, str]:
    """Return {label: reason} for every proof whose final node has a non-zero fail code.

    Useful for post-hoc diagnostics after GPU verification.  Only reads
    final-node codes — intermediate failures are summarised by propagation.
    """
    codes = node_fail_code[torch.from_numpy(final_node_indices).long().to(node_fail_code.device)]
    codes_np = codes.cpu().numpy()
    return {
        label: fail_code_name(int(codes_np[i]))
        for i, label in enumerate(proof_labels)
        if codes_np[i] != 0
    }


# ══════════════════════════════════════════════════════════════════════
#  Top-Level API
# ══════════════════════════════════════════════════════════════════════


def _select_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _verify_proofs_gpu_batched(
    plan: GlobalPlan,
    device: torch.device,
    verbose: bool = False,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Run GPU verification in sequential batches sized to fit available VRAM.

    Instead of loading all proof nodes into VRAM at once, splits the plan into
    batches where each batch's expr_buffer fits comfortably in free VRAM.
    Each batch is: load → process → collect result → free → next.

    Batch size is determined by:
        free_vram * VRAM_BUDGET_FRACTION / (max_expr_len * 4 bytes per token)
    giving the number of nodes per batch. We then find the proof boundary
    nearest to that node count using graph_offsets.
    """
    # MPS shares memory with the CPU/system — be conservative to avoid
    # starving the OS. CUDA has dedicated VRAM so can use more.
    # With custom CUDA kernels there are zero per-chunk intermediates,
    # so we can safely use 85% of VRAM (only expr_buffer + tracking).
    if device.type == "mps":
        VRAM_BUDGET_FRACTION = 0.45
    elif device.type == "cuda" and _cuda_mod.is_available():
        VRAM_BUDGET_FRACTION = 0.85
    else:
        VRAM_BUDGET_FRACTION = 0.70

    t0 = time.perf_counter()
    N = plan.num_proofs

    # Compute how many nodes fit in VRAM per batch
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
    else:
        free_bytes = 8 * 1024 ** 3  # 8 GB fallback for non-CUDA

    # ── Budget: CUDA packed path uses exact token counts ────────────
    # With packed buffer, total bytes = total_expr_tokens*2 + nodes*21
    # (21 = expr_lengths(4) + expr_hashes(8) + node_failed(1) + expr_offsets(8)).
    # No 2× safety margin needed: packed buffer never truncates/retries.
    # Torch fallback still uses per-node padded budget.
    use_packed = (device.type == "cuda" and _cuda_mod.is_available()
                  and plan.node_expr_lengths is not None)

    vram_budget = int(free_bytes * VRAM_BUDGET_FRACTION)

    if use_packed:
        # Precompute cumulative token counts per proof boundary for split search
        cum_tokens_at_proof = np.zeros(N + 1, dtype=np.int64)
        nel = plan.node_expr_lengths
        go = plan.graph_offsets
        for pi in range(N):
            n_start = int(go[pi])
            n_end = int(go[pi + 1])
            cum_tokens_at_proof[pi + 1] = cum_tokens_at_proof[pi] + int(nel[n_start:n_end].sum())

        if verbose:
            free_gb = free_bytes / 1024**3
            total_mb = (plan.total_expr_tokens * 2 + plan.total_nodes * 21) / 1024**2
            print(
                f"  Phase 3: {free_gb:.1f} GB free VRAM, "
                f"packed buffer needs {total_mb:.1f} MB",
                flush=True,
            )

        # Build split points: greedily add proofs until VRAM budget is exceeded
        split_points: list[int] = [0]
        while split_points[-1] < N:
            prev = split_points[-1]
            prev_tokens = int(cum_tokens_at_proof[prev])
            prev_nodes = int(go[prev])
            # Binary search for the last proof that fits
            lo, hi = prev + 1, N
            while lo < hi:
                mid = (lo + hi + 1) // 2
                shard_tokens = int(cum_tokens_at_proof[mid]) - prev_tokens
                shard_nodes = int(go[mid]) - prev_nodes
                shard_bytes = shard_tokens * 2 + shard_nodes * 21
                if shard_bytes <= vram_budget:
                    lo = mid
                else:
                    hi = mid - 1
            split_points.append(lo)
    else:
        budget_expr_len = min(plan.max_expr_len * 2, _MAX_EXPR_LEN_CAP)
        bytes_per_node = budget_expr_len * 2 + 13  # int16 tokens + tracking
        nodes_per_batch = max(1, vram_budget // bytes_per_node)

        if verbose:
            free_gb = free_bytes / 1024**3
            batch_gb = nodes_per_batch * bytes_per_node / 1024**3
            print(
                f"  Phase 3: {free_gb:.1f} GB free VRAM, "
                f"~{batch_gb:.1f} GB per batch, "
                f"{nodes_per_batch:,} nodes/batch",
                flush=True,
            )

        # Build batch split points by node count using graph_offsets
        split_points = [0]
        while split_points[-1] < N:
            prev = split_points[-1]
            target_nodes = int(plan.graph_offsets[prev]) + nodes_per_batch
            pi = int(np.searchsorted(plan.graph_offsets, target_nodes, side='left'))
            pi = min(pi, N)
            if pi <= prev:
                pi = prev + 1
            split_points.append(pi)

    num_batches = len(split_points) - 1
    if verbose:
        print(f"  Phase 3: {N} proofs → {num_batches} sequential batch(es)", flush=True)

    all_results: list[np.ndarray] = []
    all_fail_codes: list[np.ndarray] = []
    remaining = plan
    for batch_idx in range(num_batches):
        if batch_idx < num_batches - 1:
            chunk_size = split_points[batch_idx + 1] - split_points[batch_idx]
            shard, remaining = _split_plan(remaining, chunk_size)
        else:
            shard = remaining

        if verbose:
            print(
                f"    Batch {batch_idx + 1}/{num_batches}: "
                f"{shard.num_proofs} proofs, {shard.total_nodes:,} nodes",
                flush=True,
            )

        result, _, fail_codes = verify_proofs_gpu(shard, device, verbose=verbose)
        all_results.append(result)
        all_fail_codes.append(fail_codes)

        # Explicitly free GPU memory before next batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    proof_passed = np.concatenate(all_results)
    per_proof_fail_codes = np.concatenate(all_fail_codes)
    t_gpu = time.perf_counter() - t0
    return proof_passed, t_gpu, per_proof_fail_codes


def verify_database(
    parsed: ParsedDatabase,
    theorem_labels: list[str] | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
    check_dv: bool = True,
) -> dict[str, str | None]:
    """Verify theorems using true GPU-accelerated verification.

    Phase 1: Build dependency graphs (CPU, O(n), parallel)
    Phase 2: Pack into level-indexed GPU tensors
    Phase 3: Execute level-by-level on GPU
    Phase 4: $d post-check on CPU

    Args:
        parsed: Parsed Metamath database
        theorem_labels: Which theorems to verify (default: all)
        device: GPU device (auto-detected if None)
        verbose: Print progress
        check_dv: Whether to check $d constraints (default True)

    Returns:
        {label: failure_reason} for each theorem.
        None means the proof passed.
        A string is the failure reason — one of:
          "graph:<msg>"        — proof graph construction failed (stack error etc.)
          "ehyp_mismatch"      — essential hypothesis check failed (CUDA kernel)
          "input_failed"       — a dependency node failed (propagated)
          "conclusion_overflow"— conclusion exceeded buffer capacity (bug in pre-pass)
          "dv:<msg>"           — $d disjoint variable constraint violated
          "result_mismatch"    — final expression did not match expected conclusion
    """
    if device is None:
        device = _select_device()

    # Pre-compile all kernels before any timed work
    warmup_cuda(device)
    warmup_numba()

    if theorem_labels is None:
        theorem_labels = [
            lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"
        ]

    if not theorem_labels:
        return {}

    # ── Phase 1: Graph construction ──────────────────────────────
    t0 = time.perf_counter()
    graphs, graph_errors = build_all_proof_graphs_rs(parsed, theorem_labels, verbose=verbose)
    t_graph = time.perf_counter() - t0
    if verbose:
        print(
            f"  Phase 1 (graph construction): {t_graph:.2f}s — "
            f"{len(graphs)} graphs, {len(graph_errors)} errors"
        )
        for lbl, reason in graph_errors:
            print(f"    graph error [{lbl}]: {reason}", flush=True)

    # Build result dict — graph errors are immediate failures
    # None = passed, string = failure reason
    results: dict[str, str | None] = {}
    for lbl, reason in graph_errors:
        results[lbl] = f"graph:{reason}"

    # Any theorem with no graph (not in graph_errors either) also fails
    graph_theorem_labels = {g.theorem_label for g in graphs}
    for lbl in theorem_labels:
        if lbl not in graph_theorem_labels and lbl not in results:
            results[lbl] = "graph:unknown"

    if not graphs:
        return results

    # ── Phase 2: Level packing ───────────────────────────────────
    t1 = time.perf_counter()
    tokenizer = Tokenizer()
    # Sort to ensure deterministic token ID assignment
    for c in sorted(parsed.constants):
        tokenizer.encode_symbol(c)
    for v in sorted(parsed.variables):
        tokenizer.encode_symbol(v)

    plan = pack_levels(graphs, parsed, tokenizer, verbose=verbose)
    t_pack = time.perf_counter() - t1
    if verbose:
        print(
            f"  Phase 2 (level packing): {t_pack:.2f}s — "
            f"{plan.total_nodes} nodes, max_expr_len={plan.max_expr_len}, "
            f"V={plan.vocab_size}, {len(plan.assertion_batches)} levels"
        )

    # ── Phase 3: GPU execution ───────────────────────────────────
    proof_passed, t_gpu, per_proof_fail_codes = _verify_proofs_gpu_batched(plan, device, verbose=verbose)
    if verbose:
        n_pass = int(proof_passed.sum())
        print(
            f"  Phase 3 (GPU execution): {t_gpu:.2f}s — "
            f"{n_pass}/{len(graphs)} proofs passed"
        )

    # ── Phase 4: $d post-check ───────────────────────────────────
    dv_failures: dict[str, str] = {}
    if check_dv:
        t2 = time.perf_counter()
        proof_passed, dv_failures = _check_dv_constraints(parsed, graphs, proof_passed, verbose=verbose)
        t_dv = time.perf_counter() - t2
        if verbose:
            n_pass = int(proof_passed.sum())
            print(
                f"  Phase 4 ($d post-check): {t_dv:.2f}s — "
                f"{n_pass}/{len(graphs)} proofs passed"
            )

    # ── Build final results with failure reasons ─────────────────
    for pi, g in enumerate(graphs):
        lbl = g.theorem_label
        if proof_passed[pi]:
            results[lbl] = None
        elif lbl in dv_failures:
            results[lbl] = f"dv:{dv_failures[lbl]}"
        else:
            code = int(per_proof_fail_codes[pi])
            if code != 0:
                results[lbl] = fail_code_name(code)
            else:
                results[lbl] = "result_mismatch"

    return results
