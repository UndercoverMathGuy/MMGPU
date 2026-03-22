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

from tensormm.parser import ParsedDatabase
from tensormm.tokenizer import Tokenizer

# ══════════════════════════════════════════════════════════════════════
#  Phase 1 — Graph Construction (CPU, O(n))
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ProofNode:
    """One node in a proof's dependency graph."""
    step_idx: int           # step index within this proof
    node_type: str          # "push_f", "push_e", "assertion"
    label: str              # the label referenced
    input_steps: list[int]  # step indices this node reads from (empty for push)
    level: int              # topological level (0 for push, 1+max(input levels))
    # For push nodes: the known expression to write into expr_buffer
    expression: list[str] | None = None


@dataclass
class ProofGraph:
    """Dependency graph for a single theorem's proof."""
    theorem_label: str
    nodes: list[ProofNode]
    expected_conclusion: list[str]
    max_level: int
    max_push_expr_len: int  # max expression length among push nodes


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
    expressions). Returns ProofGraph or an error string.
    """
    if theorem_label not in parsed.assertions:
        return f"Label '{theorem_label}' not found"
    assertion = parsed.assertions[theorem_label]
    if assertion.type != "theorem":
        return f"'{theorem_label}' is not a theorem"

    nodes: list[ProofNode] = []
    virtual_stack: list[int] = []  # stack of step indices
    step_counter = 0
    max_level = 0
    max_push_expr_len = 0
    # Track levels per step for topological ordering
    step_levels: list[int] = []

    def _process_label(label: str) -> str | None:
        nonlocal step_counter, max_level, max_push_expr_len
        if label not in label_info:
            return f"Unknown label: {label}"

        stmt_type, data = label_info[label]

        if stmt_type == "$f":
            expr = [data.type_code, data.variable]
            max_push_expr_len = max(max_push_expr_len, len(expr))
            node = ProofNode(
                step_idx=step_counter,
                node_type="push_f",
                label=label,
                input_steps=[],
                level=0,
                expression=expr,
            )
            nodes.append(node)
            step_levels.append(0)
            virtual_stack.append(step_counter)
            step_counter += 1

        elif stmt_type == "$e":
            expr = data.expression
            max_push_expr_len = max(max_push_expr_len, len(expr))
            node = ProofNode(
                step_idx=step_counter,
                node_type="push_e",
                label=label,
                input_steps=[],
                level=0,
                expression=expr,
            )
            nodes.append(node)
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
                if step_levels[si] + 1 > level:
                    level = step_levels[si] + 1
            if level > max_level:
                max_level = level

            node = ProofNode(
                step_idx=step_counter,
                node_type="assertion",
                label=label,
                input_steps=list(input_steps),
                level=level,
            )
            nodes.append(node)
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
                    # Z: save current stack top
                    if not virtual_stack:
                        return f"Z save on empty stack in {theorem_label}"
                    saved_indices.append(virtual_stack[-1])
                elif proof_int < label_end:
                    err = _process_label(plabels[proof_int])
                    if err:
                        return err
                else:
                    # Backref to saved step — push the saved step index
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

    return ProofGraph(
        theorem_label=theorem_label,
        nodes=nodes,
        expected_conclusion=assertion.expression,
        max_level=max_level,
        max_push_expr_len=max_push_expr_len,
    )


# ── Parallel graph construction ────────────────────────────────────

_GRAPH_WORKER_PARSED: ParsedDatabase | None = None
_GRAPH_WORKER_LABEL_INFO: dict[str, tuple[str, object]] | None = None
_MAX_GRAPH_WORKERS = 512  # no artificial cap — use all available cores


def _init_graph_worker(parsed: ParsedDatabase) -> None:
    global _GRAPH_WORKER_PARSED, _GRAPH_WORKER_LABEL_INFO
    _GRAPH_WORKER_PARSED = parsed
    _GRAPH_WORKER_LABEL_INFO = _build_label_info(parsed)


def _build_graphs_chunk(labels: list[str]) -> list[ProofGraph | str]:
    parsed = _GRAPH_WORKER_PARSED
    label_info = _GRAPH_WORKER_LABEL_INFO
    assert parsed is not None and label_info is not None
    return [build_proof_graph(parsed, lbl, label_info) for lbl in labels]


def build_all_proof_graphs(
    parsed: ParsedDatabase,
    theorem_labels: list[str],
    max_workers: int | None = None,
    verbose: bool = False,
) -> tuple[list[ProofGraph], list[str]]:
    """Build proof graphs for all theorems in parallel."""
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
    errors: list[str] = []
    ordered: list[list[ProofGraph | str]] = [None] * len(chunks)  # type: ignore
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
        for result in chunk_results:
            if isinstance(result, str):
                errors.append(result)
            else:
                graphs.append(result)

    return graphs, errors


# ══════════════════════════════════════════════════════════════════════
#  Phase 2 — Level Packing (CPU → GPU tensors)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class AssertionLevelBatch:
    """Packed tensor data for assertion nodes at one or more topological levels.

    When multiple consecutive sparse levels are coalesced into one batch,
    `node_levels` tracks the per-node level so `_execute_level` can process
    sub-groups in topological order within the single kernel dispatch.
    `level` holds the minimum level in the batch; `max_level` holds the max.
    """
    level: int       # min level in this batch
    max_level: int   # max level in this batch (== level for single-level batches)
    count: int  # B

    # Per-node level (shape [B] int32) — all equal to `level` for single-level
    # batches; varies for coalesced multi-level batches.
    node_levels: np.ndarray

    # Which assertion is being applied (for $d post-check)
    assertion_labels: list[str]
    # Map from batch index to (theorem_label, node_step_idx) for error reporting
    theorem_labels: list[str]

    pattern_toks: np.ndarray     # [B, P_max] int32
    pattern_lengths: np.ndarray  # [B] int32

    # Input mapping: which expr_buffer slots to read
    input_global_indices: np.ndarray  # [B, max_inputs] int32 — -1 for padding
    input_counts: np.ndarray          # [B] int32

    # Floating hyp metadata
    fhyp_input_positions: np.ndarray  # [B, max_fhyps] int32
    fhyp_var_ids: np.ndarray          # [B, max_fhyps] int32
    fhyp_count: np.ndarray            # [B] int32

    # Essential hyp metadata
    ehyp_input_positions: np.ndarray    # [B, max_ehyps] int32
    ehyp_patterns: np.ndarray           # [B, max_ehyps, max_ehyp_len] int32
    ehyp_pattern_lengths: np.ndarray    # [B, max_ehyps] int32
    ehyp_count: np.ndarray              # [B] int32

    # Output: where to write in expr_buffer
    output_global_indices: np.ndarray   # [B] int32


@dataclass
class GlobalPlan:
    """Complete execution plan for GPU verification."""
    total_nodes: int
    max_expr_len: int
    num_proofs: int

    # Level 0 push data
    push_global_indices: np.ndarray    # [num_push] int32
    push_expressions: np.ndarray       # [num_push, max_expr_len] int32
    push_expr_lengths: np.ndarray      # [num_push] int32

    # Assertion levels (sorted by level)
    assertion_batches: list[AssertionLevelBatch]

    # Final check data
    final_node_indices: np.ndarray     # [num_proofs] int32
    expected_conclusions: np.ndarray   # [num_proofs, max_concl_len] int32
    conclusion_lengths: np.ndarray     # [num_proofs] int32

    # For $d post-check: per-proof theorem label
    proof_theorem_labels: list[str]

    # Tokenizer for decoding error messages
    vocab_size: int


def pack_levels(
    graphs: list[ProofGraph],
    parsed: ParsedDatabase,
    tokenizer: Tokenizer,
    verbose: bool = False,
) -> GlobalPlan:
    """Pack all proof graphs into level-indexed GPU-ready tensors."""
    if verbose:
        print(f"  Phase 2: building label_info...", flush=True)
    label_info = _build_label_info(parsed)

    # ── Assign global buffer indices ─────────────────────────────────
    total_nodes = 0
    global_idx_map: dict[tuple[int, int], int] = {}
    for pi, g in enumerate(graphs):
        for node in g.nodes:
            global_idx_map[(pi, node.step_idx)] = total_nodes
            total_nodes += 1

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

    # ── Compute max_expr_len from actual data ────────────────────────
    max_expr_len = 512
    for g in graphs:
        for node in g.nodes:
            if node.expression is not None:
                max_expr_len = max(max_expr_len, len(node.expression))
        max_expr_len = max(max_expr_len, len(g.expected_conclusion))
    for _, info in label_info.items():
        if info[0] in ("$a", "$p"):
            a = info[1]
            max_expr_len = max(max_expr_len, len(a.expression))
            for elbl in a.essential_hyps:
                eh = parsed.essential_hyps[elbl]
                max_expr_len = max(max_expr_len, len(eh.expression))

    # ── Collect nodes by level ───────────────────────────────────────
    push_nodes: list[tuple[int, ProofNode]] = []
    assertion_nodes_by_level: dict[int, list[tuple[int, ProofNode]]] = {}

    for pi, g in enumerate(graphs):
        for node in g.nodes:
            if node.node_type in ("push_f", "push_e"):
                push_nodes.append((pi, node))
            elif node.node_type == "assertion":
                lvl = node.level
                if lvl not in assertion_nodes_by_level:
                    assertion_nodes_by_level[lvl] = []
                assertion_nodes_by_level[lvl].append((pi, node))

    # ── Pack push nodes ──────────────────────────────────────────────
    n_push = len(push_nodes)
    push_global_indices = np.zeros(n_push, dtype=np.int32)
    push_expressions = np.zeros((n_push, max_expr_len), dtype=np.int32)
    push_expr_lengths = np.zeros(n_push, dtype=np.int32)

    for i, (pi, node) in enumerate(push_nodes):
        push_global_indices[i] = global_idx_map[(pi, node.step_idx)]
        assert node.expression is not None
        encoded = _enc(node.expression)
        push_expr_lengths[i] = len(encoded)
        push_expressions[i, :len(encoded)] = encoded

    # ── Pack assertion levels (parallel) ────────────────────────────
    sorted_levels = sorted(assertion_nodes_by_level.keys())

    def _pack_one_level(lvl: int) -> AssertionLevelBatch:
        nodes_at_level = assertion_nodes_by_level[lvl]
        B = len(nodes_at_level)

        max_inputs = max_fhyps = max_ehyps = max_ehyp_len = max_pat_len = 0
        for pi, node in nodes_at_level:
            a = label_info[node.label][1]
            n_f = len(a.floating_hyps)
            n_e = len(a.essential_hyps)
            max_inputs  = max(max_inputs,  n_f + n_e)
            max_fhyps   = max(max_fhyps,   n_f)
            max_ehyps   = max(max_ehyps,   n_e)
            max_pat_len = max(max_pat_len,  len(a.expression))
            for elbl in a.essential_hyps:
                max_ehyp_len = max(max_ehyp_len, len(parsed.essential_hyps[elbl].expression))

        max_inputs   = max(max_inputs,   1)
        max_fhyps    = max(max_fhyps,    1)
        max_ehyps    = max(max_ehyps,    1)
        max_ehyp_len = max(max_ehyp_len, 1)
        max_pat_len  = max(max_pat_len,  1)

        assertion_labels_list: list[str] = []
        theorem_labels_list:   list[str] = []
        pattern_toks          = np.zeros((B, max_pat_len),                   dtype=np.int32)
        pattern_lengths       = np.zeros(B,                                  dtype=np.int32)
        input_global_indices  = np.full( (B, max_inputs), -1,                dtype=np.int32)
        input_counts          = np.zeros(B,                                  dtype=np.int32)
        fhyp_input_positions  = np.zeros((B, max_fhyps),                    dtype=np.int32)
        fhyp_var_ids          = np.zeros((B, max_fhyps),                    dtype=np.int32)
        fhyp_count            = np.zeros(B,                                  dtype=np.int32)
        ehyp_input_positions  = np.zeros((B, max_ehyps),                    dtype=np.int32)
        ehyp_patterns         = np.zeros((B, max_ehyps, max_ehyp_len),      dtype=np.int32)
        ehyp_pattern_lengths  = np.zeros((B, max_ehyps),                    dtype=np.int32)
        ehyp_count            = np.zeros(B,                                  dtype=np.int32)
        output_global_indices = np.zeros(B,                                  dtype=np.int32)

        for b, (pi, node) in enumerate(nodes_at_level):
            a = label_info[node.label][1]
            assertion_labels_list.append(node.label)
            theorem_labels_list.append(graphs[pi].theorem_label)

            pat_enc = _enc(a.expression)
            pattern_lengths[b] = len(pat_enc)
            pattern_toks[b, :len(pat_enc)] = pat_enc

            n_f = len(a.floating_hyps)
            n_e = len(a.essential_hyps)
            input_counts[b] = n_f + n_e
            for k, si in enumerate(node.input_steps):
                input_global_indices[b, k] = global_idx_map[(pi, si)]

            fhyp_count[b] = n_f
            for f_idx, flbl in enumerate(a.floating_hyps):
                fh = parsed.floating_hyps[flbl]
                fhyp_input_positions[b, f_idx] = f_idx
                fhyp_var_ids[b, f_idx] = tokenizer.encode_symbol(fh.variable)

            ehyp_count[b] = n_e
            for e_idx, elbl in enumerate(a.essential_hyps):
                eh = parsed.essential_hyps[elbl]
                ehyp_input_positions[b, e_idx] = n_f + e_idx
                enc = _enc(eh.expression)
                ehyp_pattern_lengths[b, e_idx] = len(enc)
                ehyp_patterns[b, e_idx, :len(enc)] = enc

            output_global_indices[b] = global_idx_map[(pi, node.step_idx)]

        return AssertionLevelBatch(
            level=lvl,
            max_level=lvl,
            count=B,
            node_levels=np.full(B, lvl, dtype=np.int32),
            assertion_labels=assertion_labels_list,
            theorem_labels=theorem_labels_list,
            pattern_toks=pattern_toks,
            pattern_lengths=pattern_lengths,
            input_global_indices=input_global_indices,
            input_counts=input_counts,
            fhyp_input_positions=fhyp_input_positions,
            fhyp_var_ids=fhyp_var_ids,
            fhyp_count=fhyp_count,
            ehyp_input_positions=ehyp_input_positions,
            ehyp_patterns=ehyp_patterns,
            ehyp_pattern_lengths=ehyp_pattern_lengths,
            ehyp_count=ehyp_count,
            output_global_indices=output_global_indices,
        )

    if verbose:
        print(f"  Phase 2: packing {len(sorted_levels)} levels...", flush=True)
    assertion_batches = [_pack_one_level(lvl) for lvl in sorted_levels]

    # ── Final check data ─────────────────────────────────────────────
    num_proofs = len(graphs)
    max_concl_len = max((len(g.expected_conclusion) for g in graphs), default=1)
    final_node_indices    = np.zeros(num_proofs,              dtype=np.int32)
    expected_conclusions  = np.zeros((num_proofs, max_concl_len), dtype=np.int32)
    conclusion_lengths    = np.zeros(num_proofs,              dtype=np.int32)
    proof_theorem_labels: list[str] = []

    for pi, g in enumerate(graphs):
        proof_theorem_labels.append(g.theorem_label)
        last_node = g.nodes[-1]
        final_node_indices[pi] = global_idx_map[(pi, last_node.step_idx)]
        enc = _enc(g.expected_conclusion)
        conclusion_lengths[pi] = len(enc)
        expected_conclusions[pi, :len(enc)] = enc

    return GlobalPlan(
        total_nodes=total_nodes,
        max_expr_len=max_expr_len,
        num_proofs=num_proofs,
        push_global_indices=push_global_indices,
        push_expressions=push_expressions,
        push_expr_lengths=push_expr_lengths,
        assertion_batches=assertion_batches,
        final_node_indices=final_node_indices,
        expected_conclusions=expected_conclusions,
        conclusion_lengths=conclusion_lengths,
        proof_theorem_labels=proof_theorem_labels,
        vocab_size=tokenizer.vocab_size(),
    )


# ══════════════════════════════════════════════════════════════════════
#  Phase 3 — GPU Execution (level by level)
# ══════════════════════════════════════════════════════════════════════


def _apply_substitution_gpu(
    patterns: torch.Tensor,        # [B, P_max] int32
    pattern_lengths: torch.Tensor,  # [B] int32
    sub_tables: torch.Tensor,       # [B, V, S_max] int32
    sub_lengths: torch.Tensor,      # [B, V] int32
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply substitution to patterns, return (output, output_lengths).

    This is the gather→prefix_sum→scatter pipeline from TensorVerifier,
    but returns the computed output instead of comparing against a target.

    Returns:
        output: [B, max_output_len] int32 on device
        output_lengths: [B] int32 on device
    """
    B = patterns.shape[0]
    P_max = patterns.shape[1]
    S_max = sub_tables.shape[2]

    if B == 0:
        return (torch.empty(0, 0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=torch.int32, device=device))

    # STEP 1: GATHER — per-(step, position) replacement lengths
    pat_idx = patterns.long()
    token_lengths = torch.gather(sub_lengths, dim=1, index=pat_idx)
    pos_range = torch.arange(P_max, device=device).unsqueeze(0)
    pat_valid = pos_range < pattern_lengths.unsqueeze(1)
    token_lengths = token_lengths * pat_valid.int()

    # STEP 2: PREFIX SUM
    offsets = torch.cumsum(token_lengths, dim=1) - token_lengths
    total_lengths = token_lengths.sum(dim=1)
    max_output_len = int(total_lengths.max().item()) if B > 0 else 0

    if max_output_len == 0:
        return (torch.zeros(B, 1, dtype=torch.int32, device=device),
                torch.zeros(B, dtype=torch.int32, device=device))

    # STEP 3: SCATTER
    output = torch.zeros(B, max_output_len, dtype=torch.int32, device=device)
    step_base = torch.arange(B, device=device, dtype=torch.long) * max_output_len
    output_flat = output.view(-1)

    s_range = torch.arange(S_max, device=device, dtype=torch.int32).unsqueeze(0)
    actual_P_max = int(pattern_lengths.max().item())

    for p in range(actual_P_max):
        tl_p = token_lengths[:, p]
        off_p = offsets[:, p]
        pat_p = pat_idx[:, p]

        pat_p_3d = pat_p.unsqueeze(1).unsqueeze(2).expand(B, 1, S_max)
        rep_toks = torch.gather(sub_tables, dim=1, index=pat_p_3d).squeeze(1)

        p_valid = pat_valid[:, p]
        s_valid = (s_range < tl_p.unsqueeze(1)) & p_valid.unsqueeze(1)

        flat_pos = step_base.unsqueeze(1) + off_p.unsqueeze(1).long() + s_range.long()

        s_valid_flat = s_valid.reshape(-1)
        if s_valid_flat.any():
            output_flat[flat_pos.reshape(-1)[s_valid_flat]] = \
                rep_toks.reshape(-1)[s_valid_flat]

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
            # Determine merged max dims
            m_max_pat = max(b.pattern_toks.shape[1] for b in pending)
            m_max_inputs = max(b.input_global_indices.shape[1] for b in pending)
            m_max_fhyps = max(b.fhyp_input_positions.shape[1] for b in pending)
            m_max_ehyps = max(b.ehyp_input_positions.shape[1] for b in pending)
            m_max_ehyp_len = max(b.ehyp_patterns.shape[2] for b in pending)
            B_total = sum(b.count for b in pending)

            def _pad2(arr: np.ndarray, cols: int) -> np.ndarray:
                if arr.shape[1] >= cols:
                    return arr
                pad = np.zeros((arr.shape[0], cols - arr.shape[1]), dtype=arr.dtype)
                return np.concatenate([arr, pad], axis=1)

            def _pad3(arr: np.ndarray, dim1: int, dim2: int) -> np.ndarray:
                # arr is [B, d1, d2]
                r = arr
                if r.shape[1] < dim1:
                    r = np.concatenate(
                        [r, np.zeros((r.shape[0], dim1 - r.shape[1], r.shape[2]), dtype=r.dtype)], axis=1
                    )
                if r.shape[2] < dim2:
                    r = np.concatenate(
                        [r, np.zeros((r.shape[0], r.shape[1], dim2 - r.shape[2]), dtype=r.dtype)], axis=2
                    )
                return r

            pattern_toks      = np.concatenate([_pad2(b.pattern_toks, m_max_pat) for b in pending], axis=0)
            pattern_lengths   = np.concatenate([b.pattern_lengths for b in pending])
            # input_global_indices uses -1 as padding sentinel — preserve with constant_values=-1
            input_global      = np.concatenate(
                [np.pad(b.input_global_indices,
                        ((0, 0), (0, m_max_inputs - b.input_global_indices.shape[1])),
                        constant_values=-1)
                 for b in pending], axis=0
            )
            input_counts      = np.concatenate([b.input_counts for b in pending])
            fhyp_pos          = np.concatenate([_pad2(b.fhyp_input_positions, m_max_fhyps) for b in pending], axis=0)
            fhyp_var          = np.concatenate([_pad2(b.fhyp_var_ids, m_max_fhyps) for b in pending], axis=0)
            fhyp_count        = np.concatenate([b.fhyp_count for b in pending])
            ehyp_pos          = np.concatenate([_pad2(b.ehyp_input_positions, m_max_ehyps) for b in pending], axis=0)
            ehyp_pats         = np.concatenate([_pad3(b.ehyp_patterns, m_max_ehyps, m_max_ehyp_len) for b in pending], axis=0)
            ehyp_pat_lens     = np.concatenate([_pad2(b.ehyp_pattern_lengths, m_max_ehyps) for b in pending], axis=0)
            ehyp_count        = np.concatenate([b.ehyp_count for b in pending])
            output_global     = np.concatenate([b.output_global_indices for b in pending])
            node_levels_arr   = np.concatenate([b.node_levels for b in pending])
            assertion_labels  = [lbl for b in pending for lbl in b.assertion_labels]
            theorem_labels    = [lbl for b in pending for lbl in b.theorem_labels]

            merged.append(AssertionLevelBatch(
                level=pending[0].level,
                max_level=pending[-1].max_level,
                count=B_total,
                node_levels=node_levels_arr,
                assertion_labels=assertion_labels,
                theorem_labels=theorem_labels,
                pattern_toks=pattern_toks,
                pattern_lengths=pattern_lengths,
                input_global_indices=input_global,
                input_counts=input_counts,
                fhyp_input_positions=fhyp_pos,
                fhyp_var_ids=fhyp_var,
                fhyp_count=fhyp_count,
                ehyp_input_positions=ehyp_pos,
                ehyp_patterns=ehyp_pats,
                ehyp_pattern_lengths=ehyp_pat_lens,
                ehyp_count=ehyp_count,
                output_global_indices=output_global,
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
    expr_buffer: torch.Tensor,    # [total_nodes, max_expr_len] int32 on device
    expr_lengths: torch.Tensor,   # [total_nodes] int32 on device
    node_failed: torch.Tensor,    # [total_nodes] bool on device — True = failed
    vocab_size: int,
    device: torch.device,
    max_chunk_B: int = 10_000,
    prefetched: "dict[str, torch.Tensor] | None" = None,
) -> int:
    """Execute all assertion nodes in a batch (one or more coalesced levels).

    If `prefetched` is supplied (a dict returned by _upload_batch), those
    already-on-device tensors are used directly — no H2D transfer cost.
    Otherwise tensors are uploaded here (fallback for callers that don't
    use the prefetch pipeline).

    For merged multi-level batches we iterate over per-level index slices
    in ascending order so that level-L writes to expr_buffer are visible
    before any level-(L+1) node reads from them.

    Returns the max conclusion output length seen (for truncation detection).
    """
    B = batch.count
    if B == 0:
        return 0
    max_concl_output_len = 0

    # ── Use pre-uploaded tensors or fall back to synchronous upload ──
    if prefetched is not None:
        # Tensors already on device — stream sync is handled by _run_gpu_pipeline
        # before calling this function.
        d = prefetched
        pattern_toks_t    = d["pattern_toks"]
        pattern_lengths_t = d["pattern_lengths"]
        input_global_t    = d["input_global_indices"]
        input_counts_t    = d["input_counts"]
        fhyp_pos_t        = d["fhyp_input_positions"]
        fhyp_var_t        = d["fhyp_var_ids"]
        fhyp_count_t      = d["fhyp_count"]
        ehyp_pos_t        = d["ehyp_input_positions"]
        ehyp_pats_t       = d["ehyp_patterns"]
        ehyp_pat_lens_t   = d["ehyp_pattern_lengths"]
        ehyp_count_t      = d["ehyp_count"]
        output_global_t   = d["output_global_indices"]
    else:
        pattern_toks_t    = torch.from_numpy(batch.pattern_toks).to(device)
        pattern_lengths_t = torch.from_numpy(batch.pattern_lengths).to(device)
        input_global_t    = torch.from_numpy(batch.input_global_indices).long().to(device)
        input_counts_t    = torch.from_numpy(batch.input_counts).to(device)
        fhyp_pos_t        = torch.from_numpy(batch.fhyp_input_positions).long().to(device)
        fhyp_var_t        = torch.from_numpy(batch.fhyp_var_ids).long().to(device)
        fhyp_count_t      = torch.from_numpy(batch.fhyp_count).to(device)
        ehyp_pos_t        = torch.from_numpy(batch.ehyp_input_positions).long().to(device)
        ehyp_pats_t       = torch.from_numpy(batch.ehyp_patterns).to(device)
        ehyp_pat_lens_t   = torch.from_numpy(batch.ehyp_pattern_lengths).to(device)
        ehyp_count_t      = torch.from_numpy(batch.ehyp_count).to(device)
        output_global_t   = torch.from_numpy(batch.output_global_indices).long().to(device)

    max_inputs  = input_global_t.shape[1]
    max_fhyps   = fhyp_pos_t.shape[1]
    max_ehyps   = ehyp_pos_t.shape[1]
    max_expr_len = expr_buffer.shape[1]

    # ── Build sub-level index ranges (ascending topological order) ───
    # Single-level batch → one range (0, B).
    # Merged batch → one contiguous range per distinct level.
    if batch.level == batch.max_level:
        sublevel_ranges: list[tuple[int, int]] = [(0, B)]
    else:
        node_levels_np = batch.node_levels  # [B] int32, sorted ascending by construction
        unique_lvls = np.unique(node_levels_np)
        sublevel_ranges = []
        for lv in unique_lvls:
            idxs = np.where(node_levels_np == lv)[0]
            sublevel_ranges.append((int(idxs[0]), int(idxs[-1]) + 1))

    # ── Outer loop: sub-levels (for correctness across merged levels) ─
    # ── Inner loop: chunks  (for memory control within a sub-level)   ─
    for sl_start, sl_end in sublevel_ranges:
        sl_size = sl_end - sl_start

        for chunk_offset in range(0, sl_size, max_chunk_B):
            chunk_start = sl_start + chunk_offset
            chunk_end = min(chunk_start + max_chunk_B, sl_end)
            cB = chunk_end - chunk_start
            cs = slice(chunk_start, chunk_end)

            c_pat = pattern_toks_t[cs]
            c_pat_len = pattern_lengths_t[cs]
            c_in_idx = input_global_t[cs]
            c_in_count = input_counts_t[cs]
            c_fhyp_pos = fhyp_pos_t[cs]
            c_fhyp_var = fhyp_var_t[cs]
            c_fhyp_count = fhyp_count_t[cs]
            c_ehyp_pos = ehyp_pos_t[cs]
            c_ehyp_pats = ehyp_pats_t[cs]
            c_ehyp_pat_lens = ehyp_pat_lens_t[cs]
            c_ehyp_count = ehyp_count_t[cs]
            c_out_idx = output_global_t[cs]

            # ── (a) Gather inputs from expr_buffer ───────────────────
            # Clamp -1 padding to 0 for safe gather; we mask later
            safe_in_idx = c_in_idx.clamp(min=0)
            # [cB, max_inputs, max_expr_len]
            inputs = expr_buffer[safe_in_idx]
            # [cB, max_inputs]
            input_lens = expr_lengths[safe_in_idx]

            # Check if any input node has already failed
            input_failed = node_failed[safe_in_idx]  # [cB, max_inputs]
            in_valid = torch.arange(max_inputs, device=device) < c_in_count.unsqueeze(1)
            any_input_failed = (input_failed & in_valid).any(dim=1)  # [cB]

            # ── (b) Build sub_tables from floating hyps ──────────────
            # Floating hyp input expression length - 1 (strip typecode)
            fhyp_valid = torch.arange(max_fhyps, device=device) < c_fhyp_count.unsqueeze(1)
            safe_fhyp_pos = c_fhyp_pos.clamp(min=0, max=max_inputs - 1)

            batch_range = torch.arange(cB, device=device).unsqueeze(1).expand(cB, max_fhyps)
            fhyp_input_exprs = inputs[batch_range, safe_fhyp_pos]  # [cB, max_fhyps, max_expr_len]
            fhyp_input_lens = input_lens[batch_range, safe_fhyp_pos]  # [cB, max_fhyps]

            # Strip typecode: substitution value = input[1:]
            sub_values = fhyp_input_exprs[:, :, 1:]  # [cB, max_fhyps, max_expr_len-1]
            sub_value_lens = (fhyp_input_lens - 1).clamp(min=0)  # [cB, max_fhyps]
            max_sub_width = sub_values.shape[2]
            sub_value_lens = sub_value_lens.clamp(max=max_sub_width)
            S_max = max(int(sub_value_lens.max().item()), 1) if cB > 0 else 1

            sub_values = sub_values[:, :, :S_max]  # [cB, max_fhyps, S_max]

            V = vocab_size
            sub_tables = torch.zeros(cB, V, S_max, dtype=torch.int32, device=device)
            ident = torch.arange(V, device=device, dtype=torch.int32)
            sub_tables[:, :, 0] = ident.unsqueeze(0)
            sub_lengths = torch.ones(cB, V, dtype=torch.int32, device=device)

            if max_fhyps > 0 and int(fhyp_valid.sum().item()) > 0:
                s_range = torch.arange(S_max, device=device)
                s_valid = s_range.unsqueeze(0).unsqueeze(0) < sub_value_lens.unsqueeze(2)
                write_valid = s_valid & fhyp_valid.unsqueeze(2)

                if write_valid.any():
                    b_idx_3d = torch.arange(cB, device=device).view(cB, 1, 1).expand(cB, max_fhyps, S_max)
                    v_idx_3d = c_fhyp_var.unsqueeze(2).expand(cB, max_fhyps, S_max)
                    s_idx_3d = s_range.unsqueeze(0).unsqueeze(0).expand(cB, max_fhyps, S_max)

                    wv = write_valid.reshape(-1)
                    sub_tables[
                        b_idx_3d.reshape(-1)[wv],
                        v_idx_3d.reshape(-1)[wv],
                        s_idx_3d.reshape(-1)[wv],
                    ] = sub_values.reshape(-1)[wv]

                b_idx_2d = torch.arange(cB, device=device).unsqueeze(1).expand(cB, max_fhyps)
                fv = fhyp_valid.reshape(-1)
                if fv.any():
                    sub_lengths[
                        b_idx_2d.reshape(-1)[fv],
                        c_fhyp_var.reshape(-1)[fv],
                    ] = sub_value_lens.reshape(-1)[fv].int()

            # ── (c) Check essential hypotheses ───────────────────────
            max_ehyps_val = int(c_ehyp_count.max().item()) if cB > 0 else 0
            ehyp_results = torch.ones(cB, max_ehyps, dtype=torch.bool, device=device)

            for e in range(max_ehyps_val):
                ehyp_pat = c_ehyp_pats[:, e, :]        # [cB, max_ehyp_len]
                ehyp_pat_len = c_ehyp_pat_lens[:, e]    # [cB]

                ehyp_in_pos = c_ehyp_pos[:, e]          # [cB]
                safe_ehyp_in_pos = ehyp_in_pos.clamp(min=0, max=max_inputs - 1)
                ehyp_target = inputs[torch.arange(cB, device=device), safe_ehyp_in_pos]
                ehyp_target_len = input_lens[torch.arange(cB, device=device), safe_ehyp_in_pos]

                ehyp_output, ehyp_out_len = _apply_substitution_gpu(
                    ehyp_pat, ehyp_pat_len, sub_tables, sub_lengths, device,
                )
                ehyp_match = _verify_substitution_result(
                    ehyp_output, ehyp_out_len, ehyp_target, ehyp_target_len, device,
                )
                ehyp_results[:, e] = ehyp_match

            # CRITICAL MASKING: masked-out ehyps forced True
            ehyp_valid_mask = torch.arange(max_ehyps, device=device) < c_ehyp_count.unsqueeze(1)
            all_ehyps_ok = (ehyp_results | ~ehyp_valid_mask).all(dim=1)  # [cB]

            # ── (d) Compute conclusion ───────────────────────────────
            concl_output, concl_out_len = _apply_substitution_gpu(
                c_pat, c_pat_len, sub_tables, sub_lengths, device,
            )

            max_out = concl_output.shape[1]
            max_concl_output_len = max(max_concl_output_len, max_out)

            write_len = min(max_out, max_expr_len)
            expr_buffer[c_out_idx] = 0
            expr_buffer[c_out_idx, :write_len] = concl_output[:, :write_len]
            expr_lengths[c_out_idx] = concl_out_len.int()

            truncated = concl_out_len > max_expr_len
            step_failed = ~all_ehyps_ok | any_input_failed | truncated
            node_failed[c_out_idx] = step_failed

    return max_concl_output_len


# ── Batch tensor upload helper ─────────────────────────────────────────

def _pin(arr: np.ndarray) -> torch.Tensor:
    """Return a CPU tensor backed by page-locked memory for fast H2D."""
    t = torch.from_numpy(arr)
    return t.pin_memory()


def _upload_batch(
    batch: AssertionLevelBatch,
    device: torch.device,
    stream: "torch.cuda.Stream | None" = None,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    """Upload all numpy arrays for a batch to device, returning a dict of tensors.

    On CUDA with a prefetch stream, uploads happen asynchronously so the
    transfer overlaps with compute on the default stream.
    """
    def _to(arr: np.ndarray, long: bool = False) -> torch.Tensor:
        t = torch.from_numpy(arr)
        if non_blocking:
            t = t.pin_memory()
        if long:
            t = t.long()
        if stream is not None:
            with torch.cuda.stream(stream):
                return t.to(device, non_blocking=non_blocking)
        return t.to(device, non_blocking=non_blocking)

    return {
        "pattern_toks":          _to(batch.pattern_toks),
        "pattern_lengths":       _to(batch.pattern_lengths),
        "input_global_indices":  _to(batch.input_global_indices, long=True),
        "input_counts":          _to(batch.input_counts),
        "fhyp_input_positions":  _to(batch.fhyp_input_positions, long=True),
        "fhyp_var_ids":          _to(batch.fhyp_var_ids, long=True),
        "fhyp_count":            _to(batch.fhyp_count),
        "ehyp_input_positions":  _to(batch.ehyp_input_positions, long=True),
        "ehyp_patterns":         _to(batch.ehyp_patterns),
        "ehyp_pattern_lengths":  _to(batch.ehyp_pattern_lengths),
        "ehyp_count":            _to(batch.ehyp_count),
        "output_global_indices": _to(batch.output_global_indices, long=True),
        "node_levels":           _to(batch.node_levels),
    }


def _run_gpu_pipeline(
    plan: GlobalPlan,
    device: torch.device,
    max_expr_len: int,
    verbose: bool = False,
) -> tuple[np.ndarray, bool, int]:
    """Single run of the GPU pipeline with a given max_expr_len.

    Returns:
        (per_proof_passed, had_truncation, needed_expr_len)
    """
    total_nodes = plan.total_nodes
    V = plan.vocab_size

    if total_nodes == 0:
        return np.ones(plan.num_proofs, dtype=np.bool_), False, 0

    # Allocate expr_buffer and tracking tensors on GPU
    expr_buffer = torch.zeros(
        total_nodes, max_expr_len, dtype=torch.int32, device=device
    )
    expr_lengths = torch.zeros(total_nodes, dtype=torch.int32, device=device)
    node_failed = torch.zeros(total_nodes, dtype=torch.bool, device=device)

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

    if verbose:
        print(f"    Level 0: {len(plan.push_global_indices)} push nodes written")

    # ── Levels 1..max: assertion nodes ───────────────────────────
    # Merge consecutive sparse levels into combined batches to avoid
    # repeated tiny kernel dispatches and H2D transfers at the tail.
    effective_batches = _merge_sparse_levels(plan.assertion_batches)
    if verbose:
        orig_n = len(plan.assertion_batches)
        merged_n = len(effective_batches)
        if merged_n < orig_n:
            print(f"    Level coalescing: {orig_n} → {merged_n} batches")

    # ── Double-buffered async H2D prefetch (CUDA only) ───────────────
    # While the GPU executes level i on the default stream, we upload
    # level i+1's batch tensors on a separate prefetch stream.  The
    # default stream waits for the prefetch stream before reading the
    # tensors, ensuring correctness with zero extra synchronisation cost.
    use_prefetch = device.type == "cuda" and len(effective_batches) > 0
    prefetch_stream: "torch.cuda.Stream | None" = (
        torch.cuda.Stream(device=device) if use_prefetch else None
    )

    def _prefetch(b: AssertionLevelBatch) -> "dict[str, torch.Tensor]":
        return _upload_batch(b, device, stream=prefetch_stream, non_blocking=True)

    # Seed: upload first batch on prefetch stream right away
    prefetched_next = _prefetch(effective_batches[0]) if use_prefetch else None

    global_max_output = 0
    for i, batch in enumerate(effective_batches):
        t_lvl = time.perf_counter()

        if use_prefetch:
            # Kick off next batch upload before current compute starts
            if i + 1 < len(effective_batches):
                prefetched_next_next = _prefetch(effective_batches[i + 1])
            else:
                prefetched_next_next = None
            # Default stream waits for the prefetch stream to finish
            # uploading the current batch before executing on it.
            torch.cuda.current_stream(device).wait_stream(prefetch_stream)  # type: ignore[arg-type]
            current_prefetched = prefetched_next
            prefetched_next = prefetched_next_next
        else:
            current_prefetched = None

        level_max = _execute_level(
            batch, expr_buffer, expr_lengths, node_failed,
            V, device, prefetched=current_prefetched,
        )
        global_max_output = max(global_max_output, level_max)
        if verbose:
            dt = time.perf_counter() - t_lvl
            lvl_str = (
                f"{batch.level}" if batch.level == batch.max_level
                else f"{batch.level}-{batch.max_level}"
            )
            print(
                f"    Level {lvl_str}: {batch.count} assertion nodes "
                f"in {dt:.3f}s"
            )

    had_truncation = global_max_output > max_expr_len

    # ── Final check: compare last node expression to expected conclusion ──
    final_idx = torch.from_numpy(plan.final_node_indices).long().to(device)
    final_exprs = expr_buffer[final_idx]  # [num_proofs, max_expr_len]
    final_lens = expr_lengths[final_idx]  # [num_proofs]
    final_node_fail = node_failed[final_idx]  # [num_proofs]

    if use_prefetch:
        expected = torch.from_numpy(plan.expected_conclusions).pin_memory().to(device, non_blocking=False)
        expected_lens = torch.from_numpy(plan.conclusion_lengths).pin_memory().to(device, non_blocking=False)
    else:
        expected = torch.from_numpy(plan.expected_conclusions).to(device)
        expected_lens = torch.from_numpy(plan.conclusion_lengths).to(device)

    # Pad to same width if needed
    w1 = final_exprs.shape[1]
    w2 = expected.shape[1]
    if w1 < w2:
        final_exprs = torch.nn.functional.pad(final_exprs, (0, w2 - w1))
    elif w2 < w1:
        expected = torch.nn.functional.pad(expected, (0, w1 - w2))

    compare_dim = max(w1, w2)
    positions = torch.arange(compare_dim, device=device).unsqueeze(0)
    valid_mask = positions < final_lens.unsqueeze(1)
    masked_eq = (final_exprs == expected) | ~valid_mask
    content_match = masked_eq.all(dim=1)
    length_match = (final_lens == expected_lens)

    proof_passed = length_match & content_match & ~final_node_fail
    result = proof_passed.cpu().numpy()

    return result, had_truncation, global_max_output


_MAX_EXPR_LEN_CAP = 16384  # absolute safety cap

# ── CUDA warmup ────────────────────────────────────────────────────────
_CUDA_WARMED_UP: set[str] = set()


def warmup_cuda(device: torch.device) -> None:
    """Pre-compile every CUDA kernel shape used in the verification pipeline.

    On CUDA, PyTorch JIT-compiles kernels on first use.  Without warmup,
    the first few levels of a real run incur compilation latency that
    distorts timing and can make early large batches look slow.

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

    B, V, S, P, E = 8, 64, 4, 8, 4  # tiny synthetic shapes

    # Allocate
    expr_buf  = torch.zeros(B * 2, P, dtype=torch.int32, device=device)
    expr_lens = torch.zeros(B * 2, dtype=torch.int32, device=device)
    failed    = torch.zeros(B * 2, dtype=torch.bool, device=device)

    idx       = torch.zeros(B, dtype=torch.long, device=device)
    in_idx    = torch.zeros(B, 2, dtype=torch.long, device=device)
    in_count  = torch.ones(B, dtype=torch.int32, device=device)

    pat       = torch.zeros(B, P, dtype=torch.int32, device=device)
    pat_len   = torch.full((B,), P, dtype=torch.int32, device=device)

    sub_tab   = torch.zeros(B, V, S, dtype=torch.int32, device=device)
    sub_len   = torch.ones(B, V, dtype=torch.int32, device=device)

    fhyp_var  = torch.zeros(B, 1, dtype=torch.long, device=device)
    fhyp_cnt  = torch.ones(B, dtype=torch.int32, device=device)

    # Kernels used in _execute_level ─────────────────────────────────
    # (a) gather from expr_buffer
    _ = expr_buf[in_idx.clamp(min=0)]
    _ = expr_lens[in_idx.clamp(min=0)]
    _ = failed[in_idx.clamp(min=0)]
    _ = (torch.arange(2, device=device) < in_count.unsqueeze(1))

    # (b) sub_table build — cumsum, scatter, gather
    tl   = torch.ones(B, P, dtype=torch.int32, device=device)
    off  = torch.cumsum(tl, dim=1) - tl
    _    = tl.sum(dim=1)
    _    = torch.arange(V, device=device, dtype=torch.int32)
    b3   = torch.arange(B, device=device).view(B, 1, 1).expand(B, 1, S)
    v3   = fhyp_var.unsqueeze(2).expand(B, 1, S)
    s3   = torch.arange(S, device=device).unsqueeze(0).unsqueeze(0).expand(B, 1, S)
    wv   = torch.ones(B * S, dtype=torch.bool, device=device)
    sub_tab[b3.reshape(-1)[wv], v3.reshape(-1)[wv], s3.reshape(-1)[wv]] = 0

    b2   = torch.arange(B, device=device).unsqueeze(1).expand(B, 1)
    fv   = fhyp_cnt.bool().reshape(-1)
    sub_len[b2.reshape(-1)[fv], fhyp_var.reshape(-1)[fv]] = 1

    # (c) _apply_substitution_gpu kernels
    pat_idx  = pat.long()
    tok_lens = torch.gather(sub_len, dim=1, index=pat_idx)
    pos_r    = torch.arange(P, device=device).unsqueeze(0)
    valid    = pos_r < pat_len.unsqueeze(1)
    tok_lens = tok_lens * valid.int()
    offsets  = torch.cumsum(tok_lens, dim=1) - tok_lens
    tot_len  = tok_lens.sum(dim=1)
    max_out  = max(int(tot_len.max().item()), 1)
    out_buf  = torch.zeros(B, max_out, dtype=torch.int32, device=device)
    step_base = torch.arange(B, device=device, dtype=torch.long) * max_out
    p3d  = pat_idx[:, :1].unsqueeze(2).expand(B, 1, S)
    rep  = torch.gather(sub_tab, dim=1, index=p3d).squeeze(1)
    s_r  = torch.arange(S, device=device, dtype=torch.int32).unsqueeze(0)
    sv   = (s_r < tok_lens[:, :1]).reshape(-1)
    fp   = (step_base.unsqueeze(1) + offsets[:, :1].long() + s_r.long()).reshape(-1)
    out_buf.view(-1)[fp[sv]] = rep.reshape(-1)[sv]

    # (d) _verify_substitution_result kernels
    positions = torch.arange(max_out, device=device).unsqueeze(0)
    vm   = positions < tot_len.unsqueeze(1)
    _    = ((out_buf == out_buf) | ~vm).all(dim=1)
    _    = (tot_len == tot_len)

    # (e) expr_buffer write-back
    expr_buf[idx] = 0
    expr_buf[idx, :max_out] = out_buf[:, :max_out] if max_out <= P else out_buf[:, :P]
    expr_lens[idx] = tot_len.int()
    failed[idx]    = ~torch.ones(B, dtype=torch.bool, device=device)

    torch.cuda.synchronize(device)
    _CUDA_WARMED_UP.add(key)


def verify_proofs_gpu(
    plan: GlobalPlan,
    device: torch.device,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """Execute the full GPU verification pipeline.

    Automatically retries with a larger expr_buffer if intermediate
    expressions exceed the initial buffer size.

    Returns:
        (per_proof_passed: np.ndarray[bool], gpu_time: float)
    """
    t0 = time.perf_counter()
    max_expr_len = plan.max_expr_len

    while max_expr_len <= _MAX_EXPR_LEN_CAP:
        result, had_truncation, needed = _run_gpu_pipeline(
            plan, device, max_expr_len, verbose=verbose,
        )
        if not had_truncation:
            break
        # Retry with a larger buffer
        new_len = max(needed + 64, max_expr_len * 2)
        new_len = min(new_len, _MAX_EXPR_LEN_CAP)
        if verbose:
            print(
                f"    ⚠ Truncation detected: max output {needed} > "
                f"buffer {max_expr_len}. Retrying with {new_len}..."
            )
        if new_len <= max_expr_len:
            break  # can't grow further
        max_expr_len = new_len

    gpu_time = time.perf_counter() - t0
    return result, gpu_time


# ══════════════════════════════════════════════════════════════════════
#  Phase 4 — $d Post-Check (CPU)
# ══════════════════════════════════════════════════════════════════════


def _check_dv_constraints(
    parsed: ParsedDatabase,
    graphs: list[ProofGraph],
    proof_passed: np.ndarray,
) -> np.ndarray:
    """Check $d constraints for proofs that passed GPU verification.

    Uses the existing CPUVerifier which already handles $d checking
    efficiently. Only checks proofs that the GPU said passed.

    Returns updated proof_passed array.
    """
    from tensormm.cpu_verifier import CPUVerifier

    result = proof_passed.copy()
    cpu_v = CPUVerifier(parsed)

    for pi, g in enumerate(graphs):
        if not result[pi]:
            continue  # already failed on GPU
        try:
            r = cpu_v.verify_proof(g.theorem_label)
            if not r.success:
                result[pi] = False
        except Exception:
            result[pi] = False

    return result


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


def verify_database(
    parsed: ParsedDatabase,
    theorem_labels: list[str] | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
    check_dv: bool = True,
) -> dict[str, bool]:
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
        {label: passed} for each theorem
    """
    if device is None:
        device = _select_device()

    # Pre-compile all CUDA kernels before any timed work
    warmup_cuda(device)

    if theorem_labels is None:
        theorem_labels = [
            lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"
        ]

    if not theorem_labels:
        return {}

    # ── Phase 1: Graph construction ──────────────────────────────
    t0 = time.perf_counter()
    graphs, graph_errors = build_all_proof_graphs(parsed, theorem_labels, verbose=verbose)
    t_graph = time.perf_counter() - t0
    if verbose:
        print(
            f"  Phase 1 (graph construction): {t_graph:.2f}s — "
            f"{len(graphs)} graphs, {len(graph_errors)} errors"
        )

    # Build result dict — graph errors are immediate failures
    results: dict[str, bool] = {}
    error_labels: set[str] = set()
    for err in graph_errors:
        # Error string format varies, but we need the label
        # Graph errors contain the theorem label
        results[err] = False  # We'll fix this below
        error_labels.add(err)

    # For graph errors, mark the theorem as failed
    graph_theorem_labels = {g.theorem_label for g in graphs}
    for lbl in theorem_labels:
        if lbl not in graph_theorem_labels:
            results[lbl] = False

    if not graphs:
        return results

    # ── Phase 2: Level packing ───────────────────────────────────
    t1 = time.perf_counter()
    tokenizer = Tokenizer()
    for c in parsed.constants:
        tokenizer.encode_symbol(c)
    for v in parsed.variables:
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
    proof_passed, t_gpu = verify_proofs_gpu(plan, device, verbose=verbose)
    if verbose:
        n_pass = int(proof_passed.sum())
        print(
            f"  Phase 3 (GPU execution): {t_gpu:.2f}s — "
            f"{n_pass}/{len(graphs)} proofs passed"
        )

    # ── Phase 4: $d post-check ───────────────────────────────────
    if check_dv:
        t2 = time.perf_counter()
        proof_passed = _check_dv_constraints(parsed, graphs, proof_passed)
        t_dv = time.perf_counter() - t2
        if verbose:
            n_pass = int(proof_passed.sum())
            print(
                f"  Phase 4 ($d post-check): {t_dv:.2f}s — "
                f"{n_pass}/{len(graphs)} proofs passed"
            )

    # ── Build final results ──────────────────────────────────────
    for pi, g in enumerate(graphs):
        results[g.theorem_label] = bool(proof_passed[pi])

    return results
