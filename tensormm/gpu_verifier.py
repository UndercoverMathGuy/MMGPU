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


@dataclass(slots=True)
class ProofNode:
    """One node in a proof's dependency graph.

    Uses __slots__ to reduce per-object memory from ~400+ bytes (dict-based)
    to ~100 bytes. For set.mm with ~6M nodes, this saves ~1.8 GB.
    """
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
        total_nodes += len(g.nodes)
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
    used_labels: list[str] = []
    seen_labels: set[str] = set()
    for g in graphs:
        for node in g.nodes:
            if node.node_type == "assertion" and node.label not in seen_labels:
                seen_labels.add(node.label)
                used_labels.append(node.label)

    A = len(used_labels)
    del seen_labels  # no longer needed
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
    tbl_pattern_toks          = np.zeros((A, global_max_pat_len),                     dtype=np.int32)
    tbl_pattern_lengths       = np.zeros(A,                                            dtype=np.int32)
    tbl_fhyp_var_ids          = np.zeros((A, global_max_fhyps),                       dtype=np.int32)
    tbl_fhyp_count            = np.zeros(A,                                            dtype=np.int32)
    tbl_ehyp_patterns         = np.zeros((A, global_max_ehyps, global_max_ehyp_len),  dtype=np.int32)
    tbl_ehyp_pattern_lengths  = np.zeros((A, global_max_ehyps),                       dtype=np.int32)
    tbl_ehyp_count            = np.zeros(A,                                            dtype=np.int32)

    if verbose:
        sz_gb = (tbl_pattern_toks.nbytes + tbl_ehyp_patterns.nbytes) / 1e9
        print(f"  Phase 2: assertion table size: {sz_gb:.2f} GB", flush=True)

    for i, (pat_enc, n_f, fhyp_var_ids_list, n_e, ehyp_encs) in enumerate(_table_cache):
        tbl_pattern_lengths[i] = len(pat_enc)
        tbl_pattern_toks[i, :len(pat_enc)] = pat_enc
        tbl_fhyp_count[i] = n_f
        for f_idx, vid in enumerate(fhyp_var_ids_list):
            tbl_fhyp_var_ids[i, f_idx] = vid
        tbl_ehyp_count[i] = n_e
        for e_idx, enc in enumerate(ehyp_encs):
            tbl_ehyp_pattern_lengths[i, e_idx] = len(enc)
            tbl_ehyp_patterns[i, e_idx, :len(enc)] = enc
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
    # Push expressions are short (2 tokens for $f, short for $e).
    # Allocate at actual max push width — NOT max_expr_len (which can be
    # 512+ and would produce a multi-GB array for set.mm's ~10M push nodes).
    sorted_levels = sorted(assertion_nodes_by_level.keys())
    if verbose:
        print(f"  Phase 2: packing {len(push_nodes):,} push nodes, {len(sorted_levels)} levels, max_expr_len={max_expr_len}...", flush=True)
    # Single-pass: max_push_width is known from Phase 1's per-graph
    # max_push_expr_len, so we can pre-allocate the numpy array and fill
    # directly without any intermediate Python list-of-lists.
    num_push = len(push_nodes)
    max_push_width = max((g.max_push_expr_len for g in graphs), default=1)
    push_global_indices = np.empty(num_push, dtype=np.int32)
    push_expr_lengths   = np.empty(num_push, dtype=np.int32)
    push_expressions    = np.zeros((num_push, max_push_width), dtype=np.int32)

    for i, (pi, node) in enumerate(push_nodes):
        assert node.expression is not None
        enc = _enc(node.expression)
        push_global_indices[i] = int(graph_offsets[pi]) + node.step_idx
        push_expr_lengths[i] = len(enc)
        push_expressions[i, :len(enc)] = enc

    # ── Pack assertion levels ────────────────────────────────────────

    def _pack_one_level(lvl: int) -> AssertionLevelBatch:
        nodes_at_level = assertion_nodes_by_level[lvl]
        B = len(nodes_at_level)

        # Single-pass: compute dimensions AND collect per-node data
        # simultaneously. The old code iterated nodes_at_level twice
        # (once for dimensions, once for array fill) with redundant
        # label_info lookups. This merges both into one pass.
        max_inputs = max_fhyps = max_ehyps = 0
        # Pre-collect assertion info to avoid double dict lookups
        node_info: list[tuple[int, ProofNode, int, int]] = []  # (pi, node, n_f, n_e)
        for pi, node in nodes_at_level:
            a = label_info[node.label][1]
            n_f = len(a.floating_hyps)
            n_e = len(a.essential_hyps)
            node_info.append((pi, node, n_f, n_e))
            if n_f + n_e > max_inputs: max_inputs = n_f + n_e
            if n_f > max_fhyps:        max_fhyps  = n_f
            if n_e > max_ehyps:        max_ehyps  = n_e

        max_inputs = max(max_inputs, 1)
        max_fhyps  = max(max_fhyps,  1)
        max_ehyps  = max(max_ehyps,  1)

        assertion_labels_list: list[str] = []
        theorem_labels_list:   list[str] = []
        assertion_idx_arr     = np.zeros(B,                    dtype=np.int32)
        input_global_indices  = np.full((B, max_inputs), -1,   dtype=np.int32)
        input_counts          = np.zeros(B,                    dtype=np.int32)
        fhyp_input_positions  = np.zeros((B, max_fhyps),      dtype=np.int32)
        ehyp_input_positions  = np.zeros((B, max_ehyps),      dtype=np.int32)
        output_global_indices = np.zeros(B,                    dtype=np.int32)

        for b, (pi, node, n_f, n_e) in enumerate(node_info):
            assertion_labels_list.append(node.label)
            theorem_labels_list.append(graphs[pi].theorem_label)
            assertion_idx_arr[b] = label_to_idx[node.label]
            input_counts[b] = n_f + n_e
            for k, si in enumerate(node.input_steps):
                input_global_indices[b, k] = int(graph_offsets[pi]) + si

            for f_idx in range(n_f):
                fhyp_input_positions[b, f_idx] = f_idx

            for e_idx in range(n_e):
                ehyp_input_positions[b, e_idx] = n_f + e_idx

            output_global_indices[b] = int(graph_offsets[pi]) + node.step_idx

        return AssertionLevelBatch(
            level=lvl,
            max_level=lvl,
            count=B,
            node_levels=np.full(B, lvl, dtype=np.int32),
            assertion_labels=assertion_labels_list,
            theorem_labels=theorem_labels_list,
            assertion_idx=assertion_idx_arr,
            input_global_indices=input_global_indices,
            input_counts=input_counts,
            fhyp_input_positions=fhyp_input_positions,
            ehyp_input_positions=ehyp_input_positions,
            output_global_indices=output_global_indices,
            sublevel_ranges=[(0, B)],
        )

    if verbose:
        print(f"  Phase 2: packing {len(sorted_levels)} levels...", flush=True)
    assertion_batches = [_pack_one_level(lvl) for lvl in sorted_levels]
    # Free node collections — now packed into assertion_batches / push arrays
    del push_nodes, assertion_nodes_by_level, sorted_levels

    # ── Final check data ─────────────────────────────────────────────
    # Cap the stored conclusion width to max_expr_len: conclusions longer
    # than the expr_buffer use rolling-hash comparison (not token comparison),
    # so storing their full token sequences wastes memory. For set.mm this
    # avoids a num_proofs × 11548 array (1.3 GB) in favour of
    # num_proofs × 1024 (120 MB).
    num_proofs = len(graphs)
    max_concl_stored = max_expr_len
    final_node_indices         = np.zeros(num_proofs,                    dtype=np.int32)
    expected_conclusions       = np.zeros((num_proofs, max_concl_stored), dtype=np.int32)
    conclusion_lengths         = np.zeros(num_proofs,                    dtype=np.int32)
    expected_conclusion_hashes = np.zeros(num_proofs,                    dtype=np.int64)
    proof_theorem_labels: list[str] = []

    for pi, g in enumerate(graphs):
        proof_theorem_labels.append(g.theorem_label)
        last_node = g.nodes[-1]
        final_node_indices[pi] = int(graph_offsets[pi]) + last_node.step_idx
        enc = np.array(_enc(g.expected_conclusion), dtype=np.int32)
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
    )


# ══════════════════════════════════════════════════════════════════════
#  Phase 3 — GPU Execution (level by level)
# ══════════════════════════════════════════════════════════════════════

# Polynomial rolling hash base (large prime, fits int64 without overflow issues
# when multiplied by token ids up to ~100k and added).
_HASH_BASE = np.int64(1_000_000_007)


_HASH_MASK = (1 << 63) - 1  # keep within signed int64 range


def _poly_hash_np(tokens: np.ndarray) -> np.int64:
    """Compute polynomial rolling hash of a 1-D int32 token array on CPU.

    Uses Python int arithmetic (arbitrary precision) then masks to int64 so
    the result matches the GPU int64 wrap-around behaviour exactly.
    """
    base = int(_HASH_BASE)
    h = 0
    for t in tokens:
        h = (h * base + int(t)) & 0xFFFFFFFFFFFFFFFF
    # Reinterpret as signed int64
    if h >= (1 << 63):
        h -= (1 << 64)
    return np.int64(h)


def _poly_hash_gpu(
    tokens: torch.Tensor,       # [B, L] int32 — possibly padded
    lengths: torch.Tensor,      # [B] int32
    device: torch.device,
) -> torch.Tensor:              # [B] int64
    """Compute polynomial rolling hash of variable-length token rows on GPU."""
    B, L = tokens.shape
    # Only iterate up to the actual longest row — skip trailing padding.
    # For set.mm with max_expr_len=1024 but typical lengths ~100-200,
    # this saves 4-5x iterations on average.
    actual_L = min(L, int(lengths.max().item())) if B > 0 else 0
    h = torch.zeros(B, dtype=torch.int64, device=device)
    if actual_L == 0:
        return h
    base = torch.tensor(_HASH_BASE, dtype=torch.int64, device=device)
    # Convert tokens to int64 once upfront instead of per-iteration
    tokens_long = tokens[:, :actual_L].long()
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
        output: [B, max_output_len] int32 on device
        output_lengths: [B] int32 on device
    """
    B = patterns.shape[0]
    P_max = patterns.shape[1]
    max_fhyps = var_ids.shape[1]
    S_max = var_sub_values.shape[2]

    if B == 0:
        return (torch.empty(0, 0, dtype=torch.int32, device=device),
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
        return (torch.zeros(B, 1, dtype=torch.int32, device=device),
                torch.zeros(B, dtype=torch.int32, device=device))

    # STEP 4: SCATTER — fully vectorized (no Python position loop).
    # Old code looped `for p in range(actual_P_max)` which for patterns
    # of length ~200 meant 200 Python iterations × 3-4 GPU kernel launches
    # each = ~600-800 kernel launches. Now replaced with two vectorized
    # scatter operations (constants + variables).
    output = torch.zeros(B, max_output_len, dtype=torch.int32, device=device)
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


def _run_gpu_pipeline(
    plan: GlobalPlan,
    device: torch.device,
    max_expr_len: int,
    verbose: bool = False,
) -> tuple[np.ndarray, bool, int]:
    """Single run of the GPU pipeline with a given max_expr_len.

    Intermediate expressions that exceed max_expr_len are stored truncated —
    this corrupts downstream substitutions, so the caller must retry with a
    larger buffer if this happens.

    Final-step expressions that exceed max_expr_len are compared via rolling
    hash (expr_hashes), so no retry is needed for those.

    Returns:
        (per_proof_passed, had_intermediate_truncation, max_intermediate_len)
    """
    total_nodes = plan.total_nodes
    V = plan.vocab_size

    if total_nodes == 0:
        return np.ones(plan.num_proofs, dtype=np.bool_)

    # Allocate expr_buffer and tracking tensors on GPU.
    # expr_buffer uses empty (not zeros): every slot is fully overwritten
    # by either push-node writes (level 0) or assertion-node writes
    # (which zero + write via expr_buffer[c_out_idx] = 0 then assign).
    # For set.mm, this saves zeroing ~24 GB of GPU memory.
    expr_buffer  = torch.empty(total_nodes, max_expr_len, dtype=torch.int32, device=device)
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
    return proof_passed.cpu().numpy(), had_intermediate_truncation, max_intermediate

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

    B, S, P, F = 8, 4, 8, 2  # tiny synthetic shapes

    # Allocate
    expr_buf  = torch.zeros(B * 2, P, dtype=torch.int32, device=device)
    expr_lens = torch.zeros(B * 2, dtype=torch.int32, device=device)
    failed    = torch.zeros(B * 2, dtype=torch.bool, device=device)

    idx       = torch.zeros(B, dtype=torch.long, device=device)
    in_idx    = torch.zeros(B, 2, dtype=torch.long, device=device)
    in_count  = torch.ones(B, dtype=torch.int32, device=device)

    pat       = torch.ones(B, P, dtype=torch.int32, device=device)
    pat_len   = torch.full((B,), P, dtype=torch.int32, device=device)

    fhyp_var  = torch.zeros(B, F, dtype=torch.long, device=device)
    fhyp_cnt  = torch.ones(B, dtype=torch.int32, device=device)
    fhyp_valid = torch.arange(F, device=device) < fhyp_cnt.unsqueeze(1)

    var_sub_vals = torch.zeros(B, F, S, dtype=torch.int32, device=device)
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
    )
    return plan_a, plan_b


def verify_proofs_gpu(
    plan: GlobalPlan,
    device: torch.device,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """Execute the full GPU verification pipeline.

    Retries with a larger expr_buffer only when INTERMEDIATE nodes are
    truncated (which corrupts downstream substitutions). Final-step
    expressions that exceed the buffer are compared via rolling hash
    (expr_hashes), so no retry is needed for those — this handles
    quartfull's 11548-token conclusion without a 278 GB allocation.

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
    return result, gpu_time


# ══════════════════════════════════════════════════════════════════════
#  Phase 4 — $d Post-Check (CPU)
# ══════════════════════════════════════════════════════════════════════


_DV_WORKER_PARSED: ParsedDatabase | None = None


def _init_dv_worker(parsed: ParsedDatabase) -> None:
    global _DV_WORKER_PARSED
    _DV_WORKER_PARSED = parsed


def _check_dv_chunk(labels: list[str]) -> dict[str, bool]:
    from tensormm.cpu_verifier import CPUVerifier
    assert _DV_WORKER_PARSED is not None
    cpu_v = CPUVerifier(_DV_WORKER_PARSED)
    results = {}
    for lbl in labels:
        try:
            r = cpu_v.verify_proof(lbl)
            results[lbl] = r.success
        except Exception:
            results[lbl] = False
    return results


def _check_dv_constraints(
    parsed: ParsedDatabase,
    graphs: list[ProofGraph],
    proof_passed: np.ndarray,
) -> np.ndarray:
    """Check $d constraints for proofs that passed GPU verification.

    Uses the existing CPUVerifier which already handles $d checking
    efficiently. Only checks proofs that the GPU said passed.
    Runs in parallel across available CPU cores.

    Returns updated proof_passed array.
    """
    result = proof_passed.copy()
    # Collect only labels that passed GPU verification
    labels_to_check = [
        g.theorem_label for pi, g in enumerate(graphs) if result[pi]
    ]
    if not labels_to_check:
        return result

    workers = min(os.cpu_count() or 1, 32)
    chunk_size = max(1, len(labels_to_check) // (workers * 4))
    chunks = [
        labels_to_check[i: i + chunk_size]
        for i in range(0, len(labels_to_check), chunk_size)
    ]

    global _DV_WORKER_PARSED
    if sys.platform == "linux":
        _DV_WORKER_PARSED = parsed
        ctx = multiprocessing.get_context("fork")
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    else:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_dv_worker,
            initargs=(parsed,),
        )

    dv_results: dict[str, bool] = {}
    with pool as executor:
        futures = {executor.submit(_check_dv_chunk, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            dv_results.update(future.result())

    # Update result array
    label_to_pi = {g.theorem_label: pi for pi, g in enumerate(graphs)}
    for lbl, passed in dv_results.items():
        if not passed:
            result[label_to_pi[lbl]] = False

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


def _verify_proofs_gpu_multi(
    plan: GlobalPlan,
    num_gpus: int,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """Run GPU verification split evenly across multiple GPUs using threads.

    Splits the plan by proof index into num_gpus shards, runs each shard on
    a separate GPU in parallel via Python threads (GIL is released during
    CUDA ops), then concatenates results in order.
    """
    import threading

    N = plan.num_proofs
    t0 = time.perf_counter()

    # Build split points: roughly equal proof counts per GPU
    split_points = [round(N * i / num_gpus) for i in range(num_gpus + 1)]
    # split_points[0]=0, split_points[num_gpus]=N

    # Split the plan into shards
    shards: list[GlobalPlan] = []
    remaining = plan
    for i in range(num_gpus - 1):
        # Each split cuts off the first chunk from the remaining plan
        chunk_size = split_points[i + 1] - split_points[i]
        shard, remaining = _split_plan(remaining, chunk_size)
        shards.append(shard)
    shards.append(remaining)

    results: list[np.ndarray | None] = [None] * num_gpus
    errors: list[Exception | None] = [None] * num_gpus

    def _run_shard(idx: int, shard: GlobalPlan) -> None:
        dev = torch.device(f"cuda:{idx}")
        try:
            warmup_cuda(dev)
            result, _ = verify_proofs_gpu(shard, dev, verbose=verbose)
            results[idx] = result
        except Exception as e:
            errors[idx] = e

    threads = [
        threading.Thread(target=_run_shard, args=(i, shards[i]), daemon=True)
        for i in range(num_gpus)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i, err in enumerate(errors):
        if err is not None:
            raise RuntimeError(f"GPU {i} shard failed: {err}") from err

    proof_passed = np.concatenate([r for r in results if r is not None])
    t_gpu = time.perf_counter() - t0
    return proof_passed, t_gpu


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
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if num_gpus >= 2 and plan.num_proofs >= 2:
        if verbose:
            print(f"  Phase 3: distributing across {num_gpus} GPUs...", flush=True)
        proof_passed, t_gpu = _verify_proofs_gpu_multi(plan, num_gpus, verbose=verbose)
    else:
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
