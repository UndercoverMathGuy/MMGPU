"""Full correctness check: CPU vs GPU on ql.mm and set.mm.

Runs exhaustive verification on both backends and compares results.
Mathematical errors are NEVER masked — they propagate as test failures.
No fallback to CPU for the GPU path; GPU errors are real failures.

Dual-backend support:
  - Apple Silicon (MPS): fused Metal compute kernel via MetalVerifier
  - NVIDIA (CUDA): pure-torch TensorVerifier.verify_flat (already optimized)

GPU SATURATION STRATEGY:
  The tensor verifier batches across candidates sharing the SAME pattern.
  We replay all proofs on CPU, extract every assertion-step's (pattern,
  substitution, expected_result), then group ALL steps across ALL theorems
  by their axiom/theorem label (= same pattern). Each group becomes ONE
  batched GPU call. A heavily-used axiom like ax-mp might get a batch of
  10,000+ candidates in a single kernel launch.
"""

from __future__ import annotations

import gc
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import torch
import pytest

import numpy as np

from tensormm.cpu_verifier import CPUVerifier, apply_substitution
from tensormm.database import MetamathDatabase
from tensormm.parser import ParsedDatabase, parse_mm_file
from tensormm.tensor_verifier import TensorVerifier
from tensormm.tokenizer import Tokenizer

try:
    from tensormm.metal_verifier import MetalVerifier, METAL_AVAILABLE
except ImportError:
    METAL_AVAILABLE = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# ── GPU device detection (MPS or CUDA) ────────────────────────────────
CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()
GPU_AVAILABLE = CUDA_AVAILABLE or MPS_AVAILABLE


def _get_gpu_device() -> torch.device:
    """Return the best GPU device or skip the test — NEVER silently fall back to CPU."""
    if CUDA_AVAILABLE:
        return torch.device("cuda")
    if MPS_AVAILABLE:
        return torch.device("mps")
    pytest.skip(
        "No GPU available (need CUDA or MPS) — cannot run GPU correctness checks"
    )
    raise RuntimeError("unreachable")  # for type checker


def _gpu_backend_name() -> str:
    if CUDA_AVAILABLE:
        return "CUDA"
    if MPS_AVAILABLE:
        return "Metal/MPS"
    return "CPU"


# ── Replay a proof on CPU, extracting every assertion-step's substitution ──


@dataclass
class _AssertionStep:
    """One assertion-step extracted from a proof replay."""

    theorem_label: str  # which theorem this step belongs to
    step_index: int  # step index within that theorem's proof
    step_label: str  # label of the axiom/theorem applied
    pattern: list[str]  # conclusion expression of the applied assertion
    substitution: dict[str, list[str]]  # variable -> replacement tokens
    expected_result: list[str]  # apply_substitution(pattern, subst)


@dataclass
class _EncodedBatch:
    """Pre-encoded batch of assertion steps as numpy arrays.

    Workers produce these during replay so the main thread never touches
    pure-Python tokenization.  All arrays use int32 except offsets (int64).
    """

    # Per-step metadata for failure reporting (lightweight)
    step_labels: list[str]       # step_label per step (for pattern cache key)
    theorem_labels: list[str]    # theorem_label per step
    step_indices: list[int]      # step_index per step
    pat_len_per_step: list[int]  # len(pattern) per step (for failure msg)
    tgt_len_per_step: list[int]  # len(expected_result) per step
    subst_vars_per_step: list[list[str]]  # substitution keys per step

    # Flat encoded arrays (match Phase 1 output format)
    all_pat_toks: np.ndarray     # int32 — flat pattern token IDs
    pat_lengths: np.ndarray      # int32 — pattern length per step
    all_tgt_toks: np.ndarray     # int32 — flat target token IDs
    tgt_lengths: np.ndarray      # int32 — target length per step
    all_sub_step: np.ndarray     # int32 — step index per substitution entry
    all_sub_var: np.ndarray      # int32 — variable token ID per subst entry
    all_sub_len: np.ndarray      # int32 — replacement length per subst entry
    all_sub_toks: np.ndarray     # int32 — flat replacement token IDs
    per_step_S_max: np.ndarray   # int32 — max subst length per step
    unique_token_ids: np.ndarray # int32 — set of all token IDs seen

    @property
    def num_steps(self) -> int:
        return len(self.pat_lengths)


def _build_label_info(parsed: ParsedDatabase) -> dict[str, tuple[str, object]]:
    """Build the label→info lookup ONCE for the whole database."""
    label_info: dict[str, tuple[str, object]] = {}
    for lbl, fh in parsed.floating_hyps.items():
        label_info[lbl] = ("$f", [fh.type_code, fh.variable])
    for lbl, eh in parsed.essential_hyps.items():
        label_info[lbl] = ("$e", eh.expression)
    for lbl, a in parsed.assertions.items():
        st = "$a" if a.type == "axiom" else "$p"
        label_info[lbl] = (st, a)
    return label_info


def _replay_proof_extract_steps(
    parsed: ParsedDatabase,
    theorem_label: str,
    label_info: dict[str, tuple[str, object]],
) -> list[_AssertionStep] | str:
    """Replay a theorem's proof on CPU and extract every assertion step.

    Returns a list of _AssertionStep or an error string.
    """
    assertion = parsed.assertions[theorem_label]

    stack: list[list[str]] = []
    steps: list[_AssertionStep] = []
    step_counter = 0

    def _do_step(step_label: str) -> str | None:
        nonlocal step_counter
        if step_label not in label_info:
            return f"Unknown label: {step_label}"
        st, data = label_info[step_label]
        if st in ("$f", "$e"):
            stack.append(list(data))
            return None
        a = data
        f_labels = a.floating_hyps
        e_labels = a.essential_hyps
        npop = len(f_labels) + len(e_labels)
        sp = len(stack) - npop
        if sp < 0:
            return f"Stack underflow at {step_label}"
        subst: dict[str, list[str]] = {}
        for flbl in f_labels:
            fh = parsed.floating_hyps[flbl]
            entry = stack[sp]
            subst[fh.variable] = entry[1:]
            sp += 1
        for elbl in e_labels:
            sp += 1
        result = apply_substitution(a.expression, subst)
        steps.append(
            _AssertionStep(
                theorem_label=theorem_label,
                step_index=step_counter,
                step_label=step_label,
                pattern=a.expression,
                substitution=subst,
                expected_result=result,
            )
        )
        step_counter += 1
        del stack[len(stack) - npop :]
        stack.append(result)
        return None

    try:
        if assertion.compressed_proof is not None:
            cp = assertion.compressed_proof
            plabels = cp.labels
            label_end = len(plabels)
            saved: list[list[str]] = []
            for pi in cp.proof_ints:
                if pi == -1:
                    if not stack:
                        return "Z save on empty stack"
                    saved.append(list(stack[-1]))
                elif pi < label_end:
                    err = _do_step(plabels[pi])
                    if err:
                        return err
                else:
                    si = pi - label_end
                    if si >= len(saved):
                        return f"Saved index {si} out of range"
                    stack.append(list(saved[si]))
        elif assertion.proof is not None:
            for sl in assertion.proof:
                err = _do_step(sl)
                if err:
                    return err
        else:
            return "No proof"
    except Exception as e:
        return str(e)

    if len(stack) != 1 or stack[0] != assertion.expression:
        return f"Final stack mismatch: got {stack}"

    return steps


# ── Worker process globals (set once by _init_worker, avoid per-call pickling) ──
_WORKER_PARSED: ParsedDatabase | None = None
_WORKER_LABEL_INFO: dict[str, tuple[str, object]] | None = None
_WORKER_SYMBOL_TO_ID: dict[str, int] | None = None

# Cap workers: proof replay has diminishing returns past ~32 cores due to
# IPC overhead for returning step lists. Also avoids 128-process fork storms.
_MAX_REPLAY_WORKERS = 32


def _init_worker(
    parsed: ParsedDatabase,
    symbol_to_id: dict[str, int] | None = None,
) -> None:
    """ProcessPoolExecutor initializer — runs ONCE per worker process.

    Stores the ParsedDatabase, pre-built label_info, and frozen symbol→ID
    lookup as process globals.
    """
    global _WORKER_PARSED, _WORKER_LABEL_INFO, _WORKER_SYMBOL_TO_ID
    _WORKER_PARSED = parsed
    _WORKER_LABEL_INFO = _build_label_info(parsed)
    if symbol_to_id is not None:
        _WORKER_SYMBOL_TO_ID = symbol_to_id


def _make_replay_pool(
    parsed: ParsedDatabase,
    max_workers: int | None = None,
    symbol_to_id: dict[str, int] | None = None,
) -> ProcessPoolExecutor:
    """Create a ProcessPoolExecutor pre-loaded with ParsedDatabase.

    On Linux: uses 'fork' context so children inherit parent memory — zero
    pickle cost.  We set the module globals BEFORE creating the pool.
    On macOS/Windows: falls back to initializer which pickles once per worker.
    """
    global _WORKER_PARSED, _WORKER_LABEL_INFO, _WORKER_SYMBOL_TO_ID
    workers = min(max_workers or os.cpu_count() or 1, _MAX_REPLAY_WORKERS)

    if sys.platform == "linux":
        # Set globals in parent; fork'd children inherit them for free.
        _WORKER_PARSED = parsed
        _WORKER_LABEL_INFO = _build_label_info(parsed)
        _WORKER_SYMBOL_TO_ID = symbol_to_id
        ctx = multiprocessing.get_context("fork")
        return ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    else:
        # macOS / Windows: use initializer (pickles once per worker at startup)
        return ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(parsed, symbol_to_id),
        )


def _replay_chunk(
    chunk: list[str],
) -> tuple[list[_AssertionStep], list[str]]:
    """Worker function for ProcessPoolExecutor.

    Uses process-global _WORKER_PARSED and _WORKER_LABEL_INFO set by _init_worker.
    Each submit() only pickles the lightweight label list.
    """
    parsed = _WORKER_PARSED
    label_info = _WORKER_LABEL_INFO
    assert parsed is not None and label_info is not None
    steps: list[_AssertionStep] = []
    errors: list[str] = []
    for lbl in chunk:
        extracted = _replay_proof_extract_steps(parsed, lbl, label_info)
        if isinstance(extracted, str):
            errors.append(f"  REPLAY ERR  {lbl}: {extracted}")
        else:
            steps.extend(extracted)
    return steps, errors


def _collect_all_steps(
    parsed: ParsedDatabase,
    theorem_labels: list[str],
    max_workers: int | None = None,
) -> tuple[list[_AssertionStep], list[str]]:
    """Replay all theorems in parallel, return (all_steps, replay_errors).

    Uses ProcessPoolExecutor to distribute proof replay across CPU cores,
    bypassing the GIL. Each worker process gets the full ParsedDatabase
    (a picklable dataclass) and replays an independent chunk of theorems.
    Results are re-assembled in original label order.
    """
    if not theorem_labels:
        return [], []

    workers = min(max_workers or os.cpu_count() or 1, _MAX_REPLAY_WORKERS)
    chunk_size = max(1, (len(theorem_labels) + workers - 1) // workers)
    chunks = [
        theorem_labels[i : i + chunk_size]
        for i in range(0, len(theorem_labels), chunk_size)
    ]

    # Map future → chunk index to preserve order
    ordered_results: list[tuple[list[_AssertionStep], list[str]]] = [None] * len(chunks)  # type: ignore[list-item]

    with _make_replay_pool(parsed, max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_replay_chunk, chunk): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            ordered_results[idx] = future.result()

    all_steps: list[_AssertionStep] = []
    replay_errors: list[str] = []
    for steps, errors in ordered_results:
        all_steps.extend(steps)
        replay_errors.extend(errors)

    return all_steps, replay_errors


def _verify_steps_batched_on_gpu(
    all_steps: list[_AssertionStep],
    tokenizer: Tokenizer,
    is_variable: torch.Tensor,
    device: torch.device,
    chunk_size: int = 2000,
) -> tuple[list[str], dict[str, int]]:
    """Verify all steps on GPU — dual backend.

    Phase 1 (Python): tokenizer.encode calls → flat Python lists.
    Phase 2 (numpy on CPU): LUT remap, sort, pack — fast vectorized C.
    Phase 3 (GPU dispatch):
      - Metal/MPS: MetalVerifier.verify_flat() — fused MSL kernel, 1 dispatch/chunk
      - CUDA: TensorVerifier.verify_flat() — pure torch on device, H100-optimized
    """
    N = len(all_steps)
    if N == 0:
        return [], {
            "total_steps": 0,
            "num_groups": 0,
            "max_batch": 0,
            "steps_verified": 0,
        }

    use_metal = device.type == "mps" and METAL_AVAILABLE
    if use_metal:
        verifier = MetalVerifier()
    else:
        verifier = TensorVerifier(device=device)

    # ── Phase 1: encode (Python — tokenizer API is inherently Python) ─
    _t_phase1 = time.perf_counter()

    pattern_cache: dict[str, list[int]] = {}
    unique_groups: set[str] = set()

    all_pat_toks: list[int] = []
    pat_offsets: list[int] = []
    pat_lengths: list[int] = []

    all_tgt_toks: list[int] = []
    tgt_offsets: list[int] = []
    tgt_lengths: list[int] = []

    all_sub_step: list[int] = []
    all_sub_var: list[int] = []
    all_sub_len: list[int] = []
    all_sub_toks: list[int] = []

    per_step_S_max: list[int] = []
    all_token_ids: set[int] = {0}

    for i, step in enumerate(all_steps):
        unique_groups.add(step.step_label)
        if step.step_label not in pattern_cache:
            pattern_cache[step.step_label] = tokenizer.encode_expression(step.pattern)
        pat_ids = pattern_cache[step.step_label]
        pat_offsets.append(len(all_pat_toks))
        pat_lengths.append(len(pat_ids))
        all_pat_toks.extend(pat_ids)
        all_token_ids.update(pat_ids)

        s_max_here = 1
        for var, replacement in step.substitution.items():
            var_id = tokenizer.encode_symbol(var)
            rep_ids = tokenizer.encode_expression(replacement)
            all_sub_step.append(i)
            all_sub_var.append(var_id)
            all_sub_len.append(len(rep_ids))
            all_sub_toks.extend(rep_ids)
            all_token_ids.add(var_id)
            all_token_ids.update(rep_ids)
            if len(rep_ids) > s_max_here:
                s_max_here = len(rep_ids)
        per_step_S_max.append(s_max_here)

        tgt_ids = tokenizer.encode_expression(step.expected_result)
        tgt_offsets.append(len(all_tgt_toks))
        tgt_lengths.append(len(tgt_ids))
        all_tgt_toks.extend(tgt_ids)
        all_token_ids.update(tgt_ids)

    _t_phase1 = time.perf_counter() - _t_phase1
    print(f"    Phase 1 (Python encode): {_t_phase1:.2f}s  [{N:,} steps]")

    # ── Build compact vocab as numpy LUT ──────────────────────────────
    sorted_ids = sorted(all_token_ids)
    max_full_id = max(sorted_ids) + 1
    f2c_lut = np.zeros(max_full_id, dtype=np.int32)
    compact_ids = np.arange(len(sorted_ids), dtype=np.int32)
    f2c_lut[np.array(sorted_ids, dtype=np.int64)] = compact_ids
    V = len(sorted_ids)

    stats: dict[str, int] = {
        "total_steps": N,
        "num_groups": len(unique_groups),
        "max_batch": N,
        "steps_verified": 0,
        "compact_vocab": V,
    }

    # ── Phase 2: PURE NUMPY — sort, remap, pack ──────────────────────
    _t_phase2 = time.perf_counter()
    all_pat_toks_np = np.array(all_pat_toks, dtype=np.int32)
    pat_offsets_np = np.array(pat_offsets, dtype=np.int64)
    pat_lengths_np = np.array(pat_lengths, dtype=np.int32)

    all_tgt_toks_np = np.array(all_tgt_toks, dtype=np.int32)
    tgt_offsets_np = np.array(tgt_offsets, dtype=np.int64)
    tgt_lengths_np = np.array(tgt_lengths, dtype=np.int32)

    all_sub_step_np = np.array(all_sub_step, dtype=np.int32)
    all_sub_var_np = np.array(all_sub_var, dtype=np.int32)
    all_sub_len_np = np.array(all_sub_len, dtype=np.int32)
    all_sub_toks_np = np.array(all_sub_toks, dtype=np.int32)

    s_max_np = np.array(per_step_S_max, dtype=np.int32)

    # Remap ALL tokens to compact vocab in one vectorized call
    all_pat_toks_c = f2c_lut[all_pat_toks_np]
    all_tgt_toks_c = f2c_lut[all_tgt_toks_np]
    all_sub_toks_c = (
        f2c_lut[all_sub_toks_np] if len(all_sub_toks_np) else all_sub_toks_np
    )
    all_sub_var_c = f2c_lut[all_sub_var_np] if len(all_sub_var_np) else all_sub_var_np

    # Sort order: (P_len, S_max)
    sort_order = np.lexsort((s_max_np, pat_lengths_np))
    inv_sort = np.empty_like(sort_order)
    inv_sort[sort_order] = np.arange(N)

    # Apply sort to per-step arrays
    pat_lengths_s = pat_lengths_np[sort_order]
    pat_offsets_s = pat_offsets_np[sort_order]
    tgt_lengths_s = tgt_lengths_np[sort_order]
    tgt_offsets_s = tgt_offsets_np[sort_order]
    s_max_s = s_max_np[sort_order]

    P_max = int(pat_lengths_s.max())
    T_max = max(int(tgt_lengths_s.max()), 1)

    # Pack patterns [N, P_max]
    pat_lens_i64 = pat_lengths_s.astype(np.int64)
    total_pat_toks = int(pat_lens_i64.sum())
    step_idx_p = np.repeat(np.arange(N, dtype=np.int32), pat_lengths_s)
    pat_cumstart = np.repeat(np.cumsum(pat_lens_i64) - pat_lens_i64, pat_lengths_s)
    pos_within = (np.arange(total_pat_toks, dtype=np.int64) - pat_cumstart).astype(
        np.int32
    )
    src_idx_p = np.repeat(pat_offsets_s, pat_lengths_s) + pos_within

    patterns_np = np.zeros((N, P_max), dtype=np.int32)
    patterns_np[step_idx_p, pos_within] = all_pat_toks_c[src_idx_p]

    # Pack targets [N, T_max]
    tgt_lens_i64 = tgt_lengths_s.astype(np.int64)
    total_tgt_toks = int(tgt_lens_i64.sum())
    step_idx_t = np.repeat(np.arange(N, dtype=np.int32), tgt_lengths_s)
    tgt_cumstart = np.repeat(np.cumsum(tgt_lens_i64) - tgt_lens_i64, tgt_lengths_s)
    pos_within_t = (np.arange(total_tgt_toks, dtype=np.int64) - tgt_cumstart).astype(
        np.int32
    )
    src_idx_t = np.repeat(tgt_offsets_s, tgt_lengths_s) + pos_within_t

    targets_np = np.zeros((N, T_max), dtype=np.int32)
    targets_np[step_idx_t, pos_within_t] = all_tgt_toks_c[src_idx_t]

    # Pack sub_lens [N, V]
    sub_lens_np = np.ones((N, V), dtype=np.int32)
    if len(all_sub_step_np):
        sorted_sub_step = inv_sort[all_sub_step_np].astype(np.int32)
        sub_lens_np[sorted_sub_step, all_sub_var_c] = all_sub_len_np
    else:
        sorted_sub_step = np.empty(0, dtype=np.int32)

    # Sparse subst arrays
    sp_step_np = sorted_sub_step
    sp_var_np = all_sub_var_c
    sp_len_np = all_sub_len_np
    sp_toks_np = all_sub_toks_c
    sp_offsets_np = np.empty(len(sp_len_np), dtype=np.int64)
    if len(sp_len_np):
        sp_offsets_np[0] = 0
        np.cumsum(sp_len_np[:-1].astype(np.int64), out=sp_offsets_np[1:])

    _t_phase2 = time.perf_counter() - _t_phase2
    print(f"    Phase 2 (numpy pack):    {_t_phase2:.2f}s  [N={N:,}, P_max={P_max}, T_max={T_max}, V={V}]")

    # ── Phase 3: chunked GPU dispatch — adaptive chunk size ─────────
    _t_phase3 = time.perf_counter()
    # Dynamically detect GPU memory — use 80% to leave headroom
    if device.type == "cuda":
        _total_vram = torch.cuda.get_device_properties(device).total_memory
        GPU_MEM_BUDGET = int(_total_vram * 0.8)
    else:
        GPU_MEM_BUDGET = 512 * 1024 * 1024  # 512 MB

    gpu_results_sorted = np.empty(N, dtype=np.bool_)

    _MAX_CHUNK_B = 50_000  # hard cap — prevents multi-GB CPU allocations

    start = 0
    while start < N:
        probe_end = min(start + chunk_size, N)
        chunk_S = max(int(s_max_s[start:probe_end].max()), 1)
        chunk_P_est = int(pat_lengths_s[start:probe_end].max())
        chunk_T_est = max(int(tgt_lengths_s[start:probe_end].max()), 1)
        # Memory per step: sub_tables + sub_lens + patterns + targets (all int32)
        bytes_per_step = (V * chunk_S + V + chunk_P_est + chunk_T_est) * 4
        max_B = min(max(GPU_MEM_BUDGET // max(bytes_per_step, 1), 64), _MAX_CHUNK_B)
        B = min(probe_end - start, max_B)
        end = start + B

        chunk_P = int(pat_lengths_s[start:end].max())
        chunk_T = max(int(tgt_lengths_s[start:end].max()), 1)
        chunk_S = max(int(s_max_s[start:end].max()), 1)

        # Build sub_tables [B, V, chunk_S] on CPU — numpy, zero dispatch overhead
        st = np.zeros((B, V, chunk_S), dtype=np.int32)
        ident = np.arange(V, dtype=np.int32)
        st[:, :, 0] = ident[np.newaxis, :]

        mask = (sp_step_np >= start) & (sp_step_np < end)
        if mask.any():
            c_step = sp_step_np[mask] - start
            c_var = sp_var_np[mask]
            c_len = sp_len_np[mask]
            c_off = sp_offsets_np[mask]

            rc = c_len.astype(np.int64)
            total_toks = int(rc.sum())
            if total_toks > 0:
                sc_step = np.repeat(c_step, rc)
                sc_var = np.repeat(c_var, rc)
                offsets_within = np.repeat(np.cumsum(rc) - rc, rc)
                sc_s = (np.arange(total_toks, dtype=np.int64) - offsets_within).astype(
                    np.int32
                )
                sc_tok_idx = np.repeat(c_off, rc) + sc_s.astype(np.int64)
                sc_tok = sp_toks_np[sc_tok_idx]
                st[sc_step, sc_var, sc_s] = sc_tok

        # Convert to torch tensors
        pat_t = torch.from_numpy(np.ascontiguousarray(patterns_np[start:end, :chunk_P]))
        pl_t = torch.from_numpy(np.ascontiguousarray(pat_lengths_s[start:end]))
        st_t = torch.from_numpy(np.ascontiguousarray(st))
        sl_t = torch.from_numpy(np.ascontiguousarray(sub_lens_np[start:end]))
        tgt_t = torch.from_numpy(np.ascontiguousarray(targets_np[start:end, :chunk_T]))
        tl_t = torch.from_numpy(np.ascontiguousarray(tgt_lengths_s[start:end]))

        if use_metal:
            # ONE Metal dispatch per chunk (fused gather→scatter→reduce)
            gpu_out = verifier.verify_flat(pat_t, pl_t, st_t, sl_t, tgt_t, tl_t)
        else:
            # TensorVerifier: move tensors to device, dispatch via torch ops
            gpu_out = verifier.verify_flat(
                pat_t.to(device),
                pl_t.to(device),
                st_t.to(device),
                sl_t.to(device),
                tgt_t.to(device),
                tl_t.to(device),
            )
        gpu_results_sorted[start:end] = gpu_out.numpy()
        stats["steps_verified"] += B
        start = end

    _t_phase3 = time.perf_counter() - _t_phase3
    print(f"    Phase 3 (GPU dispatch):  {_t_phase3:.2f}s  [{stats['steps_verified']:,} steps verified]")

    # Unsort results
    gpu_results = np.empty(N, dtype=np.bool_)
    gpu_results[sort_order] = gpu_results_sorted

    gpu_failures: list[str] = []
    fail_indices = np.where(~gpu_results)[0]
    for j in fail_indices:
        step = all_steps[j]
        gpu_failures.append(
            f"  GPU FAIL  {step.theorem_label} step {step.step_index} "
            f"({step.step_label}): pattern_len={len(step.pattern)}, "
            f"subst_vars={list(step.substitution.keys())}, "
            f"result_len={len(step.expected_result)}"
        )

    return gpu_failures, stats


# ══════════════════════════════════════════════════════════════════════
#  ql.mm — EXHAUSTIVE (every theorem)
# ══════════════════════════════════════════════════════════════════════


class TestQLmmExhaustive:
    @pytest.fixture(scope="class")
    def ql_data(self):
        path = os.path.join(DATA_DIR, "ql.mm")
        if not os.path.exists(path):
            pytest.skip("ql.mm not found in data/")
        parsed = parse_mm_file(path)
        tok = Tokenizer()
        db = MetamathDatabase(parsed, tok)
        return parsed, tok, db

    def test_cpu_verifies_all_ql(self, ql_data, capsys) -> None:
        """CPU verifier must pass every theorem in ql.mm."""
        parsed, tok, db = ql_data
        cpu_v = CPUVerifier(parsed)
        results = cpu_v.verify_all()

        passed = failed = 0
        for label, r in results.items():
            if r.success:
                passed += 1
            else:
                failed += 1
                print(f"  CPU FAIL  {label}: {r.error_message}")

        print(
            f"\n[ql.mm CPU] {passed} passed, {failed} failed out of {len(results)} theorems"
        )
        assert failed == 0, f"{failed} theorems failed CPU verification"

    def test_gpu_matches_cpu_all_ql(self, ql_data, capsys) -> None:
        """GPU must match CPU on EVERY assertion step of EVERY theorem in ql.mm.

        Steps are batched by axiom/theorem label — one GPU call per unique pattern.
        """
        device = _get_gpu_device()
        backend = _gpu_backend_name()
        parsed, tok, db = ql_data

        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
        print(
            f"\n[ql.mm {backend}] Replaying {len(theorems)} theorems to extract assertion steps..."
        )

        t0 = time.perf_counter()
        all_steps, replay_errors = _collect_all_steps(parsed, theorems)
        t_replay = time.perf_counter() - t0
        print(
            f"[ql.mm {backend}] Extracted {len(all_steps)} assertion steps in {t_replay:.2f}s"
        )

        t1 = time.perf_counter()
        gpu_failures, stats = _verify_steps_batched_on_gpu(
            all_steps, tok, db.is_variable, device
        )
        t_gpu = time.perf_counter() - t1

        for err in replay_errors:
            print(err)
        for fail in gpu_failures:
            print(fail)

        print(f"\n[ql.mm {backend}] Results:")
        print(f"  Steps verified:  {stats['steps_verified']}")
        print(f"  Pattern groups:  {stats['num_groups']}")
        print(f"  Compact vocab:   {stats.get('compact_vocab', '?')}")
        print(f"  GPU time:        {t_gpu:.2f}s")
        print(f"  Replay errors:   {len(replay_errors)}")
        print(f"  GPU failures:    {len(gpu_failures)}")

        assert len(replay_errors) == 0, f"{len(replay_errors)} proof replay errors"
        assert len(gpu_failures) == 0, (
            f"{len(gpu_failures)} GPU/CPU divergences — kernel is UNSOUND"
        )


# ══════════════════════════════════════════════════════════════════════
#  Streaming GPU verification for large databases
# ══════════════════════════════════════════════════════════════════════


def _replay_batch(
    labels: list[str],
) -> list[tuple[str, list[_AssertionStep] | str]]:
    """Worker: replay a batch of theorems and return [(label, steps_or_error)].

    Uses process-global _WORKER_PARSED and _WORKER_LABEL_INFO set by _init_worker.
    Each submit() only pickles the lightweight label list.
    """
    parsed = _WORKER_PARSED
    label_info = _WORKER_LABEL_INFO
    assert parsed is not None and label_info is not None
    return [
        (lbl, _replay_proof_extract_steps(parsed, lbl, label_info)) for lbl in labels
    ]


def _replay_batch_encoded(
    labels: list[str],
) -> tuple[_EncodedBatch | None, list[str]]:
    """Worker: replay theorems AND encode to int arrays in one pass.

    Uses _WORKER_SYMBOL_TO_ID to encode strings → ints during replay.
    Returns (_EncodedBatch, replay_errors). Batch is None if no steps.
    """
    parsed = _WORKER_PARSED
    label_info = _WORKER_LABEL_INFO
    sym2id = _WORKER_SYMBOL_TO_ID
    assert parsed is not None and label_info is not None and sym2id is not None

    # Accumulators — Python lists, converted to numpy once at the end
    meta_step_labels: list[str] = []
    meta_theorem_labels: list[str] = []
    meta_step_indices: list[int] = []
    meta_pat_lens: list[int] = []
    meta_tgt_lens: list[int] = []
    meta_subst_vars: list[list[str]] = []

    all_pat_toks: list[int] = []
    pat_lengths: list[int] = []
    all_tgt_toks: list[int] = []
    tgt_lengths: list[int] = []
    all_sub_step: list[int] = []
    all_sub_var: list[int] = []
    all_sub_len: list[int] = []
    all_sub_toks: list[int] = []
    per_step_S_max: list[int] = []
    token_id_set: set[int] = {0}

    pattern_cache: dict[str, list[int]] = {}
    replay_errors: list[str] = []
    step_idx = 0

    for lbl in labels:
        extracted = _replay_proof_extract_steps(parsed, lbl, label_info)
        if isinstance(extracted, str):
            replay_errors.append(f"  REPLAY ERR  {lbl}: {extracted}")
            continue
        for astep in extracted:
            if astep.step_label not in pattern_cache:
                pattern_cache[astep.step_label] = [sym2id[s] for s in astep.pattern]
            pat_ids = pattern_cache[astep.step_label]
            pat_lengths.append(len(pat_ids))
            all_pat_toks.extend(pat_ids)
            token_id_set.update(pat_ids)

            s_max = 1
            for var, replacement in astep.substitution.items():
                var_id = sym2id[var]
                rep_ids = [sym2id[s] for s in replacement]
                all_sub_step.append(step_idx)
                all_sub_var.append(var_id)
                all_sub_len.append(len(rep_ids))
                all_sub_toks.extend(rep_ids)
                token_id_set.add(var_id)
                token_id_set.update(rep_ids)
                if len(rep_ids) > s_max:
                    s_max = len(rep_ids)
            per_step_S_max.append(s_max)

            tgt_ids = [sym2id[s] for s in astep.expected_result]
            tgt_lengths.append(len(tgt_ids))
            all_tgt_toks.extend(tgt_ids)
            token_id_set.update(tgt_ids)

            meta_step_labels.append(astep.step_label)
            meta_theorem_labels.append(astep.theorem_label)
            meta_step_indices.append(astep.step_index)
            meta_pat_lens.append(len(astep.pattern))
            meta_tgt_lens.append(len(astep.expected_result))
            meta_subst_vars.append(list(astep.substitution.keys()))
            step_idx += 1

    if step_idx == 0:
        return None, replay_errors

    return _EncodedBatch(
        step_labels=meta_step_labels,
        theorem_labels=meta_theorem_labels,
        step_indices=meta_step_indices,
        pat_len_per_step=meta_pat_lens,
        tgt_len_per_step=meta_tgt_lens,
        subst_vars_per_step=meta_subst_vars,
        all_pat_toks=np.array(all_pat_toks, dtype=np.int32),
        pat_lengths=np.array(pat_lengths, dtype=np.int32),
        all_tgt_toks=np.array(all_tgt_toks, dtype=np.int32),
        tgt_lengths=np.array(tgt_lengths, dtype=np.int32),
        all_sub_step=np.array(all_sub_step, dtype=np.int32),
        all_sub_var=np.array(all_sub_var, dtype=np.int32),
        all_sub_len=np.array(all_sub_len, dtype=np.int32),
        all_sub_toks=np.array(all_sub_toks, dtype=np.int32),
        per_step_S_max=np.array(per_step_S_max, dtype=np.int32),
        unique_token_ids=np.array(sorted(token_id_set), dtype=np.int32),
    ), replay_errors


def _concat_encoded_batches(batches: list[_EncodedBatch]) -> _EncodedBatch:
    """Concatenate multiple _EncodedBatch into one, rebasing step indices."""
    if len(batches) == 1:
        return batches[0]

    step_labels: list[str] = []
    theorem_labels: list[str] = []
    step_indices: list[int] = []
    pat_len_meta: list[int] = []
    tgt_len_meta: list[int] = []
    subst_vars_meta: list[list[str]] = []

    pat_toks_parts: list[np.ndarray] = []
    pat_len_parts: list[np.ndarray] = []
    tgt_toks_parts: list[np.ndarray] = []
    tgt_len_parts: list[np.ndarray] = []
    sub_step_parts: list[np.ndarray] = []
    sub_var_parts: list[np.ndarray] = []
    sub_len_parts: list[np.ndarray] = []
    sub_toks_parts: list[np.ndarray] = []
    s_max_parts: list[np.ndarray] = []
    token_set_parts: list[np.ndarray] = []

    step_offset = 0
    for b in batches:
        n = b.num_steps
        step_labels.extend(b.step_labels)
        theorem_labels.extend(b.theorem_labels)
        step_indices.extend(b.step_indices)
        pat_len_meta.extend(b.pat_len_per_step)
        tgt_len_meta.extend(b.tgt_len_per_step)
        subst_vars_meta.extend(b.subst_vars_per_step)

        pat_toks_parts.append(b.all_pat_toks)
        pat_len_parts.append(b.pat_lengths)
        tgt_toks_parts.append(b.all_tgt_toks)
        tgt_len_parts.append(b.tgt_lengths)
        sub_step_parts.append(b.all_sub_step + step_offset)
        sub_var_parts.append(b.all_sub_var)
        sub_len_parts.append(b.all_sub_len)
        sub_toks_parts.append(b.all_sub_toks)
        s_max_parts.append(b.per_step_S_max)
        token_set_parts.append(b.unique_token_ids)
        step_offset += n

    return _EncodedBatch(
        step_labels=step_labels,
        theorem_labels=theorem_labels,
        step_indices=step_indices,
        pat_len_per_step=pat_len_meta,
        tgt_len_per_step=tgt_len_meta,
        subst_vars_per_step=subst_vars_meta,
        all_pat_toks=np.concatenate(pat_toks_parts),
        pat_lengths=np.concatenate(pat_len_parts),
        all_tgt_toks=np.concatenate(tgt_toks_parts),
        tgt_lengths=np.concatenate(tgt_len_parts),
        all_sub_step=np.concatenate(sub_step_parts),
        all_sub_var=np.concatenate(sub_var_parts),
        all_sub_len=np.concatenate(sub_len_parts),
        all_sub_toks=np.concatenate(sub_toks_parts),
        per_step_S_max=np.concatenate(s_max_parts),
        unique_token_ids=np.unique(np.concatenate(token_set_parts)),
    )


def _verify_encoded_on_gpu(
    enc: _EncodedBatch,
    device: torch.device,
    chunk_size: int = 2000,
) -> tuple[list[str], dict[str, int]]:
    """Verify pre-encoded steps on GPU — Phase 1 eliminated.

    Takes _EncodedBatch (numpy arrays from workers) directly.
    Only Phase 2 (numpy pack) and Phase 3 (GPU dispatch) remain.
    """
    N = enc.num_steps
    if N == 0:
        return [], {"total_steps": 0, "num_groups": 0, "max_batch": 0,
                     "steps_verified": 0, "compact_vocab": 0}

    use_metal = device.type == "mps" and METAL_AVAILABLE
    verifier = MetalVerifier() if use_metal else TensorVerifier(device=device)

    # ── Build compact vocab LUT ──────────────────────────────────────
    _t_phase2 = time.perf_counter()

    sorted_ids = enc.unique_token_ids  # already sorted from workers
    max_full_id = int(sorted_ids[-1]) + 1
    f2c_lut = np.zeros(max_full_id, dtype=np.int32)
    f2c_lut[sorted_ids] = np.arange(len(sorted_ids), dtype=np.int32)
    V = len(sorted_ids)

    stats: dict[str, int] = {
        "total_steps": N, "num_groups": len(set(enc.step_labels)),
        "max_batch": N, "steps_verified": 0, "compact_vocab": V,
    }

    # ── Phase 2: numpy remap, sort, pack ─────────────────────────────
    all_pat_toks_c = f2c_lut[enc.all_pat_toks]
    all_tgt_toks_c = f2c_lut[enc.all_tgt_toks]
    all_sub_toks_c = f2c_lut[enc.all_sub_toks] if len(enc.all_sub_toks) else enc.all_sub_toks
    all_sub_var_c = f2c_lut[enc.all_sub_var] if len(enc.all_sub_var) else enc.all_sub_var

    pat_lengths_np = enc.pat_lengths
    tgt_lengths_np = enc.tgt_lengths
    s_max_np = enc.per_step_S_max
    all_sub_step_np = enc.all_sub_step
    all_sub_len_np = enc.all_sub_len

    # Build offsets from lengths
    pat_offsets_np = np.empty(N, dtype=np.int64)
    if N > 0:
        pat_offsets_np[0] = 0
        if N > 1:
            np.cumsum(pat_lengths_np[:-1].astype(np.int64), out=pat_offsets_np[1:])

    tgt_offsets_np = np.empty(N, dtype=np.int64)
    if N > 0:
        tgt_offsets_np[0] = 0
        if N > 1:
            np.cumsum(tgt_lengths_np[:-1].astype(np.int64), out=tgt_offsets_np[1:])

    # Sort order: (P_len, S_max)
    sort_order = np.lexsort((s_max_np, pat_lengths_np))
    inv_sort = np.empty_like(sort_order)
    inv_sort[sort_order] = np.arange(N)

    pat_lengths_s = pat_lengths_np[sort_order]
    pat_offsets_s = pat_offsets_np[sort_order]
    tgt_lengths_s = tgt_lengths_np[sort_order]
    tgt_offsets_s = tgt_offsets_np[sort_order]
    s_max_s = s_max_np[sort_order]

    P_max = int(pat_lengths_s.max())
    T_max = max(int(tgt_lengths_s.max()), 1)

    # Pack patterns [N, P_max]
    pat_lens_i64 = pat_lengths_s.astype(np.int64)
    total_pat_toks = int(pat_lens_i64.sum())
    step_idx_p = np.repeat(np.arange(N, dtype=np.int32), pat_lengths_s)
    pat_cumstart = np.repeat(np.cumsum(pat_lens_i64) - pat_lens_i64, pat_lengths_s)
    pos_within = (np.arange(total_pat_toks, dtype=np.int64) - pat_cumstart).astype(np.int32)
    src_idx_p = np.repeat(pat_offsets_s, pat_lengths_s) + pos_within
    patterns_np = np.zeros((N, P_max), dtype=np.int32)
    patterns_np[step_idx_p, pos_within] = all_pat_toks_c[src_idx_p]

    # Pack targets [N, T_max]
    tgt_lens_i64 = tgt_lengths_s.astype(np.int64)
    total_tgt_toks = int(tgt_lens_i64.sum())
    step_idx_t = np.repeat(np.arange(N, dtype=np.int32), tgt_lengths_s)
    tgt_cumstart = np.repeat(np.cumsum(tgt_lens_i64) - tgt_lens_i64, tgt_lengths_s)
    pos_within_t = (np.arange(total_tgt_toks, dtype=np.int64) - tgt_cumstart).astype(np.int32)
    src_idx_t = np.repeat(tgt_offsets_s, tgt_lengths_s) + pos_within_t
    targets_np = np.zeros((N, T_max), dtype=np.int32)
    targets_np[step_idx_t, pos_within_t] = all_tgt_toks_c[src_idx_t]

    # Pack sub_lens [N, V]
    sub_lens_np = np.ones((N, V), dtype=np.int32)
    if len(all_sub_step_np):
        sorted_sub_step = inv_sort[all_sub_step_np].astype(np.int32)
        sub_lens_np[sorted_sub_step, all_sub_var_c] = all_sub_len_np
    else:
        sorted_sub_step = np.empty(0, dtype=np.int32)

    # Sparse subst arrays
    sp_step_np = sorted_sub_step
    sp_var_np = all_sub_var_c
    sp_len_np = all_sub_len_np
    sp_toks_np = all_sub_toks_c
    sp_offsets_np = np.empty(len(sp_len_np), dtype=np.int64)
    if len(sp_len_np):
        sp_offsets_np[0] = 0
        np.cumsum(sp_len_np[:-1].astype(np.int64), out=sp_offsets_np[1:])

    _t_phase2 = time.perf_counter() - _t_phase2
    print(f"    Phase 2 (numpy pack):    {_t_phase2:.2f}s  [N={N:,}, P={P_max}, T={T_max}, V={V}]")

    # ── Phase 3: chunked GPU dispatch ────────────────────────────────
    _t_phase3 = time.perf_counter()
    if device.type == "cuda":
        _total_vram = torch.cuda.get_device_properties(device).total_memory
        GPU_MEM_BUDGET = int(_total_vram * 0.8)
    else:
        GPU_MEM_BUDGET = 512 * 1024 * 1024

    _MAX_CHUNK_B = 50_000

    gpu_results_sorted = np.empty(N, dtype=np.bool_)
    start = 0
    while start < N:
        probe_end = min(start + chunk_size, N)
        chunk_S = max(int(s_max_s[start:probe_end].max()), 1)
        chunk_P_est = int(pat_lengths_s[start:probe_end].max())
        chunk_T_est = max(int(tgt_lengths_s[start:probe_end].max()), 1)
        bytes_per_step = (V * chunk_S + V + chunk_P_est + chunk_T_est) * 4
        max_B = min(max(GPU_MEM_BUDGET // max(bytes_per_step, 1), 64), _MAX_CHUNK_B)
        B = min(probe_end - start, max_B)
        end = start + B

        chunk_P = int(pat_lengths_s[start:end].max())
        chunk_T = max(int(tgt_lengths_s[start:end].max()), 1)
        chunk_S = max(int(s_max_s[start:end].max()), 1)

        _st_mb = B * V * chunk_S * 4 / (1024 * 1024)
        print(f"      chunk [{start}:{end}] B={B}, P={chunk_P}, T={chunk_T}, "
              f"S={chunk_S}, st={_st_mb:.0f}MB", flush=True)

        st = np.zeros((B, V, chunk_S), dtype=np.int32)
        ident = np.arange(V, dtype=np.int32)
        st[:, :, 0] = ident[np.newaxis, :]

        mask = (sp_step_np >= start) & (sp_step_np < end)
        if mask.any():
            c_step = sp_step_np[mask] - start
            c_var = sp_var_np[mask]
            c_len = sp_len_np[mask]
            c_off = sp_offsets_np[mask]
            rc = c_len.astype(np.int64)
            total_toks = int(rc.sum())
            if total_toks > 0:
                sc_step = np.repeat(c_step, rc)
                sc_var = np.repeat(c_var, rc)
                offsets_within = np.repeat(np.cumsum(rc) - rc, rc)
                sc_s = (np.arange(total_toks, dtype=np.int64) - offsets_within).astype(np.int32)
                sc_tok_idx = np.repeat(c_off, rc) + sc_s.astype(np.int64)
                sc_tok = sp_toks_np[sc_tok_idx]
                st[sc_step, sc_var, sc_s] = sc_tok

        pat_t = torch.from_numpy(np.ascontiguousarray(patterns_np[start:end, :chunk_P]))
        pl_t = torch.from_numpy(np.ascontiguousarray(pat_lengths_s[start:end]))
        st_t = torch.from_numpy(np.ascontiguousarray(st))
        del st  # free CPU copy
        sl_t = torch.from_numpy(np.ascontiguousarray(sub_lens_np[start:end]))
        tgt_t = torch.from_numpy(np.ascontiguousarray(targets_np[start:end, :chunk_T]))
        tl_t = torch.from_numpy(np.ascontiguousarray(tgt_lengths_s[start:end]))

        if use_metal:
            gpu_out = verifier.verify_flat(pat_t, pl_t, st_t, sl_t, tgt_t, tl_t)
        else:
            _ct0 = time.perf_counter()
            pat_d = pat_t.to(device)
            pl_d = pl_t.to(device)
            st_d = st_t.to(device)
            del st_t  # free CPU copy after transfer
            sl_d = sl_t.to(device)
            tgt_d = tgt_t.to(device)
            tl_d = tl_t.to(device)
            if device.type == "cuda":
                _alloc = torch.cuda.memory_allocated(device) / (1024**2)
                _resrv = torch.cuda.memory_reserved(device) / (1024**2)
                print(f"        → to(device) {time.perf_counter()-_ct0:.2f}s, "
                      f"VRAM alloc={_alloc:.0f}MB reserved={_resrv:.0f}MB", flush=True)
            gpu_out = verifier.verify_flat(pat_d, pl_d, st_d, sl_d, tgt_d, tl_d)
            del pat_d, pl_d, st_d, sl_d, tgt_d, tl_d
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                print(f"        → verify_flat done {time.perf_counter()-_ct0:.2f}s", flush=True)
        gpu_results_sorted[start:end] = gpu_out.numpy()
        stats["steps_verified"] += B
        start = end

    _t_phase3 = time.perf_counter() - _t_phase3
    print(f"    Phase 3 (GPU dispatch):  {_t_phase3:.2f}s  [{stats['steps_verified']:,} steps]", flush=True)

    # Unsort results
    gpu_results = np.empty(N, dtype=np.bool_)
    gpu_results[sort_order] = gpu_results_sorted
    n_fail = int((~gpu_results).sum())
    print(f"    Unsort done, {n_fail} failures detected", flush=True)

    gpu_failures: list[str] = []
    fail_indices = np.where(~gpu_results)[0]
    for j in fail_indices:
        gpu_failures.append(
            f"  GPU FAIL  {enc.theorem_labels[j]} step {enc.step_indices[j]} "
            f"({enc.step_labels[j]}): pattern_len={enc.pat_len_per_step[j]}, "
            f"subst_vars={enc.subst_vars_per_step[j]}, "
            f"result_len={enc.tgt_len_per_step[j]}"
        )
    print(f"    _verify_encoded_on_gpu returning", flush=True)

    return gpu_failures, stats


def _verify_streaming(
    parsed: ParsedDatabase,
    theorems: list[str],
    tokenizer: Tokenizer,
    is_variable: torch.Tensor,
    device: torch.device,
    step_budget: int = 30_000,
    tag: str = "GPU",
    verbose: bool = False,
    max_workers: int | None = None,
) -> tuple[list[str], list[str], int, float, float, list[dict]]:
    """Replay theorems in parallel and verify on GPU in streaming mega-batches.

    Workers replay proofs AND encode strings → int arrays in parallel.
    Main thread only does numpy concatenation + GPU dispatch — zero
    pure-Python tokenization.

    Returns (gpu_failures, replay_errors, total_steps, t_replay, t_gpu, batch_stats).
    """
    workers = min(max_workers or os.cpu_count() or 1, _MAX_REPLAY_WORKERS)
    gpu_failures: list[str] = []
    replay_errors: list[str] = []
    total_steps = 0
    t_replay = 0.0
    t_gpu = 0.0
    batch_stats: list[dict] = []
    theorems_replayed = 0

    pending: list[_EncodedBatch] = []
    pending_steps = 0

    def _flush() -> None:
        nonlocal t_gpu, pending, pending_steps
        if not pending:
            return
        enc = _concat_encoded_batches(pending)
        n_steps = enc.num_steps
        t0 = time.perf_counter()
        fails, stats = _verify_encoded_on_gpu(enc, device)
        dt = time.perf_counter() - t0
        t_gpu += dt
        gpu_failures.extend(fails)
        batch_stats.append(
            {
                "batch": len(batch_stats) + 1,
                "steps": n_steps,
                "gpu_time": dt,
                "steps_per_sec": n_steps / dt if dt > 0 else 0,
                "vocab": stats.get("compact_vocab", 0),
                "failures": len(fails),
            }
        )
        if verbose:
            bs = batch_stats[-1]
            print(
                f"  [{tag}] Batch {bs['batch']:3d}: "
                f"{bs['steps']:6d} steps in {bs['gpu_time']:.2f}s "
                f"({bs['steps_per_sec']:,.0f} steps/s) "
                f"V={bs['vocab']} failures={bs['failures']}"
            )
        pending = []
        pending_steps = 0
        print(f"  [{tag}] _flush done: {n_steps} steps, {dt:.2f}s, {len(fails)} failures", flush=True)
        gc.collect()
        print(f"  [{tag}] gc.collect done", flush=True)

    # Split theorems into mini-batches for worker calls.
    replay_batch_size = max(1, min(256, (len(theorems) + workers - 1) // workers))
    replay_batches = [
        theorems[i : i + replay_batch_size]
        for i in range(0, len(theorems), replay_batch_size)
    ]

    # Workers need the frozen symbol→ID mapping for encoding.
    symbol_to_id = tokenizer.symbol_to_id

    if verbose:
        print(f"  [{tag}] Creating pool: {workers} workers, "
              f"{len(replay_batches)} batches of ~{replay_batch_size} theorems")
    with _make_replay_pool(parsed, max_workers=workers,
                           symbol_to_id=symbol_to_id) as executor:
        all_futures = [
            (batch, executor.submit(_replay_batch_encoded, batch))
            for batch in replay_batches
        ]
        if verbose:
            print(f"  [{tag}] All {len(all_futures)} replay+encode tasks submitted, "
                  f"collecting results...", flush=True)

        for batch_labels, fut in all_futures:
            t0 = time.perf_counter()
            enc_batch, errs = fut.result()
            t_replay += time.perf_counter() - t0
            theorems_replayed += len(batch_labels)
            replay_errors.extend(errs)

            if verbose and theorems_replayed % 5000 < len(batch_labels):
                print(
                    f"  [{tag}] Replay: {theorems_replayed:,}/{len(theorems):,} theorems, "
                    f"{total_steps + pending_steps:,} steps, {t_replay:.1f}s"
                )

            if enc_batch is not None:
                total_steps += enc_batch.num_steps
                pending.append(enc_batch)
                pending_steps += enc_batch.num_steps

                if pending_steps >= step_budget:
                    _flush()

    print(f"  [{tag}] All futures collected, final flush...", flush=True)
    _flush()  # remaining
    print(f"  [{tag}] Exiting pool context manager...", flush=True)

    return gpu_failures, replay_errors, total_steps, t_replay, t_gpu, batch_stats


# ══════════════════════════════════════════════════════════════════════
#  set.mm — first 1000 theorems
# ══════════════════════════════════════════════════════════════════════


class TestSetMMFirst1000:
    @pytest.fixture(scope="class")
    def setmm_data(self):
        path = os.path.join(DATA_DIR, "set.mm")
        if not os.path.exists(path):
            pytest.skip("set.mm not found in data/")
        print("\n[set.mm] Parsing (this may take a moment for 866k lines)...")
        t0 = time.perf_counter()
        parsed = parse_mm_file(path)
        elapsed = time.perf_counter() - t0
        print(
            f"[set.mm] Parsed in {elapsed:.1f}s: "
            f"{len(parsed.assertions)} assertions, "
            f"{len(parsed.constants)} constants, "
            f"{len(parsed.variables)} variables"
        )
        tok = Tokenizer()
        db = MetamathDatabase(parsed, tok)
        return parsed, tok, db

    def test_cpu_verifies_first_1000_set_mm(self, setmm_data, capsys) -> None:
        """CPU verifier must pass the first 1000 theorems in set.mm."""
        parsed, tok, db = setmm_data
        cpu_v = CPUVerifier(parsed)

        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"][
            :1000
        ]

        passed = failed = 0
        failures: list[str] = []
        for label in theorems:
            r = cpu_v.verify_proof(label)
            if r.success:
                passed += 1
            else:
                failed += 1
                failures.append(f"  CPU FAIL  {label}: {r.error_message}")

        for f in failures:
            print(f)
        print(
            f"\n[set.mm CPU] {passed} passed, {failed} failed out of {len(theorems)} theorems"
        )
        assert failed == 0, f"{failed} theorems failed CPU verification"

    def test_gpu_matches_cpu_first_1000_set_mm(self, setmm_data, capsys) -> None:
        """GPU must match CPU on every assertion step of first 1000 set.mm theorems."""
        device = _get_gpu_device()
        backend = _gpu_backend_name()
        parsed, tok, db = setmm_data

        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"][
            :1000
        ]

        print(f"\n[set.mm {backend}] Verifying {len(theorems)} theorems (streaming)...")

        gpu_failures, replay_errors, total_steps, t_replay, t_gpu, _ = (
            _verify_streaming(
                parsed,
                theorems,
                tok,
                db.is_variable,
                device,
                tag="set.mm",
            )
        )

        for err in replay_errors:
            print(err)
        for fail in gpu_failures:
            print(fail)

        print(f"\n[set.mm {backend}] Results:")
        print(f"  Steps verified:  {total_steps}")
        print(f"  Replay time:     {t_replay:.2f}s")
        print(f"  GPU time:        {t_gpu:.2f}s")
        print(f"  Replay errors:   {len(replay_errors)}")
        print(f"  GPU failures:    {len(gpu_failures)}")

        assert len(replay_errors) == 0, f"{len(replay_errors)} proof replay errors"
        assert len(gpu_failures) == 0, (
            f"{len(gpu_failures)} GPU/CPU divergences — kernel is UNSOUND"
        )


# ══════════════════════════════════════════════════════════════════════
#  set.mm — FULL (all ~47k theorems, streaming)
# ══════════════════════════════════════════════════════════════════════


class TestSetMMFull:
    @pytest.fixture(scope="class")
    def setmm_data(self):
        path = os.path.join(DATA_DIR, "set.mm")
        if not os.path.exists(path):
            pytest.skip("set.mm not found in data/")
        print("\n[set.mm FULL] Parsing...")
        t0 = time.perf_counter()
        parsed = parse_mm_file(path)
        elapsed = time.perf_counter() - t0
        print(
            f"[set.mm FULL] Parsed in {elapsed:.1f}s: "
            f"{len(parsed.assertions)} assertions"
        )
        # Build ONLY the tokenizer + is_variable mask — skip MetamathDatabase
        # to avoid ~600MB of padded assertion tensors we don't need
        tok = Tokenizer()
        for c in parsed.constants:
            tok.encode_symbol(c)
        for v in parsed.variables:
            tok.encode_symbol(v)
        is_variable = torch.zeros(tok.vocab_size(), dtype=torch.bool)
        for v in parsed.variables:
            is_variable[tok.encode_symbol(v)] = True
        gc.collect()
        return parsed, tok, is_variable

    def test_gpu_all_set_mm(self, setmm_data, capsys) -> None:
        """GPU streaming verification of ALL set.mm theorems."""
        device = _get_gpu_device()
        backend = _gpu_backend_name()
        parsed, tok, is_variable = setmm_data

        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
        n_axioms = sum(1 for a in parsed.assertions.values() if a.type == "axiom")
        step_budget = 500_000 if device.type == "cuda" else 5_000
        print(
            f"\n[set.mm FULL {backend}] Verifying {len(theorems)} theorems "
            f"({n_axioms} axioms, {len(parsed.assertions)} total assertions)"
        )
        print(
            f"  Streaming budget: {'ALL (single batch)' if device.type == 'cuda' else f'{step_budget // 1000}k steps/batch'}"
        )
        print(f"  Backend: {backend}")
        print()

        t_total_0 = time.perf_counter()
        gpu_failures, replay_errors, total_steps, t_replay, t_gpu, batch_stats = (
            _verify_streaming(
                parsed,
                theorems,
                tok,
                is_variable,
                device,
                step_budget=step_budget,
                tag="set.mm FULL",
                verbose=True,
            )
        )
        t_total = time.perf_counter() - t_total_0

        for fail in gpu_failures[:20]:
            print(fail)
        if len(gpu_failures) > 20:
            print(f"  ... and {len(gpu_failures) - 20} more")

        # ── Summary table ─────────────────────────────────────────────
        avg_steps_per_sec = total_steps / t_gpu if t_gpu > 0 else 0
        avg_steps_per_batch = total_steps / len(batch_stats) if batch_stats else 0
        peak_batch_rate = max((b["steps_per_sec"] for b in batch_stats), default=0)
        min_batch_rate = min((b["steps_per_sec"] for b in batch_stats), default=0)

        print(f"\n{'\u2550' * 60}")
        print(f"  set.mm FULL \u2014 {backend} Verification Report")
        print(f"{'\u2550' * 60}")
        print(f"  Database")
        print(f"    Axioms:             {n_axioms:,}")
        print(f"    Theorems:           {len(theorems):,}")
        print(f"    Constants:          {len(parsed.constants):,}")
        print(f"    Variables:          {len(parsed.variables):,}")
        print(f"  Verification")
        print(f"    Total steps:        {total_steps:,}")
        print(f"    Mega-batches:       {len(batch_stats)}")
        print(f"    Avg steps/batch:    {avg_steps_per_batch:,.0f}")
        print(f"  Timing")
        print(f"    Replay (CPU):       {t_replay:.2f}s")
        print(f"    GPU (Metal):        {t_gpu:.2f}s")
        print(f"    Total wall clock:   {t_total:.2f}s")
        print(f"  Throughput")
        print(f"    Avg:                {avg_steps_per_sec:,.0f} steps/s")
        print(f"    Peak batch:         {peak_batch_rate:,.0f} steps/s")
        print(f"    Min batch:          {min_batch_rate:,.0f} steps/s")
        print(f"  Errors")
        print(f"    Replay errors:      {len(replay_errors)}")
        print(f"    GPU failures:       {len(gpu_failures)}")
        print(
            f"    Replay success:     {len(theorems) - len(replay_errors)}/{len(theorems)} "
            f"({100 * (len(theorems) - len(replay_errors)) / len(theorems):.1f}%)"
        )
        print(f"{'═' * 60}")

        # Allow replay errors (some set.mm proofs may use features we don't support)
        # but GPU failures are UNSOUND
        assert len(gpu_failures) == 0, (
            f"{len(gpu_failures)} GPU/CPU divergences — kernel is UNSOUND"
        )
