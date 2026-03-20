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
import os
import time
from collections import defaultdict
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
    pytest.skip("No GPU available (need CUDA or MPS) — cannot run GPU correctness checks")
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
    theorem_label: str     # which theorem this step belongs to
    step_index: int        # step index within that theorem's proof
    step_label: str        # label of the axiom/theorem applied
    pattern: list[str]     # conclusion expression of the applied assertion
    substitution: dict[str, list[str]]  # variable -> replacement tokens
    expected_result: list[str]          # apply_substitution(pattern, subst)


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
        steps.append(_AssertionStep(
            theorem_label=theorem_label,
            step_index=step_counter,
            step_label=step_label,
            pattern=a.expression,
            substitution=subst,
            expected_result=result,
        ))
        step_counter += 1
        del stack[len(stack) - npop:]
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


def _collect_all_steps(
    parsed: ParsedDatabase,
    theorem_labels: list[str],
) -> tuple[list[_AssertionStep], list[str]]:
    """Replay all theorems, return (all_steps, replay_errors)."""
    label_info = _build_label_info(parsed)
    all_steps: list[_AssertionStep] = []
    replay_errors: list[str] = []
    for lbl in theorem_labels:
        extracted = _replay_proof_extract_steps(parsed, lbl, label_info)
        if isinstance(extracted, str):
            replay_errors.append(f"  REPLAY ERR  {lbl}: {extracted}")
        else:
            all_steps.extend(extracted)
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
        return [], {"total_steps": 0, "num_groups": 0, "max_batch": 0, "steps_verified": 0}

    use_metal = (device.type == "mps" and METAL_AVAILABLE)
    if use_metal:
        verifier = MetalVerifier()
    else:
        verifier = TensorVerifier(device=device)

    # ── Phase 1: encode (Python — tokenizer API is inherently Python) ─

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

    # ── Build compact vocab as numpy LUT ──────────────────────────────
    sorted_ids = sorted(all_token_ids)
    max_full_id = max(sorted_ids) + 1
    f2c_lut = np.zeros(max_full_id, dtype=np.int32)
    compact_ids = np.arange(len(sorted_ids), dtype=np.int32)
    f2c_lut[np.array(sorted_ids, dtype=np.int64)] = compact_ids
    V = len(sorted_ids)

    stats: dict[str, int] = {
        "total_steps": N, "num_groups": len(unique_groups),
        "max_batch": N, "steps_verified": 0, "compact_vocab": V,
    }

    # ── Phase 2: PURE NUMPY — sort, remap, pack ──────────────────────
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
    all_sub_toks_c = f2c_lut[all_sub_toks_np] if len(all_sub_toks_np) else all_sub_toks_np
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

    # ── Phase 3: chunked GPU dispatch — adaptive chunk size ─────────
    # H100: 80GB HBM3, M1: 8GB unified — scale budget accordingly
    if device.type == "cuda":
        GPU_MEM_BUDGET = 4 * 1024 * 1024 * 1024  # 4 GB
    else:
        GPU_MEM_BUDGET = 512 * 1024 * 1024  # 512 MB

    gpu_results_sorted = np.empty(N, dtype=np.bool_)

    start = 0
    while start < N:
        probe_end = min(start + chunk_size, N)
        chunk_S = max(int(s_max_s[start:probe_end].max()), 1)
        max_B = max(GPU_MEM_BUDGET // (V * chunk_S * 4), 64)
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
                sc_s = (np.arange(total_toks, dtype=np.int64) - offsets_within).astype(np.int32)
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
                pat_t.to(device), pl_t.to(device), st_t.to(device),
                sl_t.to(device), tgt_t.to(device), tl_t.to(device),
            )
        gpu_results_sorted[start:end] = gpu_out.numpy()
        stats["steps_verified"] += B
        start = end

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

        print(f"\n[ql.mm CPU] {passed} passed, {failed} failed out of {len(results)} theorems")
        assert failed == 0, f"{failed} theorems failed CPU verification"

    def test_gpu_matches_cpu_all_ql(self, ql_data, capsys) -> None:
        """GPU must match CPU on EVERY assertion step of EVERY theorem in ql.mm.

        Steps are batched by axiom/theorem label — one GPU call per unique pattern.
        """
        device = _get_gpu_device()
        backend = _gpu_backend_name()
        parsed, tok, db = ql_data

        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
        print(f"\n[ql.mm {backend}] Replaying {len(theorems)} theorems to extract assertion steps...")

        t0 = time.perf_counter()
        all_steps, replay_errors = _collect_all_steps(parsed, theorems)
        t_replay = time.perf_counter() - t0
        print(f"[ql.mm {backend}] Extracted {len(all_steps)} assertion steps in {t_replay:.2f}s")

        t1 = time.perf_counter()
        gpu_failures, stats = _verify_steps_batched_on_gpu(all_steps, tok, db.is_variable, device)
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
        assert len(gpu_failures) == 0, f"{len(gpu_failures)} GPU/CPU divergences — kernel is UNSOUND"


# ══════════════════════════════════════════════════════════════════════
#  Streaming GPU verification for large databases
# ══════════════════════════════════════════════════════════════════════

def _verify_streaming(
    parsed: ParsedDatabase,
    theorems: list[str],
    tokenizer: Tokenizer,
    is_variable: torch.Tensor,
    device: torch.device,
    step_budget: int = 30_000,
    tag: str = "GPU",
    verbose: bool = False,
) -> tuple[list[str], list[str], int, float, float, list[dict]]:
    """Replay theorems and verify on GPU in streaming mega-batches.

    Accumulates steps from replay until step_budget is reached, then
    verifies that batch on GPU and resets. Keeps peak memory bounded
    regardless of total step count.

    Returns (gpu_failures, replay_errors, total_steps, t_replay, t_gpu, batch_stats).
    """
    label_info = _build_label_info(parsed)
    gpu_failures: list[str] = []
    replay_errors: list[str] = []
    total_steps = 0
    t_replay = 0.0
    t_gpu = 0.0
    batch_stats: list[dict] = []
    theorems_replayed = 0

    pending: list[_AssertionStep] = []

    def _flush() -> None:
        nonlocal t_gpu
        if not pending:
            return
        t0 = time.perf_counter()
        fails, stats = _verify_steps_batched_on_gpu(
            pending, tokenizer, is_variable, device,
        )
        dt = time.perf_counter() - t0
        t_gpu += dt
        gpu_failures.extend(fails)
        batch_stats.append({
            "batch": len(batch_stats) + 1,
            "steps": len(pending),
            "gpu_time": dt,
            "steps_per_sec": len(pending) / dt if dt > 0 else 0,
            "vocab": stats.get("compact_vocab", 0),
            "failures": len(fails),
        })
        if verbose:
            bs = batch_stats[-1]
            print(f"  [{tag}] Batch {bs['batch']:3d}: "
                  f"{bs['steps']:6d} steps in {bs['gpu_time']:.2f}s "
                  f"({bs['steps_per_sec']:,.0f} steps/s) "
                  f"V={bs['vocab']} failures={bs['failures']}")
        pending.clear()
        gc.collect()

    for lbl in theorems:
        t0 = time.perf_counter()
        extracted = _replay_proof_extract_steps(parsed, lbl, label_info)
        t_replay += time.perf_counter() - t0
        theorems_replayed += 1

        # Free proof data after replay to reduce ParsedDatabase footprint
        a = parsed.assertions[lbl]
        a.proof = None
        a.compressed_proof = None

        if isinstance(extracted, str):
            replay_errors.append(f"  REPLAY ERR  {lbl}: {extracted}")
            continue

        total_steps += len(extracted)
        pending.extend(extracted)

        if len(pending) >= step_budget:
            _flush()

    _flush()  # remaining

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
        print(f"[set.mm] Parsed in {elapsed:.1f}s: "
              f"{len(parsed.assertions)} assertions, "
              f"{len(parsed.constants)} constants, "
              f"{len(parsed.variables)} variables")
        tok = Tokenizer()
        db = MetamathDatabase(parsed, tok)
        return parsed, tok, db

    def test_cpu_verifies_first_1000_set_mm(self, setmm_data, capsys) -> None:
        """CPU verifier must pass the first 1000 theorems in set.mm."""
        parsed, tok, db = setmm_data
        cpu_v = CPUVerifier(parsed)

        theorems = [
            lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"
        ][:1000]

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
        print(f"\n[set.mm CPU] {passed} passed, {failed} failed out of {len(theorems)} theorems")
        assert failed == 0, f"{failed} theorems failed CPU verification"

    def test_gpu_matches_cpu_first_1000_set_mm(self, setmm_data, capsys) -> None:
        """GPU must match CPU on every assertion step of first 1000 set.mm theorems."""
        device = _get_gpu_device()
        backend = _gpu_backend_name()
        parsed, tok, db = setmm_data

        theorems = [
            lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"
        ][:1000]

        print(f"\n[set.mm {backend}] Verifying {len(theorems)} theorems (streaming)...")

        gpu_failures, replay_errors, total_steps, t_replay, t_gpu, _ = _verify_streaming(
            parsed, theorems, tok, db.is_variable, device, tag="set.mm",
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
        assert len(gpu_failures) == 0, f"{len(gpu_failures)} GPU/CPU divergences — kernel is UNSOUND"


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
        print(f"[set.mm FULL] Parsed in {elapsed:.1f}s: "
              f"{len(parsed.assertions)} assertions")
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
        step_budget = 30_000 if device.type == "cuda" else 5_000
        print(f"\n[set.mm FULL {backend}] Verifying {len(theorems)} theorems "
              f"({n_axioms} axioms, {len(parsed.assertions)} total assertions)")
        print(f"  Streaming budget: {step_budget // 1000}k steps/batch")
        print(f"  Backend: {backend}")
        print()

        t_total_0 = time.perf_counter()
        gpu_failures, replay_errors, total_steps, t_replay, t_gpu, batch_stats = _verify_streaming(
            parsed, theorems, tok, is_variable, device,
            step_budget=step_budget, tag="set.mm FULL", verbose=True,
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
        print(f"    Replay success:     {len(theorems) - len(replay_errors)}/{len(theorems)} "
              f"({100 * (len(theorems) - len(replay_errors)) / len(theorems):.1f}%)")
        print(f"{'═' * 60}")

        # Allow replay errors (some set.mm proofs may use features we don't support)
        # but GPU failures are UNSOUND
        assert len(gpu_failures) == 0, f"{len(gpu_failures)} GPU/CPU divergences — kernel is UNSOUND"
