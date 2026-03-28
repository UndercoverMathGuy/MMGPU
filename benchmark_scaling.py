#!/usr/bin/env python3
"""GPU verification scaling benchmark.

For each multiplier N the benchmark:

  1. Takes the one-time parsed database and stamps out N copies
     (copy 0 verbatim, copies 1..N-1 with prefixed labels).
  2. Merges all copies into one ParsedDatabase.
  3. Calls verify_database ONCE on the merged database — this runs
     the full 4-phase pipeline (graph construction, level packing,
     GPU execution, $d post-check) and prints per-phase timings.

Merge time is measured but excluded from the reported benchmark time.
Parse is done once; its cost is constant and printed separately.

Single GPU per invocation.  Run the script once per machine/GPU.

Multipliers : 1×  2×  5×  10×  20×  ×  5 trials.

Usage:
    python3 benchmark_scaling.py
"""
from __future__ import annotations

import gc
import os
import platform
import subprocess
import sys
import time

import torch

from tensormm.parser import (
    parse_mm_file,
    ParsedDatabase,
    FloatingHyp,
    EssentialHyp,
    Assertion,
    CompressedProof,
)
from tensormm.gpu_verifier import verify_database

# ── Config ─────────────────────────────────────────────────────────────────────
MULTIPLIERS = [1, 2, 5, 10, 20]
TRIALS      = 5
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SETMM_PATH  = os.path.join(DATA_DIR, "set.mm")

W = 72
def _banner(title: str, c: str = "═") -> None:
    b = c * W
    print(f"\n{b}\n  {title}\n{b}\n", flush=True)

def _flush(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


# ── Hardware info ──────────────────────────────────────────────────────────────

def _get_cpu_model() -> str:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
            return out
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"

def _get_ram_gb() -> float:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / 1e9
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / 1e6
    except Exception:
        pass
    return 0.0

def _get_nvidia_driver() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0]
        return out
    except Exception:
        return "N/A"


# ── Core: stamp out N distinct copies and merge ──────────────────────────────

def _prefix_copy(db: ParsedDatabase, prefix: str) -> tuple[
        dict[str, FloatingHyp],
        dict[str, EssentialHyp],
        dict[str, Assertion],
]:
    """Return new dicts where every label and every proof-step reference
    is prefixed with `prefix`.  The copy is fully self-contained."""
    p = prefix

    new_fh: dict[str, FloatingHyp] = {}
    for lbl, fh in db.floating_hyps.items():
        new_fh[p + lbl] = FloatingHyp(label=p + lbl,
                                       type_code=fh.type_code,
                                       variable=fh.variable)

    new_eh: dict[str, EssentialHyp] = {}
    for lbl, eh in db.essential_hyps.items():
        new_eh[p + lbl] = EssentialHyp(label=p + lbl,
                                        expression=list(eh.expression))

    new_asst: dict[str, Assertion] = {}
    for lbl, a in db.assertions.items():
        new_lbl = p + lbl

        new_proof = None
        if a.proof is not None:
            new_proof = [p + step for step in a.proof]

        new_cp = None
        if a.compressed_proof is not None:
            cp = a.compressed_proof
            new_cp = CompressedProof(
                labels     = [p + l for l in cp.labels],
                proof_ints = list(cp.proof_ints),
            )

        new_asst[new_lbl] = Assertion(
            label            = new_lbl,
            type             = a.type,
            expression       = list(a.expression),
            floating_hyps    = [p + f for f in a.floating_hyps],
            essential_hyps   = [p + e for e in a.essential_hyps],
            proof            = new_proof,
            compressed_proof = new_cp,
            disjoint_vars    = list(a.disjoint_vars),
            all_disjoint_vars= set(a.all_disjoint_vars),
        )

    return new_fh, new_eh, new_asst


def _merge(base: ParsedDatabase, n: int) -> tuple[ParsedDatabase, list[str]]:
    """Create an N× merged database from a single parsed database.

    Copy 0 is embedded verbatim.  Copies 1..N-1 are stamped with __c{i}_
    on every label and proof-step reference.
    Returns (merged_db, flat list of all theorem labels in order).
    """
    merged = ParsedDatabase()
    all_theorems: list[str] = []

    # Copy 0: verbatim
    merged.constants       |= base.constants
    merged.variables       |= base.variables
    merged.floating_hyps.update(base.floating_hyps)
    merged.essential_hyps.update(base.essential_hyps)
    merged.assertions.update(base.assertions)
    all_theorems += [lbl for lbl, a in base.assertions.items()
                     if a.type == "theorem"]

    # Copies 1..N-1: prefixed
    for i in range(1, n):
        prefix = f"__c{i}_"
        merged.constants |= base.constants
        merged.variables |= base.variables
        fh, eh, asst = _prefix_copy(base, prefix)
        merged.floating_hyps.update(fh)
        merged.essential_hyps.update(eh)
        merged.assertions.update(asst)
        all_theorems += [lbl for lbl, a in asst.items()
                         if a.type == "theorem"]

    return merged, all_theorems


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── GPU discovery ─────────────────────────────────────────────────────────
    _banner("GPU Scaling Benchmark — Metamath set.mm")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        vram_gb = round(props.total_memory / 1e9, 1)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        gpu_name = "Apple Metal/MPS"
        vram_gb = 0.0
    else:
        print("FATAL: no GPU"); sys.exit(1)

    # ── Print hardware info ───────────────────────────────────────────────────
    cpu_model   = _get_cpu_model()
    cpu_cores   = os.cpu_count()
    ram_gb      = _get_ram_gb()
    driver_ver  = _get_nvidia_driver()

    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.version.cuda}")
    print(f"Driver       : {driver_ver}")
    print(f"GPU          : {gpu_name}  ({vram_gb} GB)")
    print(f"Host CPU     : {cpu_model}")
    print(f"Host cores   : {cpu_cores}")
    print(f"Host RAM     : {ram_gb:.1f} GB")
    print(f"Multipliers  : {MULTIPLIERS}")
    print(f"Trials/mult  : {TRIALS}")

    # ── Locate set.mm ─────────────────────────────────────────────────────────
    if not os.path.exists(SETMM_PATH):
        print(f"FATAL: {SETMM_PATH} not found"); sys.exit(1)
    print(f"\nset.mm       : {SETMM_PATH}  ({os.path.getsize(SETMM_PATH)/1e6:.1f} MB)")

    # ── Parse (once, constant cost) ──────────────────────────────────────────
    _banner("Parsing set.mm (once)")
    t0 = time.perf_counter()
    parsed = parse_mm_file(SETMM_PATH)
    parse_time = time.perf_counter() - t0

    base_theorems = sum(1 for a in parsed.assertions.values() if a.type == "theorem")
    base_axioms   = sum(1 for a in parsed.assertions.values() if a.type == "axiom")
    print(f"Parsed in {parse_time:.2f}s")
    print(f"  {base_theorems:,} theorems, {base_axioms:,} axioms, "
          f"{len(parsed.assertions):,} total assertions")

    # ── Warmup run (N=1, discarded) ──────────────────────────────────────────
    _banner("Warmup run (N=1, discarded)")
    print("Running full pipeline once to warm up kernels…\n", flush=True)
    warmup_merged, warmup_labels = _merge(parsed, 1)
    warmup_results = verify_database(warmup_merged, warmup_labels,
                                     device=device, verbose=True)
    warmup_pass = sum(1 for v in warmup_results.values() if v is None)
    warmup_fail = sum(1 for v in warmup_results.values() if v is not None)
    print(f"\nWarmup: {warmup_pass:,} passed, {warmup_fail:,} failed")
    del warmup_merged, warmup_labels, warmup_results
    _flush(device)
    print("Warmup complete.\n")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    all_results: list[dict] = []

    for mult in MULTIPLIERS:
        _banner(f"{mult}× set.mm  |  {TRIALS} trials")

        for trial in range(1, TRIALS + 1):
            print(f"\n{'━'*W}")
            print(f"  Trial {trial}/{TRIALS}  |  {mult}×")
            print(f"{'━'*W}\n", flush=True)

            _flush(device)

            # ── Merge N copies (timed but excluded from benchmark) ────────
            print(f"[merge] building {mult}× database…", flush=True)
            t_merge_start = time.perf_counter()
            merged, all_labels = _merge(parsed, mult)
            t_merge = time.perf_counter() - t_merge_start
            print(f"[merge] done  {t_merge:.2f}s  "
                  f"({len(all_labels):,} theorems, "
                  f"{len(merged.assertions):,} assertions)")

            # ── Full pipeline: graph → pack → GPU → $d ────────────────────
            # verify_database prints per-phase timings with verbose=True
            print(f"\n[verify] full pipeline on {len(all_labels):,} theorems…\n",
                  flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_verify_start = time.perf_counter()
            results = verify_database(merged, all_labels,
                                      device=device, verbose=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_verify = time.perf_counter() - t_verify_start

            n_pass = sum(1 for v in results.values() if v is None)
            n_fail = sum(1 for v in results.values() if v is not None)

            # ── Correctness assertions ────────────────────────────────────
            assert n_fail == 0, (
                f"FAIL: {n_fail} theorems failed at {mult}×, trial {trial}. "
                f"Failures: {[k for k,v in results.items() if v is not None][:10]}"
            )
            assert n_pass == len(all_labels), (
                f"FAIL: expected {len(all_labels)} passed, got {n_pass}"
            )

            row = {
                "mult":       mult,
                "trial":      trial,
                "theorems":   len(all_labels),
                "merge_s":    round(t_merge, 4),
                "verify_s":   round(t_verify, 4),
                "pass":       n_pass,
                "fail":       n_fail,
            }
            all_results.append(row)

            print(f"\n{'━'*W}")
            print(f"  Trial {trial}/{TRIALS}  |  {mult}×  |  ALL PASSED")
            print(f"  merge={t_merge:.3f}s (excluded)  "
                  f"verify={t_verify:.3f}s (pipeline total)")
            print(f"  {n_pass:,} theorems  "
                  f"throughput={len(all_labels)/t_verify:,.0f} theorems/s")
            print(f"{'━'*W}\n", flush=True)

            del merged, all_labels, results
            _flush(device)

    # ── Summary ────────────────────────────────────────────────────────────────
    _banner("Scaling Summary")

    print(f"GPU: {gpu_name}\n")

    hdr = "  ".join(f"  T{t}" for t in range(1, TRIALS + 1))
    print(f"  {'Mult':>5}  {'Theorems':>10}  {hdr}   {'Mean':>8}  {'Thpt/s':>10}")
    print(f"  {'─'*5}  {'─'*10}  {'  ──────'*TRIALS}   {'─'*8}  {'─'*10}")

    for m in MULTIPLIERS:
        mr = sorted([r for r in all_results if r["mult"] == m],
                     key=lambda r: r["trial"])
        if not mr:
            continue
        vs   = [r["verify_s"] for r in mr]
        mean = sum(vs) / len(vs)
        n    = mr[0]["theorems"]
        tput = n / mean
        print(f"  {m:>4}×  {n:>10,}  "
              f"{'  '.join(f'{v:6.2f}' for v in vs)}   "
              f"{mean:>8.3f}  {tput:>10,.0f}")
    print()

    # ── Scaling ratios ────────────────────────────────────────────────────────
    def _mean_verify(m: int) -> float:
        mr = [r for r in all_results if r["mult"] == m]
        return sum(r["verify_s"] for r in mr) / len(mr) if mr else float("nan")

    base_t = _mean_verify(1)
    print(f"  Scaling ratios (verify time, vs 1×)")
    print(f"  {'Mult':>5}  {'Mean(s)':>9}  {'Actual ratio':>13}  "
          f"{'Ideal':>7}  {'Efficiency':>11}")
    print(f"  {'─'*5}  {'─'*9}  {'─'*13}  {'─'*7}  {'─'*11}")
    for m in MULTIPLIERS:
        mt    = _mean_verify(m)
        ratio = mt / base_t
        eff   = m / ratio
        print(f"  {m:>4}×  {mt:>9.3f}  {ratio:>12.3f}×  "
              f"{m:>6.1f}×  {eff:>10.3f}×")
    print()

    print("═" * W)
    print("  Done.")
    print("═" * W)
    sys.exit(0)
