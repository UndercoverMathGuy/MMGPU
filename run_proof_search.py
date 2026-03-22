#!/usr/bin/env python3
"""Race: TensorMM proof search (CPU-only vs GPU-verified) vs metamath-knife.

For each .mm file, for each theorem:
  1. CPU-only forward search  → find proof, verify with CPUVerifier
  2. GPU-verified forward search → same search but GPU-checks every derivation
  3. metamath-knife → just verify the *original* proof (baseline, it can't search)

The race shows our proof search (doing something HARDER — finding proofs from
scratch) compared to knife just verifying existing proofs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

import torch

from tensormm.parser import parse_mm_file
from tensormm.proof_search import forward_search, select_axioms_and_defs

# ── Config ───────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wheeler-tests")
FILES = ["anatomy.mm", "demo0.mm"]
TIMEOUT_PER_THM = 30.0
MAX_POOL_SIZE = 500_000
MAX_DEPTH = 15
MAX_VARIABLES = 4

KNIFE_BIN = shutil.which("metamath-knife") or os.path.expanduser("~/.cargo/bin/metamath-knife")


# ── Helpers ──────────────────────────────────────────────────────────

def detect_gpu() -> str:
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_properties(0).name})"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    return "CPU only"


def run_knife(filepath: str) -> tuple[float, bool]:
    """Run metamath-knife --verify and return (elapsed_seconds, success)."""
    if not os.path.isfile(KNIFE_BIN):
        return -1.0, False
    t0 = time.perf_counter()
    r = subprocess.run(
        [KNIFE_BIN, "--verify", "--time", filepath],
        capture_output=True, text=True, timeout=60,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, r.returncode == 0


def search_one(
    db, thm_label: str, available: list[str], use_gpu: bool,
) -> dict:
    """Run forward_search for one theorem, return a results dict."""
    thm = db.assertions[thm_label]
    mel = max(30, len(thm.expression) * 3)

    result = forward_search(
        db, thm_label,
        max_depth=MAX_DEPTH,
        timeout=TIMEOUT_PER_THM,
        max_variables=MAX_VARIABLES,
        max_expr_len=mel,
        max_pool_size=MAX_POOL_SIZE,
        available_labels=available,
        use_gpu=use_gpu,
    )
    return {
        "label": thm_label,
        "target": " ".join(thm.expression),
        "found": result.success,
        "depth": result.depth_reached,
        "pool": result.pool_size,
        "tried": result.candidates_tried,
        "gpu_ok": result.gpu_verified,
        "gpu_rej": result.gpu_rejected,
        "proof": " ".join(result.proof_labels) if result.proof_labels else "",
        "verified": result.verification.success if result.verification else False,
        "elapsed": result.elapsed,
    }


# ── Race ─────────────────────────────────────────────────────────────

def race_file(filepath: str) -> None:
    name = os.path.basename(filepath)
    print(f"\n{'═' * 72}")
    print(f"  RACE: {name}")
    print(f"{'═' * 72}")

    # ── Parse ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    db = parse_mm_file(filepath)
    t_parse = time.perf_counter() - t0

    theorems = [lbl for lbl, a in db.assertions.items() if a.type == "theorem"]
    axioms = [lbl for lbl, a in db.assertions.items() if a.type == "axiom"]
    available = select_axioms_and_defs(db)

    print(f"  Parsed in {t_parse:.3f}s — {len(axioms)} axioms, {len(theorems)} theorems")
    print()

    # ── Lane 1: metamath-knife (verify existing proofs) ──────────────
    print(f"  ┌── LANE 1: metamath-knife --verify")
    knife_time, knife_ok = run_knife(filepath)
    if knife_time < 0:
        print(f"  │   SKIP (metamath-knife not found at {KNIFE_BIN})")
    else:
        tag = "✓ PASS" if knife_ok else "✗ FAIL"
        print(f"  │   {tag}  {knife_time*1000:.1f}ms (verify existing proofs)")
    print(f"  └──")
    print()

    # ── Lane 2: CPU-only forward search ──────────────────────────────
    print(f"  ┌── LANE 2: TensorMM CPU-only forward search")
    cpu_results = []
    t_cpu_total = time.perf_counter()
    for thm_label in theorems:
        print(f"  │")
        print(f"  │   {thm_label}: {' '.join(db.assertions[thm_label].expression)}")
        r = search_one(db, thm_label, available, use_gpu=False)
        cpu_results.append(r)
        tag = "✓ FOUND+VERIFIED" if r["found"] and r["verified"] else (
            "✓ FOUND" if r["found"] else "✗ NOT FOUND"
        )
        print(f"  │   {tag}  depth={r['depth']} pool={r['pool']:,} "
              f"t={r['elapsed']:.3f}s")
        if r["proof"]:
            print(f"  │   proof: {r['proof']}")
    t_cpu_total = time.perf_counter() - t_cpu_total
    cpu_found = sum(1 for r in cpu_results if r["found"])
    print(f"  │")
    print(f"  │   Summary: {cpu_found}/{len(theorems)} found in {t_cpu_total:.3f}s")
    print(f"  └──")
    print()

    # ── Lane 3: GPU-verified forward search ──────────────────────────
    print(f"  ┌── LANE 3: TensorMM GPU-verified forward search")
    gpu_results = []
    t_gpu_total = time.perf_counter()
    for thm_label in theorems:
        print(f"  │")
        print(f"  │   {thm_label}: {' '.join(db.assertions[thm_label].expression)}")
        r = search_one(db, thm_label, available, use_gpu=True)
        gpu_results.append(r)
        tag = "✓ FOUND+VERIFIED" if r["found"] and r["verified"] else (
            "✓ FOUND" if r["found"] else "✗ NOT FOUND"
        )
        gpu_tag = f" gpu={r['gpu_ok']}✓/{r['gpu_rej']}✗" if r["gpu_ok"] or r["gpu_rej"] else ""
        print(f"  │   {tag}  depth={r['depth']} pool={r['pool']:,}"
              f"{gpu_tag} t={r['elapsed']:.3f}s")
        if r["proof"]:
            print(f"  │   proof: {r['proof']}")
    t_gpu_total = time.perf_counter() - t_gpu_total
    gpu_found = sum(1 for r in gpu_results if r["found"])
    print(f"  │")
    print(f"  │   Summary: {gpu_found}/{len(theorems)} found in {t_gpu_total:.3f}s")
    print(f"  └──")

    # ── Scoreboard ───────────────────────────────────────────────────
    print()
    print(f"  ╔{'═' * 50}╗")
    print(f"  ║  SCOREBOARD — {name:<36}║")
    print(f"  ╠{'═' * 50}╣")
    if knife_time >= 0:
        print(f"  ║  metamath-knife (verify only): {knife_time*1000:>8.1f}ms     ║")
    print(f"  ║  CPU search (find+verify):      {t_cpu_total*1000:>8.1f}ms     ║")
    print(f"  ║  GPU search (find+GPU+verify):   {t_gpu_total*1000:>8.1f}ms     ║")
    print(f"  ║                                                  ║")
    print(f"  ║  CPU found: {cpu_found}/{len(theorems)}   "
          f"GPU found: {gpu_found}/{len(theorems):<20}║")
    print(f"  ╚{'═' * 50}╝")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    print(f"GPU backend: {detect_gpu()}")
    print(f"metamath-knife: {KNIFE_BIN}")
    print(f"Timeout: {TIMEOUT_PER_THM:.0f}s/theorem, pool cap: {MAX_POOL_SIZE:,}")

    for fname in FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"\n  SKIP {fname}: not found")
            continue
        race_file(path)

    print(f"\n{'═' * 72}")
    print("  Race complete.")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
