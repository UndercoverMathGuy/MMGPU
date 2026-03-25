#!/usr/bin/env python3
"""Run full set.mm verification + test suite.

Usage:
    python3 run_all.py

Executes:
  1. Full set.mm GPU verification (run_setmm.py)
  2. CUDA kernel correctness tests (test_cuda_kernels.py)
  3. Wheeler test suite — positive + negative (test_wheeler_suite.py)
"""
from __future__ import annotations

import subprocess
import sys
import time


def _run(label: str, cmd: list[str]) -> bool:
    """Run a command, print output, return success."""
    print(f"\n{'━' * 60}")
    print(f"  {label}")
    print(f"{'━' * 60}\n", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd="/app")
    dt = time.perf_counter() - t0
    ok = result.returncode == 0
    status = "PASSED" if ok else "FAILED"
    print(f"\n  [{status}] {label} ({dt:.1f}s)\n", flush=True)
    return ok


if __name__ == "__main__":
    results: list[tuple[str, bool]] = []

    # 1. Full set.mm GPU verification
    ok = _run(
        "set.mm FULL GPU verification",
        [sys.executable, "run_setmm.py"],
    )
    results.append(("set.mm verification", ok))

    # 2. CUDA kernel correctness tests (demo0, ql equivalence + memory)
    ok = _run(
        "CUDA kernel tests (demo0, ql, anatomy)",
        [sys.executable, "-m", "pytest", "-xvs",
         "tensormm/tests/test_cuda_kernels.py"],
    )
    results.append(("CUDA kernel tests", ok))

    # 3. Wheeler test suite — positive tests (GPU + CPU) + negative tests
    ok = _run(
        "Wheeler test suite (positive + negative)",
        [sys.executable, "-m", "pytest", "-xvs",
         "tensormm/tests/test_wheeler_suite.py"],
    )
    results.append(("Wheeler test suite", ok))

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Final Summary")
    print(f"{'═' * 60}")
    all_ok = True
    for label, ok in results:
        icon = "✓" if ok else "✗"
        print(f"  {icon} {label}")
        if not ok:
            all_ok = False
    print(f"{'═' * 60}")

    if all_ok:
        print(f"\n  ALL CHECKS PASSED\n")
        sys.exit(0)
    else:
        n_fail = sum(1 for _, ok in results if not ok)
        print(f"\n  {n_fail} CHECK(S) FAILED\n")
        sys.exit(1)
