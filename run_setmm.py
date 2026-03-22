#!/usr/bin/env python3
"""Standalone full set.mm GPU verification — no pytest, no silent skips.

Usage:
    python3 run_setmm.py
"""
from __future__ import annotations

import gc
import os
import sys
import time

import torch
from tensormm.parser import parse_mm_file
from tensormm.gpu_verifier import verify_database, warmup_cuda

# ── Fail loudly if no GPU ──────────────────────────────────────────
print(f"PyTorch {torch.__version__}, CUDA compiled: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    device = torch.device("cuda")
    backend = "CUDA"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    backend = "Metal/MPS"
else:
    print("FATAL: No GPU detected (need CUDA or MPS). Exiting.")
    sys.exit(1)

print(f"Using device: {device} ({backend})")

# ── CUDA warmup — force lazy init + pre-compile all kernels ───────
if device.type == "cuda":
    _t_warm = time.perf_counter()
    _ = torch.zeros(1, device=device)  # triggers context + allocator init
    torch.cuda.synchronize()
    print(f"CUDA context init: {time.perf_counter() - _t_warm:.1f}s")
    _t_warm = time.perf_counter()
    warmup_cuda(device)
    torch.cuda.synchronize()
    print(f"CUDA kernel warmup: {time.perf_counter() - _t_warm:.2f}s")
print()

# ── Locate set.mm ──────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
setmm_path = os.path.join(DATA_DIR, "set.mm")
print(f"Looking for set.mm at: {setmm_path}")
if not os.path.exists(setmm_path):
    print(f"FATAL: set.mm not found at {setmm_path}")
    print(f"Contents of {DATA_DIR}: {os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else 'DIR NOT FOUND'}")
    sys.exit(1)
print(f"Found set.mm: {os.path.getsize(setmm_path) / 1e6:.1f} MB\n")

# ── Parse ──────────────────────────────────────────────────────────
print("[set.mm FULL] Parsing...")
t0 = time.perf_counter()
parsed = parse_mm_file(setmm_path)
t_parse = time.perf_counter() - t0
print(f"[set.mm FULL] Parsed in {t_parse:.1f}s: {len(parsed.assertions)} assertions\n")

gc.collect()

# ── Run verification ───────────────────────────────────────────────
theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
n_axioms = sum(1 for a in parsed.assertions.values() if a.type == "axiom")

print(f"[set.mm FULL {backend}] Verifying {len(theorems):,} theorems "
      f"({n_axioms:,} axioms, {len(parsed.assertions):,} total assertions)")
print(f"  Backend: {backend}")
print()

t_total_0 = time.perf_counter()
gpu_results = verify_database(parsed, theorems, device=device, verbose=True)
t_total = time.perf_counter() - t_total_0

gpu_pass = sum(1 for v in gpu_results.values() if v)
gpu_fail = sum(1 for v in gpu_results.values() if not v)

# ── Summary ────────────────────────────────────────────────────────
print(f"\n{'═' * 60}")
print(f"  set.mm FULL — {backend} Verification Report")
print(f"{'═' * 60}")
print(f"  Database")
print(f"    Axioms:             {n_axioms:,}")
print(f"    Theorems:           {len(theorems):,}")
print(f"    Constants:          {len(parsed.constants):,}")
print(f"    Variables:          {len(parsed.variables):,}")
print(f"  Results")
print(f"    Passed:             {gpu_pass:,}")
print(f"    Failed:             {gpu_fail:,}")
print(f"  Timing")
print(f"    Parse:              {t_parse:.2f}s")
print(f"    Total wall clock:   {t_total:.2f}s")
print(f"{'═' * 60}")

if gpu_fail:
    print(f"\nFATAL: {gpu_fail} GPU verification failures — kernel is UNSOUND")
    sys.exit(1)
else:
    print(f"\nSUCCESS: All {gpu_pass:,} theorems verified correctly.")
    sys.exit(0)
