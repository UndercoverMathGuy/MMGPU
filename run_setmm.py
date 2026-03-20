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

# ── Fail loudly if no GPU ──────────────────────────────────────────
print(f"PyTorch {torch.__version__}, CUDA compiled: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    device = torch.device("cuda")
    backend = "CUDA"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    backend = "Metal/MPS"
else:
    print("FATAL: No GPU detected (need CUDA or MPS). Exiting.")
    sys.exit(1)

print(f"Using device: {device} ({backend})\n")

# ── Locate set.mm ──────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
setmm_path = os.path.join(DATA_DIR, "set.mm")
print(f"Looking for set.mm at: {setmm_path}")
if not os.path.exists(setmm_path):
    print(f"FATAL: set.mm not found at {setmm_path}")
    print(f"Contents of {DATA_DIR}: {os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else 'DIR NOT FOUND'}")
    sys.exit(1)
print(f"Found set.mm: {os.path.getsize(setmm_path) / 1e6:.1f} MB\n")

# ── Import project modules ────────────────────────────────────────
from tensormm.parser import parse_mm_file
from tensormm.tokenizer import Tokenizer

# Reuse the battle-tested verification functions from the test module
from tensormm.tests.test_full_correctness import _verify_streaming

# ── Parse ──────────────────────────────────────────────────────────
print("[set.mm FULL] Parsing...")
t0 = time.perf_counter()
parsed = parse_mm_file(setmm_path)
t_parse = time.perf_counter() - t0
print(f"[set.mm FULL] Parsed in {t_parse:.1f}s: {len(parsed.assertions)} assertions\n")

# ── Build tokenizer + is_variable mask ─────────────────────────────
tok = Tokenizer()
for c in parsed.constants:
    tok.encode_symbol(c)
for v in parsed.variables:
    tok.encode_symbol(v)
is_variable = torch.zeros(tok.vocab_size(), dtype=torch.bool)
for v in parsed.variables:
    is_variable[tok.encode_symbol(v)] = True
gc.collect()

# ── Run verification ───────────────────────────────────────────────
theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
n_axioms = sum(1 for a in parsed.assertions.values() if a.type == "axiom")
step_budget = 30_000 if device.type == "cuda" else 5_000

print(f"[set.mm FULL {backend}] Verifying {len(theorems):,} theorems "
      f"({n_axioms:,} axioms, {len(parsed.assertions):,} total assertions)")
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

# ── Summary ────────────────────────────────────────────────────────
avg_steps_per_sec = total_steps / t_gpu if t_gpu > 0 else 0
avg_steps_per_batch = total_steps / len(batch_stats) if batch_stats else 0
peak_batch_rate = max((b["steps_per_sec"] for b in batch_stats), default=0)
min_batch_rate = min((b["steps_per_sec"] for b in batch_stats), default=0)

print(f"\n{'═' * 60}")
print(f"  set.mm FULL — {backend} Verification Report")
print(f"{'═' * 60}")
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
print(f"    Parse:              {t_parse:.2f}s")
print(f"    Replay (CPU):       {t_replay:.2f}s")
print(f"    GPU:                {t_gpu:.2f}s")
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

if gpu_failures:
    print(f"\nFATAL: {len(gpu_failures)} GPU verification failures — kernel is UNSOUND")
    sys.exit(1)
else:
    print(f"\nSUCCESS: All {total_steps:,} steps verified correctly.")
    sys.exit(0)
