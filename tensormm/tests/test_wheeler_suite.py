"""Wheeler metamath-test suite — GPU verifier correctness tests.

Runs the david-a-wheeler/metamath-test suite against our full pipeline:
  parse → CPU replay → GPU substitution verification.

Test expectations come directly from the upstream run-testsuite script.
"bad" files should fail during either parsing or CPU proof replay.
"good" files must pass parsing, CPU replay, AND GPU verification.

See: https://github.com/david-a-wheeler/metamath-test
"""
from __future__ import annotations

import gc
import os
import time

import pytest
import torch

from tensormm.cpu_verifier import CPUVerifier
from tensormm.parser import ParsedDatabase, parse_mm_file
from tensormm.tokenizer import Tokenizer
from tensormm.database import MetamathDatabase
from tensormm.tests.test_full_correctness import (
    _collect_all_steps,
    _get_gpu_device,
    _gpu_backend_name,
    _verify_steps_batched_on_gpu,
)


WHEELER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "wheeler-tests"
)


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════


def _parse_and_verify_cpu(path: str) -> tuple[bool, str, ParsedDatabase | None]:
    """Parse a .mm file and verify all theorems with the CPU verifier.

    Returns (all_ok, detail_msg, parsed_db_or_None).
    """
    try:
        db = parse_mm_file(path)
    except Exception as e:
        return False, f"PARSE ERROR: {e}", None

    theorems = [k for k, v in db.assertions.items() if v.type == "theorem"]
    if not theorems:
        return True, "no theorems to verify", db

    v = CPUVerifier(db)
    for lbl in theorems:
        r = v.verify_proof(lbl)
        if not r.success:
            return False, f"{lbl}: {r.error_message}", db

    return True, f"{len(theorems)} theorems all pass", db


def _verify_gpu(parsed: ParsedDatabase) -> tuple[int, int, str]:
    """Run GPU substitution verification on all theorems.

    Returns (total_steps, n_failures, detail_msg).
    """
    device = _get_gpu_device()

    theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
    if not theorems:
        return 0, 0, "no theorems"

    tok = Tokenizer()
    for c in parsed.constants:
        tok.encode_symbol(c)
    for v in parsed.variables:
        tok.encode_symbol(v)
    is_variable = torch.zeros(tok.vocab_size(), dtype=torch.bool)
    for v in parsed.variables:
        is_variable[tok.encode_symbol(v)] = True

    all_steps, replay_errors = _collect_all_steps(parsed, theorems)
    if replay_errors:
        return 0, len(replay_errors), f"replay errors: {replay_errors[0]}"

    if not all_steps:
        return 0, 0, "no assertion steps extracted"

    gpu_failures, stats = _verify_steps_batched_on_gpu(
        all_steps, tok, is_variable, device
    )
    n = stats.get("steps_verified", 0)
    return n, len(gpu_failures), f"{n} steps, {len(gpu_failures)} GPU failures"


# ══════════════════════════════════════════════════════════════════════
#  Test cases — mirrors run-testsuite from david-a-wheeler/metamath-test
# ══════════════════════════════════════════════════════════════════════


def _wheeler_path(name: str) -> str:
    return os.path.join(WHEELER_DIR, name)


def _skip_if_missing(name: str) -> str:
    p = _wheeler_path(name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not found in {WHEELER_DIR}")
    return p


# ── Files that SHOULD PASS ──────────────────────────────────────────

class TestWheelerPass:
    """Files that must parse and verify correctly (CPU + GPU)."""

    @pytest.mark.parametrize("filename", [
        "anatomy.mm",
        "big-unifier.mm",
        "demo0.mm",
        "demo0-includer.mm",
        "emptyline.mm",
        "hol.mm",
        "iset.mm",
        "miu.mm",
        "nf.mm",
        "peano-fixed.mm",
        "ql.mm",
    ])
    def test_cpu_pass(self, filename: str) -> None:
        """CPU verifier must accept all theorems in this file."""
        path = _skip_if_missing(filename)
        ok, detail, _ = _parse_and_verify_cpu(path)
        assert ok, f"{filename} should PASS but: {detail}"

    @pytest.mark.parametrize("filename", [
        "anatomy.mm",
        "big-unifier.mm",
        "demo0.mm",
        "demo0-includer.mm",
        "miu.mm",
        "ql.mm",
        "hol.mm",
    ])
    def test_gpu_pass(self, filename: str) -> None:
        """GPU verifier must accept all assertion steps in this file."""
        path = _skip_if_missing(filename)
        parsed = parse_mm_file(path)
        theorems = [k for k, v in parsed.assertions.items() if v.type == "theorem"]
        if not theorems:
            pytest.skip(f"{filename} has no theorems")

        n_steps, n_fail, detail = _verify_gpu(parsed)
        print(f"  [{filename}] {detail}")
        assert n_fail == 0, f"{filename} GPU: {detail}"
        gc.collect()


# ── Files that SHOULD FAIL ──────────────────────────────────────────

class TestWheelerFail:
    """Files with deliberately broken proofs — must be rejected."""

    @pytest.mark.parametrize("filename,description", [
        ("anatomy-bad1.mm", "wrong proof step"),
        ("anatomy-bad2.mm", "too few proof steps"),
        ("anatomy-bad3.mm", "missing first step"),
        ("big-unifier-bad1.mm", "wrong unification"),
        ("big-unifier-bad2.mm", "extra proof steps"),
        ("big-unifier-bad3.mm", "wrong substitution"),
        ("demo0-bad1.mm", "reordered proof steps"),
        ("set-dist-bad1.mm", "removed $d constraint"),
    ])
    def test_cpu_rejects(self, filename: str, description: str) -> None:
        """CPU verifier must REJECT this deliberately broken file."""
        path = _skip_if_missing(filename)
        ok, detail, _ = _parse_and_verify_cpu(path)
        assert not ok, (
            f"{filename} ({description}) should FAIL but passed! "
            f"Detail: {detail}"
        )


# ── Large files — GPU-only (CPU already tested above) ───────────────

class TestWheelerLargeGPU:
    """GPU verification of the larger Wheeler test files."""

    @pytest.mark.parametrize("filename", [
        "iset.mm",
        "nf.mm",
    ])
    def test_gpu_large(self, filename: str) -> None:
        """GPU verifier must pass all steps in larger .mm files."""
        path = _skip_if_missing(filename)
        t0 = time.perf_counter()
        parsed = parse_mm_file(path)
        t_parse = time.perf_counter() - t0

        theorems = [k for k, v in parsed.assertions.items() if v.type == "theorem"]
        if not theorems:
            pytest.skip(f"{filename} has no theorems")

        print(f"\n  [{filename}] Parsed in {t_parse:.1f}s, {len(theorems)} theorems")
        n_steps, n_fail, detail = _verify_gpu(parsed)
        print(f"  [{filename}] {detail}")
        assert n_fail == 0, f"{filename} GPU: {detail}"
        gc.collect()
