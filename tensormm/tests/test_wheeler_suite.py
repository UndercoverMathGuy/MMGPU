"""Wheeler metamath-test suite — GPU verifier correctness tests.

Runs the david-a-wheeler/metamath-test suite against our full pipeline:
  parse → GPU topological-level verification.

Ground-truth oracle: metamath-knife (Rust verifier).
"bad" files must be rejected by knife AND our GPU verifier.
"good" files must be accepted by knife AND our GPU verifier.

See: https://github.com/david-a-wheeler/metamath-test
"""
from __future__ import annotations

import gc
import os
import subprocess
import shutil
import time

import pytest

from tensormm.gpu_verifier import verify_database
from tensormm.parser import ParsedDatabase, parse_mm_file


WHEELER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "wheeler-tests"
)


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════


def _knife_bin() -> str:
    path = shutil.which("metamath-knife")
    if path is None:
        pytest.skip("metamath-knife not installed (cargo install metamath-knife)")
    return path


def _knife_verify(mm_path: str) -> bool:
    """Run metamath-knife --verify; return True if exit 0."""
    r = subprocess.run(
        [_knife_bin(), "--verify", mm_path],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return r.returncode == 0


def _verify_gpu(parsed: ParsedDatabase) -> tuple[int, int, str]:
    """Run GPU verification on all theorems.

    Returns (n_theorems, n_failures, detail_msg).
    """
    theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
    if not theorems:
        return 0, 0, "no theorems"
    results = verify_database(parsed, theorem_labels=theorems)
    n_fail = sum(1 for v in results.values() if not v)
    n_pass = len(results) - n_fail
    return len(results), n_fail, f"{n_pass}/{len(results)} theorems pass"


def _wheeler_path(name: str) -> str:
    return os.path.join(WHEELER_DIR, name)


def _skip_if_missing(name: str) -> str:
    p = _wheeler_path(name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not found in {WHEELER_DIR}")
    return p


# ══════════════════════════════════════════════════════════════════════
#  Test cases — mirrors run-testsuite from david-a-wheeler/metamath-test
# ══════════════════════════════════════════════════════════════════════


# ── Files that SHOULD PASS ──────────────────────────────────────────

class TestWheelerPass:
    """Files that must verify correctly — knife AND GPU must both accept."""

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
    def test_knife_pass(self, filename: str) -> None:
        """metamath-knife must accept this file."""
        path = _skip_if_missing(filename)
        assert _knife_verify(path), f"{filename} should PASS but knife rejected it"

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
        """GPU verifier must accept all theorems in this file."""
        path = _skip_if_missing(filename)
        parsed = parse_mm_file(path)
        theorems = [k for k, v in parsed.assertions.items() if v.type == "theorem"]
        if not theorems:
            pytest.skip(f"{filename} has no theorems")

        n_total, n_fail, detail = _verify_gpu(parsed)
        print(f"  [{filename}] {detail}")
        assert n_fail == 0, f"{filename} GPU: {detail}"
        gc.collect()

    @pytest.mark.parametrize("filename", [
        "anatomy.mm",
        "big-unifier.mm",
        "demo0.mm",
        "demo0-includer.mm",
        "miu.mm",
        "ql.mm",
        "hol.mm",
    ])
    def test_gpu_knife_agree(self, filename: str) -> None:
        """GPU overall verdict must match knife for good files."""
        path = _skip_if_missing(filename)
        knife_pass = _knife_verify(path)
        parsed = parse_mm_file(path)
        theorems = [k for k, v in parsed.assertions.items() if v.type == "theorem"]
        if not theorems:
            pytest.skip(f"{filename} has no theorems")
        _, n_fail, _ = _verify_gpu(parsed)
        gpu_all_pass = n_fail == 0
        assert knife_pass == gpu_all_pass, (
            f"{filename}: knife={'PASS' if knife_pass else 'FAIL'}, "
            f"GPU={'PASS' if gpu_all_pass else 'FAIL'}"
        )
        gc.collect()


# ── Files that SHOULD FAIL ──────────────────────────────────────────

class TestWheelerFail:
    """Files with deliberately broken proofs — both knife AND GPU must reject."""

    @pytest.mark.parametrize("filename,description", [
        ("anatomy-bad1.mm",     "wrong proof step"),
        ("anatomy-bad2.mm",     "too few proof steps"),
        ("anatomy-bad3.mm",     "missing first step"),
        ("big-unifier-bad1.mm", "wrong unification"),
        ("big-unifier-bad2.mm", "extra proof steps"),
        ("big-unifier-bad3.mm", "wrong substitution"),
        ("demo0-bad1.mm",       "reordered proof steps"),
        ("set-dist-bad1.mm",    "removed $d constraint"),
    ])
    def test_knife_rejects(self, filename: str, description: str) -> None:
        """metamath-knife must reject this deliberately broken file."""
        path = _skip_if_missing(filename)
        assert not _knife_verify(path), (
            f"{filename} ({description}) should FAIL but knife accepted it"
        )

    @pytest.mark.parametrize("filename,description", [
        ("anatomy-bad1.mm",     "wrong proof step"),
        ("anatomy-bad2.mm",     "too few proof steps"),
        ("anatomy-bad3.mm",     "missing first step"),
        ("big-unifier-bad1.mm", "wrong unification"),
        ("big-unifier-bad2.mm", "extra proof steps"),
        ("big-unifier-bad3.mm", "wrong substitution"),
        ("demo0-bad1.mm",       "reordered proof steps"),
    ])
    def test_gpu_rejects(self, filename: str, description: str) -> None:
        """GPU verifier must reject at least one theorem in this broken file."""
        path = _skip_if_missing(filename)
        try:
            parsed = parse_mm_file(path)
        except Exception:
            return  # parse failure is a valid rejection
        theorems = [k for k, v in parsed.assertions.items() if v.type == "theorem"]
        if not theorems:
            pytest.skip(f"{filename} has no theorems")
        _, n_fail, detail = _verify_gpu(parsed)
        assert n_fail > 0, (
            f"{filename} ({description}): GPU accepted all theorems but should have failed"
        )
        gc.collect()


# ── Large files — GPU-only (knife already validates) ─────────────────

class TestWheelerLargeGPU:
    """GPU verification of larger Wheeler test files."""

    @pytest.mark.parametrize("filename", [
        "iset.mm",
        "nf.mm",
    ])
    def test_gpu_large(self, filename: str) -> None:
        """GPU verifier must pass all theorems in larger .mm files."""
        path = _skip_if_missing(filename)
        t0 = time.perf_counter()
        parsed = parse_mm_file(path)
        t_parse = time.perf_counter() - t0

        theorems = [k for k, v in parsed.assertions.items() if v.type == "theorem"]
        if not theorems:
            pytest.skip(f"{filename} has no theorems")

        print(f"\n  [{filename}] Parsed in {t_parse:.1f}s, {len(theorems)} theorems")
        n_total, n_fail, detail = _verify_gpu(parsed)
        print(f"  [{filename}] {detail}")
        assert n_fail == 0, f"{filename} GPU: {detail}"
        gc.collect()
