"""Full correctness: GPU verifier must agree with metamath-knife on every file.

metamath-knife (Rust) is the ground-truth oracle.  We compare its whole-file
pass/fail verdict against our GPU pipeline.  "bad" files must be rejected by
both; "good" files must be accepted by both.

For set.mm we additionally verify all ~47k theorems GPU-only (knife already
validated the file globally) and assert zero failures.
"""

from __future__ import annotations

import gc
import os
import subprocess
import shutil
import time

import torch
import pytest

from tensormm.gpu_verifier import verify_database
from tensormm.parser import parse_mm_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
WHEELER_DIR = os.path.join(DATA_DIR, "wheeler-tests")

# ── GPU device helpers ────────────────────────────────────────────────

CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()
GPU_AVAILABLE = CUDA_AVAILABLE or MPS_AVAILABLE


def _get_gpu_device() -> torch.device:
    if CUDA_AVAILABLE:
        return torch.device("cuda")
    if MPS_AVAILABLE:
        return torch.device("mps")
    pytest.skip("No GPU available (need CUDA or MPS)")
    raise RuntimeError("unreachable")


def _gpu_backend_name() -> str:
    if CUDA_AVAILABLE:
        return "CUDA"
    if MPS_AVAILABLE:
        return "Metal/MPS"
    return "CPU"


# ── metamath-knife helpers ────────────────────────────────────────────

def _knife_bin() -> str:
    """Return path to metamath-knife binary, skipping if not installed."""
    path = shutil.which("metamath-knife")
    if path is None:
        pytest.skip("metamath-knife not installed (cargo install metamath-knife)")
    return path


def knife_verify(mm_path: str) -> bool:
    """Run metamath-knife --verify on a .mm file.

    Returns True if knife exits 0 (all proofs correct), False otherwise.
    Raises pytest.skip if knife is not installed.
    """
    bin_path = _knife_bin()
    result = subprocess.run(
        [bin_path, "--verify", mm_path],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode == 0


def _skip_if_missing(path: str) -> str:
    if not os.path.exists(path):
        pytest.skip(f"File not found: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
#  Wheeler test suite — good files (knife + GPU must both pass)
# ══════════════════════════════════════════════════════════════════════


WHEELER_GOOD = [
    "anatomy.mm",
    "big-unifier.mm",
    "demo0.mm",
    "miu.mm",
    "ql.mm",
    "hol.mm",
    "demo0-includer.mm",
]

WHEELER_BAD = [
    ("anatomy-bad1.mm",       "wrong proof step"),
    ("anatomy-bad2.mm",       "too few proof steps"),
    ("anatomy-bad3.mm",       "missing first step"),
    ("big-unifier-bad1.mm",   "wrong unification"),
    ("big-unifier-bad2.mm",   "extra proof steps"),
    ("big-unifier-bad3.mm",   "wrong substitution"),
    ("demo0-bad1.mm",         "reordered proof steps"),
    ("set-dist-bad1.mm",      "removed $d constraint"),
]


class TestWheelerKnifeAgreement:
    """GPU verifier must agree with metamath-knife on the Wheeler test suite."""

    @pytest.mark.parametrize("filename", WHEELER_GOOD)
    def test_good_file_knife_passes(self, filename: str) -> None:
        """metamath-knife must accept good Wheeler files."""
        path = _skip_if_missing(os.path.join(WHEELER_DIR, filename))
        assert knife_verify(path), f"knife rejected {filename} — expected PASS"

    @pytest.mark.parametrize("filename", WHEELER_GOOD)
    def test_good_file_gpu_passes(self, filename: str) -> None:
        """GPU verifier must accept all theorems in good Wheeler files."""
        path = _skip_if_missing(os.path.join(WHEELER_DIR, filename))
        parsed = parse_mm_file(path)
        theorems = [k for k, v in parsed.assertions.items() if v.type == "theorem"]
        if not theorems:
            pytest.skip(f"{filename} has no theorems")
        results = verify_database(parsed, theorem_labels=theorems)
        n_fail = sum(1 for v in results.values() if not v)
        assert n_fail == 0, f"{filename}: {n_fail} GPU failures"
        gc.collect()

    @pytest.mark.parametrize("filename,description", WHEELER_BAD)
    def test_bad_file_knife_rejects(self, filename: str, description: str) -> None:
        """metamath-knife must reject deliberately broken files."""
        path = _skip_if_missing(os.path.join(WHEELER_DIR, filename))
        assert not knife_verify(path), (
            f"knife accepted {filename} ({description}) — expected FAIL"
        )

    @pytest.mark.parametrize("filename,description", WHEELER_BAD)
    def test_bad_file_gpu_rejects(self, filename: str, description: str) -> None:
        """GPU verifier must reject deliberately broken files.

        We expect at least one theorem to fail.  (Some bad files fail at
        parse time, which is also a valid rejection.)
        """
        path = _skip_if_missing(os.path.join(WHEELER_DIR, filename))
        try:
            parsed = parse_mm_file(path)
        except Exception:
            return  # parse rejection is a valid rejection
        theorems = [k for k, v in parsed.assertions.items() if v.type == "theorem"]
        if not theorems:
            pytest.skip(f"{filename} has no theorems — cannot test GPU rejection")
        results = verify_database(parsed, theorem_labels=theorems)
        n_fail = sum(1 for v in results.values() if not v)
        assert n_fail > 0, (
            f"{filename} ({description}): GPU accepted all theorems but should have failed"
        )
        gc.collect()


# ══════════════════════════════════════════════════════════════════════
#  ql.mm — exhaustive GPU verification (knife already validated it)
# ══════════════════════════════════════════════════════════════════════


class TestQLmmExhaustive:

    @pytest.fixture(scope="class")
    def ql_parsed(self):
        path = _skip_if_missing(os.path.join(DATA_DIR, "ql.mm"))
        return parse_mm_file(path)

    def test_knife_passes_ql(self, ql_parsed) -> None:
        """metamath-knife must pass ql.mm."""
        path = os.path.join(DATA_DIR, "ql.mm")
        assert knife_verify(path), "knife rejected ql.mm"

    def test_gpu_passes_all_ql(self, ql_parsed) -> None:
        """GPU verifier must pass every theorem in ql.mm."""
        device = _get_gpu_device()
        backend = _gpu_backend_name()
        parsed = ql_parsed

        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
        print(f"\n[ql.mm {backend}] Verifying {len(theorems)} theorems...")

        t0 = time.perf_counter()
        results = verify_database(parsed, theorems, device=device, verbose=True)
        t_gpu = time.perf_counter() - t0

        n_fail = sum(1 for v in results.values() if not v)
        n_pass = len(results) - n_fail
        print(f"[ql.mm {backend}] {n_pass}/{len(results)} pass  ({t_gpu:.2f}s)")

        assert n_fail == 0, f"{n_fail} GPU failures on ql.mm"

    def test_gpu_knife_agreement_ql(self, ql_parsed) -> None:
        """GPU overall verdict must match knife on ql.mm.

        knife says pass → GPU must have zero failures.
        """
        path = os.path.join(DATA_DIR, "ql.mm")
        knife_pass = knife_verify(path)
        parsed = ql_parsed
        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
        results = verify_database(parsed, theorems)
        gpu_all_pass = all(results.values())
        assert knife_pass == gpu_all_pass, (
            f"knife={'PASS' if knife_pass else 'FAIL'} but "
            f"GPU={'PASS' if gpu_all_pass else 'FAIL'} on ql.mm"
        )


# ══════════════════════════════════════════════════════════════════════
#  set.mm — full GPU verification
# ══════════════════════════════════════════════════════════════════════


class TestSetMMFull:

    @pytest.fixture(scope="class")
    def setmm_parsed(self):
        path = os.path.join(DATA_DIR, "set.mm")
        if not os.path.exists(path):
            pytest.skip("set.mm not found in data/")
        print("\n[set.mm] Parsing...")
        t0 = time.perf_counter()
        parsed = parse_mm_file(path)
        elapsed = time.perf_counter() - t0
        print(
            f"[set.mm] Parsed in {elapsed:.1f}s: "
            f"{len(parsed.assertions)} assertions, "
            f"{len(parsed.constants)} constants, "
            f"{len(parsed.variables)} variables"
        )
        gc.collect()
        return parsed

    def test_knife_passes_setmm(self, setmm_parsed) -> None:
        """metamath-knife must pass set.mm (the canonical Metamath database)."""
        path = os.path.join(DATA_DIR, "set.mm")
        assert knife_verify(path), "knife rejected set.mm — database is malformed"

    def test_gpu_all_set_mm(self, setmm_parsed) -> None:
        """GPU verification of ALL set.mm theorems must agree with knife.

        knife validates the whole file; GPU must produce zero failures.
        """
        device = _get_gpu_device()
        backend = _gpu_backend_name()
        parsed = setmm_parsed

        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
        n_axioms = sum(1 for a in parsed.assertions.values() if a.type == "axiom")

        print(
            f"\n[set.mm FULL {backend}] Verifying {len(theorems):,} theorems "
            f"({n_axioms:,} axioms, {len(parsed.assertions):,} total)"
        )

        t0 = time.perf_counter()
        results = verify_database(parsed, theorems, device=device, verbose=True)
        t_total = time.perf_counter() - t0

        gpu_pass = sum(1 for v in results.values() if v)
        gpu_fail = sum(1 for v in results.values() if not v)

        print(f"\n{'═' * 60}")
        print(f"  set.mm FULL — {backend} Verification Report")
        print(f"{'═' * 60}")
        print(f"  Axioms:    {n_axioms:,}    Theorems: {len(theorems):,}")
        print(f"  Passed:    {gpu_pass:,}    Failed:   {gpu_fail:,}")
        print(f"  Wall clock: {t_total:.2f}s")
        print(f"{'═' * 60}")

        assert gpu_fail == 0, (
            f"{gpu_fail} GPU failures on set.mm — kernel is UNSOUND"
        )
