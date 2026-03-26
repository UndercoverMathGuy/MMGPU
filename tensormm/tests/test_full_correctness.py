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


# Knife error headings as they appear in knife's stderr output.
_KNIFE_DIAGNOSTIC_STRINGS = {
    "ProofExcessEnd":   "Proof does not end with a single statement",
    "ProofUnderflow":   "Proof underflow",
    "StepEssenWrongType": "Wrong essential typecode",
    "StepEssenWrong":   "Wrong essential statement",
    "ProofDvViolation": "Distinct variable violation",
}


def knife_first_diagnostic(mm_path: str) -> str | None:
    """Run knife and return the first diagnostic key it emits, or None if it passes.

    Returns one of the _KNIFE_DIAGNOSTIC_STRINGS keys, or 'unknown' if the
    error message doesn't match any known pattern.
    """
    bin_path = _knife_bin()
    result = subprocess.run(
        [bin_path, "--verify", mm_path],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode == 0:
        return None
    output = result.stdout + result.stderr
    for key, text in _KNIFE_DIAGNOSTIC_STRINGS.items():
        if text in output:
            return key
    return "unknown"


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

# Each bad-file entry: (filename, knife_diagnostic_key, gpu_reason_prefix)
#
# knife_diagnostic_key: key into _KNIFE_DIAGNOSTIC_STRINGS (what knife reports)
# gpu_reason_prefix: the prefix of the GPU failure reason we expect
#   "graph:" — caught at graph construction (stack error, underflow, etc.)
#   "ehyp_mismatch" — CUDA kernel essential-hyp check failed
#   "result_mismatch" — final expression didn't match expected conclusion
#   "dv:" — $d disjoint-variable constraint violated
#
# Mapping from knife diagnostics to GPU reason:
#   ProofExcessEnd   → graph: (stack has >1 entry at end of proof)
#   ProofUnderflow   → graph: (stack underflow during graph construction)
#   StepEssenWrongType → ehyp_mismatch (type-code token doesn't match in ehyp check)
#   StepEssenWrong   → ehyp_mismatch (ehyp expression doesn't match after substitution)
#   ProofDvViolation → dv:
WHEELER_BAD = [
    ("anatomy-bad1.mm",     "ProofExcessEnd",      "graph:"),
    ("anatomy-bad2.mm",     "ProofExcessEnd",      "graph:"),
    ("anatomy-bad3.mm",     "ProofUnderflow",      "graph:"),
    ("big-unifier-bad1.mm", "StepEssenWrongType",  "ehyp_mismatch"),
    ("big-unifier-bad2.mm", "ProofExcessEnd",      "graph:"),
    ("big-unifier-bad3.mm", "StepEssenWrongType",  "ehyp_mismatch"),
    ("demo0-bad1.mm",       "StepEssenWrong",      "ehyp_mismatch"),
    ("set-dist-bad1.mm",    "ProofDvViolation",    "dv:"),
]


class TestWheelerKnifeAgreement:
    """GPU verifier must agree with metamath-knife on the Wheeler test suite,
    and must fail bad files for the same reason knife does."""

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
        n_fail = sum(1 for v in results.values() if v is not None)
        assert n_fail == 0, f"{filename}: {n_fail} GPU failures"
        gc.collect()

    @pytest.mark.parametrize("filename,knife_diag,_gpu_reason", WHEELER_BAD)
    def test_bad_file_knife_rejects(self, filename: str, knife_diag: str, _gpu_reason: str) -> None:
        """metamath-knife must reject deliberately broken files with the expected diagnostic."""
        path = _skip_if_missing(os.path.join(WHEELER_DIR, filename))
        diag = knife_first_diagnostic(path)
        assert diag is not None, (
            f"knife accepted {filename} — expected FAIL with {knife_diag!r}"
        )
        assert diag == knife_diag, (
            f"{filename}: knife emitted {diag!r}, expected {knife_diag!r}"
        )

    @pytest.mark.parametrize("filename,knife_diag,gpu_reason", WHEELER_BAD)
    def test_bad_file_gpu_rejects_with_reason(
        self, filename: str, knife_diag: str, gpu_reason: str
    ) -> None:
        """GPU verifier must reject broken files and report the same root cause as knife.

        GPU reason must start with the expected prefix:
          "graph:" for stack/structural errors caught at graph construction
          "ehyp_mismatch" for essential-hypothesis failures (knife StepEssenWrong*)
          "dv:" for $d violations
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
        failures = {lbl: reason for lbl, reason in results.items() if reason is not None}
        assert failures, (
            f"{filename}: GPU accepted all theorems but should have failed "
            f"(knife said {knife_diag!r})"
        )
        # Every failing theorem must report the expected reason category
        wrong = {
            lbl: r for lbl, r in failures.items()
            if not r.startswith(gpu_reason)
        }
        assert not wrong, (
            f"{filename}: expected all GPU failures to start with {gpu_reason!r} "
            f"(knife: {knife_diag!r}), but got: {wrong}"
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

        n_fail = sum(1 for v in results.values() if v is not None)
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
        gpu_all_pass = all(v is None for v in results.values())
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

        gpu_pass = sum(1 for v in results.values() if v is None)
        gpu_fail = sum(1 for v in results.values() if v is not None)

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
