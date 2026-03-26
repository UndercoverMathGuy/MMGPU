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
        cwd=os.path.dirname(os.path.abspath(mm_path)),
    )
    return r.returncode == 0


# Maps knife diagnostic heading text → short key
_KNIFE_DIAGNOSTICS = {
    "Proof does not end with a single statement": "ProofExcessEnd",
    "Proof underflow":                            "ProofUnderflow",
    "Wrong essential typecode":                   "StepEssenWrongType",
    "Wrong essential statement":                  "StepEssenWrong",
    "Distinct variable violation":                "ProofDvViolation",
}


def _knife_diagnostic(mm_path: str) -> str | None:
    """Run knife and return the first diagnostic key, or None if it passes."""
    r = subprocess.run(
        [_knife_bin(), "--verify", mm_path],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=os.path.dirname(os.path.abspath(mm_path)),
    )
    if r.returncode == 0:
        return None
    output = r.stdout + r.stderr
    for text, key in _KNIFE_DIAGNOSTICS.items():
        if text in output:
            return key
    return "unknown"


def _verify_gpu(parsed: ParsedDatabase) -> tuple[int, int, str]:
    """Run GPU verification on all theorems.

    Returns (n_theorems, n_failures, detail_msg).
    """
    theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
    if not theorems:
        return 0, 0, "no theorems"
    results = verify_database(parsed, theorem_labels=theorems)
    n_fail = sum(1 for v in results.values() if v is not None)
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

# (filename, knife_diagnostic_key, gpu_reason_prefix)
#
# knife_diagnostic_key: key emitted by _knife_diagnostic()
# gpu_reason_prefix: expected prefix of GPU failure reason from verify_database()
#   "graph:" — caught at graph construction (stack underflow / excess entries)
#   "ehyp_mismatch" — CUDA kernel essential-hyp check failed
#   "dv:"    — $d disjoint-variable constraint violated
_WHEELER_BAD = [
    ("anatomy-bad1.mm",     "ProofExcessEnd",     "graph:"),
    ("anatomy-bad2.mm",     "ProofExcessEnd",     "graph:"),
    ("anatomy-bad3.mm",     "ProofUnderflow",     "graph:"),
    ("big-unifier-bad1.mm", "StepEssenWrongType", "ehyp_mismatch"),
    ("big-unifier-bad2.mm", "ProofExcessEnd",     "graph:"),
    ("big-unifier-bad3.mm", "StepEssenWrongType", "ehyp_mismatch"),
    ("demo0-bad1.mm",       "StepEssenWrong",     "ehyp_mismatch"),
    ("set-dist-bad1.mm",    "ProofDvViolation",   "dv:"),
]


class TestWheelerFail:
    """Files with deliberately broken proofs — knife AND GPU must both reject,
    and must agree on the root cause."""

    @pytest.mark.parametrize("filename,knife_diag,_gpu_reason", _WHEELER_BAD)
    def test_knife_rejects(self, filename: str, knife_diag: str, _gpu_reason: str) -> None:
        """metamath-knife must reject the file with the expected diagnostic."""
        path = _skip_if_missing(filename)
        diag = _knife_diagnostic(path)
        assert diag is not None, (
            f"{filename} should FAIL but knife accepted it"
        )
        assert diag == knife_diag, (
            f"{filename}: knife emitted {diag!r}, expected {knife_diag!r}"
        )

    @pytest.mark.parametrize("filename,knife_diag,gpu_reason", _WHEELER_BAD)
    def test_gpu_rejects_with_reason(
        self, filename: str, knife_diag: str, gpu_reason: str
    ) -> None:
        """GPU verifier must reject broken files and report the same root cause as knife.

        GPU failure reason must start with the expected prefix matching knife's diagnostic:
          ProofExcessEnd / ProofUnderflow → "graph:" (caught at graph construction)
          StepEssenWrongType / StepEssenWrong → "ehyp_mismatch"
          ProofDvViolation → "dv:"
        """
        path = _skip_if_missing(filename)
        try:
            parsed = parse_mm_file(path)
        except Exception:
            return  # parse failure is a valid rejection
        theorems = [k for k, v in parsed.assertions.items() if v.type == "theorem"]
        if not theorems:
            pytest.skip(f"{filename} has no theorems")
        results = verify_database(parsed, theorem_labels=theorems)
        failures = {lbl: r for lbl, r in results.items() if r is not None}
        assert failures, (
            f"{filename}: GPU accepted all theorems but should have failed "
            f"(knife: {knife_diag!r})"
        )
        wrong = {lbl: r for lbl, r in failures.items() if not r.startswith(gpu_reason)}
        assert not wrong, (
            f"{filename}: expected all GPU failures to start with {gpu_reason!r} "
            f"(knife: {knife_diag!r}), got: {wrong}"
        )
        gc.collect()

    @pytest.mark.parametrize("filename,knife_diag,gpu_reason", _WHEELER_BAD)
    def test_knife_gpu_agree_on_failure(
        self, filename: str, knife_diag: str, gpu_reason: str
    ) -> None:
        """knife and GPU must agree: both reject, same root cause category."""
        path = _skip_if_missing(filename)
        knife_diag_actual = _knife_diagnostic(path)
        assert knife_diag_actual is not None, (
            f"{filename}: knife accepted the file — expected rejection"
        )
        assert knife_diag_actual == knife_diag, (
            f"{filename}: knife diagnostic {knife_diag_actual!r} != expected {knife_diag!r}"
        )
        try:
            parsed = parse_mm_file(path)
        except Exception:
            return
        theorems = [k for k, v in parsed.assertions.items() if v.type == "theorem"]
        if not theorems:
            pytest.skip(f"{filename} has no theorems")
        results = verify_database(parsed, theorem_labels=theorems)
        failures = {lbl: r for lbl, r in results.items() if r is not None}
        assert failures, (
            f"{filename}: knife={knife_diag_actual!r} but GPU passed everything"
        )
        wrong = {lbl: r for lbl, r in failures.items() if not r.startswith(gpu_reason)}
        assert not wrong, (
            f"{filename}: knife={knife_diag_actual!r} → expected GPU reason prefix "
            f"{gpu_reason!r}, got: {wrong}"
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
