"""Correctness tests: GPU pipeline must agree with metamath-knife on all test files.

These are the primary soundness checks — if knife passes a file and GPU fails
(or vice versa), the kernel is unsound.
"""

from __future__ import annotations

import os
import subprocess
import shutil

import pytest

from tensormm.gpu_verifier import verify_database
from tensormm.parser import parse_mm_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _knife_verify(mm_path: str) -> bool:
    """Run metamath-knife --verify; return True if exit 0."""
    knife = shutil.which("metamath-knife")
    if knife is None:
        pytest.skip("metamath-knife not installed")
    r = subprocess.run(
        [knife, "--verify", mm_path],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return r.returncode == 0


def _gpu_all_pass(mm_path: str) -> bool:
    """Run GPU pipeline; return True if all theorems pass."""
    parsed = parse_mm_file(mm_path)
    theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
    if not theorems:
        return True
    results = verify_database(parsed, theorem_labels=theorems, verbose=False)
    return all(results.values())


# ══════════════════════════════════════════════════════════════════════
#  GPU must agree with knife on all known-good files
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("filename", [
    "demo0.mm",
    "ql.mm",
    "anatomy.mm",
])
class TestGPUKnifeAgreement:

    def test_knife_passes(self, filename: str) -> None:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            pytest.skip(f"{filename} not found")
        assert _knife_verify(path), f"knife rejected {filename}"

    def test_gpu_passes(self, filename: str) -> None:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            pytest.skip(f"{filename} not found")
        assert _gpu_all_pass(path), f"GPU failed on {filename}"

    def test_gpu_knife_agree(self, filename: str) -> None:
        """GPU overall verdict must match knife — the primary soundness check."""
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            pytest.skip(f"{filename} not found")
        knife_pass = _knife_verify(path)
        gpu_pass = _gpu_all_pass(path)
        assert knife_pass == gpu_pass, (
            f"{filename}: knife={'PASS' if knife_pass else 'FAIL'}, "
            f"GPU={'PASS' if gpu_pass else 'FAIL'}"
        )
