"""Tests for tensormm.cpu_verifier."""

from __future__ import annotations

import os

from tensormm.cpu_verifier import CPUVerifier, apply_substitution
from tensormm.parser import parse_mm_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


class TestApplySubstitution:

    def test_constant_only(self) -> None:
        """Expression with no variables — substitution is a no-op."""
        result = apply_substitution(["|-", "(", "0", "+", "0", ")"], {})
        assert result == ["|-", "(", "0", "+", "0", ")"]

    def test_single_variable(self) -> None:
        result = apply_substitution(
            ["|-", "ph"],
            {"ph": ["(", "ps", "->", "ch", ")"]},
        )
        assert result == ["|-", "(", "ps", "->", "ch", ")"]

    def test_multiple_variables(self) -> None:
        result = apply_substitution(
            ["|-", "(", "ph", "->", "ps", ")"],
            {"ph": ["a", "=", "b"], "ps": ["c", "=", "d"]},
        )
        assert result == ["|-", "(", "a", "=", "b", "->", "c", "=", "d", ")"]

    def test_variable_not_in_substitution_kept(self) -> None:
        """Variables not in the subst dict are kept as-is (they're constants here)."""
        result = apply_substitution(["|-", "ph", "->", "ps"], {"ph": ["x"]})
        assert result == ["|-", "x", "->", "ps"]

    def test_empty_substitution_value(self) -> None:
        """A variable can substitute to an empty list."""
        result = apply_substitution(["wff", "ph"], {"ph": []})
        assert result == ["wff"]


class TestVerifyDemo0:

    def _get_verifier(self) -> CPUVerifier:
        parsed = parse_mm_file(os.path.join(DATA_DIR, "demo0.mm"))
        return CPUVerifier(parsed)

    def test_verify_th1(self) -> None:
        """Verify the th1 theorem from demo0.mm (the only theorem)."""
        v = self._get_verifier()
        result = v.verify_proof("th1")
        assert result.success, f"th1 failed: {result.error_message}"
        assert result.steps_verified > 0

    def test_verify_all_demo0(self) -> None:
        """Verify ALL theorems in demo0.mm — exhaustive."""
        v = self._get_verifier()
        results = v.verify_all()
        assert len(results) > 0, "No theorems found"
        for label, result in results.items():
            assert result.success, f"{label} failed: {result.error_message}"

    def test_axiom_not_verifiable(self) -> None:
        """Trying to verify an axiom should fail gracefully."""
        v = self._get_verifier()
        result = v.verify_proof("a1")
        assert not result.success
        assert "axiom" in result.error_message.lower()

    def test_nonexistent_label(self) -> None:
        v = self._get_verifier()
        result = v.verify_proof("doesnotexist")
        assert not result.success


class TestVerifyTestMini:

    def test_verify_all_test_mini(self) -> None:
        """Verify ALL theorems in test_mini.mm."""
        parsed = parse_mm_file(os.path.join(DATA_DIR, "test_mini.mm"))
        v = CPUVerifier(parsed)
        results = v.verify_all()
        assert len(results) > 0, "No theorems found in test_mini.mm"
        for label, result in results.items():
            assert result.success, f"{label} failed: {result.error_message}"
