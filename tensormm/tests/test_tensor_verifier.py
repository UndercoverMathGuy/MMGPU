"""Tests for tensormm.tensor_verifier."""

from __future__ import annotations

import torch
import pytest

from tensormm.tensor_verifier import TensorVerifier


@pytest.fixture
def tv() -> TensorVerifier:
    """Create a TensorVerifier on CPU for deterministic testing."""
    return TensorVerifier(device=torch.device("cpu"))


@pytest.fixture
def is_var() -> torch.Tensor:
    """Simple is_variable mask: tokens 10-19 are variables, rest are constants.
    vocab_size = 30 for these tests."""
    mask = torch.zeros(30, dtype=torch.bool)
    mask[10:20] = True
    return mask


class TestGatherStep:

    def test_replacement_length_lookup(self, tv: TensorVerifier, is_var: torch.Tensor) -> None:
        """Verify that gather correctly looks up per-token replacement lengths."""
        # Pattern: [1, 10, 2] — token 10 is a variable
        # Substitution: var 10 -> [5, 6, 7] (length 3)
        pattern = [1, 10, 2]
        substitutions = [{10: [5, 6, 7]}]
        cp, sub_tables, sub_lengths, _, _ = tv.prepare_substitution_tensors(
            pattern, substitutions, is_var
        )
        pattern_expanded = cp.unsqueeze(0)
        token_lengths = torch.gather(sub_lengths, dim=1, index=pattern_expanded.long())
        assert token_lengths[0, 0].item() == 1  # constant 1
        assert token_lengths[0, 1].item() == 3  # variable 10 -> [5,6,7]
        assert token_lengths[0, 2].item() == 1  # constant 2


class TestPrefixSumStep:

    def test_offset_computation(self, tv: TensorVerifier, is_var: torch.Tensor) -> None:
        """Verify cumsum-based offset computation."""
        pattern = [1, 10, 2]
        substitutions = [{10: [5, 6, 7]}]
        cp, sub_tables, sub_lengths, _, _ = tv.prepare_substitution_tensors(
            pattern, substitutions, is_var
        )
        pattern_expanded = cp.unsqueeze(0)
        token_lengths = torch.gather(sub_lengths, dim=1, index=pattern_expanded.long())
        offsets = torch.cumsum(token_lengths, dim=1) - token_lengths
        # Offsets should be [0, 1, 4] — constant(1) at 0, var(3) at 1, constant(1) at 4
        assert offsets[0, 0].item() == 0
        assert offsets[0, 1].item() == 1
        assert offsets[0, 2].item() == 4


class TestScatterStep:

    def test_output_tokens_written_correctly(self, tv: TensorVerifier, is_var: torch.Tensor) -> None:
        """Full scatter: check that the output buffer contains the right tokens."""
        # Pattern: [1, 10, 2], subst: 10 -> [5, 6, 7]
        # Expected output: [1, 5, 6, 7, 2]
        pattern = [1, 10, 2]
        substitutions = [{10: [5, 6, 7]}]
        target = [1, 5, 6, 7, 2]
        result = tv.verify_batch(pattern, substitutions, [target], is_var)
        assert result[0].item() is True


class TestEqualityCheckStep:

    def test_correct_target_matches(self, tv: TensorVerifier, is_var: torch.Tensor) -> None:
        pattern = [1, 10, 2]
        substitutions = [{10: [5, 6, 7]}]
        target = [1, 5, 6, 7, 2]
        result = tv.verify_batch(pattern, substitutions, [target], is_var)
        assert result[0].item() is True

    def test_wrong_target_rejected(self, tv: TensorVerifier, is_var: torch.Tensor) -> None:
        pattern = [1, 10, 2]
        substitutions = [{10: [5, 6, 7]}]
        wrong_target = [1, 5, 6, 8, 2]  # 8 instead of 7
        result = tv.verify_batch(pattern, substitutions, [wrong_target], is_var)
        assert result[0].item() is False


class TestBatchOfOneMatchesCPU:

    def test_single_candidate(self, tv: TensorVerifier, is_var: torch.Tensor) -> None:
        pattern = [3, 10, 4, 11, 5]
        subst = {10: [20, 21], 11: [22]}
        target = [3, 20, 21, 4, 22, 5]
        result = tv.verify_batch(pattern, [subst], [target], is_var)
        assert result[0].item() is True


class TestBatchOf1000MatchesCPU:

    def test_1000_random_candidates(self, tv: TensorVerifier, is_var: torch.Tensor) -> None:
        """Generate 1000 random substitutions, compute CPU result, compare to GPU."""
        import random
        random.seed(42)

        pattern = [1, 10, 2, 11, 3]  # 10 and 11 are variables

        substitutions = []
        targets = []
        expected = []

        for i in range(1000):
            # Random replacement lengths 1-5
            rep10 = [random.randint(1, 9) for _ in range(random.randint(1, 5))]
            rep11 = [random.randint(1, 9) for _ in range(random.randint(1, 5))]
            subst = {10: rep10, 11: rep11}
            substitutions.append(subst)

            # CPU apply_substitution
            cpu_result = []
            for tok in pattern:
                if tok in subst:
                    cpu_result.extend(subst[tok])
                else:
                    cpu_result.append(tok)

            # Half correct, half deliberately wrong
            if i % 2 == 0:
                targets.append(cpu_result)
                expected.append(True)
            else:
                wrong = cpu_result.copy()
                if wrong:
                    wrong[-1] = wrong[-1] + 1  # corrupt last token
                targets.append(wrong)
                expected.append(False)

        result = tv.verify_batch(pattern, substitutions, targets, is_var)
        for i in range(1000):
            assert result[i].item() == expected[i], f"Mismatch at candidate {i}"


class TestAllPadInput:

    def test_empty_pattern(self, tv: TensorVerifier, is_var: torch.Tensor) -> None:
        """Empty pattern should match empty target."""
        result = tv.verify_batch([], [{}], [[]], is_var)
        assert result[0].item() is True


class TestEmptySubstitution:

    def test_identity_constants_only(self, tv: TensorVerifier, is_var: torch.Tensor) -> None:
        """Pattern with only constants and empty substitution = identity."""
        pattern = [1, 2, 3, 4, 5]
        result = tv.verify_batch(pattern, [{}], [[1, 2, 3, 4, 5]], is_var)
        assert result[0].item() is True


class TestLengthMismatchDetected:

    def test_too_short_target(self, tv: TensorVerifier, is_var: torch.Tensor) -> None:
        pattern = [1, 10, 2]
        substitutions = [{10: [5, 6, 7]}]
        short_target = [1, 5, 6]  # missing tokens
        result = tv.verify_batch(pattern, substitutions, [short_target], is_var)
        assert result[0].item() is False

    def test_too_long_target(self, tv: TensorVerifier, is_var: torch.Tensor) -> None:
        pattern = [1, 10, 2]
        substitutions = [{10: [5, 6, 7]}]
        long_target = [1, 5, 6, 7, 2, 99]  # extra token
        result = tv.verify_batch(pattern, substitutions, [long_target], is_var)
        assert result[0].item() is False


class TestMPSBackend:

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available"
    )
    def test_verify_on_mps(self) -> None:
        """Verify basic operation works on MPS backend."""
        tv = TensorVerifier(device=torch.device("mps"))
        is_var = torch.zeros(30, dtype=torch.bool)
        is_var[10:20] = True
        pattern = [1, 10, 2]
        substitutions = [{10: [5, 6]}]
        target = [1, 5, 6, 2]
        result = tv.verify_batch(pattern, substitutions, [target], is_var)
        assert result[0].item() is True
