"""Negative tests: deliberately corrupted proofs MUST be rejected by the GPU kernel.

Proves the Metal kernel is genuinely verifying, not just rubber-stamping True.
Uses real assertion steps extracted from ql.mm, then corrupts them in various
ways to confirm the kernel catches every type of error.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import torch
import pytest
import numpy as np

from tensormm.cpu_verifier import CPUVerifier, apply_substitution
from tensormm.database import MetamathDatabase
from tensormm.metal_verifier import MetalVerifier, METAL_AVAILABLE
from tensormm.parser import ParsedDatabase, parse_mm_file
from tensormm.tensor_verifier import TensorVerifier
from tensormm.tokenizer import Tokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


# ── Reuse the step extraction machinery from test_full_correctness ───

@dataclass
class _AssertionStep:
    theorem_label: str
    step_index: int
    step_label: str
    pattern: list[str]
    substitution: dict[str, list[str]]
    expected_result: list[str]


def _build_label_info(parsed: ParsedDatabase) -> dict:
    label_info: dict = {}
    for lbl, fh in parsed.floating_hyps.items():
        label_info[lbl] = ("$f", [fh.type_code, fh.variable])
    for lbl, eh in parsed.essential_hyps.items():
        label_info[lbl] = ("$e", eh.expression)
    for lbl, a in parsed.assertions.items():
        st = "$a" if a.type == "axiom" else "$p"
        label_info[lbl] = (st, a)
    return label_info


def _replay_proof_extract_steps(
    parsed: ParsedDatabase,
    theorem_label: str,
    label_info: dict,
) -> list[_AssertionStep] | str:
    assertion = parsed.assertions[theorem_label]
    stack: list[list[str]] = []
    steps: list[_AssertionStep] = []
    step_counter = 0

    def _do_step(step_label: str) -> str | None:
        nonlocal step_counter
        if step_label not in label_info:
            return f"Unknown label: {step_label}"
        st, data = label_info[step_label]
        if st in ("$f", "$e"):
            stack.append(list(data))
            return None
        a = data
        f_labels = a.floating_hyps
        e_labels = a.essential_hyps
        npop = len(f_labels) + len(e_labels)
        sp = len(stack) - npop
        if sp < 0:
            return f"Stack underflow at {step_label}"
        subst: dict[str, list[str]] = {}
        for flbl in f_labels:
            fh = parsed.floating_hyps[flbl]
            entry = stack[sp]
            subst[fh.variable] = entry[1:]
            sp += 1
        for elbl in e_labels:
            sp += 1
        result = apply_substitution(a.expression, subst)
        steps.append(_AssertionStep(
            theorem_label=theorem_label,
            step_index=step_counter,
            step_label=step_label,
            pattern=a.expression,
            substitution=subst,
            expected_result=result,
        ))
        step_counter += 1
        del stack[len(stack) - npop:]
        stack.append(result)
        return None

    try:
        if assertion.compressed_proof is not None:
            cp = assertion.compressed_proof
            plabels = cp.labels
            label_end = len(plabels)
            saved: list[list[str]] = []
            for pi in cp.proof_ints:
                if pi == -1:
                    if not stack:
                        return "Z save on empty stack"
                    saved.append(list(stack[-1]))
                elif pi < label_end:
                    err = _do_step(plabels[pi])
                    if err:
                        return err
                else:
                    si = pi - label_end
                    if si >= len(saved):
                        return f"Saved index {si} out of range"
                    stack.append(list(saved[si]))
        elif assertion.proof is not None:
            for sl in assertion.proof:
                err = _do_step(sl)
                if err:
                    return err
        else:
            return "No proof"
    except Exception as e:
        return str(e)

    if len(stack) != 1 or stack[0] != assertion.expression:
        return f"Final stack mismatch: got {stack}"
    return steps


# ── Helper: build Metal-ready tensors from steps ─────────────────────

def _steps_to_metal_tensors(
    steps: list[_AssertionStep],
    tokenizer: Tokenizer,
    override_results: list[list[str]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack steps into tensors suitable for MetalVerifier.verify_flat().

    If override_results is provided, uses those instead of step.expected_result
    for the target tensors (for corruption tests).
    """
    N = len(steps)
    all_token_ids: set[int] = {0}

    # Encode patterns, substitutions, targets
    enc_patterns: list[list[int]] = []
    enc_substs: list[dict[int, list[int]]] = []
    enc_targets: list[list[int]] = []

    for i, step in enumerate(steps):
        pat_ids = tokenizer.encode_expression(step.pattern)
        enc_patterns.append(pat_ids)
        all_token_ids.update(pat_ids)

        subst_enc: dict[int, list[int]] = {}
        for var, rep in step.substitution.items():
            vid = tokenizer.encode_symbol(var)
            rids = tokenizer.encode_expression(rep)
            subst_enc[vid] = rids
            all_token_ids.add(vid)
            all_token_ids.update(rids)
        enc_substs.append(subst_enc)

        if override_results is not None:
            tgt_ids = tokenizer.encode_expression(override_results[i])
        else:
            tgt_ids = tokenizer.encode_expression(step.expected_result)
        enc_targets.append(tgt_ids)
        all_token_ids.update(tgt_ids)

    # Compact vocab
    sorted_ids = sorted(all_token_ids)
    max_full_id = max(sorted_ids) + 1
    f2c = np.zeros(max_full_id, dtype=np.int32)
    f2c[np.array(sorted_ids, dtype=np.int64)] = np.arange(len(sorted_ids), dtype=np.int32)
    V = len(sorted_ids)

    P_max = max(len(p) for p in enc_patterns)
    T_max = max((len(t) for t in enc_targets), default=1) or 1
    S_max = 1
    for s in enc_substs:
        for rep in s.values():
            S_max = max(S_max, len(rep))

    patterns = np.zeros((N, P_max), dtype=np.int32)
    pat_lengths = np.zeros(N, dtype=np.int32)
    targets = np.zeros((N, T_max), dtype=np.int32)
    tgt_lengths = np.zeros(N, dtype=np.int32)
    sub_tables = np.zeros((N, V, S_max), dtype=np.int32)
    sub_lengths = np.ones((N, V), dtype=np.int32)

    # Identity: sub_tables[n, v, 0] = v for all v
    ident = np.arange(V, dtype=np.int32)
    sub_tables[:, :, 0] = ident[np.newaxis, :]

    for i in range(N):
        p = enc_patterns[i]
        pat_lengths[i] = len(p)
        for j, tok in enumerate(p):
            patterns[i, j] = f2c[tok]

        t = enc_targets[i]
        tgt_lengths[i] = len(t)
        for j, tok in enumerate(t):
            targets[i, j] = f2c[tok]

        for vid, rep in enc_substs[i].items():
            cv = f2c[vid]
            sub_lengths[i, cv] = len(rep)
            for s, tok in enumerate(rep):
                sub_tables[i, cv, s] = f2c[tok]

    return (
        torch.from_numpy(patterns),
        torch.from_numpy(pat_lengths),
        torch.from_numpy(sub_tables),
        torch.from_numpy(sub_lengths),
        torch.from_numpy(targets),
        torch.from_numpy(tgt_lengths),
    )


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ql_steps():
    """Extract first 200 real assertion steps from ql.mm."""
    path = os.path.join(DATA_DIR, "ql.mm")
    if not os.path.exists(path):
        pytest.skip("ql.mm not found in data/")
    parsed = parse_mm_file(path)
    tok = Tokenizer()
    label_info = _build_label_info(parsed)

    theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
    all_steps: list[_AssertionStep] = []
    for lbl in theorems:
        if len(all_steps) >= 200:
            break
        extracted = _replay_proof_extract_steps(parsed, lbl, label_info)
        if isinstance(extracted, list):
            all_steps.extend(extracted)

    return all_steps[:200], tok, parsed


@pytest.fixture(scope="module")
def metal_verifier():
    if not METAL_AVAILABLE:
        pytest.skip("Metal not available")
    return MetalVerifier()


# ══════════════════════════════════════════════════════════════════════
#  2A — Corrupted Substitution Tests
# ══════════════════════════════════════════════════════════════════════

class TestCorruptedSubstitutions:

    def test_flipped_substitution_token(self, ql_steps, metal_verifier) -> None:
        """Flip one token in each substitution replacement — GPU must reject all."""
        steps, tok, parsed = ql_steps
        random.seed(77)

        # Filter to steps that actually have substitutions with tokens to flip
        candidates = [s for s in steps if any(len(r) > 0 for r in s.substitution.values())]
        test_steps = candidates[:100]
        if not test_steps:
            pytest.skip("No steps with substitutions")

        corrupted_results: list[list[str]] = []
        for step in test_steps:
            # Compute the correct result, then corrupt it by flipping a subst token
            correct = apply_substitution(step.pattern, step.substitution)
            # Corrupt the substitution: pick a random var, flip a token
            corrupt_subst = {k: list(v) for k, v in step.substitution.items()}
            vars_with_tokens = [k for k, v in corrupt_subst.items() if len(v) > 0]
            if vars_with_tokens:
                var = random.choice(vars_with_tokens)
                idx = random.randint(0, len(corrupt_subst[var]) - 1)
                # Replace with a different constant
                old = corrupt_subst[var][idx]
                choices = [c for c in parsed.constants if c != old]
                if choices:
                    corrupt_subst[var][idx] = random.choice(choices)
            corrupted_result = apply_substitution(step.pattern, corrupt_subst)
            corrupted_results.append(corrupted_result)

        pats, plens, st, sl, tgts, tlens = _steps_to_metal_tensors(
            test_steps, tok, override_results=corrupted_results,
        )
        result = metal_verifier.verify_flat(pats, plens, st, sl, tgts, tlens)

        # The correct result uses the original subst; targets use corrupted subst.
        # So we verify against the ORIGINAL expected_result
        n_pass = 0
        for i, step in enumerate(test_steps):
            if corrupted_results[i] != step.expected_result:
                assert result[i].item() is False, (
                    f"Step {i} ({step.step_label}): corrupted substitution should be rejected"
                )
            else:
                n_pass += 1  # corruption happened to produce same result (rare)

        assert n_pass < len(test_steps), "All corruptions produced identical results — bad test"

    def test_truncated_substitution(self, ql_steps, metal_verifier) -> None:
        """Truncate one variable's replacement — GPU must reject."""
        steps, tok, parsed = ql_steps
        random.seed(88)

        candidates = [s for s in steps if any(len(r) > 1 for r in s.substitution.values())]
        test_steps = candidates[:50]
        if not test_steps:
            pytest.skip("No steps with multi-token substitutions")

        corrupted_results: list[list[str]] = []
        for step in test_steps:
            corrupt_subst = {k: list(v) for k, v in step.substitution.items()}
            long_vars = [k for k, v in corrupt_subst.items() if len(v) > 1]
            if long_vars:
                var = random.choice(long_vars)
                corrupt_subst[var] = corrupt_subst[var][:-1]  # drop last token
            corrupted_results.append(apply_substitution(step.pattern, corrupt_subst))

        pats, plens, st, sl, tgts, tlens = _steps_to_metal_tensors(
            test_steps, tok, override_results=corrupted_results,
        )
        result = metal_verifier.verify_flat(pats, plens, st, sl, tgts, tlens)

        n_rejected = sum(1 for i in range(len(test_steps))
                         if corrupted_results[i] != test_steps[i].expected_result
                         and not result[i].item())
        n_corrupted = sum(1 for i in range(len(test_steps))
                          if corrupted_results[i] != test_steps[i].expected_result)

        assert n_corrupted > 0, "No truncations actually changed the result"
        assert n_rejected == n_corrupted, (
            f"Only {n_rejected}/{n_corrupted} truncated substitutions were rejected"
        )


# ══════════════════════════════════════════════════════════════════════
#  2B — Corrupted Target Tests
# ══════════════════════════════════════════════════════════════════════

class TestCorruptedTargets:

    def test_flip_one_token(self, ql_steps, metal_verifier) -> None:
        """Flip one token in the expected result — GPU must reject."""
        steps, tok, _ = ql_steps
        random.seed(99)

        test_steps = [s for s in steps if len(s.expected_result) > 1][:100]
        corrupted: list[list[str]] = []
        for step in test_steps:
            r = list(step.expected_result)
            idx = random.randint(0, len(r) - 1)
            r[idx] = r[idx] + "_CORRUPT"  # guaranteed different token
            corrupted.append(r)

        pats, plens, st, sl, tgts, tlens = _steps_to_metal_tensors(
            test_steps, tok, override_results=corrupted,
        )
        result = metal_verifier.verify_flat(pats, plens, st, sl, tgts, tlens)
        assert not result.any(), f"{result.sum().item()} corrupted targets passed — kernel is UNSOUND"

    def test_delete_last_token(self, ql_steps, metal_verifier) -> None:
        """Delete the last token from each target — GPU must reject."""
        steps, tok, _ = ql_steps
        test_steps = [s for s in steps if len(s.expected_result) > 1][:100]

        corrupted = [list(s.expected_result)[:-1] for s in test_steps]
        pats, plens, st, sl, tgts, tlens = _steps_to_metal_tensors(
            test_steps, tok, override_results=corrupted,
        )
        result = metal_verifier.verify_flat(pats, plens, st, sl, tgts, tlens)
        assert not result.any(), f"{result.sum().item()} short targets passed — length check broken"

    def test_append_extra_token(self, ql_steps, metal_verifier) -> None:
        """Append an extra token to each target — GPU must reject."""
        steps, tok, parsed = ql_steps
        test_steps = steps[:100]

        corrupted = [list(s.expected_result) + ["("] for s in test_steps]
        pats, plens, st, sl, tgts, tlens = _steps_to_metal_tensors(
            test_steps, tok, override_results=corrupted,
        )
        result = metal_verifier.verify_flat(pats, plens, st, sl, tgts, tlens)
        assert not result.any(), f"{result.sum().item()} long targets passed — length check broken"

    def test_swap_adjacent_tokens(self, ql_steps, metal_verifier) -> None:
        """Swap two adjacent tokens in each target — GPU must reject."""
        steps, tok, _ = ql_steps
        random.seed(42)

        test_steps = [s for s in steps if len(s.expected_result) > 2][:100]
        corrupted: list[list[str]] = []
        for step in test_steps:
            r = list(step.expected_result)
            idx = random.randint(0, len(r) - 2)
            r[idx], r[idx + 1] = r[idx + 1], r[idx]
            corrupted.append(r)

        pats, plens, st, sl, tgts, tlens = _steps_to_metal_tensors(
            test_steps, tok, override_results=corrupted,
        )
        result = metal_verifier.verify_flat(pats, plens, st, sl, tgts, tlens)

        # Some swaps of identical adjacent tokens won't change anything
        n_actually_different = sum(
            1 for i in range(len(test_steps))
            if corrupted[i] != list(test_steps[i].expected_result)
        )
        n_rejected = sum(
            1 for i in range(len(test_steps))
            if corrupted[i] != list(test_steps[i].expected_result)
            and not result[i].item()
        )
        assert n_actually_different > 0, "No swaps actually changed the result"
        assert n_rejected == n_actually_different, (
            f"Only {n_rejected}/{n_actually_different} swapped targets were rejected"
        )


# ══════════════════════════════════════════════════════════════════════
#  2C — Mixed Batch Soundness
# ══════════════════════════════════════════════════════════════════════

class TestMixedBatchSoundness:

    def test_500_mixed_correct_and_corrupted(self, ql_steps, metal_verifier) -> None:
        """Build a batch of 500 steps: 250 correct + 250 corrupted.

        Verify the kernel returns True for correct, False for corrupted.
        Tests batch-independence — no cross-contamination between steps.
        """
        steps, tok, parsed = ql_steps
        random.seed(123)

        # Use first 250 steps twice: once correct, once corrupted
        base_steps = steps[:250]
        N = len(base_steps) * 2
        all_steps_doubled = base_steps + base_steps
        override_results: list[list[str]] = []

        # First 250: correct targets
        for step in base_steps:
            override_results.append(list(step.expected_result))

        # Second 250: corrupted targets
        for step in base_steps:
            r = list(step.expected_result)
            if len(r) > 0:
                idx = random.randint(0, len(r) - 1)
                r[idx] = r[idx] + "_BAD"
            else:
                r = ["_BAD"]
            override_results.append(r)

        pats, plens, st, sl, tgts, tlens = _steps_to_metal_tensors(
            all_steps_doubled, tok, override_results=override_results,
        )
        result = metal_verifier.verify_flat(pats, plens, st, sl, tgts, tlens)

        # First 250 should be True
        correct_half = result[:len(base_steps)]
        assert correct_half.all(), (
            f"{(~correct_half).sum().item()} correct steps were rejected — kernel has false negatives"
        )

        # Second 250 should be False
        corrupted_half = result[len(base_steps):]
        assert not corrupted_half.any(), (
            f"{corrupted_half.sum().item()} corrupted steps passed — kernel has false positives"
        )


# ══════════════════════════════════════════════════════════════════════
#  2D — Edge Cases
# ══════════════════════════════════════════════════════════════════════

class TestNegativeEdgeCases:

    @pytest.fixture
    def mv(self):
        if not METAL_AVAILABLE:
            pytest.skip("Metal not available")
        return MetalVerifier()

    def test_empty_pattern_nonempty_target(self, mv) -> None:
        """Empty pattern + non-empty target → must be False."""
        # P_max=1 (need at least 1 col), but pat_len=0
        patterns = torch.zeros(1, 1, dtype=torch.int32)
        pat_lengths = torch.tensor([0], dtype=torch.int32)
        sub_tables = torch.zeros(1, 2, 1, dtype=torch.int32)
        sub_tables[0, :, 0] = torch.arange(2, dtype=torch.int32)
        sub_lengths = torch.ones(1, 2, dtype=torch.int32)
        targets = torch.tensor([[1]], dtype=torch.int32)
        tgt_lengths = torch.tensor([1], dtype=torch.int32)

        result = mv.verify_flat(patterns, pat_lengths, sub_tables, sub_lengths, targets, tgt_lengths)
        assert result[0].item() is False

    def test_nonempty_pattern_empty_target(self, mv) -> None:
        """Non-empty pattern with variable substitution → non-empty output → empty target → False."""
        # Pattern [0, 1], var 1 -> [2] (len 1). Output = [0, 2], target = empty
        V = 3
        S_max = 1
        patterns = torch.tensor([[0, 1]], dtype=torch.int32)
        pat_lengths = torch.tensor([2], dtype=torch.int32)
        sub_tables = torch.zeros(1, V, S_max, dtype=torch.int32)
        for v in range(V):
            sub_tables[0, v, 0] = v
        sub_lengths = torch.ones(1, V, dtype=torch.int32)
        targets = torch.zeros(1, 1, dtype=torch.int32)
        tgt_lengths = torch.tensor([0], dtype=torch.int32)

        result = mv.verify_flat(patterns, pat_lengths, sub_tables, sub_lengths, targets, tgt_lengths)
        assert result[0].item() is False

    def test_correct_tokens_wrong_order(self, mv) -> None:
        """Correct substitution tokens in wrong order → False."""
        # Pattern [0, 1, 2], var 1 -> [3, 4]. Correct output: [0, 3, 4, 2]
        # Target: [0, 4, 3, 2] — same tokens, wrong order
        V = 5
        S_max = 2
        patterns = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        pat_lengths = torch.tensor([3], dtype=torch.int32)
        sub_tables = torch.zeros(1, V, S_max, dtype=torch.int32)
        for v in range(V):
            sub_tables[0, v, 0] = v
        sub_tables[0, 1, 0] = 3
        sub_tables[0, 1, 1] = 4
        sub_lengths = torch.ones(1, V, dtype=torch.int32)
        sub_lengths[0, 1] = 2
        targets = torch.tensor([[0, 4, 3, 2]], dtype=torch.int32)  # swapped 3,4
        tgt_lengths = torch.tensor([4], dtype=torch.int32)

        result = mv.verify_flat(patterns, pat_lengths, sub_tables, sub_lengths, targets, tgt_lengths)
        assert result[0].item() is False

    def test_all_variable_pattern_wrong_lengths(self, mv) -> None:
        """All-variable pattern with wrong substitution lengths → wrong output length → False."""
        # Pattern [1, 2], both variables. Subst: 1->[3,4], 2->[5] → output [3,4,5] len=3
        # Target length = 2 → must fail
        V = 6
        S_max = 2
        patterns = torch.tensor([[1, 2]], dtype=torch.int32)
        pat_lengths = torch.tensor([2], dtype=torch.int32)
        sub_tables = torch.zeros(1, V, S_max, dtype=torch.int32)
        for v in range(V):
            sub_tables[0, v, 0] = v
        sub_tables[0, 1, 0] = 3
        sub_tables[0, 1, 1] = 4
        sub_tables[0, 2, 0] = 5
        sub_lengths = torch.ones(1, V, dtype=torch.int32)
        sub_lengths[0, 1] = 2
        sub_lengths[0, 2] = 1
        targets = torch.tensor([[3, 4]], dtype=torch.int32)  # missing token 5
        tgt_lengths = torch.tensor([2], dtype=torch.int32)

        result = mv.verify_flat(patterns, pat_lengths, sub_tables, sub_lengths, targets, tgt_lengths)
        assert result[0].item() is False
