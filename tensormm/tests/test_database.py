"""Tests for tensormm.database."""

from __future__ import annotations

import os

import torch

from tensormm.database import MetamathDatabase
from tensormm.parser import parse_mm_file
from tensormm.tokenizer import Tokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _build_demo0_db() -> tuple[MetamathDatabase, Tokenizer]:
    parsed = parse_mm_file(os.path.join(DATA_DIR, "demo0.mm"))
    tok = Tokenizer()
    db = MetamathDatabase(parsed, tok)
    return db, tok


class TestDatabaseFromDemo0:

    def test_num_assertions(self) -> None:
        db, _ = _build_demo0_db()
        # demo0.mm has: tze, tpl, weq, wim, a1, a2, mp, th1 = 8 assertions
        assert db.num_assertions() == 8

    def test_conclusion_shapes(self) -> None:
        db, _ = _build_demo0_db()
        assert db.conclusions.ndim == 2
        assert db.conclusions.shape[0] == 8
        assert db.conclusion_lengths.shape == (8,)
        assert db.conclusions.dtype == torch.int32
        assert db.conclusion_lengths.dtype == torch.int32

    def test_assertion_labels_order(self) -> None:
        db, _ = _build_demo0_db()
        expected = ["tze", "tpl", "weq", "wim", "a1", "a2", "mp", "th1"]
        assert db.assertion_labels == expected

    def test_get_conclusion_decodes_correctly(self) -> None:
        db, tok = _build_demo0_db()
        # tze: "term 0"
        tze_conclusion = db.get_conclusion("tze")
        decoded = tok.decode_expression(tze_conclusion.tolist())
        assert decoded == ["term", "0"]

        # a2: "|- ( t + 0 ) = t"
        a2_conclusion = db.get_conclusion("a2")
        decoded = tok.decode_expression(a2_conclusion.tolist())
        assert decoded == ["|-", "(", "t", "+", "0", ")", "=", "t"]

    def test_floating_hyp_shapes(self) -> None:
        db, _ = _build_demo0_db()
        # mp has 2 mandatory floating hyps (wp, wq)
        mp_fhyps = db.get_floating_hyps("mp")
        assert mp_fhyps.shape == (2, 2)
        assert mp_fhyps.dtype == torch.int32

        # tze has 0 mandatory floating hyps (no variables in "term 0")
        tze_fhyps = db.get_floating_hyps("tze")
        assert tze_fhyps.shape == (0, 2)

    def test_essential_hyp_shapes(self) -> None:
        db, _ = _build_demo0_db()
        # mp has 2 essential hyps
        mp_ehyps = db.get_essential_hyps("mp")
        assert mp_ehyps.shape[0] == 2
        assert mp_ehyps.dtype == torch.int32

        # a1 has 0 essential hyps
        a1_ehyps = db.get_essential_hyps("a1")
        assert a1_ehyps.shape[0] == 0


class TestIsVariableMask:

    def test_variables_are_marked(self) -> None:
        db, tok = _build_demo0_db()
        # t, r, s, P, Q should be variables
        for var in ["t", "r", "s", "P", "Q"]:
            tid = tok.encode_symbol(var)
            assert db.is_variable[tid].item() is True, f"{var} should be a variable"

    def test_constants_are_not_variables(self) -> None:
        db, tok = _build_demo0_db()
        for const in ["0", "+", "=", "->", "(", ")", "term", "wff", "|-"]:
            tid = tok.encode_symbol(const)
            assert db.is_variable[tid].item() is False, f"{const} should not be a variable"

    def test_pad_is_not_variable(self) -> None:
        db, _ = _build_demo0_db()
        assert db.is_variable[Tokenizer.PAD_TOKEN].item() is False


class TestConclusionPadding:

    def test_padding_uses_pad_token(self) -> None:
        db, _ = _build_demo0_db()
        # tze conclusion is "term 0" (length 2), max is longer
        # so remaining positions should be PAD
        idx = db.label_to_index["tze"]
        length = db.conclusion_lengths[idx].item()
        padded_part = db.conclusions[idx, length:]
        if padded_part.numel() > 0:
            assert (padded_part == Tokenizer.PAD_TOKEN).all()

    def test_lengths_match_actual_content(self) -> None:
        db, tok = _build_demo0_db()
        for lbl in db.assertion_labels:
            idx = db.label_to_index[lbl]
            length = db.conclusion_lengths[idx].item()
            conclusion = db.conclusions[idx, :length]
            # No PAD tokens in the actual content
            assert (conclusion != Tokenizer.PAD_TOKEN).all(), f"PAD in content of {lbl}"
            # Decode should match the parsed expression
            decoded = tok.decode_expression(conclusion.tolist())
            assert decoded == db.parsed.assertions[lbl].expression
