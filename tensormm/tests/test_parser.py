"""Tests for tensormm.parser."""

from __future__ import annotations

import os
import tempfile

import pytest

from tensormm.parser import (
    Assertion,
    CompressedProof,
    EssentialHyp,
    FloatingHyp,
    ParsedDatabase,
    _decompress_proof,
    parse_mm_file,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _write_tmp_mm(content: str) -> str:
    """Write content to a temp .mm file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".mm")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


class TestParseConstants:

    def test_parse_constants(self) -> None:
        path = _write_tmp_mm("$c 0 + = -> ( ) term wff |- $.")
        db = parse_mm_file(path)
        os.unlink(path)
        assert db.constants == {"0", "+", "=", "->", "(", ")", "term", "wff", "|-"}

    def test_multiple_constant_declarations(self) -> None:
        path = _write_tmp_mm("$c a b $. $c c d $.")
        db = parse_mm_file(path)
        os.unlink(path)
        assert db.constants == {"a", "b", "c", "d"}


class TestParseVariables:

    def test_parse_variables(self) -> None:
        path = _write_tmp_mm("$c wff $. $v ph ps ch $.")
        db = parse_mm_file(path)
        os.unlink(path)
        assert db.variables == {"ph", "ps", "ch"}


class TestParseFloatingHyp:

    def test_parse_floating_hyp(self) -> None:
        path = _write_tmp_mm("$c wff $. $v ph $. wph $f wff ph $.")
        db = parse_mm_file(path)
        os.unlink(path)
        assert "wph" in db.floating_hyps
        fh = db.floating_hyps["wph"]
        assert fh.type_code == "wff"
        assert fh.variable == "ph"
        assert fh.label == "wph"


class TestParseEssentialHyp:

    def test_parse_essential_hyp(self) -> None:
        path = _write_tmp_mm(
            "$c |- wff $. $v ph $. wph $f wff ph $. ${ min $e |- ph $. $}"
        )
        db = parse_mm_file(path)
        os.unlink(path)
        assert "min" in db.essential_hyps
        eh = db.essential_hyps["min"]
        assert eh.expression == ["|-", "ph"]


class TestParseAxiom:

    def test_parse_axiom_with_mandatory_hyps(self) -> None:
        path = _write_tmp_mm(
            "$c |- wff ( ) -> $.\n"
            "$v ph ps $.\n"
            "wph $f wff ph $.\n"
            "wps $f wff ps $.\n"
            "${ min $e |- ph $.\n"
            "   maj $e |- ( ph -> ps ) $.\n"
            "   mp $a |- ps $. $}\n"
        )
        db = parse_mm_file(path)
        os.unlink(path)
        assert "mp" in db.assertions
        mp = db.assertions["mp"]
        assert mp.type == "axiom"
        assert mp.expression == ["|-", "ps"]
        # Both wph and wps should be mandatory (ph and ps appear in e-hyps/conclusion)
        assert "wph" in mp.floating_hyps
        assert "wps" in mp.floating_hyps
        assert mp.essential_hyps == ["min", "maj"]


class TestParseTheoremWithProof:

    def test_parse_theorem_uncompressed(self) -> None:
        """Parse th1 from test_mini.mm which has an uncompressed proof."""
        path = os.path.join(DATA_DIR, "test_mini.mm")
        db = parse_mm_file(path)
        assert "th1" in db.assertions
        th1 = db.assertions["th1"]
        assert th1.type == "theorem"
        assert th1.expression == ["|-", "(", "ps", "->", "ps", ")"]
        assert th1.proof == ["wps", "th1.1", "ax-1"]
        assert th1.compressed_proof is None


class TestScoping:

    def test_scoping_limits_essential_hyps(self) -> None:
        """Essential hyps inside ${ ... $} should not leak to assertions outside."""
        path = _write_tmp_mm(
            "$c |- wff ( ) -> $.\n"
            "$v ph ps $.\n"
            "wph $f wff ph $.\n"
            "wps $f wff ps $.\n"
            "${ inner $e |- ph $.\n"
            "   ax-inner $a |- ( ph -> ph ) $. $}\n"
            "ax-outer $a |- ( ps -> ps ) $.\n"
        )
        db = parse_mm_file(path)
        os.unlink(path)
        # ax-inner should have the essential hyp
        assert db.assertions["ax-inner"].essential_hyps == ["inner"]
        # ax-outer should NOT have the essential hyp (it's out of scope)
        assert db.assertions["ax-outer"].essential_hyps == []

    def test_scoping_limits_floating_hyps(self) -> None:
        """Floating hyps inside ${ ... $} should not be visible outside."""
        path = _write_tmp_mm(
            "$c |- wff term $.\n"
            "$v ph $.\n"
            "wph $f wff ph $.\n"
            "${ $v t $. tt $f term t $.\n"
            "   ax-in $a |- ph $. $}\n"
            "ax-out $a |- ph $.\n"
        )
        db = parse_mm_file(path)
        os.unlink(path)
        # ax-in should see tt as mandatory (t doesn't appear in expr, but ph does)
        # Actually t does NOT appear in "|- ph", so tt is NOT mandatory
        assert "tt" not in db.assertions["ax-in"].floating_hyps
        # ax-out cannot see tt at all (out of scope)
        assert "tt" not in db.assertions["ax-out"].floating_hyps


class TestMandatoryHypFiltering:

    def test_only_used_floating_hyps_are_mandatory(self) -> None:
        """Extra floating hyps in scope but typing unused variables must NOT be mandatory."""
        path = _write_tmp_mm(
            "$c |- wff class $.\n"
            "$v ph ps A $.\n"
            "wph $f wff ph $.\n"
            "wps $f wff ps $.\n"
            "cA  $f class A $.\n"  # A is in scope but unused
            "ax-test $a |- ph $.\n"
        )
        db = parse_mm_file(path)
        os.unlink(path)
        ax = db.assertions["ax-test"]
        # Only wph should be mandatory (ph appears in conclusion)
        # wps and cA should NOT be mandatory
        assert ax.floating_hyps == ["wph"]

    def test_floating_hyps_from_essential_hyps_are_mandatory(self) -> None:
        """Variables appearing in essential hyps (not just conclusion) must have mandatory floats."""
        path = _write_tmp_mm(
            "$c |- wff ( ) -> $.\n"
            "$v ph ps ch $.\n"
            "wph $f wff ph $.\n"
            "wps $f wff ps $.\n"
            "wch $f wff ch $.\n"
            "${ hyp1 $e |- ph $.\n"
            "   hyp2 $e |- ( ph -> ps ) $.\n"
            "   ax-test2 $a |- ps $. $}\n"
        )
        db = parse_mm_file(path)
        os.unlink(path)
        ax = db.assertions["ax-test2"]
        # ph and ps appear in e-hyps/conclusion. ch does NOT.
        assert "wph" in ax.floating_hyps
        assert "wps" in ax.floating_hyps
        assert "wch" not in ax.floating_hyps


class TestParseDemo0:

    def test_parse_demo0_full(self) -> None:
        """Full parse of demo0.mm — check all assertions are present."""
        path = os.path.join(DATA_DIR, "demo0.mm")
        db = parse_mm_file(path)

        # Constants
        assert "0" in db.constants
        assert "+" in db.constants
        assert "=" in db.constants
        assert "term" in db.constants
        assert "wff" in db.constants
        assert "|-" in db.constants

        # Variables
        assert db.variables == {"t", "r", "s", "P", "Q"}

        # Floating hyps
        assert set(db.floating_hyps.keys()) == {"tt", "tr", "ts", "wp", "wq"}

        # Axioms
        for lbl in ["tze", "tpl", "weq", "wim", "a1", "a2", "mp"]:
            assert lbl in db.assertions, f"Missing assertion: {lbl}"
            assert db.assertions[lbl].type == "axiom"

        # Theorem
        assert "th1" in db.assertions
        assert db.assertions["th1"].type == "theorem"
        assert db.assertions["th1"].proof is not None

        # mp should have 2 essential hyps and 2 floating hyps
        mp = db.assertions["mp"]
        assert len(mp.essential_hyps) == 2
        # P and Q appear in e-hyps/conclusion, so wp and wq are mandatory
        assert "wp" in mp.floating_hyps
        assert "wq" in mp.floating_hyps

    def test_demo0_th1_proof(self) -> None:
        """Check that th1's proof is correctly parsed."""
        path = os.path.join(DATA_DIR, "demo0.mm")
        db = parse_mm_file(path)
        th1 = db.assertions["th1"]
        expected_proof = [
            "tt", "tze", "tpl", "tt", "weq", "tt", "tt", "weq", "tt", "a2",
            "tt", "tze", "tpl", "tt", "weq", "tt", "tze", "tpl", "tt", "weq",
            "tt", "tt", "weq", "wim", "tt", "a2", "tt", "tze", "tpl", "tt",
            "tt", "a1", "mp", "mp",
        ]
        assert th1.proof == expected_proof

    def test_demo0_a1_mandatory_hyps(self) -> None:
        """a1 uses t, r, s — so tt, tr, ts should be mandatory. wp, wq should NOT."""
        path = os.path.join(DATA_DIR, "demo0.mm")
        db = parse_mm_file(path)
        a1 = db.assertions["a1"]
        assert "tt" in a1.floating_hyps
        assert "tr" in a1.floating_hyps
        assert "ts" in a1.floating_hyps
        assert "wp" not in a1.floating_hyps
        assert "wq" not in a1.floating_hyps


class TestParseDisjointVars:

    def test_parse_disjoint_vars(self) -> None:
        path = _write_tmp_mm(
            "$c |- wff $.\n"
            "$v ph ps ch $.\n"
            "wph $f wff ph $.\n"
            "wps $f wff ps $.\n"
            "wch $f wff ch $.\n"
            "$d ph ps $.\n"
            "$d ps ch $.\n"
            "${ hyp $e |- ph $.\n"
            "   ax-d $a |- ps $. $}\n"
        )
        db = parse_mm_file(path)
        os.unlink(path)
        ax = db.assertions["ax-d"]
        # (ph, ps) should be in dvs since both are mandatory variables
        assert ("ph", "ps") in ax.disjoint_vars or ("ps", "ph") in ax.disjoint_vars
        # (ps, ch) should NOT be in dvs since ch is not mandatory
        assert ("ch", "ps") not in ax.disjoint_vars
        assert ("ps", "ch") not in ax.disjoint_vars


class TestCompressedProofDecompression:

    def test_simple_compressed_decode(self) -> None:
        """Test the base-20/base-5 decoding algorithm."""
        # A=0, B=1, ..., T=19
        cp = _decompress_proof(["(", "lab1", "lab2", ")", "AAB"])
        assert cp.labels == ["lab1", "lab2"]
        assert cp.proof_ints == [0, 0, 1]

    def test_z_save(self) -> None:
        """Z should decode to -1 (save to heap)."""
        cp = _decompress_proof(["(", ")", "AZB"])
        assert cp.proof_ints == [0, -1, 1]

    def test_multi_digit_encoding(self) -> None:
        """U-Y are non-terminal digits: 5*cur + (ord(ch) - 84)."""
        # U = ord('U')-84 = 1, so UA = 20*1 + 0 = 20
        cp = _decompress_proof(["(", ")", "UA"])
        assert cp.proof_ints == [20]

        # UB = 20*1 + 1 = 21
        cp = _decompress_proof(["(", ")", "UB"])
        assert cp.proof_ints == [21]

        # VA = 20*2 + 0 = 40  (V = ord('V')-84 = 2)
        cp = _decompress_proof(["(", ")", "VA"])
        assert cp.proof_ints == [40]

        # UUA = 20*(5*1+1) + 0 = 20*6 = 120
        cp = _decompress_proof(["(", ")", "UUA"])
        assert cp.proof_ints == [120]

    def test_compressed_proof_with_split_tokens(self) -> None:
        """The encoded string may be split across multiple tokens."""
        cp = _decompress_proof(["(", "lab1", ")", "AB", "CD"])
        assert cp.labels == ["lab1"]
        # AB = [0, 1], CD = [2, 3]
        assert cp.proof_ints == [0, 1, 2, 3]
