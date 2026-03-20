"""Tests for tensormm.tokenizer."""

from tensormm.tokenizer import Tokenizer


class TestTokenizer:

    def test_pad_token_is_zero(self) -> None:
        tok = Tokenizer()
        assert tok.PAD_TOKEN == 0
        assert tok.is_pad(0)
        assert not tok.is_pad(1)

    def test_encode_decode_roundtrip(self) -> None:
        tok = Tokenizer()
        symbols = ["|-", "wff", "(", "ph", "->", "ps", ")"]
        ids = tok.encode_expression(symbols)
        decoded = tok.decode_expression(ids)
        assert decoded == symbols

    def test_unique_ids(self) -> None:
        tok = Tokenizer()
        id1 = tok.encode_symbol("ph")
        id2 = tok.encode_symbol("ps")
        id3 = tok.encode_symbol("ph")  # same symbol again
        assert id1 != id2
        assert id1 == id3  # idempotent
        assert id1 != tok.PAD_TOKEN
        assert id2 != tok.PAD_TOKEN

    def test_encode_expression_multiple_symbols(self) -> None:
        tok = Tokenizer()
        expr1 = tok.encode_expression(["|-", "ph"])
        expr2 = tok.encode_expression(["|-", "ps"])
        # "|-" should get the same ID in both
        assert expr1[0] == expr2[0]
        # "ph" and "ps" should differ
        assert expr1[1] != expr2[1]

    def test_decode_ignores_pad(self) -> None:
        tok = Tokenizer()
        tok.encode_expression(["a", "b", "c"])
        ids_with_pad = [1, 0, 2, 0, 3]
        decoded = tok.decode_expression(ids_with_pad)
        assert decoded == ["a", "b", "c"]

    def test_vocab_size(self) -> None:
        tok = Tokenizer()
        assert tok.vocab_size() == 1  # just PAD
        tok.encode_symbol("x")
        assert tok.vocab_size() == 2
        tok.encode_symbol("y")
        assert tok.vocab_size() == 3
        tok.encode_symbol("x")  # duplicate
        assert tok.vocab_size() == 3  # unchanged
