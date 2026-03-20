"""Bidirectional mapping between Metamath symbol strings and integer token IDs."""

from __future__ import annotations


class Tokenizer:
    """Maps Metamath symbols to integer token IDs and back.

    Token 0 is ALWAYS reserved as the PAD token.
    Token IDs are assigned sequentially starting from 1 as symbols are encountered.
    """

    PAD_TOKEN: int = 0

    def __init__(self) -> None:
        self.symbol_to_id: dict[str, int] = {}
        self.id_to_symbol: dict[int, str] = {}
        self.next_id: int = 1

    def encode_symbol(self, symbol: str) -> int:
        """Get or create a token ID for a symbol. Returns the integer ID."""
        if symbol in self.symbol_to_id:
            return self.symbol_to_id[symbol]
        token_id = self.next_id
        self.symbol_to_id[symbol] = token_id
        self.id_to_symbol[token_id] = symbol
        self.next_id += 1
        return token_id

    def encode_expression(self, symbols: list[str]) -> list[int]:
        """Convert a list of symbol strings to a list of token IDs."""
        return [self.encode_symbol(s) for s in symbols]

    def decode_expression(self, token_ids: list[int]) -> list[str]:
        """Convert a list of token IDs back to symbol strings. Ignores PAD tokens."""
        return [self.id_to_symbol[tid] for tid in token_ids if tid != self.PAD_TOKEN]

    def vocab_size(self) -> int:
        """Return total number of tokens including PAD."""
        return self.next_id

    def is_pad(self, token_id: int) -> bool:
        """Check if a token ID is the PAD token."""
        return token_id == self.PAD_TOKEN
