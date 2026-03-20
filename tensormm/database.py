"""Tensorized representation of a parsed Metamath database."""

from __future__ import annotations

import torch

from tensormm.parser import ParsedDatabase
from tensormm.tokenizer import Tokenizer


class MetamathDatabase:
    """Builds and holds tensor representations of a parsed Metamath database.

    All tensors use int32 for MPS compatibility. Tensors are created on CPU;
    callers can move them to device as needed.
    """

    def __init__(self, parsed: ParsedDatabase, tokenizer: Tokenizer) -> None:
        self.parsed = parsed
        self.tokenizer = tokenizer

        # Encode all symbols into the tokenizer
        for c in parsed.constants:
            tokenizer.encode_symbol(c)
        for v in parsed.variables:
            tokenizer.encode_symbol(v)

        # Build variable mask: is_variable[token_id] = True if token is a variable
        vs = tokenizer.vocab_size()
        self._is_variable = torch.zeros(vs, dtype=torch.bool)
        for v in parsed.variables:
            tid = tokenizer.encode_symbol(v)
            self._is_variable[tid] = True

        # Build ordered list of assertion labels (preserving insertion order)
        self.assertion_labels: list[str] = list(parsed.assertions.keys())
        self.label_to_index: dict[str, int] = {
            lbl: idx for idx, lbl in enumerate(self.assertion_labels)
        }

        # Build padded conclusion tensor [num_assertions, max_conclusion_len]
        encoded_conclusions = [
            tokenizer.encode_expression(parsed.assertions[lbl].expression)
            for lbl in self.assertion_labels
        ]
        if encoded_conclusions:
            max_clen = max(len(c) for c in encoded_conclusions)
            self._conclusion_lengths = torch.tensor(
                [len(c) for c in encoded_conclusions], dtype=torch.int32
            )
            self._conclusions = torch.full(
                (len(encoded_conclusions), max_clen),
                Tokenizer.PAD_TOKEN,
                dtype=torch.int32,
            )
            for i, enc in enumerate(encoded_conclusions):
                self._conclusions[i, :len(enc)] = torch.tensor(enc, dtype=torch.int32)
        else:
            self._conclusions = torch.empty(0, 0, dtype=torch.int32)
            self._conclusion_lengths = torch.empty(0, dtype=torch.int32)

        # Build per-assertion hypothesis tensors
        # floating_hyp_expressions[label] -> tensor [num_f_hyps, 2] (typecode_id, var_id)
        # essential_hyp_expressions[label] -> tensor [num_e_hyps, max_ehyp_len] (padded)
        self.floating_hyp_expressions: dict[str, torch.Tensor] = {}
        self.essential_hyp_expressions: dict[str, torch.Tensor] = {}

        for lbl in self.assertion_labels:
            assertion = parsed.assertions[lbl]

            # Floating hyps: each is (typecode, variable)
            f_data: list[list[int]] = []
            for flbl in assertion.floating_hyps:
                fh = parsed.floating_hyps[flbl]
                f_data.append([
                    tokenizer.encode_symbol(fh.type_code),
                    tokenizer.encode_symbol(fh.variable),
                ])
            if f_data:
                self.floating_hyp_expressions[lbl] = torch.tensor(f_data, dtype=torch.int32)
            else:
                self.floating_hyp_expressions[lbl] = torch.empty(0, 2, dtype=torch.int32)

            # Essential hyps: padded token sequences
            e_encoded: list[list[int]] = []
            for elbl in assertion.essential_hyps:
                eh = parsed.essential_hyps[elbl]
                e_encoded.append(tokenizer.encode_expression(eh.expression))
            if e_encoded:
                max_elen = max(len(e) for e in e_encoded)
                e_tensor = torch.full(
                    (len(e_encoded), max_elen),
                    Tokenizer.PAD_TOKEN,
                    dtype=torch.int32,
                )
                for j, enc in enumerate(e_encoded):
                    e_tensor[j, :len(enc)] = torch.tensor(enc, dtype=torch.int32)
                self.essential_hyp_expressions[lbl] = e_tensor
            else:
                self.essential_hyp_expressions[lbl] = torch.empty(0, 0, dtype=torch.int32)

    @property
    def is_variable(self) -> torch.Tensor:
        """Boolean mask [vocab_size]: True if token ID is a variable."""
        return self._is_variable

    @property
    def conclusions(self) -> torch.Tensor:
        """Padded conclusions tensor [num_assertions, max_conclusion_len]."""
        return self._conclusions

    @property
    def conclusion_lengths(self) -> torch.Tensor:
        """Conclusion lengths [num_assertions]."""
        return self._conclusion_lengths

    def get_conclusion(self, label: str) -> torch.Tensor:
        """Return the unpadded conclusion tensor for a given assertion label."""
        idx = self.label_to_index[label]
        length = self._conclusion_lengths[idx].item()
        return self._conclusions[idx, :length]

    def get_floating_hyps(self, label: str) -> torch.Tensor:
        """Return floating hyp tensor [num_f_hyps, 2] for assertion label."""
        return self.floating_hyp_expressions[label]

    def get_essential_hyps(self, label: str) -> torch.Tensor:
        """Return padded essential hyp tensor [num_e_hyps, max_len] for assertion label."""
        return self.essential_hyp_expressions[label]

    def get_variables_in_assertion(self, label: str) -> set[int]:
        """Return set of variable token IDs appearing in the assertion's conclusion."""
        conclusion = self.get_conclusion(label)
        return {
            tid.item()
            for tid in conclusion
            if self._is_variable[tid]
        }

    def num_assertions(self) -> int:
        """Return total number of assertions (axioms + theorems)."""
        return len(self.assertion_labels)
