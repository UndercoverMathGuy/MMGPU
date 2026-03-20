"""GPU-accelerated batch verification via tensorized substitution + equality.

The four-step pipeline:
  1. GATHER  — look up per-token replacement lengths
  2. PREFIX SUM — cumsum to compute output write offsets
  3. SCATTER — vectorized write of replacement tokens into output buffer
  4. REDUCE  — element-wise equality check, masked, reduced to per-candidate bool

All tensors use int32 for MPS compatibility.
Scatter is fully vectorized (no Python loops over P or S) to saturate the GPU.

Two APIs:
  - verify_batch(): dict-based, convenient for small jobs and tests
  - verify_prebuilt(): tensor-in-tensor-out, zero Python per call, for real workloads
"""

from __future__ import annotations

from collections import defaultdict

import torch


def _select_device() -> torch.device:
    """Auto-detect best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TensorVerifier:
    """Batch-verifies substitution+equality checks on GPU.

    Given B candidate substitutions applied to the same pattern, checks which
    candidates produce a target expression. The batch dimension is across
    candidates (different substitutions), not across different patterns.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or _select_device()

    # ──────────────────────────────────────────────────────────────────
    #  LOW-LEVEL: pure-tensor pipeline (no Python dicts in hot path)
    # ──────────────────────────────────────────────────────────────────

    def verify_prebuilt(
        self,
        pattern: torch.Tensor,
        sub_tables: torch.Tensor,
        sub_lengths: torch.Tensor,
        target_tensor: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Verify a batch with pre-built tensors already on device.

        This is the hot-path kernel — ZERO Python dicts, ZERO CPU↔GPU transfers.
        All inputs must already be on self.device.

        Args:
            pattern: [P] int32 — token IDs of the pattern (full vocab, not compact)
            sub_tables: [B, V, S_max] int32 — replacement lookup table
                        sub_tables[b, token_id, s] = replacement token at offset s
            sub_lengths: [B, V] int32 — replacement length per token per candidate
                         For constants: 1 (identity). For variables: len(replacement).
            target_tensor: [B, T_max] int32 — padded target expressions
            target_lengths: [B] int32 — actual length of each target

        Returns:
            [B] bool tensor on CPU.
        """
        device = self.device
        B = sub_tables.shape[0]
        P = pattern.shape[0]
        S_max = sub_tables.shape[2]

        if B == 0:
            return torch.empty(0, dtype=torch.bool)
        if P == 0:
            return (target_lengths == 0).cpu()

        # STEP 1: GATHER
        pattern_expanded = pattern.unsqueeze(0).expand(B, P).long()
        token_lengths = torch.gather(sub_lengths, dim=1, index=pattern_expanded)

        # STEP 2: PREFIX SUM
        offsets = torch.cumsum(token_lengths, dim=1) - token_lengths
        total_lengths = token_lengths.sum(dim=1)
        max_output_len = int(total_lengths.max().item())

        # STEP 3: SCATTER
        s_range = torch.arange(S_max, device=device, dtype=torch.int32).reshape(1, 1, S_max)
        valid = s_range < token_lengths.unsqueeze(2)
        write_pos = offsets.unsqueeze(2) + s_range

        pat_idx = pattern_expanded.unsqueeze(2).expand(B, P, S_max)
        replacement_toks = torch.gather(sub_tables, dim=1, index=pat_idx)

        output = torch.zeros(B, max_output_len, dtype=torch.int32, device=device)
        batch_idx = torch.arange(B, device=device, dtype=torch.long).reshape(B, 1, 1).expand(B, P, S_max)

        valid_flat = valid.reshape(-1)
        output[batch_idx.reshape(-1)[valid_flat],
               write_pos.reshape(-1)[valid_flat].long()] = replacement_toks.reshape(-1)[valid_flat]

        # STEP 4: REDUCE
        T_max = int(target_tensor.shape[1])
        compare_dim = max(max_output_len, T_max)
        length_match = (total_lengths == target_lengths)

        if max_output_len < compare_dim:
            output = torch.nn.functional.pad(output, (0, compare_dim - max_output_len))
        if T_max < compare_dim:
            target_tensor = torch.nn.functional.pad(target_tensor, (0, compare_dim - T_max))

        positions = torch.arange(compare_dim, device=device).unsqueeze(0)
        valid_mask = positions < total_lengths.unsqueeze(1)
        masked_eq = (output == target_tensor) | ~valid_mask
        content_match = masked_eq.all(dim=1)

        return (length_match & content_match).cpu()

    # ──────────────────────────────────────────────────────────────────
    #  HIGH-LEVEL: dict-based convenience API (for tests / small jobs)
    # ──────────────────────────────────────────────────────────────────

    def prepare_substitution_tensors(
        self,
        pattern: list[int],
        substitutions: list[dict[int, list[int]]],
        is_variable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], dict[int, int]]:
        """Build compact substitution lookup tables for a batch of candidates.

        Uses compact vocabulary: only token IDs appearing in the pattern get
        a compact index. This reduces the V dimension from ~30k to ~50-100.

        Returns:
            compact_pattern, sub_tables, sub_lengths, compact_to_full, full_to_compact
        """
        B = len(substitutions)

        full_to_compact: dict[int, int] = {0: 0}
        compact_to_full: list[int] = [0]

        def _get_compact(full_id: int) -> int:
            if full_id not in full_to_compact:
                cid = len(compact_to_full)
                full_to_compact[full_id] = cid
                compact_to_full.append(full_id)
            return full_to_compact[full_id]

        for tid in pattern:
            _get_compact(tid)
        for subst in substitutions:
            for var_id, replacement in subst.items():
                _get_compact(var_id)
                for tid in replacement:
                    _get_compact(tid)

        V = len(compact_to_full)
        S_max = 1
        for subst in substitutions:
            for replacement in subst.values():
                if len(replacement) > S_max:
                    S_max = len(replacement)

        sub_lengths = torch.ones(B, V, dtype=torch.int32)
        identity = torch.arange(V, dtype=torch.int32).unsqueeze(0).unsqueeze(2)
        sub_tables = torch.zeros(B, V, S_max, dtype=torch.int32)
        sub_tables[:, :, 0:1] = identity.expand(B, V, 1)

        for b, subst in enumerate(substitutions):
            for var_id, replacement in subst.items():
                cid = full_to_compact[var_id]
                sub_lengths[b, cid] = len(replacement)
                for s, tid in enumerate(replacement):
                    sub_tables[b, cid, s] = full_to_compact[tid]

        compact_pattern = torch.tensor(
            [full_to_compact[tid] for tid in pattern], dtype=torch.int32
        )

        return compact_pattern, sub_tables, sub_lengths, compact_to_full, full_to_compact

    def verify_batch(
        self,
        pattern: list[int],
        substitutions: list[dict[int, list[int]]],
        targets: list[list[int]],
        is_variable: torch.Tensor,
    ) -> torch.Tensor:
        """Verify a batch of (substitution, target) pairs against a pattern.

        Convenience wrapper around verify_prebuilt. Builds compact vocab, tensors,
        moves to device, calls verify_prebuilt.
        """
        B = len(substitutions)
        if B == 0:
            return torch.empty(0, dtype=torch.bool)

        P = len(pattern)
        if P == 0:
            return torch.tensor([len(t) == 0 for t in targets], dtype=torch.bool)

        device = self.device
        compact_pattern, sub_tables, sub_lengths, compact_to_full, full_to_compact = \
            self.prepare_substitution_tensors(pattern, substitutions, is_variable)

        compact_pattern = compact_pattern.to(device)
        sub_tables = sub_tables.to(device)
        sub_lengths = sub_lengths.to(device)

        # Build target tensor using compact IDs
        max_target_len = max(len(t) for t in targets) if targets else 0
        target_tensor = torch.zeros(B, max(max_target_len, 1), dtype=torch.int32, device=device)
        target_lengths_list = []
        for b, target in enumerate(targets):
            target_lengths_list.append(len(target))
            for j, tid in enumerate(target):
                target_tensor[b, j] = full_to_compact.get(tid, -1)
        target_lengths = torch.tensor(target_lengths_list, dtype=torch.int32, device=device)

        return self.verify_prebuilt(compact_pattern, sub_tables, sub_lengths,
                                    target_tensor, target_lengths)

    def verify_multi_pattern(
        self,
        items: list[tuple[list[int], dict[int, list[int]], list[int]]],
        is_variable: torch.Tensor,
    ) -> list[bool]:
        """Verify a list of (pattern, substitution, target) triples efficiently.

        Groups items by pattern and fires one batched GPU call per pattern group.
        """
        if not items:
            return []

        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for i, (pat, _subst, _tgt) in enumerate(items):
            groups[tuple(pat)].append(i)

        results = [False] * len(items)

        for pat_key, indices in groups.items():
            pattern = list(pat_key)
            batch_substs = [items[i][1] for i in indices]
            batch_targets = [items[i][2] for i in indices]
            gpu_out = self.verify_batch(pattern, batch_substs, batch_targets, is_variable)
            for j, idx in enumerate(indices):
                results[idx] = gpu_out[j].item()

        return results

    # ──────────────────────────────────────────────────────────────────
    #  FLAT KERNEL: one call for N heterogeneous steps (different patterns)
    # ──────────────────────────────────────────────────────────────────

    def verify_flat(
        self,
        patterns: torch.Tensor,
        pattern_lengths: torch.Tensor,
        sub_tables: torch.Tensor,
        sub_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Verify N heterogeneous (pattern, substitution, target) triples in ONE kernel.

        Unlike verify_prebuilt (same pattern for all B), this handles DIFFERENT
        patterns per step. All N steps are processed in parallel in a single
        gather→prefix_sum→scatter→reduce pass. ONE kernel launch, ONE transfer.

        Memory optimization: uses a flat 1D output buffer with per-step base offsets
        to avoid materializing [N, P_max, S_max] long index tensors. Peak memory
        is O(N × max_output_len) instead of O(N × P_max × S_max).

        All inputs must already be on self.device.

        Args:
            patterns:       [N, P_max] int32 — padded pattern token IDs (0 = pad)
            pattern_lengths:[N] int32 — actual pattern length per step
            sub_tables:     [N, V, S_max] int32 — per-step replacement lookup
            sub_lengths:    [N, V] int32 — per-step replacement lengths
            targets:        [N, T_max] int32 — padded target expressions
            target_lengths: [N] int32 — actual target length per step

        Returns:
            [N] bool tensor on CPU.
        """
        device = self.device
        N = patterns.shape[0]
        P_max = patterns.shape[1]
        S_max = sub_tables.shape[2]

        if N == 0:
            return torch.empty(0, dtype=torch.bool)

        # STEP 1: GATHER — per-(step, position) replacement lengths [N, P_max]
        pat_idx_gather = patterns.long()  # [N, P_max]
        token_lengths = torch.gather(sub_lengths, dim=1, index=pat_idx_gather)  # [N, P_max]
        pos_range = torch.arange(P_max, device=device).unsqueeze(0)  # [1, P_max]
        pat_valid = pos_range < pattern_lengths.unsqueeze(1)  # [N, P_max]
        token_lengths = token_lengths * pat_valid.int()

        # STEP 2: PREFIX SUM [N, P_max]
        offsets = torch.cumsum(token_lengths, dim=1) - token_lengths  # [N, P_max]
        total_lengths = token_lengths.sum(dim=1)  # [N]
        max_output_len = int(total_lengths.max().item())

        if max_output_len == 0:
            return (target_lengths == 0).cpu()

        # STEP 3: SCATTER — flat 1D output buffer to minimize memory
        # output_flat[step_base[n] + offsets[n,p] + s] = sub_tables[n, patterns[n,p], s]
        # where step_base[n] = n * max_output_len
        #
        # We process P positions in a loop to avoid [N, P_max, S_max] intermediates.
        # Each iteration creates only [N, S_max] tensors — memory is O(N × S_max).

        output = torch.zeros(N, max_output_len, dtype=torch.int32, device=device)
        step_base = torch.arange(N, device=device, dtype=torch.long) * max_output_len
        output_flat = output.view(-1)  # [N * max_output_len]

        s_range = torch.arange(S_max, device=device, dtype=torch.int32).unsqueeze(0)  # [1, S_max]

        # Precompute max pattern length to bound the loop (avoids iterating over padding)
        actual_P_max = int(pattern_lengths.max().item())

        for p in range(actual_P_max):
            # For each step: replacement length, write offset, pattern token at position p
            tl_p = token_lengths[:, p]  # [N] int32
            off_p = offsets[:, p]  # [N] int32
            pat_p = pat_idx_gather[:, p]  # [N] long — which vocab entry

            # Gather replacement tokens: sub_tables[n, pat_p[n], :] → [N, S_max]
            pat_p_3d = pat_p.unsqueeze(1).unsqueeze(2).expand(N, 1, S_max)  # [N, 1, S_max]
            rep_toks = torch.gather(sub_tables, dim=1, index=pat_p_3d).squeeze(1)  # [N, S_max]

            # Valid mask: s < tl_p[n] AND p < pattern_lengths[n]
            p_valid = pat_valid[:, p]  # [N] bool
            s_valid = (s_range < tl_p.unsqueeze(1)) & p_valid.unsqueeze(1)  # [N, S_max]

            # Flat write positions: step_base[n] + off_p[n] + s
            flat_pos = step_base.unsqueeze(1) + off_p.unsqueeze(1).long() + s_range.long()  # [N, S_max]

            # Scatter — masked write (no .any() sync, just mask)
            s_valid_flat = s_valid.reshape(-1)
            output_flat[flat_pos.reshape(-1)[s_valid_flat]] = rep_toks.reshape(-1)[s_valid_flat]

        # STEP 4: REDUCE — compare output to targets [N]
        T_max = int(targets.shape[1])
        compare_dim = max(max_output_len, T_max)
        length_match = (total_lengths == target_lengths)

        if max_output_len < compare_dim:
            output = torch.nn.functional.pad(output, (0, compare_dim - max_output_len))
        if T_max < compare_dim:
            targets = torch.nn.functional.pad(targets, (0, compare_dim - T_max))

        positions = torch.arange(compare_dim, device=device).unsqueeze(0)
        valid_mask = positions < total_lengths.unsqueeze(1)
        masked_eq = (output == targets) | ~valid_mask
        content_match = masked_eq.all(dim=1)

        return (length_match & content_match).cpu()
