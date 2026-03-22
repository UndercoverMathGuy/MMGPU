"""Tests for tensormm.gpu_verifier — true GPU-accelerated verification.

Validates graph construction, level packing, GPU execution, and the
full pipeline against the CPU verifier as oracle.
"""
from __future__ import annotations

import os

import pytest
import torch

from tensormm.cpu_verifier import CPUVerifier
from tensormm.gpu_verifier import (
    build_proof_graph,
    build_all_proof_graphs,
    pack_levels,
    verify_database,
    verify_proofs_gpu,
    _build_label_info,
)
from tensormm.parser import ParsedDatabase, parse_mm_file
from tensormm.tokenizer import Tokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _load_mm(name: str) -> ParsedDatabase:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not found in data/")
    return parse_mm_file(path)


# ══════════════════════════════════════════════════════════════════════
#  Phase 1: Graph Construction
# ══════════════════════════════════════════════════════════════════════


class TestGraphConstruction:

    def test_demo0_all_graphs_build(self) -> None:
        """All demo0.mm theorems should produce valid ProofGraphs."""
        parsed = _load_mm("demo0.mm")
        label_info = _build_label_info(parsed)
        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
        assert len(theorems) > 0

        for lbl in theorems:
            result = build_proof_graph(parsed, lbl, label_info)
            assert not isinstance(result, str), f"Graph build failed for {lbl}: {result}"
            assert result.theorem_label == lbl
            assert len(result.nodes) > 0
            assert result.max_level >= 0

    def test_ql_all_graphs_build(self) -> None:
        """All ql.mm theorems should produce valid ProofGraphs."""
        parsed = _load_mm("ql.mm")
        label_info = _build_label_info(parsed)
        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]

        errors = []
        for lbl in theorems:
            result = build_proof_graph(parsed, lbl, label_info)
            if isinstance(result, str):
                errors.append(f"{lbl}: {result}")

        assert len(errors) == 0, f"{len(errors)} graph build errors:\n" + "\n".join(errors[:10])

    def test_graph_structure_demo0(self) -> None:
        """Verify graph structure for a specific demo0 theorem."""
        parsed = _load_mm("demo0.mm")
        label_info = _build_label_info(parsed)

        # Pick the first theorem
        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
        g = build_proof_graph(parsed, theorems[0], label_info)
        assert not isinstance(g, str), f"Failed: {g}"

        # All push nodes should be level 0
        for node in g.nodes:
            if node.node_type in ("push_f", "push_e"):
                assert node.level == 0
                assert node.expression is not None
                assert len(node.input_steps) == 0
            elif node.node_type == "assertion":
                # Assertions with 0 inputs (e.g., nullary axioms) can be level 0
                assert node.level >= 0
                if node.input_steps:
                    assert node.level >= 1

    def test_compressed_proof_z_backref(self) -> None:
        """ql.mm uses compressed proofs — Z backrefs must work."""
        parsed = _load_mm("ql.mm")
        label_info = _build_label_info(parsed)
        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]

        # Find a theorem with compressed proof
        found_compressed = False
        for lbl in theorems:
            a = parsed.assertions[lbl]
            if a.compressed_proof is not None:
                g = build_proof_graph(parsed, lbl, label_info)
                assert not isinstance(g, str), f"Failed on compressed proof {lbl}: {g}"
                found_compressed = True
                break
        assert found_compressed, "No compressed proofs found in ql.mm"

    def test_parallel_graph_construction(self) -> None:
        """Parallel graph construction should match serial."""
        parsed = _load_mm("demo0.mm")
        label_info = _build_label_info(parsed)
        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]

        # Serial
        serial = []
        for lbl in theorems:
            r = build_proof_graph(parsed, lbl, label_info)
            if not isinstance(r, str):
                serial.append(r.theorem_label)

        # Parallel
        graphs, errors = build_all_proof_graphs(parsed, theorems, max_workers=2)
        parallel = [g.theorem_label for g in graphs]

        assert set(serial) == set(parallel)

    def test_level_assignment(self) -> None:
        """Topological levels must be consistent with dependencies."""
        parsed = _load_mm("ql.mm")
        label_info = _build_label_info(parsed)
        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]

        for lbl in theorems[:50]:  # check first 50
            g = build_proof_graph(parsed, lbl, label_info)
            if isinstance(g, str):
                continue
            node_levels = {n.step_idx: n.level for n in g.nodes}
            for node in g.nodes:
                if node.node_type == "assertion":
                    for inp in node.input_steps:
                        assert node_levels[inp] < node.level, (
                            f"Level violation in {lbl}: node {node.step_idx} at level "
                            f"{node.level} depends on node {inp} at level {node_levels[inp]}"
                        )


# ══════════════════════════════════════════════════════════════════════
#  Phase 2: Level Packing
# ══════════════════════════════════════════════════════════════════════


class TestLevelPacking:

    def test_demo0_pack(self) -> None:
        """Level packing on demo0.mm should produce valid tensor shapes."""
        parsed = _load_mm("demo0.mm")
        tokenizer = Tokenizer()
        for c in parsed.constants:
            tokenizer.encode_symbol(c)
        for v in parsed.variables:
            tokenizer.encode_symbol(v)

        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
        graphs, errors = build_all_proof_graphs(parsed, theorems)
        assert len(graphs) > 0

        plan = pack_levels(graphs, parsed, tokenizer)
        assert plan.total_nodes > 0
        assert plan.max_expr_len >= 1
        assert plan.num_proofs == len(graphs)
        assert len(plan.push_global_indices) > 0
        # push_expressions is stored at actual max push width, not max_expr_len
        assert plan.push_expressions.shape[0] == len(plan.push_global_indices)
        assert plan.push_expressions.shape[1] >= 1

    def test_ql_pack(self) -> None:
        """Level packing on ql.mm should handle essential hyps."""
        parsed = _load_mm("ql.mm")
        tokenizer = Tokenizer()
        for c in parsed.constants:
            tokenizer.encode_symbol(c)
        for v in parsed.variables:
            tokenizer.encode_symbol(v)

        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
        graphs, _ = build_all_proof_graphs(parsed, theorems)
        plan = pack_levels(graphs, parsed, tokenizer)

        # ql.mm should have assertions with essential hyps in the table
        assert plan.assertion_table.ehyp_count.max() > 0, \
            "ql.mm should have assertions with essential hypotheses"

    def test_max_expr_len_not_hardcoded(self) -> None:
        """max_expr_len should be computed from data, not hardcoded small."""
        parsed = _load_mm("ql.mm")
        tokenizer = Tokenizer()
        for c in parsed.constants:
            tokenizer.encode_symbol(c)
        for v in parsed.variables:
            tokenizer.encode_symbol(v)

        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
        graphs, _ = build_all_proof_graphs(parsed, theorems)
        plan = pack_levels(graphs, parsed, tokenizer)

        # Should be at least 512 (our floor)
        assert plan.max_expr_len >= 512


# ══════════════════════════════════════════════════════════════════════
#  Phase 3: GPU Execution + Full Pipeline
# ══════════════════════════════════════════════════════════════════════


class TestFullPipeline:

    def test_demo0_full(self) -> None:
        """Full pipeline on demo0.mm must match CPU verifier."""
        parsed = _load_mm("demo0.mm")
        device = torch.device("cpu")

        gpu_results = verify_database(parsed, device=device, verbose=True)
        cpu_v = CPUVerifier(parsed)
        cpu_results = cpu_v.verify_all()

        for lbl, cpu_r in cpu_results.items():
            assert lbl in gpu_results, f"GPU missing theorem {lbl}"
            assert gpu_results[lbl] == cpu_r.success, (
                f"Divergence on {lbl}: GPU={gpu_results[lbl]}, "
                f"CPU={cpu_r.success} (err={cpu_r.error_message})"
            )

    def test_ql_full(self) -> None:
        """Full pipeline on ql.mm must match CPU verifier."""
        parsed = _load_mm("ql.mm")
        device = torch.device("cpu")

        gpu_results = verify_database(parsed, device=device, verbose=True)
        cpu_v = CPUVerifier(parsed)
        cpu_results = cpu_v.verify_all()

        divergences = []
        for lbl, cpu_r in cpu_results.items():
            if lbl not in gpu_results:
                divergences.append(f"GPU missing: {lbl}")
                continue
            if gpu_results[lbl] != cpu_r.success:
                divergences.append(
                    f"{lbl}: GPU={gpu_results[lbl]}, CPU={cpu_r.success} "
                    f"(err={cpu_r.error_message})"
                )

        assert len(divergences) == 0, (
            f"{len(divergences)} divergences:\n" + "\n".join(divergences[:20])
        )

    def test_ehyp_masking_no_false_failures(self) -> None:
        """Assertions with different ehyp counts must not cause false failures.

        ql.mm has a mix of assertions with 0 and 2+ essential hyps.
        If ehyp masking is broken, we'd see false failures on assertions
        with fewer ehyps than the batch max.
        """
        parsed = _load_mm("ql.mm")
        device = torch.device("cpu")
        gpu_results = verify_database(parsed, device=device, check_dv=False)

        # CPU oracle (without $d for fair comparison)
        cpu_v = CPUVerifier(parsed)
        cpu_results = cpu_v.verify_all()

        # Count how many GPU says pass vs CPU says pass
        gpu_pass = sum(1 for v in gpu_results.values() if v)
        cpu_pass = sum(1 for v in cpu_results.values() if v.success)

        # GPU should pass at least as many as CPU (without $d, GPU might pass more)
        # But critically: no FALSE FAILURES (GPU fail when CPU pass)
        false_failures = [
            lbl for lbl, cpu_r in cpu_results.items()
            if cpu_r.success and lbl in gpu_results and not gpu_results[lbl]
        ]
        assert len(false_failures) == 0, (
            f"{len(false_failures)} false GPU failures (CPU pass, GPU fail):\n"
            + "\n".join(false_failures[:20])
        )


# ══════════════════════════════════════════════════════════════════════
#  GPU device tests (when available)
# ══════════════════════════════════════════════════════════════════════


CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


class TestGPUDevice:

    @pytest.mark.skipif(not (CUDA_AVAILABLE or MPS_AVAILABLE), reason="No GPU")
    def test_demo0_on_gpu(self) -> None:
        """demo0.mm on actual GPU device."""
        parsed = _load_mm("demo0.mm")
        gpu_results = verify_database(parsed, verbose=True)

        cpu_v = CPUVerifier(parsed)
        cpu_results = cpu_v.verify_all()
        for lbl, cpu_r in cpu_results.items():
            assert gpu_results.get(lbl) == cpu_r.success

    @pytest.mark.skipif(not (CUDA_AVAILABLE or MPS_AVAILABLE), reason="No GPU")
    def test_ql_on_gpu(self) -> None:
        """ql.mm on actual GPU device."""
        parsed = _load_mm("ql.mm")
        gpu_results = verify_database(parsed, verbose=True)

        cpu_v = CPUVerifier(parsed)
        cpu_results = cpu_v.verify_all()
        divergences = [
            lbl for lbl, cpu_r in cpu_results.items()
            if gpu_results.get(lbl) != cpu_r.success
        ]
        assert len(divergences) == 0, f"{len(divergences)} divergences"
