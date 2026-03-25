"""Tests for tensormm.gpu_verifier — graph construction, level packing, GPU execution.

Full-pipeline correctness is validated against metamath-knife as the oracle.
"""
from __future__ import annotations

import os
import subprocess
import shutil

import pytest
import torch

from tensormm.gpu_verifier import (
    build_proof_graph,
    build_all_proof_graphs,
    pack_levels,
    verify_database,
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


def _knife_verify(mm_path: str) -> bool:
    """Run metamath-knife --verify; return True if exit 0."""
    knife = shutil.which("metamath-knife")
    if knife is None:
        pytest.skip("metamath-knife not installed")
    r = subprocess.run(
        [knife, "--verify", mm_path],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return r.returncode == 0


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

        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
        g = build_proof_graph(parsed, theorems[0], label_info)
        assert not isinstance(g, str), f"Failed: {g}"

        for node in g.nodes:
            if node.node_type in ("push_f", "push_e"):
                assert node.level == 0
                assert node.expression is not None
                assert len(node.input_steps) == 0
            elif node.node_type == "assertion":
                assert node.level >= 0
                if node.input_steps:
                    assert node.level >= 1

    def test_compressed_proof_z_backref(self) -> None:
        """ql.mm uses compressed proofs — Z backrefs must work."""
        parsed = _load_mm("ql.mm")
        label_info = _build_label_info(parsed)
        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]

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

        serial = []
        for lbl in theorems:
            r = build_proof_graph(parsed, lbl, label_info)
            if not isinstance(r, str):
                serial.append(r.theorem_label)

        graphs, errors = build_all_proof_graphs(parsed, theorems, max_workers=2)
        parallel = [g.theorem_label for g in graphs]

        assert set(serial) == set(parallel)

    def test_level_assignment(self) -> None:
        """Topological levels must be consistent with dependencies."""
        parsed = _load_mm("ql.mm")
        label_info = _build_label_info(parsed)
        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]

        for lbl in theorems[:50]:
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

        assert plan.max_expr_len >= 512


# ══════════════════════════════════════════════════════════════════════
#  Phase 3: Full Pipeline — GPU vs metamath-knife
# ══════════════════════════════════════════════════════════════════════


class TestFullPipeline:

    def test_demo0_full(self) -> None:
        """Full pipeline on demo0.mm: GPU must pass all knife-validated theorems."""
        path = os.path.join(DATA_DIR, "demo0.mm")
        if not os.path.exists(path):
            pytest.skip("demo0.mm not found")
        assert _knife_verify(path), "knife rejected demo0.mm"

        parsed = _load_mm("demo0.mm")
        results = verify_database(parsed, device=torch.device("cpu"), verbose=True)
        n_fail = sum(1 for v in results.values() if not v)
        assert n_fail == 0, f"{n_fail} GPU failures on demo0.mm"

    def test_ql_full(self) -> None:
        """Full pipeline on ql.mm: GPU must pass all knife-validated theorems."""
        path = os.path.join(DATA_DIR, "ql.mm")
        if not os.path.exists(path):
            pytest.skip("ql.mm not found")
        assert _knife_verify(path), "knife rejected ql.mm"

        parsed = _load_mm("ql.mm")
        results = verify_database(parsed, device=torch.device("cpu"), verbose=True)
        n_fail = sum(1 for v in results.values() if not v)
        assert n_fail == 0, f"{n_fail} GPU failures on ql.mm"

    def test_ehyp_masking_no_false_failures(self) -> None:
        """Assertions with different ehyp counts must not cause false failures.

        ql.mm has a mix of assertions with 0 and 2+ essential hyps.
        If ehyp masking is broken, we'd see false failures on assertions
        with fewer ehyps than the batch max.
        """
        parsed = _load_mm("ql.mm")
        results = verify_database(parsed, device=torch.device("cpu"), check_dv=False)
        n_fail = sum(1 for v in results.values() if not v)
        assert n_fail == 0, (
            f"{n_fail} false GPU failures on ql.mm (ehyp masking broken?)"
        )


# ══════════════════════════════════════════════════════════════════════
#  GPU device tests (when available)
# ══════════════════════════════════════════════════════════════════════


CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


class TestGPUDevice:

    @pytest.mark.skipif(not (CUDA_AVAILABLE or MPS_AVAILABLE), reason="No GPU")
    def test_demo0_on_gpu(self) -> None:
        """demo0.mm on actual GPU device — must agree with knife."""
        path = os.path.join(DATA_DIR, "demo0.mm")
        if not os.path.exists(path):
            pytest.skip("demo0.mm not found")
        assert _knife_verify(path), "knife rejected demo0.mm"

        parsed = _load_mm("demo0.mm")
        results = verify_database(parsed, verbose=True)
        n_fail = sum(1 for v in results.values() if not v)
        assert n_fail == 0

    @pytest.mark.skipif(not (CUDA_AVAILABLE or MPS_AVAILABLE), reason="No GPU")
    def test_ql_on_gpu(self) -> None:
        """ql.mm on actual GPU device — must agree with knife."""
        path = os.path.join(DATA_DIR, "ql.mm")
        if not os.path.exists(path):
            pytest.skip("ql.mm not found")
        assert _knife_verify(path), "knife rejected ql.mm"

        parsed = _load_mm("ql.mm")
        results = verify_database(parsed, verbose=True)
        n_fail = sum(1 for v in results.values() if not v)
        assert n_fail == 0, f"{n_fail} divergences on ql.mm"
