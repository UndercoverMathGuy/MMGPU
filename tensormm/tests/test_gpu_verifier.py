"""Tests for tensormm.gpu_verifier — graph construction, level packing, GPU execution.

Full-pipeline correctness is validated against metamath-knife as the oracle.
"""
from __future__ import annotations

import os
import subprocess
import shutil
import time

import pytest
import torch

from tensormm.gpu_verifier import (
    build_proof_graph,
    build_all_proof_graphs,
    build_all_proof_graphs_rs,
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
            assert result.num_nodes > 0
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

        from tensormm.gpu_verifier import NODE_PUSH_F, NODE_PUSH_E, NODE_ASSERTION
        for i in range(g.num_nodes):
            nt = g.node_types[i]
            if nt == NODE_PUSH_F or nt == NODE_PUSH_E:
                assert g.node_levels[i] == 0
                # push nodes have no inputs in CSR
                assert g.input_offsets[i] == g.input_offsets[i + 1]
            elif nt == NODE_ASSERTION:
                assert g.node_levels[i] >= 0
                n_inputs = g.input_offsets[i + 1] - g.input_offsets[i]
                if n_inputs > 0:
                    assert g.node_levels[i] >= 1

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
            from tensormm.gpu_verifier import NODE_ASSERTION
            # node_levels[i] == g.node_levels[i] (step_idx == array index)
            for i in range(g.num_nodes):
                if g.node_types[i] == NODE_ASSERTION:
                    lvl = int(g.node_levels[i])
                    inp_start = int(g.input_offsets[i])
                    inp_end   = int(g.input_offsets[i + 1])
                    for k in range(inp_start, inp_end):
                        dep_step = int(g.input_data[k])
                        dep_lvl  = int(g.node_levels[dep_step])
                        assert dep_lvl < lvl, (
                            f"Level violation in {lbl}: node {i} at level "
                            f"{lvl} depends on node {dep_step} at level {dep_lvl}"
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
        n_fail = sum(1 for v in results.values() if v is not None)
        assert n_fail == 0, f"{n_fail} GPU failures on demo0.mm"

    def test_ql_full(self) -> None:
        """Full pipeline on ql.mm: GPU must pass all knife-validated theorems."""
        path = os.path.join(DATA_DIR, "ql.mm")
        if not os.path.exists(path):
            pytest.skip("ql.mm not found")
        assert _knife_verify(path), "knife rejected ql.mm"

        parsed = _load_mm("ql.mm")
        results = verify_database(parsed, device=torch.device("cpu"), verbose=True)
        n_fail = sum(1 for v in results.values() if v is not None)
        assert n_fail == 0, f"{n_fail} GPU failures on ql.mm"

    def test_ehyp_masking_no_false_failures(self) -> None:
        """Assertions with different ehyp counts must not cause false failures.

        ql.mm has a mix of assertions with 0 and 2+ essential hyps.
        If ehyp masking is broken, we'd see false failures on assertions
        with fewer ehyps than the batch max.
        """
        parsed = _load_mm("ql.mm")
        results = verify_database(parsed, device=torch.device("cpu"), check_dv=False)
        n_fail = sum(1 for v in results.values() if v is not None)
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
        n_fail = sum(1 for v in results.values() if v is not None)
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
        n_fail = sum(1 for v in results.values() if v is not None)
        assert n_fail == 0, f"{n_fail} divergences on ql.mm"


# ══════════════════════════════════════════════════════════════════════
#  set.mm CPU preprocessing benchmark
# ══════════════════════════════════════════════════════════════════════


class TestSetMMBenchmark:
    """Timed CPU preprocessing benchmark on the full set.mm database.

    Skipped when set.mm is absent.  Prints phase-by-phase wall times and
    asserts basic sanity (node count, zero pack errors) but does NOT run
    GPU execution — this is purely a preprocessing throughput test.
    """

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(DATA_DIR, "set.mm")),
        reason="set.mm not found in data/",
    )
    def test_setmm_preprocessing_timing(self, capsys) -> None:
        """Parse → graph construction → level packing on full set.mm, with timing."""
        set_mm_path = os.path.join(DATA_DIR, "set.mm")

        # ── Parse ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        parsed = parse_mm_file(set_mm_path)
        t_parse = time.perf_counter() - t0

        tokenizer = Tokenizer()
        for c in parsed.constants:
            tokenizer.encode_symbol(c)
        for v in parsed.variables:
            tokenizer.encode_symbol(v)

        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
        n_theorems = len(theorems)

        # ── Phase 1a: Rust graph construction (with tokenizer for pre-encoding)
        t0 = time.perf_counter()
        graphs_rs, errors_rs = build_all_proof_graphs_rs(
            parsed, theorems, tokenizer=tokenizer, verbose=True
        )
        t_phase1_rs = time.perf_counter() - t0

        # ── Phase 1b: Python graph construction (for comparison) ───────
        t0 = time.perf_counter()
        graphs, errors = build_all_proof_graphs(parsed, theorems, verbose=False)
        t_phase1_py = time.perf_counter() - t0

        # Use Rust graphs for Phase 2
        t0 = time.perf_counter()
        plan = pack_levels(graphs_rs, parsed, tokenizer, verbose=True)
        t_phase2 = time.perf_counter() - t0

        t_total_rs = t_parse + t_phase1_rs + t_phase2

        with capsys.disabled():
            print(f"\n{'═'*60}")
            print(f"  set.mm preprocessing benchmark")
            print(f"{'─'*60}")
            print(f"  theorems       : {n_theorems:>10,}")
            print(f"  total nodes    : {plan.total_nodes:>10,}")
            print(f"  graph errors   : {len(errors_rs):>10,}")
            print(f"{'─'*60}")
            print(f"  parse          : {t_parse:>8.2f}s")
            print(f"  phase 1 Rust   : {t_phase1_rs:>7.2f}s  ({n_theorems/t_phase1_rs:,.0f} theorems/s)")
            print(f"  phase 1 Python : {t_phase1_py:>7.2f}s  ({n_theorems/t_phase1_py:,.0f} theorems/s)")
            print(f"  phase 2 (pack) : {t_phase2:>8.2f}s  ({plan.total_nodes/t_phase2:,.0f} nodes/s)")
            print(f"  total (Rust p1): {t_total_rs:>8.2f}s")
            print(f"{'═'*60}")

        assert plan.total_nodes > 1_000_000, f"Expected >1M nodes for set.mm, got {plan.total_nodes}"
        assert len(errors_rs) == 0, f"{len(errors_rs)} Rust graph build errors on set.mm"
        assert plan.num_proofs == len(graphs_rs)
