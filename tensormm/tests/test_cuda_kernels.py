"""Tests for tensormm.cuda_kernels — custom CUDA kernel correctness.

Tests run on CUDA GPU only (skipped if unavailable).  Full-pipeline tests
compare the CUDA path against metamath-knife as the ground-truth oracle.
Torch-path vs CUDA-path equivalence tests remain as internal consistency checks.
"""
from __future__ import annotations

import os
import subprocess
import shutil

import numpy as np
import pytest
import torch

from tensormm.gpu_verifier import (
    build_all_proof_graphs,
    pack_levels,
    verify_database,
    _run_gpu_pipeline,
    _run_gpu_pipeline_cuda,
)
from tensormm.parser import parse_mm_file
from tensormm.tokenizer import Tokenizer
from tensormm import cuda_kernels as _cuda_mod


CUDA_AVAILABLE = torch.cuda.is_available()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="No CUDA GPU")


def _load_mm(name: str):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not found in data/")
    return parse_mm_file(path)


def _build_plan(parsed, theorems=None):
    """Build a GlobalPlan from a parsed database."""
    if theorems is None:
        theorems = [lbl for lbl, a in parsed.assertions.items() if a.type == "theorem"]
    tokenizer = Tokenizer()
    for c in parsed.constants:
        tokenizer.encode_symbol(c)
    for v in parsed.variables:
        tokenizer.encode_symbol(v)
    graphs, _ = build_all_proof_graphs(parsed, theorems)
    plan = pack_levels(graphs, parsed, tokenizer)
    return plan, graphs


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
#  Kernel availability
# ══════════════════════════════════════════════════════════════════════


class TestCudaKernelAvailability:

    def test_jit_compiles(self) -> None:
        """Custom CUDA kernels should JIT-compile on a CUDA machine."""
        assert _cuda_mod.is_available(), (
            "CUDA kernels failed to compile — check nvcc / CUDA toolkit"
        )

    def test_module_has_launchers(self) -> None:
        """Compiled module should expose all 3 launcher functions."""
        mod = _cuda_mod.get_module()
        assert mod is not None
        assert hasattr(mod, "push_nodes_launch")
        assert hasattr(mod, "execute_assertion_launch")
        assert hasattr(mod, "final_check_launch")


# ══════════════════════════════════════════════════════════════════════
#  CUDA vs Torch path equivalence (internal consistency)
# ══════════════════════════════════════════════════════════════════════


class TestCudaVsTorch:

    def test_demo0_equivalence(self) -> None:
        """CUDA and torch paths must produce identical results on demo0.mm."""
        parsed = _load_mm("demo0.mm")
        plan, _ = _build_plan(parsed)
        device = torch.device("cuda")

        torch_result, _, _ = _run_gpu_pipeline(
            plan, torch.device("cpu"), plan.max_expr_len, verbose=False
        )
        cuda_result, _, _ = _run_gpu_pipeline_cuda(
            plan, device, plan.max_expr_len, verbose=False
        )

        np.testing.assert_array_equal(
            cuda_result, torch_result,
            err_msg="CUDA and torch paths diverge on demo0.mm"
        )

    def test_ql_equivalence(self) -> None:
        """CUDA and torch paths must produce identical results on ql.mm."""
        parsed = _load_mm("ql.mm")
        plan, _ = _build_plan(parsed)
        device = torch.device("cuda")

        torch_result, _, _ = _run_gpu_pipeline(
            plan, torch.device("cpu"), plan.max_expr_len, verbose=False
        )
        cuda_result, _, _ = _run_gpu_pipeline_cuda(
            plan, device, plan.max_expr_len, verbose=False
        )

        np.testing.assert_array_equal(
            cuda_result, torch_result,
            err_msg="CUDA and torch paths diverge on ql.mm"
        )

    def test_anatomy_equivalence(self) -> None:
        """CUDA and torch paths must produce identical results on anatomy.mm."""
        parsed = _load_mm("anatomy.mm")
        plan, _ = _build_plan(parsed)
        device = torch.device("cuda")

        torch_result, _, _ = _run_gpu_pipeline(
            plan, torch.device("cpu"), plan.max_expr_len, verbose=False
        )
        cuda_result, _, _ = _run_gpu_pipeline_cuda(
            plan, device, plan.max_expr_len, verbose=False
        )

        np.testing.assert_array_equal(
            cuda_result, torch_result,
            err_msg="CUDA and torch paths diverge on anatomy.mm"
        )


# ══════════════════════════════════════════════════════════════════════
#  Full pipeline correctness (CUDA path vs metamath-knife oracle)
# ══════════════════════════════════════════════════════════════════════


class TestCudaFullPipeline:

    def test_demo0_vs_knife(self) -> None:
        """Full CUDA pipeline on demo0.mm must agree with metamath-knife."""
        path = os.path.join(DATA_DIR, "demo0.mm")
        assert _knife_verify(path), "knife rejected demo0.mm"

        parsed = _load_mm("demo0.mm")
        results = verify_database(parsed, device=torch.device("cuda"), verbose=False)
        n_fail = sum(1 for v in results.values() if not v)
        assert n_fail == 0, f"{n_fail} CUDA failures on demo0.mm"

    def test_anatomy_vs_knife(self) -> None:
        """Full CUDA pipeline on anatomy.mm must agree with metamath-knife."""
        path = os.path.join(DATA_DIR, "anatomy.mm")
        if not os.path.exists(path):
            pytest.skip("anatomy.mm not found")
        assert _knife_verify(path), "knife rejected anatomy.mm"

        parsed = _load_mm("anatomy.mm")
        results = verify_database(parsed, device=torch.device("cuda"), verbose=False)
        n_fail = sum(1 for v in results.values() if not v)
        assert n_fail == 0, f"{n_fail} CUDA failures on anatomy.mm"

    def test_ql_vs_knife(self) -> None:
        """Full CUDA pipeline on ql.mm must agree with metamath-knife."""
        path = os.path.join(DATA_DIR, "ql.mm")
        assert _knife_verify(path), "knife rejected ql.mm"

        parsed = _load_mm("ql.mm")
        results = verify_database(parsed, device=torch.device("cuda"), verbose=False)
        n_fail = sum(1 for v in results.values() if not v)
        assert n_fail == 0, f"{n_fail} CUDA failures on ql.mm"


# ══════════════════════════════════════════════════════════════════════
#  Memory usage verification
# ══════════════════════════════════════════════════════════════════════


class TestCudaMemory:

    def test_no_intermediate_allocations(self) -> None:
        """CUDA path should use only expr_buffer + tracking + table memory."""
        parsed = _load_mm("ql.mm")
        plan, _ = _build_plan(parsed)
        device = torch.device("cuda")

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        _run_gpu_pipeline_cuda(plan, device, plan.max_expr_len, verbose=False)

        peak_bytes = torch.cuda.max_memory_allocated(device)
        expected_base = (
            plan.total_expr_tokens * 2
            + plan.total_nodes * 13
            + (plan.total_nodes + 1) * 8
        )
        assert peak_bytes < expected_base * 3, (
            f"Peak GPU memory {peak_bytes / 1e9:.2f} GB exceeds "
            f"3x expected base {expected_base * 3 / 1e9:.2f} GB — "
            f"intermediate allocations may be leaking"
        )
