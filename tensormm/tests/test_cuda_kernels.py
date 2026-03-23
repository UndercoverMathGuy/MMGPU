"""Tests for tensormm.cuda_kernels — custom CUDA kernel correctness.

Tests run on CUDA GPU only (skipped if unavailable). Each test compares
the CUDA kernel output against the existing PyTorch tensor-op path as
an oracle, ensuring bit-exact equivalence.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from tensormm.cpu_verifier import CPUVerifier
from tensormm.gpu_verifier import (
    build_all_proof_graphs,
    pack_levels,
    verify_database,
    _build_label_info,
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
    graphs, errors = build_all_proof_graphs(parsed, theorems)
    plan = pack_levels(graphs, parsed, tokenizer)
    return plan, graphs


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
#  CUDA vs Torch path equivalence
# ══════════════════════════════════════════════════════════════════════


class TestCudaVsTorch:

    def test_demo0_equivalence(self) -> None:
        """CUDA and torch paths must produce identical results on demo0.mm."""
        parsed = _load_mm("demo0.mm")
        plan, _ = _build_plan(parsed)
        device = torch.device("cuda")

        # Run torch path
        torch_result, torch_trunc, torch_max = _run_gpu_pipeline(
            plan, torch.device("cpu"), plan.max_expr_len, verbose=False
        )

        # Run CUDA path
        cuda_result, cuda_trunc, cuda_max = _run_gpu_pipeline_cuda(
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


# ══════════════════════════════════════════════════════════════════════
#  Full pipeline correctness (CUDA path vs CPU oracle)
# ══════════════════════════════════════════════════════════════════════


class TestCudaFullPipeline:

    def test_demo0_vs_cpu(self) -> None:
        """Full CUDA pipeline on demo0.mm must match CPU verifier."""
        parsed = _load_mm("demo0.mm")
        device = torch.device("cuda")

        gpu_results = verify_database(parsed, device=device, verbose=False)
        cpu_v = CPUVerifier(parsed)
        cpu_results = cpu_v.verify_all()

        for lbl, cpu_r in cpu_results.items():
            assert lbl in gpu_results, f"GPU missing theorem {lbl}"
            assert gpu_results[lbl] == cpu_r.success, (
                f"Divergence on {lbl}: GPU={gpu_results[lbl]}, "
                f"CPU={cpu_r.success} (err={cpu_r.error_message})"
            )

    def test_ql_vs_cpu(self) -> None:
        """Full CUDA pipeline on ql.mm must match CPU verifier."""
        parsed = _load_mm("ql.mm")
        device = torch.device("cuda")

        gpu_results = verify_database(parsed, device=device, verbose=False)
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


# ══════════════════════════════════════════════════════════════════════
#  Memory usage verification
# ══════════════════════════════════════════════════════════════════════


class TestCudaMemory:

    def test_no_intermediate_allocations(self) -> None:
        """CUDA path should use only expr_buffer + tracking + table memory.

        Verifies that peak GPU memory stays within expected bounds (no
        multi-GB intermediate tensors from gather/scatter/replication).
        """
        parsed = _load_mm("ql.mm")
        plan, _ = _build_plan(parsed)
        device = torch.device("cuda")

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        _run_gpu_pipeline_cuda(plan, device, plan.max_expr_len, verbose=False)

        peak_bytes = torch.cuda.max_memory_allocated(device)
        # Expected: expr_buffer + tracking + assertion table
        # expr_buffer: total_nodes × max_expr_len × 4
        # tracking: total_nodes × 13
        # assertion table: small relative to expr_buffer
        expected_base = plan.total_nodes * (plan.max_expr_len * 4 + 13)
        # Allow 2x for assertion table + small temporaries
        assert peak_bytes < expected_base * 2, (
            f"Peak GPU memory {peak_bytes / 1e9:.2f} GB exceeds "
            f"2x expected base {expected_base * 2 / 1e9:.2f} GB — "
            f"intermediate allocations may be leaking"
        )
