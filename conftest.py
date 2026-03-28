"""Session-level pytest configuration.

Warms up Numba JIT kernels once before any test runs.  Because the kernels use
cache=True the compiled LLVM is written to __pycache__ on first call and
reloaded on subsequent runs — so this warmup is only slow the very first time.
"""
from __future__ import annotations


def pytest_configure(config):  # noqa: ARG001
    """Pre-compile Numba kernels at session start so tests don't pay JIT cost."""
    from tensormm.gpu_verifier import warmup_numba
    warmup_numba()
