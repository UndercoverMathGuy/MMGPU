"""Fused Metal compute kernel for Metamath proof verification.

Replaces the PyTorch MPS verify_flat pipeline (which dispatches ~30 × P_max
GPU ops per chunk) with a SINGLE Metal compute dispatch per chunk.

Each GPU thread handles one verification step:
  gather → prefix_sum → scatter → reduce
all in thread-local memory.  No intermediate device-memory tensors.

Uses pyobjc to call the Metal 3 API directly from Python.
Metal 4 MTL4* types will be a drop-in upgrade once pyobjc binds them.
"""

from __future__ import annotations

import ctypes
import struct
from typing import Optional

import torch

try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


# ── MSL kernel source ────────────────────────────────────────────────
# One thread per step (thread_position_in_grid.x = step index).
# Buffers:
#   0: patterns      [N, P_max]   int32  (row-major)
#   1: pat_lengths    [N]          int32
#   2: sub_tables     [N, V, S_max] int32 (row-major)
#   3: sub_lengths    [N, V]       int32
#   4: targets        [N, T_max]   int32
#   5: tgt_lengths    [N]          int32
#   6: results        [N]          uint8  (1=pass, 0=fail)
#   7: params         {N, P_max, V, S_max, T_max}  int32[5]

_MSL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct Params {
    int N;
    int P_max;
    int V;
    int S_max;
    int T_max;
};

kernel void verify_fused(
    device const int*    patterns    [[buffer(0)]],
    device const int*    pat_lengths [[buffer(1)]],
    device const int*    sub_tables  [[buffer(2)]],
    device const int*    sub_lengths [[buffer(3)]],
    device const int*    targets     [[buffer(4)]],
    device const int*    tgt_lengths [[buffer(5)]],
    device uchar*        results     [[buffer(6)]],
    device const Params* params      [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    const int N     = params->N;
    const int P_max = params->P_max;
    const int V     = params->V;
    const int S_max = params->S_max;
    const int T_max = params->T_max;

    if ((int)tid >= N) return;

    const int n = (int)tid;
    const int pat_len = pat_lengths[n];
    const int tgt_len = tgt_lengths[n];

    // Pointers into row-major arrays
    device const int* my_pat = patterns + n * P_max;
    device const int* my_sub = sub_tables + n * V * S_max;
    device const int* my_sl  = sub_lengths + n * V;
    device const int* my_tgt = targets + n * T_max;

    // Phase 1+2+3: gather, compute output length, and compare on-the-fly.
    // We don't need to materialize the full output — just stream-compare
    // against the target as we generate tokens.
    int out_pos = 0;
    bool match = true;

    for (int p = 0; p < pat_len && match; p++) {
        int tok = my_pat[p];
        int rep_len = my_sl[tok];
        device const int* rep = my_sub + tok * S_max;

        for (int s = 0; s < rep_len && match; s++) {
            if (out_pos >= tgt_len) {
                match = false;
            } else {
                if (rep[s] != my_tgt[out_pos]) {
                    match = false;
                }
                out_pos++;
            }
        }
    }

    // Length check: output must be exactly tgt_len
    if (out_pos != tgt_len) match = false;

    results[n] = match ? 1 : 0;
}
"""


class MetalVerifier:
    """Fused Metal compute kernel for batch verification.

    Drop-in replacement for TensorVerifier.verify_flat() with ~50-100×
    fewer GPU dispatches (1 per chunk instead of ~30 × P_max).
    """

    def __init__(self) -> None:
        if not METAL_AVAILABLE:
            raise RuntimeError("pyobjc-framework-Metal not available")

        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device found")

        # Compile MSL at init time (one-time cost ~5ms)
        lib, err = self.device.newLibraryWithSource_options_error_(
            _MSL_SOURCE, None, None
        )
        if lib is None:
            raise RuntimeError(f"MSL compilation failed: {err}")

        func = lib.newFunctionWithName_("verify_fused")
        if func is None:
            raise RuntimeError("verify_fused function not found in compiled library")

        self._pso, err = self.device.newComputePipelineStateWithFunction_error_(
            func, None
        )
        if self._pso is None:
            raise RuntimeError(f"Pipeline state creation failed: {err}")

        self._queue = self.device.newCommandQueue()
        self._max_threads = self._pso.maxTotalThreadsPerThreadgroup()

    # ── Public API ────────────────────────────────────────────────────

    def verify_flat(
        self,
        patterns: torch.Tensor,
        pattern_lengths: torch.Tensor,
        sub_tables: torch.Tensor,
        sub_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Verify N heterogeneous steps in ONE Metal dispatch.

        Signature matches TensorVerifier.verify_flat() exactly.
        All inputs can be on any device (MPS, CPU) — data is copied to
        shared Metal buffers.  Returns [N] bool tensor on CPU.

        Args:
            patterns:       [N, P_max] int32
            pattern_lengths:[N] int32
            sub_tables:     [N, V, S_max] int32
            sub_lengths:    [N, V] int32
            targets:        [N, T_max] int32
            target_lengths: [N] int32
        """
        N = patterns.shape[0]
        if N == 0:
            return torch.empty(0, dtype=torch.bool)

        P_max = patterns.shape[1]
        V = sub_tables.shape[1]
        S_max = sub_tables.shape[2]
        T_max = targets.shape[1]

        # Move all tensors to CPU contiguous int32
        def _to_bytes(t: torch.Tensor) -> bytes:
            return t.detach().cpu().contiguous().to(torch.int32).numpy().tobytes()

        pat_bytes = _to_bytes(patterns)
        pl_bytes = _to_bytes(pattern_lengths)
        st_bytes = _to_bytes(sub_tables)
        sl_bytes = _to_bytes(sub_lengths)
        tgt_bytes = _to_bytes(targets)
        tl_bytes = _to_bytes(target_lengths)

        # Params struct: {N, P_max, V, S_max, T_max} as int32[5]
        params_bytes = struct.pack("5i", N, P_max, V, S_max, T_max)

        shared = Metal.MTLResourceStorageModeShared

        # Create Metal buffers
        buf_pat = self.device.newBufferWithBytes_length_options_(pat_bytes, len(pat_bytes), shared)
        buf_pl = self.device.newBufferWithBytes_length_options_(pl_bytes, len(pl_bytes), shared)
        buf_st = self.device.newBufferWithBytes_length_options_(st_bytes, len(st_bytes), shared)
        buf_sl = self.device.newBufferWithBytes_length_options_(sl_bytes, len(sl_bytes), shared)
        buf_tgt = self.device.newBufferWithBytes_length_options_(tgt_bytes, len(tgt_bytes), shared)
        buf_tl = self.device.newBufferWithBytes_length_options_(tl_bytes, len(tl_bytes), shared)
        buf_params = self.device.newBufferWithBytes_length_options_(params_bytes, len(params_bytes), shared)

        # Output buffer: N bytes (uint8)
        buf_results = self.device.newBufferWithLength_options_(N, shared)

        # Encode and dispatch
        cb = self._queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(self._pso)

        enc.setBuffer_offset_atIndex_(buf_pat, 0, 0)
        enc.setBuffer_offset_atIndex_(buf_pl, 0, 1)
        enc.setBuffer_offset_atIndex_(buf_st, 0, 2)
        enc.setBuffer_offset_atIndex_(buf_sl, 0, 3)
        enc.setBuffer_offset_atIndex_(buf_tgt, 0, 4)
        enc.setBuffer_offset_atIndex_(buf_tl, 0, 5)
        enc.setBuffer_offset_atIndex_(buf_results, 0, 6)
        enc.setBuffer_offset_atIndex_(buf_params, 0, 7)

        # Grid: one thread per step
        grid = Metal.MTLSizeMake(N, 1, 1)
        tpg = Metal.MTLSizeMake(min(N, self._max_threads), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)

        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        # Read results
        raw = buf_results.contents().as_buffer(N)
        result_bytes = bytes(raw)
        result_list = [bool(b) for b in result_bytes]
        return torch.tensor(result_list, dtype=torch.bool)
