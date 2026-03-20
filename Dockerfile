# ── MMGPU: GPU-Accelerated Metamath Proof Verification ──────────────
# Target: RunPod — NVIDIA H100 (x86_64, CUDA 12.x, 80GB HBM3)
#
# Build:
#   docker build -t mmgpu-cuda .
#
# Run on RunPod (GPU is auto-exposed, no --gpus needed):
#   docker run mmgpu-cuda
# ────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python 3.12 + pip + wget
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.6 support + deps
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu126 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
    numpy pytest

# Download set.mm from Metamath GitHub
RUN mkdir -p data \
    && wget -q --show-progress -O data/set.mm \
    https://raw.githubusercontent.com/metamath/set.mm/develop/set.mm \
    && echo "set.mm: $(wc -c < data/set.mm) bytes"

# Copy project source (after data layer for better caching)
COPY pyproject.toml .
COPY tensormm/ tensormm/

# Smoke test: verify torch + project imports work
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA compiled: {torch.version.cuda}')"
RUN python3 -c "from tensormm.tensor_verifier import TensorVerifier; print('TensorVerifier OK')"

# Default: run ONLY the full set.mm verification on GPU
ENTRYPOINT ["python3", "-m", "pytest"]
CMD ["tensormm/tests/test_full_correctness.py::TestSetMMFull::test_gpu_all_set_mm", "-v", "-s", "--tb=short"]
