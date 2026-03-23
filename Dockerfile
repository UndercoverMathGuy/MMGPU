# ── MMGPU: GPU-Accelerated Metamath Proof Verification ──────────────
# Target: RunPod — NVIDIA H100 (x86_64, CUDA 12.x, 80GB HBM3)
#
# Build:
#   docker build -t mmgpu-cuda .
#
# Run on RunPod (GPU is auto-exposed, no --gpus needed):
#   docker run mmgpu-cuda
# ────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTORCH_ALLOC_CONF=expandable_segments:True

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

# Copy project source
COPY pyproject.toml .
COPY tensormm/ tensormm/
COPY run_setmm.py .

# Default: run full set.mm GPU verification (no pytest, no silent skips)
CMD ["python3", "run_setmm.py"]
