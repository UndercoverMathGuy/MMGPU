# ── MMGPU: GPU-Accelerated Metamath Proof Verification ──────────────
# Target: RunPod — NVIDIA H100 (x86_64, CUDA 12.x, 80GB HBM3)
#
# Build:
#   docker build -t mmgpu-cuda .
#
# Run on RunPod (GPU is auto-exposed, no --gpus needed):
#   docker run mmgpu-cuda
#
# Run specific test:
#   docker run mmgpu-cuda python3 -m pytest tensormm/tests/test_full_correctness.py::TestSetMMFull -v -s
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

# Copy local test data first (small files, changes rarely → cached layer)
COPY data/demo0.mm data/test_mini.mm data/

# Download set.mm and ql.mm from Metamath GitHub (excluded from build context)
RUN wget -q --show-progress -O data/set.mm \
    https://raw.githubusercontent.com/metamath/set.mm/develop/set.mm \
    && wget -q --show-progress -O data/ql.mm \
    https://raw.githubusercontent.com/metamath/set.mm/develop/ql.mm \
    && echo "set.mm: $(wc -c < data/set.mm) bytes" \
    && echo "ql.mm: $(wc -c < data/ql.mm) bytes"

# Copy project source (after data layer for better caching)
COPY pyproject.toml .
COPY tensormm/ tensormm/

# Smoke test: verify torch imports and CUDA compile capability
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA compiled: {torch.version.cuda}')"
RUN python3 -c "from tensormm.tensor_verifier import TensorVerifier; print('TensorVerifier OK')"
RUN python3 -c "from tensormm.parser import parse_mm_file; p = parse_mm_file('data/demo0.mm'); print(f'Parser OK: {len(p.assertions)} assertions')"

# Default: run full test suite (all 74+ tests including full set.mm)
ENTRYPOINT ["python3", "-m", "pytest"]
CMD ["tensormm/tests/", "-v", "-s", "--tb=short"]
