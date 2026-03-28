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

# Install Python 3.12 + pip + wget + Rust toolchain deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    g++ gcc wget curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed to build metamath-knife)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Build and install metamath-knife (the Rust reference verifier — used as oracle in tests)
RUN cargo install --git https://github.com/metamath/metamath-knife --locked

WORKDIR /app

# Install PyTorch with CUDA 12.6 support + deps (includes maturin for Rust ext)
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu126 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
    numpy pytest ninja maturin numba

# Build and install the Rust extension (mmgpu_rs) with maturin
# Copied early so Rust builds are cached unless rust_ext/ changes
COPY rust_ext/ rust_ext/
RUN cd rust_ext && maturin build --release --interpreter python3 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
    target/wheels/mmgpu_rs-*.whl \
    && rm -rf target

# Download set.mm from Metamath GitHub
RUN mkdir -p data \
    && wget -q --show-progress -O data/set.mm \
    https://raw.githubusercontent.com/metamath/set.mm/develop/set.mm \
    && echo "set.mm: $(wc -c < data/set.mm) bytes"

# Download test files + wheeler-tests suite from david-a-wheeler/metamath-test
ENV MM_TEST=https://raw.githubusercontent.com/david-a-wheeler/metamath-test/master
RUN wget -q -O data/demo0.mm    $MM_TEST/demo0.mm \
    && wget -q -O data/ql.mm    $MM_TEST/ql.mm \
    && wget -q -O data/anatomy.mm $MM_TEST/anatomy.mm \
    && mkdir -p data/wheeler-tests \
    && for f in \
    anatomy.mm anatomy-bad1.mm anatomy-bad2.mm anatomy-bad3.mm \
    big-unifier.mm big-unifier-bad1.mm big-unifier-bad2.mm big-unifier-bad3.mm \
    demo0.mm demo0-bad1.mm demo0-includee.mm demo0-includer.mm \
    emptyline.mm hol.mm iset.mm miu.mm nf.mm peano-fixed.mm ql.mm \
    set-dist-bad1.mm set.2010-08-29.mm; do \
    wget -q -O "data/wheeler-tests/$f" "$MM_TEST/$f"; \
    done \
    && echo "wheeler-tests: $(ls data/wheeler-tests/ | wc -l) files"

# Copy project source
COPY pyproject.toml .
COPY conftest.py .
COPY tensormm/ tensormm/
COPY run_setmm.py .
COPY run_all.py .

# Default: run set.mm verification + full test suite
CMD ["python3", "run_all.py"]
