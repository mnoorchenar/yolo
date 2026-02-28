FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV / ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Install CPU-only PyTorch BEFORE requirements.txt ──────────────────────────
# Without this, pip resolves torch from PyPI and pulls ~2 GB of NVIDIA CUDA
# wheels (cublas, cudnn, nccl, cusparse…) that are useless on a CPU-only Space.
# Installing from the PyTorch CPU wheel index first satisfies the torch
# dependency so the next pip install step never touches CUDA packages.
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Remaining Python deps ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

# Create writable runtime dirs
RUN mkdir -p data/uploads data/datasets runs static/results

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "app.py"]