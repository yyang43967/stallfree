"""Shared Modal definitions: pinned images, the weights volume, and the model revision."""

from __future__ import annotations

import modal

PYTHON = "3.12"
MODEL_REPO = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
WEIGHTS_DIR = "/weights"
MODEL_DIR = f"{WEIGHTS_DIR}/qwen3-0.6b-{MODEL_REVISION[:7]}"

# Development GPUs: L4 preferred, A10 accepted when no L4 is available. Every result records
# which one ran. The benchmark uses L40S.
GPU = ["L4", "A10"]

# Prebuilt FlashAttention-2 wheel for torch 2.10 / CUDA 13 / CPython 3.12, pinned by hash.
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3%2Bcu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    "#sha256=910d8db9def162de5b7c15474b933e7e2371e93733b980e9d3c07cd3bf2f568e"
)

# Versions match uv.lock, except torch, which comes from the CUDA 13 index here.
PINNED = [
    "transformers==5.23.0",
    "tokenizers==0.23.3",
    "safetensors==0.8.0",
    "huggingface-hub==2.4.0",
    "numpy==2.5.5",
    "pytest==9.2.0",
    "fastapi==0.163.2",
    "uvicorn==0.54.0",
    "openai==3.53.0",
    "httpx==0.28.1",
]

weights_volume = modal.Volume.from_name("stallfree-weights", create_if_missing=True)

download_image = modal.Image.debian_slim(python_version=PYTHON).pip_install(
    "huggingface-hub==2.4.0"
)

gpu_image = (
    modal.Image.debian_slim(python_version=PYTHON)
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install(*PINNED)
    .env({"HF_HUB_OFFLINE": "1", "STALLFREE_MODEL_DIR": MODEL_DIR})
    .add_local_file("pyproject.toml", "/root/pyproject.toml")
    .add_local_dir("tests", "/root/tests")
    .add_local_python_source("stallfree")
)
