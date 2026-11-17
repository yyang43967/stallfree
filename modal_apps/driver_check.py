"""Report the GPU, driver, and CUDA version of a Modal L4 host (minimal image, no torch).

uv run modal run modal_apps/driver_check.py
"""

from __future__ import annotations

import subprocess

import modal

from modal_apps.common import GPU, PYTHON

app = modal.App("stallfree-driver-check")


@app.function(image=modal.Image.debian_slim(python_version=PYTHON), gpu=GPU, timeout=120)
def driver_info() -> str:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    header = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout
    cuda = next((line for line in header.splitlines() if "CUDA Version" in line), "").strip()
    return f"{query}\n{cuda}"


@app.local_entrypoint()
def main() -> None:
    print(driver_info.remote())
