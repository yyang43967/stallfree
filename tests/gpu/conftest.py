"""GPU tests run only on a CUDA machine (on Modal); everything here is marked ``gpu``."""

from __future__ import annotations

import os

import pytest
import torch


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "tests/gpu/" in str(item.fspath):
            item.add_marker(pytest.mark.gpu)
            if not torch.cuda.is_available():
                item.add_marker(pytest.mark.skip(reason="needs CUDA"))


@pytest.fixture(scope="session")
def model_dir() -> str:
    path = os.environ.get("STALLFREE_MODEL_DIR")
    if not path:
        pytest.skip("STALLFREE_MODEL_DIR not set")
    return path
