"""Load a Hugging Face Qwen3 checkpoint (safetensors) into ``Qwen3ForCausalLM``."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from stallfree.model.qwen3 import Qwen3Config, Qwen3ForCausalLM

# Present in checkpoints of tied models in some exports; the tied model has no separate parameter.
_TIED_LM_HEAD = "lm_head.weight"


def load_model(
    model_dir: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> Qwen3ForCausalLM:
    """Build the model from ``config.json`` and copy every tensor from ``*.safetensors`` by name.

    Raises:
        ValueError: if the checkpoint lacks a parameter the model has, has a tensor the model
            does not, or a shape differs.
    """
    model_dir = Path(model_dir)
    cfg = Qwen3Config.from_pretrained(model_dir)
    with torch.device(device):
        model = Qwen3ForCausalLM(cfg)
    model.to(dtype=dtype)
    params = dict(model.named_parameters())
    loaded: set[str] = set()

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no .safetensors files in {model_dir}")
    for path in files:
        with safe_open(str(path), framework="pt", device=str(device)) as f:
            for name in f.keys():  # noqa: SIM118  (safe_open has no __iter__)
                if name == _TIED_LM_HEAD and cfg.tie_word_embeddings:
                    continue
                if name not in params:
                    raise ValueError(f"unexpected tensor {name!r} in {path.name}")
                tensor = f.get_tensor(name)
                param = params[name]
                if tensor.shape != param.shape:
                    raise ValueError(f"{name}: shape {tuple(tensor.shape)} != {tuple(param.shape)}")
                with torch.no_grad():
                    param.copy_(tensor.to(dtype))
                loaded.add(name)

    missing = sorted(set(params) - loaded)
    if missing:
        raise ValueError(f"checkpoint is missing {missing}")
    model.eval()
    return model
