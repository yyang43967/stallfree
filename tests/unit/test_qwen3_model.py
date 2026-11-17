"""Qwen3 forward pass and weight loading against Hugging Face transformers (fp32, CPU).

Attention here is ``DenseCausalAttention`` (no cache); the paged backend is tested separately.
Contract: the predicted token at every position is identical, and logits agree within
``torch.testing.assert_close``'s default fp32 tolerances.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from helpers.tiny_qwen3 import (
    TINY_CONFIG,
    DenseCausalAttention,
    hf_greedy,
    hf_logits,
    make_hf_model,
    save_hf_model,
)
from safetensors.torch import load_file, save_file
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from stallfree.model.loader import load_model
from stallfree.model.qwen3 import Qwen3Config, Qwen3ForCausalLM


@pytest.fixture(scope="module")
def hf_model() -> HFQwen3ForCausalLM:
    return make_hf_model(seed=0)


@pytest.fixture(scope="module")
def model_dir(hf_model: HFQwen3ForCausalLM, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_hf_model(hf_model, tmp_path_factory.mktemp("tiny_qwen3"))


@pytest.fixture(scope="module")
def model(model_dir: Path) -> Qwen3ForCausalLM:
    return load_model(model_dir, dtype=torch.float32)


def random_ids(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(TINY_CONFIG["vocab_size"]) for _ in range(n)]  # type: ignore[call-overload]


def our_logits(model: Qwen3ForCausalLM, seqs: list[list[int]]) -> list[torch.Tensor]:
    """Run all sequences as one flat batch; return per-sequence logits for every position."""
    flat = torch.tensor([t for s in seqs for t in s])
    positions = torch.cat([torch.arange(len(s)) for s in seqs])
    with torch.no_grad():
        attn = DenseCausalAttention([len(s) for s in seqs])
        logits = model.compute_logits(model(flat, positions, attn))
    return list(logits.split([len(s) for s in seqs]))


def assert_matches(got: torch.Tensor, ref: torch.Tensor) -> None:
    assert torch.equal(got.argmax(-1), ref.argmax(-1)), "predicted tokens differ"
    torch.testing.assert_close(got, ref)  # default fp32 tolerances


# --- config ---------------------------------------------------------------------------------

# Fields of Qwen/Qwen3-0.6B config.json (revision c1899de), older format with top-level rope_theta.
REAL_QWEN3_06B = {
    "attention_bias": False,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "max_position_embeddings": 40960,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "tie_word_embeddings": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


def test_config_parses_real_qwen3_0_6b() -> None:
    cfg = Qwen3Config.from_dict(REAL_QWEN3_06B)
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.head_dim) == (28, 1024, 128)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_kv_groups) == (16, 8, 2)
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.tie_word_embeddings


def test_config_parses_rope_parameters_format() -> None:
    cfg = Qwen3Config.from_dict(
        {
            **REAL_QWEN3_06B,
            "rope_theta": None,
            "rope_parameters": {"rope_theta": 5000.0, "rope_type": "default"},
        }
    )
    assert cfg.rope_theta == 5000.0


@pytest.mark.parametrize(
    "override",
    [
        {"hidden_act": "gelu"},
        {"attention_bias": True},
        {"use_sliding_window": True},
        {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
        {"num_key_value_heads": 3},
    ],
)
def test_config_rejects_unsupported_features(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Qwen3Config.from_dict({**REAL_QWEN3_06B, **override})


# --- loading --------------------------------------------------------------------------------


def test_loader_copies_every_weight(model: Qwen3ForCausalLM, hf_model: HFQwen3ForCausalLM) -> None:
    ref = hf_model.state_dict()
    ours = dict(model.named_parameters())
    assert model.lm_head is None, "tied embeddings: no separate lm_head parameter"
    assert set(ours) == set(ref) - {"lm_head.weight"}
    for name, p in ours.items():
        assert torch.equal(p, ref[name]), name


def _copy_checkpoint(src: Path, dst: Path, drop: str | None = None, add: str | None = None) -> Path:
    dst.mkdir()
    (dst / "config.json").write_text((src / "config.json").read_text())
    tensors = load_file(str(src / "model.safetensors"))
    if drop:
        del tensors[drop]
    if add:
        tensors[add] = torch.zeros(1)
    save_file(tensors, str(dst / "model.safetensors"))
    return dst


def test_loader_rejects_missing_tensor(model_dir: Path, tmp_path: Path) -> None:
    name = "model.layers.1.self_attn.k_norm.weight"
    d = _copy_checkpoint(model_dir, tmp_path / "ckpt", drop=name)
    with pytest.raises(ValueError, match="missing"):
        load_model(d, dtype=torch.float32)


def test_loader_rejects_unexpected_tensor(model_dir: Path, tmp_path: Path) -> None:
    d = _copy_checkpoint(model_dir, tmp_path / "ckpt", add="model.layers.0.extra.weight")
    with pytest.raises(ValueError, match="unexpected"):
        load_model(d, dtype=torch.float32)


def test_loader_casts_to_requested_dtype(model_dir: Path) -> None:
    m = load_model(model_dir, dtype=torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in m.parameters())
    assert m.model.rotary_emb.inv_freq.dtype == torch.float32, "RoPE frequencies stay fp32"


# --- forward --------------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1, 2, 7, 37, 100])
def test_single_sequence_matches_hf(
    model: Qwen3ForCausalLM, hf_model: HFQwen3ForCausalLM, length: int
) -> None:
    ids = random_ids(length, seed=length)
    (got,) = our_logits(model, [ids])
    assert_matches(got, hf_logits(hf_model, ids))


def test_flat_batch_matches_each_sequence_alone(
    model: Qwen3ForCausalLM, hf_model: HFQwen3ForCausalLM
) -> None:
    seqs = [random_ids(n, seed=100 + n) for n in (5, 37, 1, 64, 12)]
    for got, ids in zip(our_logits(model, seqs), seqs, strict=True):
        assert_matches(got, hf_logits(hf_model, ids))


def test_positions_are_per_sequence(model: Qwen3ForCausalLM) -> None:
    """Same tokens at different positions must give different results (RoPE is applied)."""
    ids = random_ids(6, seed=7)
    with torch.no_grad():
        a = model(torch.tensor(ids), torch.arange(6), DenseCausalAttention([6]))
        b = model(torch.tensor(ids), torch.arange(6) + 50, DenseCausalAttention([6]))
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_greedy_without_cache_matches_hf_generate(
    model: Qwen3ForCausalLM, hf_model: HFQwen3ForCausalLM, seed: int
) -> None:
    prompt = random_ids(9 + 11 * seed, seed=200 + seed)
    tokens = list(prompt)
    for _ in range(30):
        with torch.no_grad():
            hidden = model(
                torch.tensor(tokens), torch.arange(len(tokens)), DenseCausalAttention([len(tokens)])
            )
            tokens.append(int(model.compute_logits(hidden[-1:]).argmax()))
    assert tokens[len(prompt) :] == hf_greedy(hf_model, prompt, 30)
