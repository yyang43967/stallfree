"""Command-line options of the server map onto the engine configuration."""

from __future__ import annotations

from stallfree.config import AttentionBackendName
from stallfree.server.api import engine_config, parse_args


def test_defaults_keep_mixed_batching_off() -> None:
    cfg = engine_config(parse_args(["--model", "m"]), eos_token_id=7)
    assert cfg.model == "m"
    assert cfg.attention is AttentionBackendName.TORCH_REF
    assert not cfg.scheduler.enable_mixed_batching
    assert cfg.scheduler.max_num_batched_tokens == 4096
    assert cfg.scheduler.eos_token_id == 7


def test_mixed_batching_flag_and_budget() -> None:
    args = parse_args(
        [
            "--model",
            "m",
            "--enable-mixed-batching",
            "--max-num-batched-tokens",
            "512",
            "--max-num-seqs",
            "64",
            "--attention",
            "flash",
            "--device",
            "cuda",
            "--block-size",
            "256",
        ]
    )
    cfg = engine_config(args, eos_token_id=None)
    assert cfg.scheduler.enable_mixed_batching
    assert cfg.scheduler.max_num_batched_tokens == 512
    assert cfg.scheduler.max_num_seqs == 64
    assert cfg.attention is AttentionBackendName.FLASH
    assert (cfg.device, cfg.block_size, cfg.num_blocks) == ("cuda", 256, None)
