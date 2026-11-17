"""The synchronous engine: configuration, request lifecycle, and step outputs."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from helpers.tiny_qwen3 import TINY_CONFIG, hf_greedy
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from stallfree.config import AttentionBackendName, EngineConfig, SchedulerConfig
from stallfree.engine import LLMEngine
from stallfree.sequence import FinishReason, SamplingParams

BS = 4


def make_engine(model_dir: Path, num_blocks: int = 64, **sched: object) -> LLMEngine:
    scheduler = SchedulerConfig(
        max_num_seqs=8, max_num_batched_tokens=64, max_model_len=256, watermark=0.0, **sched
    )  # type: ignore[arg-type]
    cfg = EngineConfig(
        model=str(model_dir),
        num_blocks=num_blocks,
        block_size=BS,
        dtype="float32",
        scheduler=scheduler,
    )
    return LLMEngine(cfg)


def prompt(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    vocab: int = TINY_CONFIG["vocab_size"]  # type: ignore[assignment]
    return [rng.randrange(vocab) for _ in range(n)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_blocks": 0},
        {"block_size": 0},
        {"dtype": "float64x"},
        {"attention": AttentionBackendName.FLASH, "block_size": 16},
    ],
)
def test_invalid_engine_config(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {"model": "unused", "num_blocks": 8, "block_size": 4}
    with pytest.raises(ValueError):
        EngineConfig(**{**base, **kwargs})  # type: ignore[arg-type]


def test_flash_backend_not_available_yet(tiny_model_dir: Path) -> None:
    flash = AttentionBackendName.FLASH
    cfg = EngineConfig(model=str(tiny_model_dir), num_blocks=8, attention=flash)
    with pytest.raises(NotImplementedError):
        LLMEngine(cfg)


def test_max_model_len_cannot_exceed_the_model(tiny_model_dir: Path) -> None:
    limit: int = TINY_CONFIG["max_position_embeddings"]  # type: ignore[assignment]
    too_long = SchedulerConfig(max_model_len=limit + 1)
    cfg = EngineConfig(model=str(tiny_model_dir), num_blocks=8, block_size=BS, scheduler=too_long)
    with pytest.raises(ValueError):
        LLMEngine(cfg)


def test_generate_matches_hf_greedy(
    tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM
) -> None:
    engine = make_engine(tiny_model_dir)
    p = prompt(9, seed=1)
    (out,) = engine.generate([p], SamplingParams(max_tokens=12, ignore_eos=True))
    assert out == hf_greedy(tiny_hf_model, p, 12)
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks


def test_step_with_nothing_to_do_returns_nothing(tiny_model_dir: Path) -> None:
    engine = make_engine(tiny_model_dir)
    assert engine.step() == []
    assert not engine.has_unfinished()


def test_request_ids_increase_and_unfit_requests_are_rejected(tiny_model_dir: Path) -> None:
    engine = make_engine(tiny_model_dir, num_blocks=4)
    one = SamplingParams(max_tokens=1)
    assert [engine.add_request(prompt(3, s), one) for s in range(3)] == [0, 1, 2]
    with pytest.raises(ValueError):
        engine.add_request(prompt(15, 9), SamplingParams(max_tokens=2))  # 17 tokens > 4 blocks


def test_step_outputs_report_tokens_and_finish(tiny_model_dir: Path) -> None:
    engine = make_engine(tiny_model_dir)
    sid = engine.add_request(prompt(5, 2), SamplingParams(max_tokens=3, ignore_eos=True))
    outs = [engine.step() for _ in range(3)]
    assert [len(o) for o in outs] == [1, 1, 1]
    assert [o[0].seq_id for o in outs] == [sid] * 3
    assert [o[0].finished for o in outs] == [False, False, True]
    assert outs[-1][0].finish_reason is FinishReason.LENGTH
    assert not engine.has_unfinished()


def test_eos_finishes_with_stop(tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM) -> None:
    p = prompt(6, seed=3)
    ref = hf_greedy(tiny_hf_model, p, 5)
    eos = ref[2]  # make the third generated token the end-of-sequence token
    stop_at = ref.index(eos)
    engine = make_engine(tiny_model_dir, eos_token_id=eos)
    sid = engine.add_request(p, SamplingParams(max_tokens=5))
    outs = []
    while engine.has_unfinished():
        outs += engine.step()
    assert [o.token_id for o in outs] == ref[: stop_at + 1]
    assert outs[-1].seq_id == sid and outs[-1].finish_reason is FinishReason.STOP


def test_abort_frees_blocks(tiny_model_dir: Path) -> None:
    engine = make_engine(tiny_model_dir)
    sid = engine.add_request(prompt(9, 4), SamplingParams(max_tokens=20))
    engine.step()
    assert engine.block_manager.num_used_blocks > 0
    assert engine.abort(sid)
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks
    assert not engine.has_unfinished()
    assert not engine.abort(sid)
    assert engine.step() == []


def test_random_sampling_is_reproducible_by_seed(tiny_model_dir: Path) -> None:
    params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=10, ignore_eos=True)
    prompts = [prompt(5, 10), prompt(8, 11)]
    a = make_engine(tiny_model_dir).generate(prompts, params)
    b = make_engine(tiny_model_dir).generate(prompts, params)
    assert a == b


def test_batched_requests_each_match_hf(
    tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM
) -> None:
    engine = make_engine(tiny_model_dir)
    prompts = [prompt(n, seed=20 + n) for n in (4, 9, 6)]
    outs = engine.generate(prompts, SamplingParams(max_tokens=10, ignore_eos=True))
    assert outs == [hf_greedy(tiny_hf_model, p, 10) for p in prompts]


def test_each_request_uses_its_own_sampling_params(
    tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM
) -> None:
    """A greedy request decoded in the same steps as a high-temperature one stays greedy."""
    engine = make_engine(tiny_model_dir)
    greedy_prompt, random_prompt = prompt(7, seed=30), prompt(5, seed=31)
    params = [
        SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True),
        SamplingParams(temperature=5.0, max_tokens=12, ignore_eos=True),
    ]
    greedy_out, random_out = engine.generate([greedy_prompt, random_prompt], params)
    assert greedy_out == hf_greedy(tiny_hf_model, greedy_prompt, 12)
    assert random_out != hf_greedy(tiny_hf_model, random_prompt, 12)
