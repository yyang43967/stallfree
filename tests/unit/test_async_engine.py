"""The asyncio front end: correctness under concurrency, cancellation, backpressure, failures,
metrics, and a responsive event loop while a step runs."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest
from helpers.tiny_qwen3 import TINY_CONFIG, hf_greedy
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from stallfree.async_engine import AsyncEngine, EngineDeadError, QueueFullError
from stallfree.config import EngineConfig, SchedulerConfig
from stallfree.engine import LLMEngine
from stallfree.sequence import FinishReason, SamplingParams

BS = 4


def make_async(model_dir: Path, num_blocks: int = 64, **kw: int) -> AsyncEngine:
    cfg = EngineConfig(
        model=str(model_dir),
        num_blocks=num_blocks,
        block_size=BS,
        dtype="float32",
        scheduler=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=64,
            max_model_len=256,
            watermark=0.0,
            eos_token_id=None,
        ),
    )
    return AsyncEngine(LLMEngine(cfg), **kw)


def prompt(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    vocab: int = TINY_CONFIG["vocab_size"]  # type: ignore[assignment]
    return [rng.randrange(vocab) for _ in range(n)]


def run[R](coro: Coroutine[Any, Any, R], timeout: float = 30.0) -> R:
    """asyncio.run with an overall deadline, so a broken engine fails the test instead of
    hanging it."""
    return asyncio.run(asyncio.wait_for(coro, timeout))


def greedy(n: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)


async def collect(ae: AsyncEngine, p: list[int], params: SamplingParams) -> list[int]:
    return [out.token_id async for out in ae.generate(p, params)]


async def wait_idle(ae: AsyncEngine, timeout: float = 10.0) -> None:
    """Until every accepted request has finished or been aborted and the engine is empty."""
    deadline = time.monotonic() + timeout
    while ae.num_active or ae.engine.has_unfinished():
        assert time.monotonic() < deadline, "engine did not become idle"
        await asyncio.sleep(0.01)


def test_single_request_matches_hf(tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM) -> None:
    async def main() -> list[int]:
        ae = make_async(tiny_model_dir)
        await ae.start()
        try:
            return await collect(ae, prompt(9, 1), greedy(12))
        finally:
            await ae.stop()

    assert run(main()) == hf_greedy(tiny_hf_model, prompt(9, 1), 12)


def test_concurrent_requests_are_each_correct(
    tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM
) -> None:
    prompts = [prompt(3 + 2 * i, 10 + i) for i in range(8)]
    lengths = [5 + i for i in range(8)]

    async def main() -> list[list[int]]:
        ae = make_async(tiny_model_dir)
        await ae.start()
        try:
            return await asyncio.gather(
                *(collect(ae, p, greedy(n)) for p, n in zip(prompts, lengths, strict=True))
            )
        finally:
            await ae.stop()

    outs = run(main())
    assert outs == [hf_greedy(tiny_hf_model, p, n) for p, n in zip(prompts, lengths, strict=True)]


def test_staggered_arrivals(tiny_model_dir: Path, tiny_hf_model: HFQwen3ForCausalLM) -> None:
    prompts = [prompt(4 + i, 30 + i) for i in range(5)]

    async def delayed(ae: AsyncEngine, p: list[int], delay: float) -> list[int]:
        await asyncio.sleep(delay)
        return await collect(ae, p, greedy(10))

    async def main() -> list[list[int]]:
        ae = make_async(tiny_model_dir)
        await ae.start()
        try:
            return await asyncio.gather(*(delayed(ae, p, 0.02 * i) for i, p in enumerate(prompts)))
        finally:
            await ae.stop()

    assert run(main()) == [hf_greedy(tiny_hf_model, p, 10) for p in prompts]


def test_consumer_that_stops_early_aborts_and_frees_blocks(tiny_model_dir: Path) -> None:
    async def main() -> tuple[int, int, int]:
        ae = make_async(tiny_model_dir)
        await ae.start()
        try:
            got = 0
            async for _ in ae.generate(prompt(9, 2), greedy(200)):
                got += 1
                if got == 3:
                    break
            steps_at_stop = ae.engine.num_steps
            await wait_idle(ae)
            # Stopped within a step or two, not after generating the remaining ~197 tokens.
            assert ae.engine.num_steps - steps_at_stop <= 2
            bm = ae.engine.block_manager
            return got, bm.num_free_blocks, bm.num_blocks
        finally:
            await ae.stop()

    got, free, total = run(main())
    assert got == 3 and free == total


def test_cancelled_task_aborts_and_frees_blocks(tiny_model_dir: Path) -> None:
    async def main() -> tuple[int, int]:
        ae = make_async(tiny_model_dir)
        await ae.start()
        try:
            task = asyncio.create_task(collect(ae, prompt(9, 3), greedy(200)))
            while ae.engine.block_manager.num_used_blocks == 0:
                await asyncio.sleep(0.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            steps_at_cancel = ae.engine.num_steps
            await wait_idle(ae)
            assert ae.engine.num_steps - steps_at_cancel <= 2
            return ae.engine.block_manager.num_free_blocks, ae.engine.block_manager.num_blocks
        finally:
            await ae.stop()

    free, total = run(main())
    assert free == total


def test_queue_limit_rejects_then_recovers(tiny_model_dir: Path) -> None:
    async def main() -> list[int]:
        ae = make_async(tiny_model_dir, max_queued_requests=2)
        await ae.start()
        try:
            a = asyncio.create_task(collect(ae, prompt(5, 4), greedy(8)))
            b = asyncio.create_task(collect(ae, prompt(5, 5), greedy(8)))
            await asyncio.sleep(0)  # let both register
            with pytest.raises(QueueFullError):
                await collect(ae, prompt(5, 6), greedy(8))
            await asyncio.gather(a, b)
            return await collect(ae, prompt(5, 6), greedy(8))  # room again
        finally:
            await ae.stop()

    assert len(run(main())) == 8


def test_request_that_can_never_fit_is_rejected_up_front(tiny_model_dir: Path) -> None:
    async def main() -> None:
        ae = make_async(tiny_model_dir, num_blocks=4)  # 16 tokens of cache
        await ae.start()
        try:
            with pytest.raises(ValueError):
                await collect(ae, prompt(10, 7), greedy(10))
            assert ae.num_active == 0
        finally:
            await ae.stop()

    run(main())


def test_impossible_request_is_rejected_without_the_loop(tiny_model_dir: Path) -> None:
    """Rejection happens in generate() itself, not after a round trip through the loop."""

    async def main() -> None:
        ae = make_async(tiny_model_dir, num_blocks=4)  # loop never started
        with pytest.raises(ValueError):
            await asyncio.wait_for(collect(ae, prompt(10, 7), greedy(10)), timeout=1.0)

    run(main())


def test_engine_failure_reaches_every_waiter(tiny_model_dir: Path) -> None:
    async def main() -> None:
        ae = make_async(tiny_model_dir)

        def broken_step() -> list[object]:
            raise RuntimeError("device lost")

        ae.engine.step = broken_step  # type: ignore[method-assign]
        await ae.start()
        results = await asyncio.gather(
            collect(ae, prompt(5, 8), greedy(5)),
            collect(ae, prompt(6, 9), greedy(5)),
            return_exceptions=True,
        )
        assert all(isinstance(r, EngineDeadError) for r in results)
        with pytest.raises(EngineDeadError):
            await collect(ae, prompt(5, 10), greedy(5))
        with pytest.raises(RuntimeError):
            await ae.stop()

    run(main())


def test_metrics_are_logged_per_request(
    tiny_model_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async def main() -> None:
        ae = make_async(tiny_model_dir)
        real_step = ae.engine.step

        def slow_step() -> list[object]:
            time.sleep(0.05)
            return real_step()  # type: ignore[return-value]

        ae.engine.step = slow_step  # type: ignore[method-assign]
        await ae.start()
        try:
            await collect(ae, prompt(6, 11), greedy(6))
            async for _ in ae.generate(prompt(6, 12), greedy(50)):
                break
            await wait_idle(ae)
        finally:
            await ae.stop()

    with caplog.at_level(logging.INFO, logger="stallfree.metrics"):
        run(main())
    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "stallfree.metrics"]
    done, aborted = records
    assert done["finish_reason"] == FinishReason.LENGTH.value
    assert (done["prompt_tokens"], done["completion_tokens"]) == (6, 6)
    assert 0 <= done["queue_s"] <= done["ttft_s"] <= done["e2e_s"]
    # Queue time ends when the first token's step starts; TTFT includes that 50 ms step.
    assert done["ttft_s"] - done["queue_s"] >= 0.04
    assert done["tpot_s"] >= 0.04
    assert aborted["finish_reason"] == FinishReason.ABORT.value
    # The consumer read one token; a step already running when it stopped may add more, because
    # an abort takes effect at the next step boundary.
    assert 1 <= aborted["completion_tokens"] < 50


def test_event_loop_keeps_running_while_a_step_runs(tiny_model_dir: Path) -> None:
    """step() runs in a worker thread; a slow step must not freeze other coroutines."""

    async def main() -> tuple[int, bool]:
        ae = make_async(tiny_model_dir)
        real_step = ae.engine.step
        step_thread: list[bool] = []

        def slow_step() -> list[object]:
            step_thread.append(threading.current_thread() is not threading.main_thread())
            time.sleep(0.2)
            return real_step()  # type: ignore[return-value]

        ae.engine.step = slow_step  # type: ignore[method-assign]
        await ae.start()
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        try:
            await collect(ae, prompt(5, 13), greedy(2))
        finally:
            t.cancel()
            await ae.stop()
        return ticks, all(step_thread)

    ticks, off_main = run(main())
    assert off_main, "step must run off the event-loop thread"
    assert ticks >= 20, f"event loop ticked only {ticks} times during ~0.4 s of steps"
