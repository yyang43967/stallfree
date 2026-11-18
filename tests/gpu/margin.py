"""bf16 correctness on the real model: the margin rule, the teacher-forced check, and a report.

For each prompt the reference is Hugging Face bf16, eager attention, one prompt at a time, greedy.
The engine's output passes if it is identical, or if at the first differing position the
reference's top-1 minus top-2 logit gap is at most ``BF16_MARGIN_TAU``. The teacher-forced check
runs the reference's full sequence through the engine's model in one prefill and requires the same
top-1 token wherever the reference's gap is above ``BF16_MARGIN_TAU``.
"""

from __future__ import annotations

import collections
import json
import math
import os
from typing import Any

import torch

from stallfree.attention.flash import FlashAttention
from stallfree.attention.metadata import SegmentSpec, build_attention_metadata
from stallfree.attention.torch_ref import TorchRefAttention
from stallfree.config import AttentionBackendName, EngineConfig, SchedulerConfig
from stallfree.engine import LLMEngine
from stallfree.kv_cache import KVCache
from stallfree.sequence import SamplingParams
from tests.gpu.calibration import (
    MAX_NEW,
    chat_ids,
    first_diff,
    greedy,
    load,
    reference_gaps,
)
from tests.gpu.prompts import PROMPTS

# Calibrated on an A10 against Hugging Face's own bf16 noise; see the "bf16 margin threshold"
# entry in docs/decisions.md and docs/data/tau_calibration.json.
BF16_MARGIN_TAU = 0.5

IM_END, ENDOFTEXT = 151645, 151643
EOS = {IM_END, ENDOFTEXT}


def hf_self_divergence() -> dict[str, Any]:
    """HF's own divergence counts from the calibration JSON named by ``$STALLFREE_CALIBRATION``
    (written by the same run, so on the same GPU), for side-by-side reporting."""
    path = os.environ.get("STALLFREE_CALIBRATION")
    if not path or not os.path.exists(path):
        return {"note": "no calibration from this run"}
    with open(path) as f:
        cal = json.load(f)
    out: dict[str, Any] = {"gpu": cal["gpu"]}
    for name, rows in cal["divergences"].items():
        out[name] = {
            "diverged": len(rows),
            "gaps": sorted(r["ref_gap"] for r in rows if r["ref_gap"] is not None),
        }
    return out


def judge(ref: list[int], ours: list[int], gaps: list[float]) -> dict[str, Any]:
    """Exact, or first divergence with the reference's gap there, and whether the rule passes."""
    if ours[: len(ref)] == ref and (len(ours) == len(ref) or ref[-1] in EOS):
        return {"exact": True, "position": None, "gap": None, "passes": True}
    t = first_diff(ref, ours)
    assert t is not None
    gap = gaps[t] if t < len(gaps) else None
    passes = gap is not None and gap <= BF16_MARGIN_TAU
    return {
        "exact": False,
        "position": t,
        "gap": gap,
        "passes": passes,
        "ref_token": ref[t] if t < len(ref) else None,
        "our_token": ours[t] if t < len(ours) else None,
    }


@torch.inference_mode()
def all_row_logits(engine: LLMEngine, tokens: list[int], backend: type) -> torch.Tensor:
    """Logits at every position of one prefill of ``tokens`` on a fresh cache."""
    cfg = engine.model.config
    bs = engine.config.block_size
    blocks = math.ceil(len(tokens) / bs)
    cache = KVCache(
        num_layers=cfg.num_hidden_layers,
        num_blocks=blocks,
        block_size=bs,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    md = build_attention_metadata([SegmentSpec(0, len(tokens), tuple(range(blocks)))], bs, "cuda")
    ids = torch.tensor(tokens, device="cuda")
    hidden = engine.model(ids, torch.arange(len(tokens), device="cuda"), backend(cache, md))
    return engine.model.compute_logits(hidden).float()


def scheduler_config() -> SchedulerConfig:
    """Mixed batching is turned on by ``STALLFREE_MIXED_BUDGET=<tokens>``; a small budget makes
    sure prompts are really chunked (``max_num_seqs`` must not exceed the budget)."""
    budget = int(os.environ.get("STALLFREE_MIXED_BUDGET", "0"))
    if budget:
        return SchedulerConfig(
            max_model_len=4096,
            eos_token_id=IM_END,
            enable_mixed_batching=True,
            max_num_batched_tokens=budget,
            max_num_seqs=min(256, budget),
        )
    return SchedulerConfig(max_model_len=4096, eos_token_id=IM_END)


def make_engine(model_dir: str, num_blocks: int | None = None) -> LLMEngine:
    return LLMEngine(
        EngineConfig(
            model=model_dir,
            num_blocks=num_blocks,
            dtype="bfloat16",
            device="cuda",
            attention=AttentionBackendName.FLASH,
            scheduler=scheduler_config(),
        )
    )


def generate_counting_chunks(
    engine: LLMEngine, prompts: list[list[int]], params: SamplingParams
) -> tuple[list[list[int]], list[int]]:
    """``engine.generate`` that also counts, per prompt, the steps that computed part of its
    prompt (1 = the whole prompt in one step; more = chunked prefill)."""
    ids = [engine.add_request(p, params) for p in prompts]
    prompt_len = dict(zip(ids, (len(p) for p in prompts), strict=True))
    chunks = dict.fromkeys(ids, 0)
    schedule = engine.scheduler.schedule

    def recording_schedule() -> Any:
        plan = schedule()
        for e in plan.entries:
            if e.start_pos < prompt_len[e.seq_id]:
                chunks[e.seq_id] += 1
        return plan

    engine.scheduler.schedule = recording_schedule  # type: ignore[method-assign]
    outputs: dict[int, list[int]] = {i: [] for i in ids}
    try:
        while engine.has_unfinished():
            for out in engine.step():
                outputs[out.seq_id].append(out.token_id)
    finally:
        del engine.scheduler.schedule
    return [outputs[i] for i in ids], [chunks[i] for i in ids]


def run_margin_suite(model_dir: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    prompts = [chat_ids(tok, p) for p in PROMPTS]
    hf = load(model_dir, torch.bfloat16, "eager")
    refs = greedy(hf, tok, prompts, 1, EOS)
    gaps = [reference_gaps(hf, p, r) for p, r in zip(prompts, refs, strict=True)]

    engine = make_engine(model_dir)
    greedy_params = SamplingParams(max_tokens=MAX_NEW)
    single, single_chunks = [], []
    for p in prompts:
        outs, chunks = generate_counting_chunks(engine, [p], greedy_params)
        single.append(outs[0])
        single_chunks.append(chunks[0])
    batched, batched_chunks = generate_counting_chunks(engine, prompts, greedy_params)
    prefill_chunks = {"single": single_chunks, "batched": batched_chunks}
    results: dict[str, list[dict[str, Any]]] = {"single": [], "batched": []}
    for mode, outs in (("single", single), ("batched", batched)):
        for i, (r, o) in enumerate(zip(refs, outs, strict=True)):
            results[mode].append(
                {
                    "prompt": i,
                    "prompt_len": len(prompts[i]),
                    "prefill_chunks": prefill_chunks[mode][i],
                    "ref_len": len(r),
                    "our_len": len(o),
                    **judge(r, o, gaps[i]),
                }
            )

    # Teacher forcing: the reference's sequence through our model, flash and the reference backend.
    tf: dict[str, Any] = {
        "positions": 0,
        "checked_above_tau": 0,
        "flash_mismatches": [],
        "torch_ref_mismatches": [],
        "max_abs_logit_diff_flash_vs_torch_ref": 0.0,
    }
    for i, (p, r) in enumerate(zip(prompts, refs, strict=True)):
        seq = p + r[:-1]
        rows = slice(len(p) - 1, len(seq))
        flash = all_row_logits(engine, seq, FlashAttention)[rows]
        ref_backend = all_row_logits(engine, seq, TorchRefAttention)[rows]
        tf["max_abs_logit_diff_flash_vs_torch_ref"] = max(
            tf["max_abs_logit_diff_flash_vs_torch_ref"], float((flash - ref_backend).abs().max())
        )
        for j, (g, want) in enumerate(zip(gaps[i], r, strict=True)):
            tf["positions"] += 1
            if g <= BF16_MARGIN_TAU:
                continue
            tf["checked_above_tau"] += 1
            if int(flash[j].argmax()) != want:
                tf["flash_mismatches"].append({"prompt": i, "position": j, "gap": g})
            if int(ref_backend[j].argmax()) != want:
                tf["torch_ref_mismatches"].append({"prompt": i, "position": j, "gap": g})

    report = {
        "gpu": torch.cuda.get_device_name(),
        "tau": BF16_MARGIN_TAU,
        "num_prompts": len(prompts),
        "memory": engine.memory_report.summary() if engine.memory_report else None,
        "results": results,
        "summary": {},
        "hf_self_divergence_same_run": hf_self_divergence(),
        "mixed_batching_budget": int(os.environ.get("STALLFREE_MIXED_BUDGET", "0")) or None,
        "teacher_forced": tf,
    }
    for mode, rows in results.items():
        div = [x for x in rows if not x["exact"]]
        report["summary"][mode] = {
            "diverged": len(div),
            "failed_rule": sum(not x["passes"] for x in rows),
            "gap_histogram": dict(sorted(collections.Counter(str(x["gap"]) for x in div).items())),
            "prompts_chunked": sum(x["prefill_chunks"] > 1 for x in rows),
            "prefill_chunks_histogram": dict(
                sorted(collections.Counter(x["prefill_chunks"] for x in rows).items())
            ),
        }
    del engine
    torch.cuda.empty_cache()
    report["preemption"] = run_preemption(model_dir, tok, hf)
    return report


def run_preemption(model_dir: str, tok: Any, hf: Any) -> dict[str, Any]:
    """Four prompts of 202-215 tokens (one 256-token block each) that grow by 100 tokens into a
    second block, on 6 blocks: four requests need 8 blocks, so some must be preempted."""
    groups = [[PROMPTS[(8 * i + j) % len(PROMPTS)] for j in range(18)] for i in range(4)]
    prompts = [chat_ids(tok, "Answer each briefly:\n" + "\n".join(g)) for g in groups]
    new = 100
    with torch.inference_mode():
        refs = []
        for p in prompts:
            gen = hf.generate(
                input_ids=torch.tensor([p], device="cuda"),
                attention_mask=torch.ones(1, len(p), dtype=torch.long, device="cuda"),
                max_new_tokens=new,
                min_new_tokens=new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            refs.append(gen[0, len(p) :].tolist())
    gaps = [reference_gaps(hf, p, r) for p, r in zip(prompts, refs, strict=True)]
    engine = make_engine(model_dir, num_blocks=6)
    outs = engine.generate(prompts, SamplingParams(max_tokens=new, ignore_eos=True))
    rows = [
        {"prompt_len": len(p), **judge(r, o, g)}
        for p, r, o, g in zip(prompts, refs, outs, gaps, strict=True)
    ]
    return {
        "num_preemptions": engine.num_preemptions,
        "blocks": 6,
        "block_size": 256,
        "results": rows,
    }
