"""Measure how far Hugging Face's own greedy output drifts under bf16 noise (CUDA).

Reference R: bf16, eager attention, one prompt at a time. Each noise source reruns the same
prompts with one thing changed, and every divergence from R is recorded with R's top-1 minus
top-2 logit gap at the first differing position:

- B: bf16 eager, prompts in batches of 8 (left padding); the required source.
- S: bf16 SDPA attention, one at a time (a different attention kernel).
- F: fp32 eager, one at a time (no bf16 rounding at all).

The distribution of R's gaps over every generated position is reported too, so a threshold can
be judged against how often near-ties occur.
"""

from __future__ import annotations

import statistics
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.gpu.prompts import PROMPTS

MAX_NEW = 64
BATCH = 8
THRESHOLDS = (0.0625, 0.125, 0.25, 0.5, 1.0)


def chat_ids(tok: Any, prompt: str) -> list[int]:
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(tok.encode(text, add_special_tokens=False))


def load(model_dir: str, dtype: torch.dtype, attn: str) -> Any:
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, attn_implementation=attn)
    return model.to("cuda").eval()


def strip_eos(tokens: list[int], eos: set[int]) -> list[int]:
    for i, t in enumerate(tokens):
        if t in eos:
            return tokens[: i + 1]
    return tokens


@torch.inference_mode()
def greedy(
    model: Any, tok: Any, prompts: list[list[int]], batch: int, eos: set[int]
) -> list[list[int]]:
    outs: list[list[int]] = []
    for i in range(0, len(prompts), batch):
        group = prompts[i : i + batch]
        width = max(len(p) for p in group)
        pad = tok.pad_token_id
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in group], device="cuda")
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in group], device="cuda")
        gen = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=pad,
        )
        outs += [strip_eos(gen[j, width:].tolist(), eos) for j in range(len(group))]
    return outs


@torch.inference_mode()
def reference_gaps(model: Any, prompt: list[int], new: list[int]) -> list[float]:
    """R's top-1 minus top-2 logit (fp32) at each position that produced a generated token."""
    ids = torch.tensor([prompt + new[:-1]], device="cuda")
    logits = model(input_ids=ids).logits[0, len(prompt) - 1 :].float()
    top2 = logits.topk(2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).tolist()


def first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def run_calibration(model_dir: str) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(model_dir)
    eos = {151645, 151643}  # <|im_end|>, <|endoftext|>
    prompts = [chat_ids(tok, p) for p in PROMPTS]

    bf16 = load(model_dir, torch.bfloat16, "eager")
    ref = greedy(bf16, tok, prompts, 1, eos)
    gaps = [reference_gaps(bf16, p, r) for p, r in zip(prompts, ref, strict=True)]
    sources = {"B_bf16_eager_batch8": greedy(bf16, tok, prompts, BATCH, eos)}
    del bf16
    torch.cuda.empty_cache()
    sdpa = load(model_dir, torch.bfloat16, "sdpa")
    sources["S_bf16_sdpa_batch1"] = greedy(sdpa, tok, prompts, 1, eos)
    del sdpa
    torch.cuda.empty_cache()
    fp32 = load(model_dir, torch.float32, "eager")
    sources["F_fp32_eager_batch1"] = greedy(fp32, tok, prompts, 1, eos)
    del fp32

    divergences: dict[str, list[dict[str, Any]]] = {}
    for name, outs in sources.items():
        rows = []
        for i, (r, o) in enumerate(zip(ref, outs, strict=True)):
            t = first_diff(r, o)
            if t is None:
                continue
            gap = gaps[i][t] if t < len(gaps[i]) else None  # None: one side ended (EOS) earlier
            rows.append(
                {
                    "prompt": i,
                    "prompt_len": len(prompts[i]),
                    "position": t,
                    "ref_token": r[t] if t < len(r) else None,
                    "other_token": o[t] if t < len(o) else None,
                    "ref_gap": gap,
                }
            )
        divergences[name] = rows

    all_gaps = sorted(g for gs in gaps for g in gs)

    def pct(q: float) -> float:
        return all_gaps[min(len(all_gaps) - 1, int(q * len(all_gaps)))]

    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "num_prompts": len(prompts),
        "max_new_tokens": MAX_NEW,
        "reference": "bf16, eager attention, batch 1, greedy",
        "reference_lengths": [len(r) for r in ref],
        "reference_gap_distribution": {
            "positions": len(all_gaps),
            "min": all_gaps[0],
            "p01": pct(0.01),
            "p05": pct(0.05),
            "median": statistics.median(all_gaps),
            "count_below": {str(x): sum(g < x for g in all_gaps) for x in THRESHOLDS},
        },
        "divergences": divergences,
    }
