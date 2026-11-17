"""Request validation, and turning generated ids back into text incrementally."""

from __future__ import annotations

import random

import pytest
from helpers.byte_tokenizer import ByteTokenizer
from pydantic import ValidationError

from stallfree.server.protocol import ChatCompletionRequest
from stallfree.tokenizer import REPLACEMENT_CHAR, HFChatTokenizer, IncrementalDetokenizer

QWEN = "Qwen/Qwen3-0.6B"
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def request(**kw: object) -> ChatCompletionRequest:
    base: dict[str, object] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    return ChatCompletionRequest.model_validate({**base, **kw})


def test_minimal_request_and_defaults() -> None:
    r = request()
    assert r.stream is False and r.requested_max_tokens() is None
    assert r.chat() == [{"role": "user", "content": "hi"}]
    assert r.unsupported() == []


def test_max_completion_tokens_wins_over_max_tokens() -> None:
    assert request(max_tokens=5, max_completion_tokens=7).requested_max_tokens() == 7
    assert request(max_tokens=5).requested_max_tokens() == 5


def test_content_parts_are_joined() -> None:
    r = request(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            }
        ]
    )
    assert r.chat()[0]["content"] == "ab"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", 2),
        ("stop", "x"),
        ("stop", ["x"]),
        ("seed", 1),
        ("tools", [{"type": "function"}]),
        ("tool_choice", "auto"),
        ("logprobs", True),
        ("response_format", {"type": "json_object"}),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.1),
        ("logit_bias", {"1": 1.0}),
    ],
)
def test_unsupported_fields_are_reported(field: str, value: object) -> None:
    assert request(**{field: value}).unsupported() == [field]


def test_harmless_or_default_fields_are_accepted() -> None:
    r = request(
        n=1,
        tool_choice="none",
        logprobs=False,
        presence_penalty=0.0,
        user="abc",
        stream_options={"include_usage": True},
        chat_template_kwargs={"enable_thinking": False},
    )
    assert r.unsupported() == []
    assert r.stream_options is not None and r.stream_options.include_usage
    assert r.chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.parametrize(
    "bad",
    [
        {"messages": []},
        {"max_tokens": 0},
        {"temperature": -1},
        {"top_p": 0},
        {"messages": [{"role": "tool", "content": "x"}]},
    ],
)
def test_invalid_requests_fail_validation(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        request(**bad)


# --- incremental detokenization --------------------------------------------------------------


def stream_text(ids: list[int]) -> tuple[list[str], str]:
    detok = IncrementalDetokenizer(ByteTokenizer())
    deltas = [detok.add(i) for i in ids]
    return deltas, "".join(deltas) + detok.flush()


def test_ascii_comes_out_one_token_at_a_time() -> None:
    deltas, full = stream_text(list(b"hello"))
    assert deltas == ["h", "e", "l", "l", "o"] and full == "hello"


def test_multibyte_character_is_held_until_complete() -> None:
    ids = list("aé😀b".encode())  # é is 2 bytes, the emoji 4
    deltas, full = stream_text(ids)
    assert deltas == ["a", "", "é", "", "", "", "😀", "b"]
    assert full == "aé😀b"
    assert all(REPLACEMENT_CHAR not in d for d in deltas)


def test_end_of_turn_token_is_not_shown() -> None:
    _, full = stream_text([*b"ok", 0])
    assert full == "ok"


def test_flush_reports_a_dangling_partial_character() -> None:
    _, full = stream_text(list("é".encode())[:1])
    assert full == REPLACEMENT_CHAR


@pytest.mark.parametrize("seed", range(10))
def test_deltas_always_add_up_to_the_full_decode(seed: int) -> None:
    rng = random.Random(seed)
    alphabet = "abc xyz 你好世界 ünïcödé 😀🍜 \n"
    text = "".join(rng.choice(alphabet) for _ in range(60))
    _, full = stream_text(list(text.encode()))
    assert full == text


# --- the real Qwen3 chat template (tokenizer files only; cached in CI) ----------------------


@pytest.fixture(scope="module")
def qwen() -> HFChatTokenizer:
    return HFChatTokenizer(QWEN, revision=QWEN_REVISION)


def test_enable_thinking_changes_the_rendered_prompt(qwen: HFChatTokenizer) -> None:
    chat = [{"role": "user", "content": "What is 2 + 2?"}]
    off = qwen.render(chat, {"enable_thinking": False})
    on = qwen.render(chat, {"enable_thinking": True})
    default = qwen.render(chat, {})
    assert off.endswith("<think>\n\n</think>\n\n"), "thinking off pre-fills an empty think block"
    assert not on.endswith("</think>\n\n")
    assert default == on, "Qwen3 thinks by default"
    assert qwen.encode_chat(chat, {"enable_thinking": False}) != qwen.encode_chat(chat, {})


def test_qwen_eos_is_the_end_of_turn_token(qwen: HFChatTokenizer) -> None:
    assert qwen.eos_token_id == 151645  # <|im_end|>


def test_incremental_decoding_with_qwen(qwen: HFChatTokenizer) -> None:
    text = "Stall-free serving: 料盒 and 灶台 😀 — done."
    ids = qwen._tok.encode(text, add_special_tokens=False)
    detok = IncrementalDetokenizer(qwen)
    out = "".join(detok.add(i) for i in ids) + detok.flush()
    assert out == text
