"""A tokenizer whose token ids are UTF-8 bytes (0-255), matching the tiny test model's 256-token
vocabulary. Multi-byte characters span several tokens, which exercises incremental decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ByteTokenizer:
    eos_token_id: int | None = 0  # NUL ends a turn; never shown in text

    def __init__(self) -> None:
        self.last_template_kwargs: dict[str, Any] = {}

    def encode_chat(
        self, messages: Sequence[Mapping[str, str]], template_kwargs: Mapping[str, Any]
    ) -> list[int]:
        self.last_template_kwargs = dict(template_kwargs)
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant: "
        if template_kwargs.get("enable_thinking") is False:
            text += "(no thinking) "
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(i for i in ids if i != self.eos_token_id).decode("utf-8", errors="replace")
