"""Text in, token ids out (chat template), and token ids back to text, incrementally.

The engine works on token ids only. The API layer turns a chat into ids with the model's chat
template and turns generated ids back into text one token at a time for streaming.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

REPLACEMENT_CHAR = "�"  # what a decoder emits for an incomplete UTF-8 sequence


class ChatTokenizer(Protocol):
    eos_token_id: int | None

    def encode_chat(
        self, messages: Sequence[Mapping[str, str]], template_kwargs: Mapping[str, Any]
    ) -> list[int]:
        """Render the chat with the model's template (ready for the assistant's turn) and
        tokenize it."""
        ...

    def decode(self, ids: Sequence[int]) -> str:
        """Text for ``ids``, without special tokens such as end-of-turn markers."""
        ...


class HFChatTokenizer:
    """``ChatTokenizer`` over a Hugging Face tokenizer directory."""

    def __init__(self, path: str, revision: str | None = None) -> None:
        from transformers import AutoTokenizer  # deferred: only the server needs it

        self._tok: Any = AutoTokenizer.from_pretrained(path, revision=revision)
        self.eos_token_id: int | None = self._tok.eos_token_id

    def render(
        self, messages: Sequence[Mapping[str, str]], template_kwargs: Mapping[str, Any]
    ) -> str:
        text: str = self._tok.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        return text

    def encode_chat(
        self, messages: Sequence[Mapping[str, str]], template_kwargs: Mapping[str, Any]
    ) -> list[int]:
        ids: list[int] = self._tok.encode(
            self.render(messages, template_kwargs), add_special_tokens=False
        )
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        text: str = self._tok.decode(list(ids), skip_special_tokens=True)
        return text


class IncrementalDetokenizer:
    """Turns a growing list of token ids into text deltas that are safe to send.

    Decoding token by token breaks multi-byte characters and merges that span tokens. Instead,
    each step decodes a short window ending at the newest token and emits only the part past what
    was already sent; if the window ends in an incomplete character (the decoder's replacement
    character), nothing is emitted until a later token completes it. ``flush`` emits whatever
    remains when the sequence ends.
    """

    def __init__(self, tokenizer: ChatTokenizer) -> None:
        self._tok = tokenizer
        self._ids: list[int] = []
        self._prefix = 0  # start of the decode window
        self._read = 0  # tokens whose text has been emitted

    def add(self, token_id: int) -> str:
        self._ids.append(token_id)
        prefix_text = self._tok.decode(self._ids[self._prefix : self._read])
        text = self._tok.decode(self._ids[self._prefix :])
        if len(text) > len(prefix_text) and not text.endswith(REPLACEMENT_CHAR):
            delta = text[len(prefix_text) :]
            self._prefix = self._read
            self._read = len(self._ids)
            return delta
        return ""

    def flush(self) -> str:
        prefix_text = self._tok.decode(self._ids[self._prefix : self._read])
        text = self._tok.decode(self._ids[self._prefix :])
        self._prefix = self._read = len(self._ids)
        return text[len(prefix_text) :]
