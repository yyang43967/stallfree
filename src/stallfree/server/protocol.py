"""OpenAI chat-completions request and response shapes (the subset this server implements).

No torch import. Unknown request fields are accepted and ignored unless they would change what
the client gets back; those are rejected with an OpenAI-shaped 400 rather than silently dropped.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TextPart(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[TextPart] | None = None

    def text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions``. Extra fields are kept so they can be checked."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stream: bool = False
    stream_options: StreamOptions | None = None
    # Not in the OpenAI schema; accepted as vLLM does.
    chat_template_kwargs: dict[str, Any] | None = None
    ignore_eos: bool = False

    # Fields that change the response and are not implemented.
    n: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[Any] | None = None
    tool_choice: Any = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    response_format: Any = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None

    def unsupported(self) -> list[str]:
        """Names of fields set to something this server cannot honour."""
        bad = []
        if self.n not in (None, 1):
            bad.append("n")
        for name in ("stop", "seed", "tools", "response_format", "logit_bias", "top_logprobs"):
            if getattr(self, name):
                bad.append(name)
        if self.tool_choice not in (None, "none"):
            bad.append("tool_choice")
        if self.logprobs:
            bad.append("logprobs")
        for name in ("presence_penalty", "frequency_penalty"):
            if getattr(self, name):
                bad.append(name)
        return bad

    def requested_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens

    def chat(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.text()} for m in self.messages]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: str | None


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class Delta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "stallfree"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


def error_body(
    message: str, type_: str, param: str | None = None, code: str | None = None
) -> dict[str, Any]:
    return {"error": {"message": message, "type": type_, "param": param, "code": code}}


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def now() -> int:
    return int(time.time())
