import hashlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from anthropic import AsyncAnthropicBedrock
from anthropic.types import Message as AnthropicMessage

from app.bedrock.dev_logging import log_invocation_to_file, log_response_to_file

logger = logging.getLogger(__name__)


def _cache_prefix_key(system: str | list | None, tools: list | None) -> str:
    """Short hash of the cacheable request prefix (system + tools), to spot drift in logs."""
    payload = json.dumps({"system": system, "tools": tools}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class StreamStarted:
    """The response has started (message_start received)."""


@dataclass(frozen=True)
class StreamText:
    """A raw text delta from the model."""

    text: str


@dataclass(frozen=True)
class StreamDone:
    """The complete response.

    llm_internal_response_id is the logged row, if the handler wrote one (see log_llm_call).
    """

    message: AnthropicMessage
    llm_internal_response_id: int | None = None


StreamEvent = StreamStarted | StreamText | StreamDone


async def stream_message_events(
    async_client: AsyncAnthropicBedrock,
    *,
    messages: list[dict],
    model: str,
    max_tokens: int,
    system: str | list | None = None,
    **extra_api_kwargs,
) -> AsyncIterator[StreamEvent]:
    """Stream one Messages API call as events: started, raw text deltas, then the final message.

    Args:
        async_client: Bedrock client for the region to call.
        messages: Request messages.
        model: Model id.
        max_tokens: Output token limit.
        system: System prompt.
        **extra_api_kwargs: Other request params (tools, tool_choice, thinking...).

    Yields:
        StreamStarted once, StreamText per text delta, then StreamDone.
    """
    request_kwargs = {"system": system, **extra_api_kwargs} if system is not None else extra_api_kwargs
    logger.debug(f"Cache prefix key: {_cache_prefix_key(system, extra_api_kwargs.get('tools'))}")
    log_path = log_invocation_to_file(model, system, messages, extra=extra_api_kwargs or None)
    async with async_client.messages.stream(
        max_tokens=max_tokens, messages=messages, model=model, **request_kwargs
    ) as stream:
        async for event in stream:
            if event.type == "message_start":
                yield StreamStarted()
            elif event.type == "text" and event.text:
                yield StreamText(event.text)
        message = await stream.get_final_message()
    log_response_to_file(message, log_path)
    yield StreamDone(message)
