"""The chat agent loop: the model calls tools over several iterations, then answers."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from anthropic.types import Message as AnthropicMessage
from anthropic.types import ToolUseBlock
from pydantic import ValidationError

from app.bedrock import BedrockHandler
from app.bedrock.bedrock_stream import StreamDone, StreamText
from app.bedrock.call_log import LlmCallContext
from app.bedrock.service import calculate_completion_cost, extract_cache_tokens
from app.bedrock.tool_schema import ToolDef
from app.chat.cache_control import apply_compaction_aware_cache_control
from app.chat.schemas import Sources
from app.chat.tools import RegisteredTool, ToolCall, ToolContext, ToolExecResult
from app.compaction.service import estimate_prefix_tokens
from app.config import settings
from app.database.table import async_db_session

logger = logging.getLogger(__name__)

# Separates text from successive iterations in the displayed answer.
ITERATION_TEXT_SEPARATOR = "\n\n"

UNKNOWN_TOOL_MESSAGE = "Unknown tool. Use only the tools you have been given."
TOOL_FAILED_MESSAGE = "The tool failed. Try a different request, or answer without it."
CALL_LIMIT_MESSAGE = "Tool call limit reached for this step. Use the results you already have."
NOT_RUN_MESSAGE = "Not run: the response ended before this call could be made."
TURN_LIMIT_MESSAGE = "Tool call limit reached for this turn: answer now with the information you have."
TOOL_TIMEOUT_MESSAGE = "The tool took too long and was stopped. Try a narrower request, or answer without it."
RESULT_TRUNCATED_NOTE = "\n\n[Result truncated: it was too long to return in full.]"


@dataclass(frozen=True)
class TextDelta:
    """Answer text to append to what has been displayed so far."""

    text: str


@dataclass(frozen=True)
class ToolStatus:
    """Progress of one tool call, for display."""

    id: str
    tool: str
    state: Literal["running", "done", "error"]
    label: str


@dataclass(frozen=True)
class SourcesUpdated:
    """A tool added citations or sources; the values are the turn's full lists so far."""

    citations: list[dict]
    sources: Sources


@dataclass(frozen=True)
class ModelCallUsage:
    """Usage and cost of one model call in the loop."""

    llm_internal_response_id: int | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    completion_cost: Decimal
    stop_reason: str | None

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool_use block and what happened to it, for analytics."""

    iteration: int
    tool_use_id: str
    tool_name: str
    args: dict | None
    is_error: bool
    duration_ms: int
    result_chars: int
    llm_internal_response_id: int | None = None


@dataclass
class AgentLoopResult:
    """Everything the loop produced for one chat turn.

    Args:
        display_text: The answer as shown: all iterations' text.
        raw_messages: API messages appended this turn, exactly as sent.
        citations: Deduplicated citation dicts from all tool calls.
        sources: Deduplicated sources from all tool calls.
        model_calls: Usage per model call, in order.
        tool_calls: Every tool_use block, in order.
    """

    display_text: str = ""
    raw_messages: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    sources: Sources = field(default_factory=Sources)
    model_calls: list[ModelCallUsage] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    @property
    def stop_reason(self) -> str | None:
        return self.model_calls[-1].stop_reason if self.model_calls else None

    @property
    def input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.model_calls)

    @property
    def output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.model_calls)

    @property
    def cache_read_tokens(self) -> int:
        return sum(call.cache_read_tokens for call in self.model_calls)

    @property
    def cache_write_tokens(self) -> int:
        return sum(call.cache_write_tokens for call in self.model_calls)

    @property
    def completion_cost(self) -> Decimal:
        return sum((call.completion_cost for call in self.model_calls), Decimal(0))


@dataclass(frozen=True)
class LoopFinished:
    """The loop has finished; always the last event."""

    result: AgentLoopResult


AgentLoopEvent = TextDelta | ToolStatus | SourcesUpdated | LoopFinished


async def run_agent_loop(
    handler: BedrockHandler,
    *,
    history: list[dict],
    tools: list[RegisteredTool[Any]],
    ctx: ToolContext,
    should_compact: bool = False,
    request_kwargs: dict | None = None,
) -> AsyncIterator[AgentLoopEvent]:
    """Run model calls and tool calls until the model answers or the iteration cap is reached.

    Tool calls run one at a time, each on its own short DB session, committed when the tool
    returns and rolled back if it raises or times out; no session is held across model calls. A
    failing or timed-out tool becomes an is_error tool_result; the loop itself only raises for
    model call errors.

    Tool choice is always left to the model (auto); it is never sent, so it can never invalidate
    the messages cache. The iteration cap, context budget and per-turn tool-result char budget are
    all enforced softly instead: once any is reached, a short text nudge is added to the pending
    tool-results message asking the model to answer now, and no further tool call is executed this
    turn (a `tool_use` after that point gets an is_error result, not a run). The char budget is
    also checked before each call within an iteration, and each result is truncated to
    `agent_loop_max_result_chars_per_call`, so a turn's tool results are at most the budget plus
    one capped result. A model that keeps calling tools after the nudge gets is_error results and
    one more auto call as a grace period; only if it calls tools again do we force
    `tool_choice: none` as a last resort (logged, and the one case that can break the cache -
    expected to be rare).

    Args:
        handler: Handler for the chat model, with the system prompt set.
        history: Formatted messages ending with this turn's user message, without cache markers.
        tools: Tools to offer, in a stable order.
        ctx: Per-request context passed to executors.
        should_compact: Passed to `apply_compaction_aware_cache_control` for the first call.
        request_kwargs: Extra request params (e.g. thinking) for every call.

    Yields:
        TextDelta, ToolStatus and SourcesUpdated events as they happen, then LoopFinished.

    Raises:
        BedrockError: A model call failed.
    """
    result = AgentLoopResult()
    tools_by_name = {tool.name: tool for tool in tools}
    tool_specs = [tool.tool_def.spec for tool in tools]
    max_iterations = settings.agent_loop_max_iterations

    context_budget = int(settings.chat_model_context_window_tokens * settings.agent_loop_context_budget_fraction)

    at_limit = False
    calls_since_limit = 0
    llm_call_context = LlmCallContext(purpose="chat_loop", chat_id=ctx.chat.id, message_id=ctx.user_message.id)
    iteration = 0
    while True:
        iteration += 1
        unmarked_messages = history + result.raw_messages
        over_token_budget = estimate_prefix_tokens(unmarked_messages) > context_budget
        over_char_budget = _total_result_chars(result) > settings.agent_loop_max_result_chars_per_turn
        if not at_limit and (iteration >= max_iterations or over_token_budget or over_char_budget):
            if over_token_budget:
                logger.warning(f"Agent loop request is over the {context_budget}-token budget; asking it to answer")
            if over_char_budget:
                logger.warning(
                    f"Agent loop tool results are over the "
                    f"{settings.agent_loop_max_result_chars_per_turn}-char budget; asking it to answer"
                )
            at_limit = True
            _append_turn_limit_guidance(result.raw_messages)

        call_kwargs = dict(request_kwargs or {})
        if tool_specs:
            call_kwargs["tools"] = tool_specs
            if at_limit and calls_since_limit >= 2:
                logger.warning("Agent loop forcing tool_choice: none as a last resort after the turn limit")
                call_kwargs["tool_choice"] = {"type": "none"}
        if at_limit:
            calls_since_limit += 1
        request_messages = apply_compaction_aware_cache_control(
            unmarked_messages, should_compact=should_compact and iteration == 1
        )

        message = None
        llm_internal_response_id = None
        iteration_has_text = False
        async for event in handler.stream_events(request_messages, context=llm_call_context, **call_kwargs):
            if isinstance(event, StreamDone):
                message = event.message
                llm_internal_response_id = event.llm_internal_response_id
            elif isinstance(event, StreamText) and event.text:
                display = event.text
                if result.display_text and not iteration_has_text:
                    display = ITERATION_TEXT_SEPARATOR + display
                iteration_has_text = True
                result.display_text += display
                yield TextDelta(display)

        usage = _model_call_usage(handler, message, llm_internal_response_id)
        result.model_calls.append(usage)

        assistant_content = [block.model_dump(mode="json", exclude_none=True) for block in message.content]
        if assistant_content:
            result.raw_messages.append({"role": "assistant", "content": assistant_content})

        tool_uses = [block for block in message.content if isinstance(block, ToolUseBlock)]
        if not tool_uses:
            break
        if message.stop_reason != "tool_use":
            # e.g. max_tokens mid tool_use: every tool_use still needs a tool_result for replay.
            logger.warning(f"Loop stopped with {len(tool_uses)} unanswered tool_use blocks ({message.stop_reason})")
            result.raw_messages.append(
                {"role": "user", "content": [_tool_result(block.id, NOT_RUN_MESSAGE, True) for block in tool_uses]}
            )
            break

        if at_limit:
            # The model kept calling tools after being told the turn is over: refuse and retry.
            for block in tool_uses:
                call = ToolCall(block.id, iteration, usage.llm_internal_response_id)
                result.tool_calls.append(_record(call, block, None, True, 0, TURN_LIMIT_MESSAGE))
            result.raw_messages.append(
                {"role": "user", "content": [_tool_result(block.id, TURN_LIMIT_MESSAGE, True) for block in tool_uses]}
            )
            continue

        tool_results = []
        call_limit = settings.agent_loop_max_calls_per_iteration
        for position, block in enumerate(tool_uses):
            call = ToolCall(
                tool_use_id=block.id, iteration=iteration, llm_internal_response_id=usage.llm_internal_response_id
            )
            if position >= call_limit:
                tool_results.append(_tool_result(block.id, CALL_LIMIT_MESSAGE, True))
                result.tool_calls.append(_record(call, block, None, True, 0, CALL_LIMIT_MESSAGE))
                continue
            if _total_result_chars(result) > settings.agent_loop_max_result_chars_per_turn:
                tool_results.append(_tool_result(block.id, TURN_LIMIT_MESSAGE, True))
                result.tool_calls.append(_record(call, block, None, True, 0, TURN_LIMIT_MESSAGE))
                continue
            async for event in _run_tool_call(tools_by_name.get(block.name), block, call, ctx, result, tool_results):
                yield event
        result.raw_messages.append({"role": "user", "content": tool_results})

    yield LoopFinished(result)


async def _run_tool_call(
    tool: RegisteredTool[Any] | None,
    block: ToolUseBlock,
    call: ToolCall,
    ctx: ToolContext,
    result: AgentLoopResult,
    tool_results: list[dict],
) -> AsyncIterator[ToolStatus | SourcesUpdated]:
    """Validate and execute one tool_use block, appending its tool_result and record.

    Yields:
        Running, then done or error status (none for unknown tools or invalid input), and
        SourcesUpdated if the call added citations or sources.
    """
    if tool is None:
        logger.warning(f"Model called unknown tool {block.name!r}")
        tool_results.append(_tool_result(block.id, UNKNOWN_TOOL_MESSAGE, True))
        result.tool_calls.append(_record(call, block, None, True, 0, UNKNOWN_TOOL_MESSAGE))
        return

    if not tool.is_available(ctx):
        tool_results.append(_tool_result(block.id, tool.unavailable_message, True))
        result.tool_calls.append(_record(call, block, None, True, 0, tool.unavailable_message))
        return

    ignored = tool.tool_def.unknown_fields(block.input)
    ignored_note = ignored_fields_note(tool.tool_def, ignored) if ignored else ""
    if ignored:
        logger.warning(f"Tool {block.name!r} called with unrecognised input fields {ignored}")

    try:
        args = tool.tool_def.parse(block.input)
    except ValidationError as e:
        error_text = f"Invalid input: {format_validation_error(e)}"
        if ignored_note:
            error_text = f"{error_text}\n\n{ignored_note}"
        logger.warning(f"Invalid input for tool {block.name!r}: {error_text}")
        tool_results.append(_tool_result(block.id, error_text, True))
        result.tool_calls.append(_record(call, block, None, True, 0, error_text))
        return

    yield ToolStatus(block.id, tool.name, "running", tool.running_label(args))
    start = time.monotonic()
    exec_result: ToolExecResult | None = None
    timed_out = False
    timeout = settings.agent_loop_tool_timeout_seconds
    try:
        async with async_db_session() as db_session:
            exec_result = await asyncio.wait_for(tool.executor(ctx, args, call, db_session), timeout)
    except TimeoutError:
        timed_out = True
        logger.error(f"Tool {tool.name!r} timed out after {timeout}s")
    except Exception:
        logger.exception(f"Tool {tool.name!r} failed")
    duration_ms = int((time.monotonic() - start) * 1000)

    sources_changed = False
    if exec_result is None:
        content, is_error = (TOOL_TIMEOUT_MESSAGE if timed_out else TOOL_FAILED_MESSAGE), True
    else:
        content, is_error = exec_result.content, exec_result.is_error
        if not is_error:
            sources_changed = _merge_citations(result, exec_result)

    max_chars = settings.agent_loop_max_result_chars_per_call
    if len(content) > max_chars:
        logger.warning(f"Tool {tool.name!r} returned {len(content)} chars; truncating to {max_chars}")
        content = content[: max(max_chars - len(RESULT_TRUNCATED_NOTE), 0)] + RESULT_TRUNCATED_NOTE

    if ignored_note:
        content = f"{content}\n\n{ignored_note}"
    tool_results.append(_tool_result(block.id, content, is_error))
    result.tool_calls.append(
        _record(call, block, args.model_dump(mode="json", by_alias=True), is_error, duration_ms, content)
    )
    if is_error:
        yield ToolStatus(block.id, tool.name, "error", tool.error_label)
    else:
        yield ToolStatus(block.id, tool.name, "done", tool.done_label(args))
    if sources_changed:
        yield SourcesUpdated(list(result.citations), result.sources)


def _model_call_usage(
    handler: BedrockHandler, message: AnthropicMessage, llm_internal_response_id: int | None
) -> ModelCallUsage:
    """Usage and cost of a loop model call, which the handler has already logged to llm_internal_response."""
    cache_read_tokens, cache_write_tokens = extract_cache_tokens(message.usage)
    return ModelCallUsage(
        llm_internal_response_id=llm_internal_response_id,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        completion_cost=calculate_completion_cost(
            handler.llm,
            message.usage.input_tokens,
            message.usage.output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        ),
        stop_reason=message.stop_reason,
    )


def ignored_fields_note(tool_def: ToolDef, ignored: list[str]) -> str:
    """Tell the model which input fields were ignored and which fields the tool has.

    Args:
        tool_def: The tool that was called.
        ignored: The unrecognised input fields.
    """
    return (
        f"Note: these input fields are not recognised and were ignored: {', '.join(f'{name!r}' for name in ignored)}. "
        f"The fields are: {', '.join(tool_def.field_names)}."
    )


def format_validation_error(error: ValidationError) -> str:
    """Compact, model-readable summary of a ValidationError."""
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in error.errors())


def _append_turn_limit_guidance(raw_messages: list[dict]) -> None:
    """Nudge the model to answer now, by adding text to the pending tool-results message.

    A no-op if there is no pending tool-results message yet (the limit was hit before any tool
    ran this turn): the state machine in `run_agent_loop` still enforces the limit from the next
    call onward, just without a visible nudge for this one call.
    """
    if raw_messages and raw_messages[-1]["role"] == "user":
        raw_messages[-1]["content"].append({"type": "text", "text": TURN_LIMIT_MESSAGE})


def _total_result_chars(result: AgentLoopResult) -> int:
    """Chars of tool_result content returned so far this turn, excluding error placeholders."""
    return sum(call.result_chars for call in result.tool_calls if not call.is_error)


def _tool_result(tool_use_id: str, content: str, is_error: bool) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def _record(
    call: ToolCall, block: ToolUseBlock, args: dict | None, is_error: bool, duration_ms: int, content: str
) -> ToolCallRecord:
    return ToolCallRecord(
        iteration=call.iteration,
        tool_use_id=block.id,
        tool_name=block.name,
        args=args,
        is_error=is_error,
        duration_ms=duration_ms,
        result_chars=len(content),
        llm_internal_response_id=call.llm_internal_response_id,
    )


def _merge_citations(result: AgentLoopResult, exec_result: ToolExecResult) -> bool:
    """Add a tool's citations and sources to the turn's, skipping duplicates.

    Returns:
        Whether anything was added.
    """
    before = (len(result.citations), result.sources)
    seen = {json.dumps(c, sort_keys=True) for c in result.citations}
    for citation in exec_result.citations:
        key = json.dumps(citation, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.citations.append(citation)

    merged = {}
    for name in Sources.model_fields:
        existing = getattr(result.sources, name) or []
        incoming = getattr(exec_result.sources, name) or []
        keys = {(s.pretty_name, s.url) for s in existing}
        combined = list(existing)
        for source in incoming:
            if (source.pretty_name, source.url) not in keys:
                keys.add((source.pretty_name, source.url))
                combined.append(source)
        merged[name] = combined or None
    result.sources = Sources(**merged)
    return before != (len(result.citations), result.sources)
