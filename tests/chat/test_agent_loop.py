"""Offline tests for the chat agent loop, driven by a scripted fake Bedrock handler."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from pydantic import Field
from sqlalchemy import delete, select, text

from app.bedrock.bedrock_stream import StreamDone, StreamStarted, StreamText
from app.bedrock.call_log import LlmCallContext
from app.bedrock.tool_schema import ToolArgs, ToolDef
from app.central_guidance.tools import SearchCentralGuidanceArgs
from app.chat.agent_loop import (
    CALL_LIMIT_MESSAGE,
    NOT_RUN_MESSAGE,
    RESULT_TRUNCATED_NOTE,
    TOOL_FAILED_MESSAGE,
    TOOL_TIMEOUT_MESSAGE,
    TURN_LIMIT_MESSAGE,
    UNKNOWN_TOOL_MESSAGE,
    LoopFinished,
    SourcesUpdated,
    TextDelta,
    ToolStatus,
    run_agent_loop,
)
from app.chat.schemas import CentralGuidanceSource, Sources
from app.chat.tools import (
    RegisteredTool,
    ToolContext,
    ToolExecResult,
    _exec_search_central_guidance,
    build_available_tools,
)
from app.config import settings
from app.database.models import Theme
from app.database.table import AsyncEngineProvider, async_db_session

HISTORY = [{"role": "user", "content": "What is OASIS?"}]


# --- fakes ---


def _message(*content, stop_reason: str = "end_turn") -> Message:
    return Message(
        id="msg",
        type="message",
        role="assistant",
        model="m",
        content=list(content),
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=100, output_tokens=10, cache_read_input_tokens=50, cache_creation_input_tokens=5),
    )


def _text(value: str) -> TextBlock:
    return TextBlock(type="text", text=value)


def _tool_use(name: str, tool_input: dict, block_id: str) -> ToolUseBlock:
    return ToolUseBlock(id=block_id, type="tool_use", name=name, input=tool_input)


class FakeHandler:
    """Returns scripted responses, streaming their text in small chunks, and records requests."""

    def __init__(self, *responses: Message):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.llm = SimpleNamespace(input_cost_per_token=0.000003, output_cost_per_token=0.000015)
        self.logged = 0

    def stream_events(self, messages, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), **copy.deepcopy(kwargs)})
        response = self.responses.pop(0)

        async def events():
            yield StreamStarted()
            for block in response.content:
                if block.type == "text":
                    for i in range(0, len(block.text), 3):
                        yield StreamText(block.text[i : i + 3])
            self.logged += 1
            yield StreamDone(response, llm_internal_response_id=self.logged)

        return events()


class EchoArgs(ToolArgs):
    """Echo the query."""

    query: str = Field(description="Query")


ECHO = ToolDef("echo", EchoArgs, strict=False)


def _registered(
    executor,
    tool_def: ToolDef = ECHO,
    enabled: bool = True,
    is_available=lambda ctx: True,
    unavailable_message: str = "Not available",
) -> RegisteredTool:
    return RegisteredTool(
        tool_def=tool_def,
        executor=executor,
        enabled=lambda: enabled,
        is_available=is_available,
        unavailable_message=unavailable_message,
        running_label=lambda args: f"Echoing {args.query}",
        done_label=lambda args: f"Echoed {args.query}",
        error_label="Couldn't echo",
    )


async def _echo(ctx, args, call, db_session):
    return ToolExecResult(content=f"<echo>{args.query}</echo>")


def _ctx() -> ToolContext:
    return ToolContext(
        chat=SimpleNamespace(id=1),
        user_id=1,
        user_message=SimpleNamespace(id=1, content="What is OASIS?"),
        history=[],
        input_data=SimpleNamespace(use_rag=True, use_gov_uk_search_api=False, use_smart_targets=False),
    )


async def _run(handler, tools, ctx=None, **kwargs) -> tuple[list, object]:
    events = [e async for e in run_agent_loop(handler, history=HISTORY, tools=tools, ctx=ctx or _ctx(), **kwargs)]
    assert isinstance(events[-1], LoopFinished)
    return events, events[-1].result


def _tool_results(result) -> list[dict]:
    return [m for m in result.raw_messages if m["role"] == "user"][-1]["content"]


# --- tests ---


async def test_answer_without_tools():
    handler = FakeHandler(_message(_text("OASIS is a campaign framework.")))

    events, result = await _run(handler, [_registered(_echo)])

    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "OASIS is a campaign framework."
    assert result.display_text == "OASIS is a campaign framework."
    assert result.raw_messages == [
        {"role": "assistant", "content": [{"type": "text", "text": "OASIS is a campaign framework."}]}
    ]
    assert len(result.model_calls) == 1
    assert handler.logged == 1
    assert result.model_calls[0].llm_internal_response_id == 1
    assert handler.requests[0]["context"] == LlmCallContext(purpose="chat_loop", chat_id=1, message_id=1)


async def test_tool_call_then_answer():
    handler = FakeHandler(
        _message(_text("Let me check."), _tool_use("echo", {"query": "oasis"}, "toolu_1"), stop_reason="tool_use"),
        _message(_text("It's a framework.")),
    )

    events, result = await _run(handler, [_registered(_echo)])

    statuses = [e for e in events if isinstance(e, ToolStatus)]
    assert [(s.id, s.state, s.label) for s in statuses] == [
        ("toolu_1", "running", "Echoing oasis"),
        ("toolu_1", "done", "Echoed oasis"),
    ]
    assert result.display_text == "Let me check.\n\nIt's a framework."
    assert _tool_results(result) == [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "<echo>oasis</echo>"}]
    assert all(request["tools"] == [ECHO.spec] for request in handler.requests)
    assert all("tool_choice" not in request for request in handler.requests)
    assert result.tool_calls[0].args == {"query": "oasis"}
    assert result.tool_calls[0].is_error is False


async def test_later_requests_replay_turn_messages_verbatim():
    handler = FakeHandler(
        _message(_tool_use("echo", {"query": "a"}, "toolu_1"), stop_reason="tool_use"),
        _message(_tool_use("echo", {"query": "b"}, "toolu_2"), stop_reason="tool_use"),
        _message(_text("done")),
    )

    _, result = await _run(handler, [_registered(_echo)])

    def strip_cache_control(messages):
        return [
            {**m, "content": [{k: v for k, v in b.items() if k != "cache_control"} for b in m["content"]]}
            if isinstance(m["content"], list)
            else m
            for m in messages
        ]

    third_request = strip_cache_control(handler.requests[2]["messages"])
    assert third_request == [*HISTORY, *result.raw_messages[:4]]
    assert "cache_control" not in str(result.raw_messages)


async def test_two_tool_calls_answered_in_one_message_in_order():
    handler = FakeHandler(
        _message(
            _tool_use("echo", {"query": "a"}, "toolu_1"),
            _tool_use("echo", {"query": "b"}, "toolu_2"),
            stop_reason="tool_use",
        ),
        _message(_text("done")),
    )

    _, result = await _run(handler, [_registered(_echo)])

    assert [r["tool_use_id"] for r in _tool_results(result)] == ["toolu_1", "toolu_2"]
    assert [m["role"] for m in result.raw_messages] == ["assistant", "user", "assistant"]


async def test_invalid_input_is_error_and_loop_continues():
    handler = FakeHandler(
        _message(_tool_use("echo", {"wrong": 1}, "toolu_1"), stop_reason="tool_use"),
        _message(_text("done")),
    )

    events, result = await _run(handler, [_registered(_echo)])

    tool_result = _tool_results(result)[0]
    assert tool_result["is_error"] is True
    assert "query" in tool_result["content"]
    assert not [e for e in events if isinstance(e, ToolStatus)]
    assert result.display_text == "done"


async def test_unknown_tool_is_error():
    handler = FakeHandler(
        _message(_tool_use("nope", {"query": "a"}, "toolu_1"), stop_reason="tool_use"),
        _message(_text("done")),
    )

    _, result = await _run(handler, [_registered(_echo)])

    assert _tool_results(result) == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": UNKNOWN_TOOL_MESSAGE, "is_error": True}
    ]


async def test_unavailable_tool_gets_is_error_with_no_status_line():
    handler = FakeHandler(
        _message(_tool_use("echo", {"query": "a"}, "toolu_1"), stop_reason="tool_use"),
        _message(_text("done")),
    )
    executed = []

    async def recording_echo(ctx, args, call, db_session):
        executed.append(args.query)
        return ToolExecResult(content=args.query)

    tool = _registered(
        recording_echo, is_available=lambda ctx: False, unavailable_message="Turn on the 'Echo' option to use this."
    )
    events, result = await _run(handler, [tool])

    assert executed == []
    assert not [e for e in events if isinstance(e, ToolStatus)]
    assert _tool_results(result) == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "Turn on the 'Echo' option to use this.",
            "is_error": True,
        }
    ]
    assert result.tool_calls[0].is_error is True


async def test_calls_over_the_per_iteration_cap_are_answered_with_errors():
    handler = FakeHandler(
        _message(
            _tool_use("echo", {"query": "a"}, "toolu_1"),
            _tool_use("echo", {"query": "b"}, "toolu_2"),
            stop_reason="tool_use",
        ),
        _message(_text("done")),
    )
    executed = []

    async def recording_echo(ctx, args, call, db_session):
        executed.append(args.query)
        return ToolExecResult(content=args.query)

    with patch.object(settings, "agent_loop_max_calls_per_iteration", 1):
        _, result = await _run(handler, [_registered(recording_echo)])

    assert executed == ["a"]
    results = _tool_results(result)
    assert [r["tool_use_id"] for r in results] == ["toolu_1", "toolu_2"]
    assert results[1] == {
        "type": "tool_result",
        "tool_use_id": "toolu_2",
        "content": CALL_LIMIT_MESSAGE,
        "is_error": True,
    }


async def test_executor_exception_gives_generic_error():
    handler = FakeHandler(
        _message(_tool_use("echo", {"query": "a"}, "toolu_1"), stop_reason="tool_use"),
        _message(_text("done")),
    )

    async def failing(ctx, args, call, db_session):
        raise RuntimeError("connection to secret-host.internal refused")

    events, result = await _run(handler, [_registered(failing)])

    tool_result = _tool_results(result)[0]
    assert tool_result == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": TOOL_FAILED_MESSAGE,
        "is_error": True,
    }
    assert "secret-host" not in str(result.raw_messages)
    assert [(s.state, s.label) for s in events if isinstance(s, ToolStatus)] == [
        ("running", "Echoing a"),
        ("error", "Couldn't echo"),
    ]


async def test_truncated_tool_use_is_not_run_but_answered():
    handler = FakeHandler(
        _message(_text("Checking"), _tool_use("echo", {"query": "a"}, "toolu_1"), stop_reason="max_tokens")
    )
    executed = []

    async def recording_echo(ctx, args, call, db_session):
        executed.append(args.query)
        return ToolExecResult(content=args.query)

    _, result = await _run(handler, [_registered(recording_echo)])

    assert executed == []
    assert result.raw_messages[-1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": NOT_RUN_MESSAGE, "is_error": True}],
    }
    assert result.stop_reason == "max_tokens"


async def test_turn_limit_nudges_the_model_and_it_complies():
    handler = FakeHandler(
        _message(_tool_use("echo", {"query": "a"}, "toolu_1"), stop_reason="tool_use"),
        _message(_text("final")),
    )

    with patch.object(settings, "agent_loop_max_iterations", 2):
        _, result = await _run(handler, [_registered(_echo)])

    assert not any("tool_choice" in r for r in handler.requests)
    assert handler.requests[1]["tools"] == [ECHO.spec]
    last_message = handler.requests[1]["messages"][-1]
    assert last_message["role"] == "user"
    assert {"type": "text", "text": TURN_LIMIT_MESSAGE} in last_message["content"]
    assert result.display_text == "final"
    assert len(result.model_calls) == 2


async def test_turn_limit_ignored_then_forced_after_a_grace_call():
    handler = FakeHandler(
        _message(_tool_use("echo", {"query": "a"}, "toolu_1"), stop_reason="tool_use"),
        _message(_tool_use("echo", {"query": "b"}, "toolu_2"), stop_reason="tool_use"),
        _message(_tool_use("echo", {"query": "c"}, "toolu_3"), stop_reason="tool_use"),
        _message(_text("final")),
    )
    executed = []

    async def recording_echo(ctx, args, call, db_session):
        executed.append(args.query)
        return ToolExecResult(content=args.query)

    with patch.object(settings, "agent_loop_max_iterations", 2):
        _, result = await _run(handler, [_registered(recording_echo)])

    # Only the first (normal) call ran the tool; the two after the limit were refused.
    assert executed == ["a"]
    assert [r.get("tool_choice") for r in handler.requests] == [None, None, None, {"type": "none"}]
    tool_results = [
        block
        for m in result.raw_messages
        if m["role"] == "user"
        for block in m["content"]
        if block.get("type") == "tool_result"
    ]
    assert tool_results == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a"},
        {"type": "tool_result", "tool_use_id": "toolu_2", "content": TURN_LIMIT_MESSAGE, "is_error": True},
        {"type": "tool_result", "tool_use_id": "toolu_3", "content": TURN_LIMIT_MESSAGE, "is_error": True},
    ]
    # The nudge was appended alongside toolu_1's result, before the second call.
    assert result.raw_messages[1]["content"][-1] == {"type": "text", "text": TURN_LIMIT_MESSAGE}
    assert result.display_text == "final"


async def test_no_tools_means_no_tool_params():
    handler = FakeHandler(_message(_text("hi")))

    await _run(handler, [])

    assert "tools" not in handler.requests[0]
    assert "tool_choice" not in handler.requests[0]


async def test_citations_and_sources_deduplicated_across_calls():
    handler = FakeHandler(
        _message(
            _tool_use("echo", {"query": "a"}, "toolu_1"),
            _tool_use("echo", {"query": "b"}, "toolu_2"),
            stop_reason="tool_use",
        ),
        _message(_text("done")),
    )

    async def citing(ctx, args, call, db_session):
        return ToolExecResult(
            content="x",
            citations=[{"docname": "OASIS", "docurl": "https://example.com/oasis"}],
            sources=Sources(
                central_guidance_sources=[CentralGuidanceSource(pretty_name="OASIS", url="https://example.com/oasis")]
            ),
        )

    events, result = await _run(handler, [_registered(citing)])

    assert result.citations == [{"docname": "OASIS", "docurl": "https://example.com/oasis"}]
    assert len(result.sources.central_guidance_sources) == 1
    assert len([e for e in events if isinstance(e, SourcesUpdated)]) == 1


async def test_cache_marker_on_last_assistant_only_for_first_call_when_compacting():
    long_text = "long " * 1000
    history = [
        {"role": "user", "content": long_text},
        {"role": "assistant", "content": long_text},
        {"role": "user", "content": "new question"},
    ]
    handler = FakeHandler(
        _message(_tool_use("echo", {"query": "a"}, "toolu_1"), stop_reason="tool_use"),
        _message(_text("done")),
    )

    events = [
        e
        async for e in run_agent_loop(
            handler, history=history, tools=[_registered(_echo)], ctx=_ctx(), should_compact=True
        )
    ]

    first, second = handler.requests[0]["messages"], handler.requests[1]["messages"]
    assert "cache_control" in first[1]["content"][-1]
    assert isinstance(first[2]["content"], str)
    assert "cache_control" in second[-1]["content"][-1]
    assert isinstance(second[1]["content"], str)
    assert isinstance(events[-1], LoopFinished)


async def test_usage_summed_across_calls():
    handler = FakeHandler(
        _message(_tool_use("echo", {"query": "a"}, "toolu_1"), stop_reason="tool_use"),
        _message(_text("done")),
    )

    _, result = await _run(handler, [_registered(_echo)])

    assert (result.input_tokens, result.output_tokens, result.cache_read_tokens, result.cache_write_tokens) == (
        200,
        20,
        100,
        10,
    )
    assert result.completion_cost == sum(call.completion_cost for call in result.model_calls)
    assert result.completion_cost > 0


# --- tool registry ---


def test_build_available_tools_sorted_and_stable():
    first = [tool.tool_def.spec for tool in build_available_tools()]
    second = [tool.tool_def.spec for tool in build_available_tools()]

    assert first == second
    assert [spec["name"] for spec in first] == sorted(spec["name"] for spec in first)


def test_build_available_tools_filters_by_deploy_level_enabled_flag():
    """`enabled` (constant per deploy) filters the list; `is_available` (per-chat) does not."""
    on_tool = _registered(_echo, tool_def=ToolDef("on_tool", EchoArgs, strict=False), enabled=True)
    off_tool = _registered(
        _echo, tool_def=ToolDef("off_tool", EchoArgs, strict=False), enabled=False, is_available=lambda ctx: True
    )

    with patch("app.chat.tools.TOOL_REGISTRY", {"on_tool": on_tool, "off_tool": off_tool}):
        assert [t.name for t in build_available_tools()] == ["on_tool"]


def test_build_available_tools_not_affected_by_per_chat_options():
    """A tool the deploy enables stays in the list even when this chat's is_available is false."""
    ctx_off = ToolContext(
        **{
            **_ctx().__dict__,
            "input_data": SimpleNamespace(use_rag=False, use_gov_uk_search_api=False, use_smart_targets=False),
        }
    )

    tools = {t.name: t for t in build_available_tools()}
    assert "search_central_guidance" in tools
    assert tools["search_central_guidance"].is_available(ctx_off) is False


async def test_central_guidance_relevance_check_gets_a_cut_user_query():
    """Regression: a pasted document must not be sent whole to the chunk relevance evaluator."""
    ctx = ToolContext(
        **{**_ctx().__dict__, "user_message": SimpleNamespace(id=1, content="start " + "x" * 50_000 + " end")}
    )
    search = AsyncMock(return_value=("content", [], False))

    with (
        patch("app.chat.tools.search_central_guidance_for_tool", new=search),
        patch.object(settings, "helper_call_max_query_chars", 1000),
    ):
        await _exec_search_central_guidance(
            ctx, SearchCentralGuidanceArgs(queries=["q"]), SimpleNamespace(llm_internal_response_id=None), None
        )

    user_query = search.await_args.kwargs["user_query"]
    assert user_query.startswith("start ")
    assert user_query.endswith(" end")
    assert len(user_query) < 1100


# --- real database ---


async def test_each_tool_call_gets_its_own_session_committed_or_rolled_back():
    """Each executor runs on a short session of its own: a success is committed straight away, a
    failure rolls back only its own writes, and no connection is held during model calls."""
    run_id = uuid4().hex
    pool = AsyncEngineProvider.get().pool
    checked_out_during_model_calls = []

    class PoolCheckingHandler(FakeHandler):
        def stream_events(self, messages, **kwargs):
            checked_out_during_model_calls.append(pool.checkedout())
            return super().stream_events(messages, **kwargs)

    handler = PoolCheckingHandler(
        _message(
            _tool_use("write_db", {"query": "a"}, "toolu_1"),
            _tool_use("broken_db", {"query": "b"}, "toolu_2"),
            _tool_use("write_db", {"query": "c"}, "toolu_3"),
            stop_reason="tool_use",
        ),
        _message(_text("done")),
    )
    sessions = []

    async def write_db(ctx, args, call, db_session):
        sessions.append(db_session)
        db_session.add(Theme(title=f"{run_id}-{args.query}", subtitle="test"))
        await db_session.flush()
        return ToolExecResult(content=f"wrote {args.query}")

    async def broken_db(ctx, args, call, db_session):
        sessions.append(db_session)
        db_session.add(Theme(title=f"{run_id}-{args.query}", subtitle="test"))
        await db_session.flush()
        await db_session.execute(text("SELECT * FROM table_that_does_not_exist"))
        return ToolExecResult(content="unreachable")

    tools = [
        _registered(write_db, ToolDef("write_db", EchoArgs, strict=False)),
        _registered(broken_db, ToolDef("broken_db", EchoArgs, strict=False)),
    ]
    checked_out_before = pool.checkedout()

    _, result = await _run(handler, tools)

    results = _tool_results(result)
    assert results[0]["content"] == "wrote a"
    assert results[1]["is_error"] is True
    assert results[2]["content"] == "wrote c"
    assert len({id(s) for s in sessions}) == 3
    assert checked_out_during_model_calls == [checked_out_before, checked_out_before]
    async with async_db_session() as db_session:
        titles = (await db_session.scalars(select(Theme.title).where(Theme.title.startswith(run_id)))).all()
        assert sorted(titles) == [f"{run_id}-a", f"{run_id}-c"]
        await db_session.execute(delete(Theme).where(Theme.title.startswith(run_id)))


async def test_over_budget_on_the_first_call_still_offers_tools_without_tool_choice():
    handler = FakeHandler(_message(_text("answer")))

    with patch.object(settings, "chat_model_context_window_tokens", 1):
        await _run(handler, [_registered(_echo)])

    # Over budget before any tool ran this turn: no pending message to nudge, but tools stay
    # offered and tool_choice is still never sent.
    assert "tool_choice" not in handler.requests[0]
    assert handler.requests[0]["tools"] == [ECHO.spec]


async def test_result_char_budget_nudges_the_model_once_crossed():
    handler = FakeHandler(
        _message(_tool_use("echo", {"query": "a"}, "toolu_1"), stop_reason="tool_use"),
        _message(_text("final")),
    )

    with patch.object(settings, "agent_loop_max_result_chars_per_turn", 1):
        _, result = await _run(handler, [_registered(_echo)])

    assert not any("tool_choice" in r for r in handler.requests)
    last_message = handler.requests[1]["messages"][-1]
    assert last_message["role"] == "user"
    assert {"type": "text", "text": TURN_LIMIT_MESSAGE} in last_message["content"]
    assert result.display_text == "final"


async def test_result_char_budget_only_counts_successful_results():
    handler = FakeHandler(
        _message(_tool_use("boom", {"query": "a"}, "toolu_1"), stop_reason="tool_use"),
        _message(_tool_use("echo", {"query": "b"}, "toolu_2"), stop_reason="tool_use"),
        _message(_text("final")),
    )

    async def boom(ctx, args, call, db_session):
        raise RuntimeError("boom")

    tools = [_registered(boom, tool_def=ToolDef("boom", EchoArgs, strict=False)), _registered(_echo)]
    with patch.object(settings, "agent_loop_max_result_chars_per_turn", 1):
        _, result = await _run(handler, tools)

    # The failed call's error text doesn't count toward the budget; only the successful echo does.
    assert {"type": "text", "text": TURN_LIMIT_MESSAGE} not in handler.requests[1]["messages"][-1]["content"]
    assert {"type": "text", "text": TURN_LIMIT_MESSAGE} in handler.requests[2]["messages"][-1]["content"]
    assert not any("tool_choice" in r for r in handler.requests)
    assert result.display_text == "final"


async def test_result_char_budget_is_checked_before_each_call_in_an_iteration():
    handler = FakeHandler(
        _message(
            _tool_use("echo", {"query": "a"}, "toolu_1"),
            _tool_use("echo", {"query": "b"}, "toolu_2"),
            stop_reason="tool_use",
        ),
        _message(_text("final")),
    )
    executed = []

    async def recording_echo(ctx, args, call, db_session):
        executed.append(args.query)
        return ToolExecResult(content=args.query * 10)

    with patch.object(settings, "agent_loop_max_result_chars_per_turn", 5):
        _, result = await _run(handler, [_registered(recording_echo)])

    # The first result crosses the budget, so the second call in the same response isn't run.
    assert executed == ["a"]
    results = _tool_results(result)
    assert results[1] == {
        "type": "tool_result",
        "tool_use_id": "toolu_2",
        "content": TURN_LIMIT_MESSAGE,
        "is_error": True,
    }
    assert [(c.tool_use_id, c.is_error) for c in result.tool_calls] == [("toolu_1", False), ("toolu_2", True)]
    assert result.display_text == "final"


async def test_long_tool_result_is_truncated_to_the_per_call_cap():
    handler = FakeHandler(
        _message(_tool_use("echo", {"query": "a"}, "toolu_1"), stop_reason="tool_use"),
        _message(_text("final")),
    )

    async def long_echo(ctx, args, call, db_session):
        return ToolExecResult(content="x" * 500)

    cap = len(RESULT_TRUNCATED_NOTE) + 10
    with patch.object(settings, "agent_loop_max_result_chars_per_call", cap):
        _, result = await _run(handler, [_registered(long_echo)])

    assert _tool_results(result)[0]["content"] == "x" * 10 + RESULT_TRUNCATED_NOTE
    assert result.tool_calls[0].result_chars == cap


async def test_tool_that_times_out_gets_an_error_and_its_writes_are_rolled_back():
    run_id = uuid4().hex
    handler = FakeHandler(
        _message(_tool_use("echo", {"query": "a"}, "toolu_1"), stop_reason="tool_use"),
        _message(_text("done")),
    )

    async def slow(ctx, args, call, db_session):
        db_session.add(Theme(title=f"{run_id}-{args.query}", subtitle="test"))
        await db_session.flush()
        await asyncio.sleep(10)
        return ToolExecResult(content="unreachable")

    with patch.object(settings, "agent_loop_tool_timeout_seconds", 0.05):
        events, result = await _run(handler, [_registered(slow)])

    assert _tool_results(result)[0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": TOOL_TIMEOUT_MESSAGE,
        "is_error": True,
    }
    assert [s.state for s in events if isinstance(s, ToolStatus)] == ["running", "error"]
    assert result.display_text == "done"
    async with async_db_session() as db_session:
        titles = (await db_session.scalars(select(Theme.title).where(Theme.title.startswith(run_id)))).all()
        assert titles == []


class PagedArgs(ToolArgs):
    """Read a stretch."""

    query: str = Field(description="Query")
    start_chunk: int = Field(default=1, ge=1)


PAGED = ToolDef("paged", PagedArgs, strict=False)


async def _paged(ctx, args, call, db_session):
    return ToolExecResult(content=f"<paged>{args.query} from {args.start_chunk}</paged>")


async def test_ignored_input_fields_are_reported_in_the_result_and_the_call_still_runs():
    handler = FakeHandler(
        _message(
            _tool_use("paged", {"query": "oasis", "start": 25, "colour": "red"}, "toolu_1"), stop_reason="tool_use"
        ),
        _message(_text("done")),
    )

    _, result = await _run(handler, [_registered(_paged, PAGED)])

    tool_result = _tool_results(result)[0]
    assert "is_error" not in tool_result
    assert tool_result["content"].startswith("<paged>oasis from 1</paged>")
    assert "ignored: 'start', 'colour'." in tool_result["content"]
    assert "The fields are: query, start_chunk." in tool_result["content"]


async def test_no_note_when_every_input_field_is_recognised():
    handler = FakeHandler(
        _message(_tool_use("paged", {"query": "oasis", "start_chunk": 25}, "toolu_1"), stop_reason="tool_use"),
        _message(_text("done")),
    )

    _, result = await _run(handler, [_registered(_paged, PAGED)])

    assert _tool_results(result)[0]["content"] == "<paged>oasis from 25</paged>"


async def test_ignored_fields_are_also_reported_when_the_input_is_invalid():
    handler = FakeHandler(
        _message(_tool_use("paged", {"queries": "oasis"}, "toolu_1"), stop_reason="tool_use"),
        _message(_text("done")),
    )

    _, result = await _run(handler, [_registered(_paged, PAGED)])

    tool_result = _tool_results(result)[0]
    assert tool_result["is_error"] is True
    assert "query: Field required" in tool_result["content"]
    assert "ignored: 'queries'." in tool_result["content"]
