import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from anthropic.types.message import Message as AnthropicMessage
from anthropic.types.usage import Usage
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.bedrock import call_log
from app.bedrock.call_log import LlmCallContext, log_llm_call
from app.database.models import LLM, Chat, LlmInternalResponse, Message

INSERT = "app.bedrock.call_log.DbOperations.insert_llm_internal_response_id_query"


def make_llm():
    llm = MagicMock()
    llm.input_cost_per_token = 1e-6
    llm.output_cost_per_token = 5e-6
    llm.cache_read_cost_multiplier = None
    return llm


def make_response(cache_read=3, cache_write=7):
    return AnthropicMessage(
        id="msg_test",
        type="message",
        role="assistant",
        content=[],
        model="m",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )


def session_patch(session=None):
    session = session or MagicMock()
    session.scalar = getattr(session, "scalar", None) or AsyncMock()

    @asynccontextmanager
    async def _session():
        yield session

    return patch("app.bedrock.call_log.async_db_session", _session)


async def test_writes_usage_cost_and_context_and_returns_the_row_id():
    with session_patch(), patch(INSERT, new_callable=AsyncMock, return_value=MagicMock(id=42)) as insert:
        row_id = await log_llm_call(
            make_llm(), make_response(), LlmCallContext(purpose="title", chat_id=1, message_id=2)
        )

    assert row_id == 42
    kwargs = insert.call_args.kwargs
    assert (kwargs["tokens_in"], kwargs["tokens_out"]) == (10, 5)
    assert (kwargs["cache_read_tokens"], kwargs["cache_write_tokens"]) == (3, 7)
    assert kwargs["completion_cost"] > 0
    assert (kwargs["purpose"], kwargs["chat_id"], kwargs["message_id"]) == ("title", 1, 2)


async def test_chat_id_is_looked_up_from_message_id_when_not_given():
    session = MagicMock()
    session.scalar = AsyncMock(return_value=99)
    with session_patch(session), patch(INSERT, new_callable=AsyncMock, return_value=MagicMock(id=1)) as insert:
        await log_llm_call(make_llm(), make_response(), LlmCallContext(purpose="x", message_id=5))

    assert insert.call_args.kwargs["chat_id"] == 99
    assert insert.call_args.kwargs["message_id"] == 5


async def test_returns_none_and_does_not_raise_when_the_insert_fails():
    with session_patch(), patch(INSERT, new_callable=AsyncMock, side_effect=RuntimeError("db down")):
        assert await log_llm_call(make_llm(), make_response(), LlmCallContext(purpose="x")) is None


async def test_returns_none_when_the_session_cannot_be_opened():
    @asynccontextmanager
    async def broken_session():
        raise ConnectionError("no connection")
        yield

    with patch("app.bedrock.call_log.async_db_session", broken_session):
        assert await log_llm_call(make_llm(), make_response(), LlmCallContext(purpose="x")) is None


async def test_returns_none_when_the_response_is_malformed():
    with session_patch():
        assert await log_llm_call(make_llm(), {"not": "a message"}, LlmCallContext(purpose="x")) is None


async def test_rejected_chat_and_message_ids_are_dropped_and_the_row_still_written():
    rejected = IntegrityError("insert", {}, Exception("fk violation"))
    with session_patch(), patch(INSERT, new_callable=AsyncMock, side_effect=[rejected, MagicMock(id=8)]) as insert:
        row_id = await log_llm_call(
            make_llm(), make_response(), LlmCallContext(purpose="x", chat_id=999, message_id=999)
        )

    assert row_id == 8
    retry = insert.call_args_list[1].kwargs
    assert (retry["chat_id"], retry["message_id"], retry["purpose"]) == (None, None, "x")


async def test_an_integrity_error_with_no_ids_is_not_retried():
    rejected = IntegrityError("insert", {}, Exception("other"))
    with session_patch(), patch(INSERT, new_callable=AsyncMock, side_effect=rejected) as insert:
        assert await log_llm_call(make_llm(), make_response(), LlmCallContext(purpose="x")) is None

    insert.assert_awaited_once()


async def test_a_hung_database_times_out_instead_of_blocking_the_call():
    async def hang(**kwargs):
        await asyncio.sleep(60)

    with (
        session_patch(),
        patch(INSERT, side_effect=hang),
        patch.object(call_log, "LOG_TIMEOUT_SECONDS", 0.05),
    ):
        assert await log_llm_call(make_llm(), make_response(), LlmCallContext(purpose="x")) is None


async def test_row_is_committed_on_its_own_session(db_session, chat):
    """End to end against the test DB: the row is visible from another session straight away."""
    llm = (await db_session.execute(select(LLM).where(LLM.input_cost_per_token.isnot(None)).limit(1))).scalar_one()
    user_message = (
        await db_session.execute(
            select(Message).join(Chat, Chat.id == Message.chat_id).where(Chat.uuid == chat.uuid).limit(1)
        )
    ).scalar_one()

    row_id = await log_llm_call(
        llm, make_response(), LlmCallContext(purpose="integration_test", message_id=user_message.id)
    )

    assert row_id is not None
    row = (await db_session.execute(select(LlmInternalResponse).where(LlmInternalResponse.id == row_id))).scalar_one()
    assert row.purpose == "integration_test"
    assert row.message_id == user_message.id
    assert row.chat_id == user_message.chat_id
