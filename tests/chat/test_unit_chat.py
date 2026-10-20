import logging
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from app.chat.prompts import build_chat_system_prompt
from app.chat.schemas import (
    CentralGuidanceSource,
    GovUkSearchSource,
    MessageDefaults,
    SmartTargetsSource,
    Sources,
    UserDocumentSource,
)
from app.config import settings
from app.database.table import async_db_session

logger = logging.getLogger(__name__)


pytestmark = [
    pytest.mark.messages,
    pytest.mark.unit,
]


@pytest.fixture(autouse=True)
def _reset_system_prompt_last_known_good():
    """`build_chat_system_prompt` caches the last successful API-sourced segment at module
    scope (see app.chat.prompts); tests that exercise failures must not see another test's
    success (or vice versa)."""
    import app.chat.prompts as prompts_module

    prompts_module._LAST_GOOD_SEGMENTS.clear()
    prompts_module._last_dynamic_block_hash = None
    yield
    prompts_module._LAST_GOOD_SEGMENTS.clear()
    prompts_module._last_dynamic_block_hash = None


def test_message_defaults():
    message = MessageDefaults(chat_id=1, auth_session_id=2, llm_id=3)
    assert isinstance(UUID(message.uuid, version=4), UUID)
    assert message.chat_id == 1
    assert message.auth_session_id == 2
    assert not message.interrupted
    assert message.llm_id == 3
    assert message.tokens == 0


@patch("app.chat.schemas.uuid4")
def test_auto_generated_uuid(mock_uuid4):
    mock_uuid4.side_effect = [
        UUID("123e4567-e89b-12d3-a456-426614174000"),
        UUID("223e4567-e89b-12d3-a456-426614174000"),
    ]

    message1 = MessageDefaults(chat_id=1, auth_session_id=2, llm_id=3)
    message2 = MessageDefaults(chat_id=1, auth_session_id=2, llm_id=3)

    assert message1.uuid != message2.uuid
    assert message1.uuid == "123e4567-e89b-12d3-a456-426614174000"
    assert message2.uuid == "223e4567-e89b-12d3-a456-426614174000"


@pytest.mark.asyncio
async def test_return_type():
    async with async_db_session() as db_session:
        response = await build_chat_system_prompt(db_session)
        logging.info(f"Generated system prompt: {response}")
    assert isinstance(response, list)
    assert all(block.get("type") == "text" for block in response)


@pytest.mark.asyncio
async def test_cache_control_present_when_enabled():
    with patch.object(settings, "system_prompt_caching_enabled", True):
        async with async_db_session() as db_session:
            response = await build_chat_system_prompt(db_session)
    assert all("cache_control" in block for block in response)


@pytest.mark.asyncio
async def test_cache_control_absent_when_disabled():
    with patch.object(settings, "system_prompt_caching_enabled", False):
        async with async_db_session() as db_session:
            response = await build_chat_system_prompt(db_session)
    assert all("cache_control" not in block for block in response)


def test_empty_sources_serialization():
    """Test that empty Sources serializes to empty JSON object and deserializes back correctly."""
    # Create empty Sources
    sources = Sources()

    # Serialize to JSON
    json_str = sources.model_dump_json(exclude_none=True)

    # Verify it's an empty JSON object
    assert json_str == "{}"

    # Deserialize back
    deserialized = Sources.model_validate_json(json_str)

    # Verify all fields are None
    assert deserialized.central_guidance_sources is None
    assert deserialized.user_document_sources is None
    assert deserialized.gov_uk_search_sources is None
    assert deserialized.smart_targets_sources is None


def test_sources_with_single_source_type():
    """Test that Sources with only one source type serializes and deserializes correctly."""
    # Create Sources with only central_guidance_sources
    sources = Sources(
        central_guidance_sources=[
            CentralGuidanceSource(pretty_name="Document 1", url="https://example.com/doc1"),
            CentralGuidanceSource(pretty_name="Document 2", url="https://example.com/doc2"),
        ]
    )

    # Serialize to JSON
    json_str = sources.model_dump_json(exclude_none=True)

    # Verify JSON only contains central_guidance_sources key
    import json

    parsed = json.loads(json_str)
    assert "central_guidance_sources" in parsed
    assert "user_document_sources" not in parsed
    assert "gov_uk_search_sources" not in parsed
    assert "smart_targets_sources" not in parsed
    assert len(parsed["central_guidance_sources"]) == 2

    # Deserialize back
    deserialized = Sources.model_validate_json(json_str)

    # Verify the data matches
    assert deserialized.central_guidance_sources is not None
    assert len(deserialized.central_guidance_sources) == 2
    assert deserialized.central_guidance_sources[0].pretty_name == "Document 1"
    assert deserialized.central_guidance_sources[0].url == "https://example.com/doc1"
    assert deserialized.central_guidance_sources[1].pretty_name == "Document 2"
    assert deserialized.central_guidance_sources[1].url == "https://example.com/doc2"

    # Verify other fields are None
    assert deserialized.user_document_sources is None
    assert deserialized.gov_uk_search_sources is None
    assert deserialized.smart_targets_sources is None


def test_sources_with_multiple_source_types():
    """Test that Sources with multiple source types serializes and deserializes correctly."""
    # Create Sources with multiple source types
    sources = Sources(
        central_guidance_sources=[CentralGuidanceSource(pretty_name="Central Doc", url="https://example.com/central")],
        gov_uk_search_sources=[GovUkSearchSource(pretty_name="Gov.UK Page", url="https://www.gov.uk/page")],
        user_document_sources=[UserDocumentSource(pretty_name="User Doc", url="https://example.com/user")],
    )

    # Serialize to JSON
    json_str = sources.model_dump_json(exclude_none=True)

    # Verify JSON contains exactly the three keys
    import json

    parsed = json.loads(json_str)
    assert "central_guidance_sources" in parsed
    assert "gov_uk_search_sources" in parsed
    assert "user_document_sources" in parsed
    assert "smart_targets_sources" not in parsed  # This one should be excluded

    # Deserialize back
    deserialized = Sources.model_validate_json(json_str)

    # Verify all populated data matches original
    assert deserialized.central_guidance_sources is not None
    assert len(deserialized.central_guidance_sources) == 1
    assert deserialized.central_guidance_sources[0].pretty_name == "Central Doc"

    assert deserialized.gov_uk_search_sources is not None
    assert len(deserialized.gov_uk_search_sources) == 1
    assert deserialized.gov_uk_search_sources[0].pretty_name == "Gov.UK Page"

    assert deserialized.user_document_sources is not None
    assert len(deserialized.user_document_sources) == 1
    assert deserialized.user_document_sources[0].pretty_name == "User Doc"

    # Verify unpopulated field is None
    assert deserialized.smart_targets_sources is None


def test_nested_source_objects_round_trip():
    """Test that complex nested source objects preserve all data in round-trip serialization."""
    # Create Sources with multiple items in each list
    sources = Sources(
        central_guidance_sources=[
            CentralGuidanceSource(pretty_name="Central 1", url="https://example.com/central1"),
            CentralGuidanceSource(pretty_name="Central 2", url="https://example.com/central2"),
            CentralGuidanceSource(pretty_name="Central 3", url="https://example.com/central3"),
        ],
        user_document_sources=[
            UserDocumentSource(pretty_name="User 1", url="https://example.com/user1"),
            UserDocumentSource(pretty_name="User 2", url="https://example.com/user2"),
        ],
        gov_uk_search_sources=[
            GovUkSearchSource(pretty_name="Gov 1", url="https://www.gov.uk/1"),
        ],
        smart_targets_sources=[
            SmartTargetsSource(pretty_name="Smart 1", url="https://example.com/smart1"),
            SmartTargetsSource(pretty_name="Smart 2", url="https://example.com/smart2"),
        ],
    )

    # Serialize to JSON
    json_str = sources.model_dump_json(exclude_none=True)

    # Deserialize back
    deserialized = Sources.model_validate_json(json_str)

    # Verify all data is preserved
    assert deserialized.central_guidance_sources
    assert deserialized.user_document_sources
    assert deserialized.gov_uk_search_sources
    assert deserialized.smart_targets_sources

    assert len(deserialized.central_guidance_sources) == 3
    assert deserialized.central_guidance_sources[0].pretty_name == "Central 1"
    assert deserialized.central_guidance_sources[1].pretty_name == "Central 2"
    assert deserialized.central_guidance_sources[2].pretty_name == "Central 3"

    assert len(deserialized.user_document_sources) == 2
    assert deserialized.user_document_sources[0].url == "https://example.com/user1"
    assert deserialized.user_document_sources[1].url == "https://example.com/user2"

    assert len(deserialized.gov_uk_search_sources) == 1
    assert deserialized.gov_uk_search_sources[0].pretty_name == "Gov 1"

    assert len(deserialized.smart_targets_sources) == 2
    assert deserialized.smart_targets_sources[0].url == "https://example.com/smart1"
    assert deserialized.smart_targets_sources[1].url == "https://example.com/smart2"

    # Verify the round-trip preserves all fields
    json_str_2 = deserialized.model_dump_json(exclude_none=True)
    assert json_str == json_str_2


# Tests for Smart Targets disable functionality


@pytest.mark.asyncio
async def test_smart_targets_disabled_skips_services():
    """Test that services are not called when Smart Targets is disabled."""
    # Use context managers for clarity and to avoid pytest patch parameter issues in CI
    with patch("app.chat.prompts.CAMPAIGN_BENCHMARKS_SERVICE_DISABLED", True):
        with patch("app.chat.prompts.SmartTargetsService") as mock_smart:
            with patch("app.chat.prompts.BmdbEditionService") as mock_bmdb:
                async with async_db_session() as db_session:
                    response = await build_chat_system_prompt(db_session)

    mock_smart.return_value.get_available_metrics.assert_not_called()
    mock_bmdb.get_latest_edition.assert_not_called()
    assert isinstance(response, list)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_smart_targets_enabled_calls_services():
    """Test that services are called when Smart Targets is enabled."""
    with patch("app.chat.prompts.CAMPAIGN_BENCHMARKS_SERVICE_DISABLED", False):
        with patch("app.chat.prompts.SmartTargetsService") as mock_smart:
            with patch("app.chat.prompts.BmdbEditionService") as mock_bmdb:
                mock_smart.return_value.get_available_metrics = AsyncMock(return_value=["metric1", "metric2"])

                mock_edition = AsyncMock()
                mock_edition.version_number = "1.0"
                mock_edition.date_received = "2026-03-01"
                mock_edition.latest_campaign_end_date = "2026-02-28"
                mock_edition.earliest_campaign_end_date = "2024-01-01"
                mock_edition.n_campaigns = 50
                mock_edition.max_media_spend = 100000
                mock_edition.min_media_spend = 1000
                mock_bmdb.get_latest_edition = AsyncMock(return_value=mock_edition)

                async with async_db_session() as db_session:
                    response = await build_chat_system_prompt(db_session)

    mock_smart.return_value.get_available_metrics.assert_called_once()
    mock_bmdb.get_latest_edition.assert_called_once()
    full_text = " ".join(block["text"] for block in response)
    assert "campaign benchmarks tool" in full_text
    assert "metric1" in full_text


@pytest.mark.asyncio
async def test_smart_targets_error_continues_gracefully():
    """Test that system prompt builds even if Smart Targets service fails."""
    from app.smart_targets.exceptions import GetSmartTargetsMetricsError

    with patch("app.chat.prompts.CAMPAIGN_BENCHMARKS_SERVICE_DISABLED", False):
        with patch("app.chat.prompts.SmartTargetsService") as mock_smart:
            with patch("app.chat.prompts.BmdbEditionService") as mock_bmdb:
                mock_smart.return_value.get_available_metrics = AsyncMock(
                    side_effect=GetSmartTargetsMetricsError("Connection failed")
                )

                mock_edition = AsyncMock()
                mock_edition.version_number = "1.0"
                mock_bmdb.get_latest_edition = AsyncMock(return_value=mock_edition)

                async with async_db_session() as db_session:
                    response = await build_chat_system_prompt(db_session)

    assert isinstance(response, list)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_bmdb_error_continues_gracefully():
    """Test that system prompt builds even if BMDB service fails."""
    from app.bmdb.exceptions import GetBenchmarkDatabaseEditionError

    with patch("app.chat.prompts.CAMPAIGN_BENCHMARKS_SERVICE_DISABLED", False):
        with patch("app.chat.prompts.SmartTargetsService") as mock_smart:
            with patch("app.chat.prompts.BmdbEditionService") as mock_bmdb:
                mock_smart.return_value.get_available_metrics = AsyncMock(return_value=["metric1"])
                mock_bmdb.get_latest_edition = AsyncMock(
                    side_effect=GetBenchmarkDatabaseEditionError("API unavailable")
                )

                async with async_db_session() as db_session:
                    response = await build_chat_system_prompt(db_session)

    assert isinstance(response, list)
    assert "metric1" in " ".join(block["text"] for block in response)


@pytest.mark.asyncio
def _mock_edition() -> AsyncMock:
    edition = AsyncMock()
    edition.version_number = "1.0"
    edition.date_received = "2026-03-01"
    edition.latest_campaign_end_date = "2026-02-28"
    edition.earliest_campaign_end_date = "2024-01-01"
    edition.n_campaigns = 50
    edition.max_media_spend = 100000
    edition.min_media_spend = 1000
    return edition


@pytest.mark.asyncio
async def test_smart_targets_metrics_are_sorted():
    """API order isn't deterministic; the prompt must not vary just because the API reordered."""
    with patch("app.chat.prompts.CAMPAIGN_BENCHMARKS_SERVICE_DISABLED", False):
        with patch("app.chat.prompts.SmartTargetsService") as mock_smart:
            with patch("app.chat.prompts.BmdbEditionService") as mock_bmdb:
                mock_smart.return_value.get_available_metrics = AsyncMock(return_value=["zebra", "apple"])
                mock_bmdb.get_latest_edition = AsyncMock(return_value=_mock_edition())

                async with async_db_session() as db_session:
                    response = await build_chat_system_prompt(db_session)

    full_text = " ".join(block["text"] for block in response)
    assert full_text.index("apple") < full_text.index("zebra")


@pytest.mark.asyncio
async def test_smart_targets_metrics_fall_back_to_last_known_good_on_transient_failure():
    """A later failure re-uses the last successful value instead of dropping the segment,
    which would otherwise change the (cached) system prompt for every chat in this process."""
    from app.smart_targets.exceptions import GetSmartTargetsMetricsError

    with patch("app.chat.prompts.CAMPAIGN_BENCHMARKS_SERVICE_DISABLED", False):
        with patch("app.chat.prompts.SmartTargetsService") as mock_smart:
            with patch("app.chat.prompts.BmdbEditionService") as mock_bmdb:
                mock_bmdb.get_latest_edition = AsyncMock(return_value=_mock_edition())
                mock_smart.return_value.get_available_metrics = AsyncMock(return_value=["metric1"])
                async with async_db_session() as db_session:
                    await build_chat_system_prompt(db_session)

                mock_smart.return_value.get_available_metrics = AsyncMock(
                    side_effect=GetSmartTargetsMetricsError("down")
                )
                async with async_db_session() as db_session:
                    response = await build_chat_system_prompt(db_session)

    assert "metric1" in " ".join(block["text"] for block in response)


def _agent_turn(should_compact: bool = False, prefix_before_current_turn: list[dict] | None = None):
    from app.chat.service import _AgentTurn

    return _AgentTurn(
        chat=MagicMock(id=1),
        input_data=MagicMock(),
        llm=MagicMock(),
        llm_obj=MagicMock(),
        system=[{"type": "text", "text": "system"}],
        user_message=MagicMock(id=10, content="question"),
        history=[],
        formatted_messages=[{"role": "user", "content": "question"}],
        prefix_before_current_turn=prefix_before_current_turn or [],
        ai_message_defaults=MessageDefaults(chat_id=1, auth_session_id=2, llm_id=3),
        should_compact=should_compact,
        last_assistant_message_id=9,
        request_kwargs={"thinking": {"type": "disabled"}},
    )


def _loop_result():
    from decimal import Decimal

    from app.chat.agent_loop import AgentLoopResult, ModelCallUsage

    def usage(input_tokens, cache_read):
        return ModelCallUsage(
            llm_internal_response_id=None,
            input_tokens=input_tokens,
            output_tokens=5,
            cache_read_tokens=cache_read,
            cache_write_tokens=7,
            completion_cost=Decimal("0.01"),
            stop_reason="end_turn",
        )

    return AgentLoopResult(
        display_text="The colour answer",
        raw_messages=[{"role": "assistant", "content": [{"type": "text", "text": "The color answer"}]}],
        citations=[{"docname": "Doc", "docurl": "https://example.com"}],
        sources=Sources(central_guidance_sources=[CentralGuidanceSource(pretty_name="Doc", url="https://example.com")]),
        model_calls=[usage(10, 3), usage(20, 4)],
    )


@patch("app.chat.service.ChatTable")
@patch("app.chat.service.MessageTable")
def test_agent_turn_save_writes_summed_usage_and_raw_messages(mock_message_table_cls, mock_chat_table_cls):
    mock_message_repo = MagicMock()
    mock_message_table_cls.return_value = mock_message_repo
    mock_message_repo.create.return_value = MagicMock(id=2, chat_id=1)
    turn = _agent_turn()

    turn._save(_loop_result())

    _, user_update = mock_message_repo.update.call_args[0]
    assert (user_update["tokens"], user_update["cache_read_tokens"], user_update["cache_write_tokens"]) == (30, 7, 14)
    assistant = mock_message_repo.create.call_args[0][0]
    assert assistant["content"] == "The colour answer"
    assert assistant["raw_messages"][0]["content"][0]["text"] == "The color answer"
    assert assistant["tokens"] == 10
    assert str(assistant["completion_cost"]) == "0.02"
    assert assistant["citation"] == user_update["citation"] == '[{"docname": "Doc", "docurl": "https://example.com"}]'
    assert assistant["parent_message_id"] == 10


@patch("app.chat.service.schedule_compaction")
@patch("app.chat.service.ChatTable")
@patch("app.chat.service.MessageTable")
def test_agent_turn_save_schedules_compaction_with_tools(mock_message_table_cls, mock_chat_table_cls, mock_schedule):
    mock_message_table_cls.return_value.create.return_value = MagicMock(id=2, chat_id=1)
    turn = _agent_turn(should_compact=True)
    turn._tool_specs = [{"name": "search"}]

    turn._save(_loop_result())

    kwargs = mock_schedule.call_args.kwargs
    assert kwargs["up_to_message_id"] == 9
    assert kwargs["prefix_tokens"] == 10 + 3 + 7
    assert kwargs["extra_api_kwargs"] == {"thinking": {"type": "disabled"}, "tools": [{"name": "search"}]}


@patch("app.chat.service.schedule_compaction")
@patch("app.chat.service.ChatTable")
@patch("app.chat.service.MessageTable")
def test_agent_turn_compaction_prefix_keeps_a_trailing_tool_result(
    mock_message_table_cls, mock_chat_table_cls, mock_schedule
):
    """A stored turn that ends with a tool_result is merged into the current user message in
    formatted_messages; the compaction prefix must keep it, or the request ends on a bare tool_use."""
    long_text = "long " * 1000
    tool_result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "r"}]}
    prefix = [
        {"role": "user", "content": long_text},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "x", "input": {}}]},
        tool_result,
    ]
    mock_message_table_cls.return_value.create.return_value = MagicMock(id=2, chat_id=1)
    turn = _agent_turn(should_compact=True, prefix_before_current_turn=prefix)

    turn._save(_loop_result())

    sent = mock_schedule.call_args.kwargs["formatted_messages"]
    assert sent[-1] == tool_result
    assert "cache_control" in sent[1]["content"][-1]
    assert sent[0] == prefix[0]


async def test_record_tool_calls_adds_one_row_per_record():
    from contextlib import asynccontextmanager

    from app.chat.agent_loop import ToolCallRecord
    from app.chat.service import _record_tool_calls

    added = []

    class FakeSession:
        def add_all(self, rows):
            added.extend(rows)

    @asynccontextmanager
    async def fake_async_db_session():
        yield FakeSession()

    record = ToolCallRecord(
        iteration=1,
        tool_use_id="toolu_1",
        tool_name="search_gov_uk",
        args={"queries": ["x"]},
        is_error=False,
        duration_ms=120,
        result_chars=500,
        llm_internal_response_id=77,
    )

    with patch("app.chat.service.async_db_session", fake_async_db_session):
        await _record_tool_calls(2, [record])

    assert [(r.message_id, r.llm_internal_response_id, r.tool_name, r.args, r.duration_ms) for r in added] == [
        (2, 77, "search_gov_uk", {"queries": ["x"]}, 120)
    ]


@pytest.mark.parametrize("entry_point", ["chat_add_message", "chat_create"])
async def test_non_streaming_turn_runs_after_the_request_session_closes(entry_point):
    """The request-scope writes are committed and the session closed before the agent loop runs (M5)."""
    from contextlib import asynccontextmanager

    from app.chat import service

    session_open = False
    open_during_run = []

    @asynccontextmanager
    async def fake_async_db_session():
        nonlocal session_open
        session_open = True
        try:
            yield AsyncMock()
        finally:
            session_open = False

    async def run():
        open_during_run.append(session_open)
        return MagicMock(client_response=lambda: {}, dict=lambda: {})

    turn = MagicMock(run=run)
    chat = MagicMock(id=1, user_id=1, client_response=lambda: {}, dict=lambda: {})
    with (
        patch.object(service, "async_db_session", fake_async_db_session),
        patch.object(service, "chat_create_message", AsyncMock(return_value=turn)),
        patch.object(service, "_chat_message_with_documents", AsyncMock(side_effect=lambda c, s, data: data)),
        patch.object(service, "ChatTable", return_value=MagicMock(create=MagicMock(return_value=chat))),
        patch.object(service, "ChatWithLatestMessage", MagicMock()),
        patch.object(service, "ChatCreateMessageInput", MagicMock()),
    ):
        data = MagicMock(stream=False, use_case_id=None, to_dict=lambda: {})
        if entry_point == "chat_add_message":
            await service.chat_add_message(chat, data)
        else:
            await service.chat_create(data)

    assert open_during_run == [False]
