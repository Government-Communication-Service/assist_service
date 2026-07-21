import logging
from unittest.mock import AsyncMock, patch
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
from app.database.table import async_db_session

logger = logging.getLogger(__name__)


pytestmark = [
    pytest.mark.messages,
    pytest.mark.unit,
]


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
    with patch("app.chat.prompts.SMART_TARGETS_SERVICE_DISABLED", True):
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
    with patch("app.chat.prompts.SMART_TARGETS_SERVICE_DISABLED", False):
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
    assert "Smart Targets" in full_text
    assert "metric1" in full_text


@pytest.mark.asyncio
async def test_smart_targets_error_continues_gracefully():
    """Test that system prompt builds even if Smart Targets service fails."""
    from app.smart_targets.exceptions import GetSmartTargetsMetricsError

    with patch("app.chat.prompts.SMART_TARGETS_SERVICE_DISABLED", False):
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

    with patch("app.chat.prompts.SMART_TARGETS_SERVICE_DISABLED", False):
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
