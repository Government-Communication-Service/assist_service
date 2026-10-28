"""Unit tests for the per-turn <context> block (app/chat/prompts.py)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.chat.prompts import build_turn_context_block
from app.chat.tools import ToolContext


def _ctx(
    document_uuids=None,
    use_rag=True,
    use_gov_uk_search_api=False,
    use_smart_targets=False,
) -> ToolContext:
    return ToolContext(
        chat=SimpleNamespace(id=1),
        user_id=1,
        user_message=SimpleNamespace(id=1, content="hi"),
        history=[],
        input_data=SimpleNamespace(
            use_rag=use_rag, use_gov_uk_search_api=use_gov_uk_search_api, use_smart_targets=use_smart_targets
        ),
        document_uuids=document_uuids or [],
    )


async def test_block_is_wrapped_in_context_tags_and_states_the_time():
    with patch("app.chat.prompts.datetime") as mock_datetime:
        mock_datetime.now.return_value.strftime.return_value = "1 January 2026, 09:00"
        block = await build_turn_context_block(_ctx(), AsyncMock())

    assert block.startswith("<context>\n")
    assert block.endswith("\n</context>")
    assert "1 January 2026, 09:00" in block


async def test_lists_each_tool_as_available_or_turned_off_with_its_message():
    with patch("app.chat.tools.USE_RAG", True):
        block = await build_turn_context_block(_ctx(use_rag=False, use_gov_uk_search_api=True), AsyncMock())

    assert "- search_central_guidance: turned off. Central documents search is turned off" in block
    assert "- search_gov_uk: available" in block


async def test_lists_attached_documents():
    db_session = AsyncMock()
    rows = [
        SimpleNamespace(uuid="u1", name="a.docx", chunks=87, characters=64_250),
        SimpleNamespace(uuid="u2", name="b.txt", chunks=1, characters=0),
    ]
    db_session.execute.return_value = MagicMock(fetchall=lambda: rows)

    block = await build_turn_context_block(_ctx(document_uuids=["u1", "u2"]), db_session)

    assert "Documents attached to this chat" in block
    assert "- a.docx [id: u1] 87 chunks, ~64,250 characters" in block
    assert "- b.txt [id: u2] 1 chunk, ~0 characters" in block


async def test_no_documents_section_when_nothing_attached():
    block = await build_turn_context_block(_ctx(document_uuids=[]), AsyncMock())

    assert "Documents attached" not in block


async def test_audience_segments_listed_when_selected():
    segment = SimpleNamespace(uuid="11111111-1111-1111-1111-111111111111", pretty_name="Engaged Citizens")
    with patch("app.chat.prompts.AudienceSegmentsService.get_segment_library", AsyncMock(return_value=[segment])):
        block = await build_turn_context_block(_ctx(), AsyncMock(), audience_segment_uuids=[segment.uuid])

    assert "Audience segments selected for this chat" in block
    assert "- Engaged Citizens" in block
