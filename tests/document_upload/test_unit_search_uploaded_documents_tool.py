"""Unit tests for the search_uploaded_documents tool's duplicate removal and DB use (search and DB mocked)."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.document_upload.service import search_uploaded_documents_for_tool

SERVICE = "app.document_upload.service"


@pytest.mark.parametrize("remove_duplicates, expected_exclude_ids", [(False, set()), (True, {"already-returned"})])
async def test_previously_returned_chunks_are_only_excluded_when_removing_duplicates(
    remove_duplicates, expected_exclude_ids
):
    db_session = MagicMock()
    db_session.execute = AsyncMock(return_value=MagicMock())
    select_chunks = AsyncMock(return_value=([], 0))
    read_session = MagicMock()

    @asynccontextmanager
    async def short_session():
        yield read_session

    previously_returned = AsyncMock(return_value={"already-returned"})
    with (
        patch(f"{SERVICE}.async_db_session", short_session),
        patch(f"{SERVICE}.get_previously_returned_opensearch_ids", previously_returned),
        patch(f"{SERVICE}.select_chunks_for_tool", select_chunks),
        patch(f"{SERVICE}.get_document_chunk_mappings", AsyncMock(return_value=[])),
    ):
        _, _, failed = await search_uploaded_documents_for_tool(
            queries=["q"],
            document_uuids=["doc-1"],
            message_id=1,
            llm_internal_response_id=None,
            db_session=db_session,
            remove_duplicates=remove_duplicates,
        )

    assert not failed
    assert select_chunks.await_args.args[2] == expected_exclude_ids
    if remove_duplicates:
        assert previously_returned.await_args.args[0] is read_session
    else:
        previously_returned.assert_not_awaited()


async def test_no_session_is_open_during_the_search():
    """The duplicates read closes before OpenSearch is queried, and the tool's session is used only after it."""
    events = []

    def record_db_use(*args, **kwargs):
        events.append("tool_session")
        return MagicMock()

    db_session = MagicMock()
    db_session.execute = AsyncMock(side_effect=record_db_use)
    db_session.add = MagicMock(side_effect=record_db_use)

    @asynccontextmanager
    async def short_session():
        events.append("read_open")
        yield MagicMock()
        events.append("read_closed")

    async def search(*args):
        events.append("opensearch")
        return [], 0

    with (
        patch(f"{SERVICE}.async_db_session", short_session),
        patch(f"{SERVICE}.get_previously_returned_opensearch_ids", AsyncMock(return_value=set())),
        patch(f"{SERVICE}.select_chunks_for_tool", search),
        patch(f"{SERVICE}.get_document_chunk_mappings", AsyncMock(return_value=[])),
    ):
        await search_uploaded_documents_for_tool(
            queries=["q"],
            document_uuids=["doc-1"],
            message_id=1,
            llm_internal_response_id=None,
            db_session=db_session,
            remove_duplicates=True,
        )

    assert events[:3] == ["read_open", "read_closed", "opensearch"]
    assert set(events[3:]) == {"tool_session"}
