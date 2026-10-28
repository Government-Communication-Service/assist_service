"""Tests of the uploaded-document tools against the database: reading in order, positions and scoping."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from app.database.models import (
    AuthSession,
    Chat,
    Document,
    DocumentChunk,
    Message,
    MessageDocumentChunkMapping,
    SearchIndex,
    User,
)
from app.document_upload.constants import PERSONAL_DOCUMENTS_INDEX_NAME
from app.document_upload.personal_document_rag import ScoredChunk
from app.document_upload.service import (
    _scope_documents,
    read_uploaded_document_for_tool,
    search_uploaded_documents_for_tool,
)
from app.document_upload.tools import READ_UPLOADED_DOCUMENT, SEARCH_UPLOADED_DOCUMENTS
from app.document_upload.utils import get_document_chunk_mappings, get_previously_returned_opensearch_ids

SERVICE = "app.document_upload.service"

pytestmark = pytest.mark.usefixtures("test_db")


@pytest.fixture(autouse=True)
def lookups_use_test_session(db_session):
    """The search reads in its own short session; point it at the test's session to see its unflushed rows."""

    @asynccontextmanager
    async def same_session():
        yield db_session

    with patch(f"{SERVICE}.async_db_session", same_session):
        yield


async def _message_id(db_session) -> int:
    user = User()
    db_session.add(user)
    await db_session.flush()
    auth_session = AuthSession(user_id=user.id)
    db_session.add(auth_session)
    await db_session.flush()
    chat = Chat(user_id=user.id, title="Test Chat", from_open_chat=True)
    db_session.add(chat)
    await db_session.flush()
    message = Message(
        chat_id=chat.id, content="q", role="user", tokens=0, auth_session_id=auth_session.id, interrupted=False
    )
    db_session.add(message)
    await db_session.flush()
    return message.id


async def _document(db_session, name: str, contents: list[str]) -> tuple[Document, list[DocumentChunk]]:
    """A document whose chunks are inserted in the order given, so their ids run in that order."""
    search_index = (
        await db_session.execute(select(SearchIndex).where(SearchIndex.name == PERSONAL_DOCUMENTS_INDEX_NAME))
    ).scalar_one()
    document = Document(name=name, is_central=False)
    db_session.add(document)
    await db_session.flush()
    chunks = []
    for i, content in enumerate(contents):
        chunk = DocumentChunk(
            search_index_id=search_index.id,
            document_id=document.id,
            name="CompositeElement",
            content=content,
            id_opensearch=f"{document.uuid}-{i}",
        )
        db_session.add(chunk)
        await db_session.flush()
        chunks.append(chunk)
    return document, chunks


def _chunk_numbers(text: str) -> list[int]:
    return [int(part.split('"')[0]) for part in text.split('<chunk n="')[1:]]


async def test_read_pages_through_a_document_in_order_with_continuation_notes(db_session):
    document, _ = await _document(db_session, "Report.docx", [f"{i}" * 100 for i in range(1, 6)])
    message_id = await _message_id(db_session)
    uuid = str(document.uuid)

    with patch(f"{SERVICE}.UPLOADED_DOCUMENTS_MAX_READ_CHARS", 250):
        first, citations, failed = await read_uploaded_document_for_tool(uuid, 1, [uuid], message_id, db_session)
        second, _, _ = await read_uploaded_document_for_tool(uuid, 3, [uuid], message_id, db_session)
        third, _, _ = await read_uploaded_document_for_tool(uuid, 5, [uuid], message_id, db_session)

    assert not failed
    assert citations == [{"docname": "Report.docx", "docurl": None}]
    assert _chunk_numbers(first) == [1, 2]
    assert "Chunks 1-2 of 5" in first and f'document_id="{uuid}" and start_chunk=3' in first
    assert _chunk_numbers(second) == [3, 4]
    assert "start_chunk=5" in second
    assert _chunk_numbers(third) == [5]
    assert "Chunks 5-5 of 5. End of document." in third


async def test_read_returns_at_least_one_chunk_even_if_it_is_over_the_limit(db_session):
    document, _ = await _document(db_session, "Big.docx", ["x" * 500, "y" * 10])
    message_id = await _message_id(db_session)
    uuid = str(document.uuid)

    with patch(f"{SERVICE}.UPLOADED_DOCUMENTS_MAX_READ_CHARS", 100):
        text, _, failed = await read_uploaded_document_for_tool(uuid, 1, [uuid], message_id, db_session)

    assert not failed
    assert _chunk_numbers(text) == [1]
    assert "start_chunk=2" in text


async def test_read_past_the_end_says_how_many_chunks_there_are(db_session):
    document, _ = await _document(db_session, "Short.docx", ["a", "b"])
    message_id = await _message_id(db_session)
    uuid = str(document.uuid)

    text, citations, failed = await read_uploaded_document_for_tool(uuid, 3, [uuid], message_id, db_session)

    assert failed and citations == []
    assert "2 chunk(s)" in text and "past the end" in text


async def test_read_of_a_document_with_no_chunks_says_it_has_no_readable_text_yet(db_session):
    document, _ = await _document(db_session, "Empty.docx", [])
    message_id = await _message_id(db_session)
    uuid = str(document.uuid)

    text, _, failed = await read_uploaded_document_for_tool(uuid, 1, [uuid], message_id, db_session)

    assert failed
    assert "no readable text yet" in text


async def test_read_refuses_documents_that_are_not_attached_to_the_chat(db_session):
    attached, _ = await _document(db_session, "Attached.docx", ["a"])
    other, _ = await _document(db_session, "Other.docx", ["secret"])
    message_id = await _message_id(db_session)

    text, _, failed = await read_uploaded_document_for_tool(
        str(other.uuid), 1, [str(attached.uuid)], message_id, db_session
    )

    assert failed
    assert "secret" not in text
    assert str(attached.uuid) in text


async def test_chunks_read_are_recorded_so_remove_duplicates_skips_them(db_session):
    document, chunks = await _document(db_session, "Notes.docx", ["a" * 10, "b" * 10, "c" * 10])
    message_id = await _message_id(db_session)
    uuid = str(document.uuid)

    with patch(f"{SERVICE}.UPLOADED_DOCUMENTS_MAX_READ_CHARS", 15):
        await read_uploaded_document_for_tool(uuid, 1, [uuid], message_id, db_session)
    await db_session.flush()

    rows = (
        (
            await db_session.execute(
                select(MessageDocumentChunkMapping).where(MessageDocumentChunkMapping.message_id == message_id)
            )
        )
        .scalars()
        .all()
    )
    assert [r.document_chunk_id for r in rows] == [chunks[0].id]
    assert rows[0].use_document_chunk is True
    assert await get_previously_returned_opensearch_ids(db_session, message_id) == {chunks[0].id_opensearch}


async def test_search_with_remove_duplicates_skips_chunks_returned_earlier(db_session):
    document, chunks = await _document(db_session, "Notes.docx", ["a", "b"])
    message_id = await _message_id(db_session)
    uuid = str(document.uuid)
    with patch(
        f"{SERVICE}.select_chunks_for_tool", AsyncMock(return_value=([ScoredChunk(chunks[0].id_opensearch, 1.0, 1)], 0))
    ):
        await search_uploaded_documents_for_tool(["q"], [uuid], message_id, None, db_session)
    await db_session.flush()

    select_chunks = AsyncMock(return_value=([], 1))
    with patch(f"{SERVICE}.select_chunks_for_tool", select_chunks):
        text, _, failed = await search_uploaded_documents_for_tool(
            ["q"], [uuid], message_id, None, db_session, remove_duplicates=True
        )

    assert not failed
    assert select_chunks.await_args.args[2] == {chunks[0].id_opensearch}
    assert "1 matching passage(s) were already provided" in text


async def test_positions_number_the_active_chunks_within_each_document(db_session):
    first, first_chunks = await _document(db_session, "First.docx", ["a", "b", "c"])
    second, second_chunks = await _document(db_session, "Second.docx", ["d", "e"])
    first_chunks[0].deleted_at = first_chunks[0].created_at  # soft-deleted chunks don't count
    await db_session.flush()
    message_id = await _message_id(db_session)
    hits = [ScoredChunk(c.id_opensearch, 1.0, 1) for c in (second_chunks[1], first_chunks[2], first_chunks[0])]

    results = await get_document_chunk_mappings(
        hits, [str(first.uuid), str(second.uuid)], message_id, db_session, use_chunk=True
    )

    assert [(r.document.name, r.document_chunk.content, r.position, r.total_chunks) for r in results] == [
        ("Second.docx", "e", 2, 2),
        ("First.docx", "c", 2, 2),
    ]


async def test_hits_outside_the_searched_documents_are_dropped(db_session):
    first, first_chunks = await _document(db_session, "First.docx", ["a"])
    second, second_chunks = await _document(db_session, "Second.docx", ["b"])
    message_id = await _message_id(db_session)
    hits = [ScoredChunk(first_chunks[0].id_opensearch, 1.0, 1), ScoredChunk(second_chunks[0].id_opensearch, 1.0, 1)]

    results = await get_document_chunk_mappings(hits, [str(first.uuid)], message_id, db_session)

    assert [r.document.name for r in results] == ["First.docx"]


async def test_search_results_are_in_document_order_and_labelled_with_position(db_session):
    document, chunks = await _document(db_session, "Guide.docx", ["one", "two", "three"])
    message_id = await _message_id(db_session)
    uuid = str(document.uuid)
    best_first = [ScoredChunk(chunks[2].id_opensearch, 9.0, 5), ScoredChunk(chunks[0].id_opensearch, 1.0, 3)]

    with patch(f"{SERVICE}.select_chunks_for_tool", AsyncMock(return_value=(best_first, 0))):
        text, citations, failed = await search_uploaded_documents_for_tool(["q"], [uuid], message_id, None, db_session)

    assert not failed
    assert text.index("chunk 1 of 3") < text.index("chunk 3 of 3")
    assert f"<document-id>{uuid}</document-id>" in text
    assert citations == [{"docname": "Guide.docx", "docurl": None}]


async def test_search_passes_only_the_requested_documents_to_the_search(db_session):
    first, _ = await _document(db_session, "First.docx", ["a"])
    second, _ = await _document(db_session, "Second.docx", ["b"])
    message_id = await _message_id(db_session)
    attached = [str(first.uuid), str(second.uuid)]
    select_chunks = AsyncMock(return_value=([], 0))

    with patch(f"{SERVICE}.select_chunks_for_tool", select_chunks):
        await search_uploaded_documents_for_tool(
            ["q"], attached, message_id, None, db_session, document_ids=[str(second.uuid).upper()]
        )

    assert select_chunks.await_args.args[1] == [str(second.uuid)]


async def test_search_with_an_unattached_document_id_fails_without_searching(db_session):
    first, _ = await _document(db_session, "First.docx", ["a"])
    message_id = await _message_id(db_session)
    select_chunks = AsyncMock()

    with patch(f"{SERVICE}.select_chunks_for_tool", select_chunks):
        text, _, failed = await search_uploaded_documents_for_tool(
            ["q"], [str(first.uuid)], message_id, None, db_session, document_ids=["not-attached"]
        )

    assert failed
    assert "not-attached" in text
    select_chunks.assert_not_awaited()


@pytest.mark.parametrize("requested", [None, [], ["  "]])
def test_scope_defaults_to_every_attached_document(requested):
    assert _scope_documents(requested, ["u1", "u2"]) == (["u1", "u2"], None)


def test_scope_is_case_insensitive_and_drops_repeats():
    assert _scope_documents(["U2", "u2"], ["u1", "u2"]) == (["u2"], None)


async def test_read_of_a_range_returns_only_those_chunks(db_session):
    document, _ = await _document(db_session, "Range.docx", [f"{i}" * 10 for i in range(1, 8)])
    message_id = await _message_id(db_session)
    uuid = str(document.uuid)

    text, _, failed = await read_uploaded_document_for_tool(uuid, 3, [uuid], message_id, db_session, end_chunk=5)

    assert not failed
    assert _chunk_numbers(text) == [3, 4, 5]
    assert "Chunks 3-5 of 7" in text and "start_chunk=6" in text


async def test_read_of_a_long_range_is_cut_at_the_limit_and_says_where_to_continue(db_session):
    document, _ = await _document(db_session, "Long.docx", ["x" * 100 for _ in range(6)])
    message_id = await _message_id(db_session)
    uuid = str(document.uuid)

    with patch(f"{SERVICE}.UPLOADED_DOCUMENTS_MAX_READ_CHARS", 250):
        text, _, _ = await read_uploaded_document_for_tool(uuid, 2, [uuid], message_id, db_session, end_chunk=6)

    assert _chunk_numbers(text) == [2, 3]
    assert "start_chunk=4" in text


async def test_read_range_ending_past_the_last_chunk_stops_at_the_end_of_the_document(db_session):
    document, _ = await _document(db_session, "Tail.docx", ["a", "b", "c"])
    message_id = await _message_id(db_session)
    uuid = str(document.uuid)

    text, _, _ = await read_uploaded_document_for_tool(uuid, 2, [uuid], message_id, db_session, end_chunk=50)

    assert _chunk_numbers(text) == [2, 3]
    assert "End of document." in text


async def test_read_range_ending_before_it_starts_is_rejected(db_session):
    document, _ = await _document(db_session, "Bad.docx", ["a", "b", "c"])
    message_id = await _message_id(db_session)
    uuid = str(document.uuid)

    text, _, failed = await read_uploaded_document_for_tool(uuid, 3, [uuid], message_id, db_session, end_chunk=2)

    assert failed
    assert "before start_chunk" in text


def test_read_args_reject_an_end_chunk_before_the_start_chunk():
    with pytest.raises(ValidationError, match="end_chunk"):
        READ_UPLOADED_DOCUMENT.parse({"document_id": "u1", "start_chunk": 5, "end_chunk": 4})
    assert READ_UPLOADED_DOCUMENT.parse({"document_id": "u1", "start_chunk": 5, "end_chunk": 5}).end_chunk == 5


async def test_search_result_says_which_documents_were_searched_when_scoped(db_session):
    first, _ = await _document(db_session, "Alpha.docx", ["a"])
    second, _ = await _document(db_session, "Beta.docx", ["b"])
    third, _ = await _document(db_session, "Gamma.docx", ["c"])
    message_id = await _message_id(db_session)
    attached = [str(d.uuid) for d in (first, second, third)]

    with patch(f"{SERVICE}.select_chunks_for_tool", AsyncMock(return_value=([], 0))):
        text, _, _ = await search_uploaded_documents_for_tool(
            ["q"], attached, message_id, None, db_session, document_ids=[attached[2], attached[0]]
        )

    assert "Searched 2 of 3 attached documents: Alpha.docx, Gamma.docx." in text
    assert "Beta.docx" not in text


async def test_search_result_says_all_documents_were_searched_when_unscoped(db_session):
    first, _ = await _document(db_session, "Alpha.docx", ["a"])
    second, _ = await _document(db_session, "Beta.docx", ["b"])
    message_id = await _message_id(db_session)
    attached = [str(first.uuid), str(second.uuid)]

    with patch(f"{SERVICE}.select_chunks_for_tool", AsyncMock(return_value=([], 0))):
        text, _, _ = await search_uploaded_documents_for_tool(["q"], attached, message_id, None, db_session)

    assert "Searched all 2 attached document(s)." in text


@pytest.mark.parametrize(
    "document_ids, expected",
    [
        ("05776226-8ff0-49a1-a1f4-44ad55ce0542", ["05776226-8ff0-49a1-a1f4-44ad55ce0542"]),
        ("Report.docx", ["Report.docx"]),
        ('["a", "b"]', ["a", "b"]),
        (["a"], ["a"]),
        (None, None),
    ],
)
def test_search_args_accept_a_single_document_id_as_a_bare_string(document_ids, expected):
    args = SEARCH_UPLOADED_DOCUMENTS.parse({"queries": ["q"], "document_ids": document_ids})

    assert args.document_ids == expected
