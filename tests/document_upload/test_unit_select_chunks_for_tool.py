"""Unit tests for choosing uploaded-document chunks for the search_uploaded_documents tool."""

from unittest.mock import AsyncMock, patch

from app.document_upload.personal_document_rag import (
    MULTI_QUERY_BOOST,
    UPLOADED_DOCUMENTS_MAX_RESULT_CHARS,
    ScoredChunk,
    select_chunks_for_tool,
)

OPS = "app.document_upload.personal_document_rag.AsyncOpenSearchOperations"


def _hit(chunk_id: str, chars: int = 100, score: float = 1.0) -> dict:
    return {"_id": chunk_id, "_score": score, "_source": {"chunk_content": "x" * chars}}


async def test_returns_hits_best_first_as_scored_chunks():
    hits = [_hit("a", chars=10, score=1.0), _hit("b", chars=20, score=3.0)]
    with patch(f"{OPS}.search_multiple_document_chunks", AsyncMock(return_value=hits)):
        selected, skipped = await select_chunks_for_tool(["q"], ["doc-1"], exclude_ids=set())

    assert selected == [ScoredChunk("b", 3.0, 20), ScoredChunk("a", 1.0, 10)]
    assert skipped == 0


async def test_chunks_found_by_several_queries_have_scores_summed_and_boosted():
    search = AsyncMock(side_effect=[[_hit("a", score=2.0)], [_hit("a", score=3.0), _hit("b", score=4.0)]])
    with patch(f"{OPS}.search_multiple_document_chunks", search):
        selected, _ = await select_chunks_for_tool(["q1", "q2"], ["doc-1"], exclude_ids=set())

    assert [c.id_opensearch for c in selected] == ["a", "b"]
    assert selected[0].score == 5.0 * (1 + MULTI_QUERY_BOOST)
    assert selected[1].score == 4.0


async def test_already_returned_chunks_are_skipped_and_counted_once():
    search = AsyncMock(side_effect=[[_hit("a"), _hit("b")], [_hit("a")]])
    with patch(f"{OPS}.search_multiple_document_chunks", search):
        selected, skipped = await select_chunks_for_tool(["q1", "q2"], ["doc-1"], exclude_ids={"a"})

    assert [c.id_opensearch for c in selected] == ["b"]
    assert skipped == 1


async def test_stops_adding_chunks_at_the_character_limit():
    limit = UPLOADED_DOCUMENTS_MAX_RESULT_CHARS
    hits = [
        _hit("big", chars=limit - 10, score=9.0),
        _hit("too-big", chars=50, score=8.0),
        _hit("small", chars=10, score=1.0),
    ]
    with patch(f"{OPS}.search_multiple_document_chunks", AsyncMock(return_value=hits)):
        selected, _ = await select_chunks_for_tool(["q"], ["doc-1"], exclude_ids=set())

    assert [c.id_opensearch for c in selected] == ["big", "small"]
