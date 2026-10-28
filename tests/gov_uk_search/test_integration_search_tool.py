"""search_gov_uk_for_tool against the test database, with the Search API, page fetches and relevancy check stubbed."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.future import select

from app.database.models import GovUkSearchQuery, GovUkSearchResult
from app.gov_uk_search.prompts import TOOL_RESULT_RELEVANCE_CHECK_FAILED
from app.gov_uk_search.service import search_gov_uk_for_tool
from app.gov_uk_search.tools import SEARCH_GOV_UK
from tests.central_guidance.test_integration_central_guidance import create_loop_call, create_test_message

SERVICE = "app.gov_uk_search.service"


@pytest.fixture(autouse=True)
def lookups_use_test_session(db_session):
    """The search reads in its own short session; point it at the test's session to see its unflushed rows."""

    @asynccontextmanager
    async def same_session():
        yield db_session

    with patch(f"{SERVICE}.async_db_session", same_session):
        yield


def _result(n: int) -> dict:
    return {"title": f"Page {n}", "description": f"About {n}", "link": f"/page-{n}"}


async def _search(db_session, message_id, queries, results, decide, relevancy_response_id, **kwargs):
    async def assess(handler, query, pages, context):
        return [decide(page) for page in pages], relevancy_response_id

    with (
        patch(f"{SERVICE}._fetch_search_results", AsyncMock(side_effect=results)),
        patch(f"{SERVICE}.extract_content_from_gov_uk", AsyncMock(side_effect=lambda link: f"Body of {link}")),
        patch(f"{SERVICE}._assess_relevancy", side_effect=assess),
    ):
        return await search_gov_uk_for_tool(
            SEARCH_GOV_UK.parse({"queries": queries, **kwargs}),
            user_query="q",
            message_id=message_id,
            llm_internal_response_id=await create_loop_call(db_session),
            db_session=db_session,
        )


async def _rows(db_session, message_id):
    results = (
        await db_session.execute(
            select(GovUkSearchResult)
            .filter(GovUkSearchResult.message_id == message_id)
            .order_by(GovUkSearchResult.position)
        )
    ).scalars()
    queries = (
        await db_session.execute(select(GovUkSearchQuery).filter(GovUkSearchQuery.message_id == message_id))
    ).scalars()
    return list(queries), list(results)


async def test_search_records_queries_and_pages_and_later_skips_used_pages(db_session):
    message_id = await create_test_message(db_session, "What is the guidance?")
    relevancy_id = await create_loop_call(db_session)

    content, citations, failed = await _search(
        db_session, message_id, ["a", "b"], [[_result(1)], [_result(2)]], lambda p: p["title"] == "Page 2", relevancy_id
    )

    assert not failed and [c["docname"] for c in citations] == ["Page 2"]
    queries, results = await _rows(db_session, message_id)
    assert len(queries) == 2
    by_query = {q.id: q for q in queries}
    assert [(r.url, r.is_used, r.llm_internal_response_id) for r in results] == [
        ("https://www.gov.uk/page-1", False, relevancy_id),
        ("https://www.gov.uk/page-2", True, relevancy_id),
    ]
    assert "q=a" in by_query[results[0].gov_uk_search_query_id].content
    assert "q=b" in by_query[results[1].gov_uk_search_query_id].content

    content, citations, failed = await _search(
        db_session, message_id, ["c"], [[_result(2), _result(3)]], lambda p: True, relevancy_id, remove_duplicates=True
    )

    assert not failed and [c["docname"] for c in citations] == ["Page 3"]
    assert "1 matching page(s) were already provided" in content


async def test_failed_relevancy_check_is_an_error_and_still_recorded(db_session):
    message_id = await create_test_message(db_session, "What is the guidance?")

    content, citations, failed = await _search(
        db_session, message_id, ["a"], [[_result(1), _result(2)]], lambda p: None, None
    )

    assert failed and citations == []
    assert TOOL_RESULT_RELEVANCE_CHECK_FAILED in content
    queries, results = await _rows(db_session, message_id)
    assert len(queries) == 1
    assert [(r.is_used, r.llm_internal_response_id) for r in results] == [(False, None), (False, None)]
