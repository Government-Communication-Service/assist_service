from unittest.mock import MagicMock, patch

import pytest

from app.bedrock import BedrockHandler, RunMode
from app.config import LLM_DEFAULT_MODEL
from app.database.table import LLMTable
from app.gov_uk_search.service import (
    extract_urls_from_user_prompt,
    get_relevant_documents_from_gov_uk_search,
    get_search_queries,
    is_document_relevant,
)


async def response():
    document_relevancy = {"content": {"input": {"is_relevant": True}, "type": "tool_use"}}
    return document_relevancy


@patch("app.bedrock.bedrock.BedrockHandler")
@pytest.mark.asyncio
async def test_is_document_relevant(mock: MagicMock):
    mock.invoke_async.return_value = response()
    llm = mock

    role = "user"
    query = "Just a test."
    document = "TEST DOC"
    title = "TEST TITLE"

    wrapped_document = await is_document_relevant(
        llm=llm, role=role, query=query, non_rag_document=document, non_rag_document_title=title
    )

    assert "content" in wrapped_document
    assert wrapped_document.get("content").get("input").get("is_relevant") is True
    assert wrapped_document.get("content").get("type") == "tool_use"


@pytest.mark.asyncio
async def test_get_relevant_documents_from_gov_uk_search(db_session):
    llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
    web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

    relevant_documents, citations, search_cost = await get_relevant_documents_from_gov_uk_search(
        llm=web_browsing_llm,
        role="user",
        query="What is the current guidance on contempt of court?",
        m_user_id=1,
        db_session=db_session,
    )

    assert len(relevant_documents) > 0, (
        "Expected some documents from GOV.UK Search when searching for contempt of court guidance, received 0."
    )
    assert citations[0].get("url") != ""


@pytest.mark.asyncio
async def test_get_search_terms_matching_user_query(db_session):
    llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
    web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

    search_terms, llm_internal_response_id_query, search_params, cost = await get_search_queries(
        llm=web_browsing_llm,
        role="user",
        query="What are current UK Personal Income Tax rates?",
        db_session=db_session,
    )

    assert len(search_terms) > 0
    assert llm_internal_response_id_query is not None
    assert search_params is not None
    for param in search_params:
        assert param in [
            "count",
            "order_by",
            "descending_order",
            "date_range",
            "content_purpose_supergroup",
            "content_purpose_subgroup",
            "is_political",
        ]
    assert cost > 0.0, f"Cost of creating search queries with an LLM should be greater than 0, got {cost}"


@pytest.mark.asyncio
async def test_extract_urls_from_query(user_id):
    llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
    web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

    url1 = "http://www.example.com/"
    url2 = "https://aws.amazon.com/"
    url3 = "https://www.gov.uk/"

    urls = await extract_urls_from_user_prompt(
        llm=web_browsing_llm, role="user", query=f"I'd like to sumarise {url1}, {url2}, {url3}?"
    )

    assert len(urls) == 3
    assert len([u for u in urls if u.startswith(url1)]) > 0
    assert len([u for u in urls if u.startswith(url2)]) > 0
    assert len([u for u in urls if u.startswith(url3)]) > 0
