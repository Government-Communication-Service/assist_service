"""Unit tests for the search_gov_uk loop tool (network, content fetch and relevancy mocked)."""

import asyncio
from contextlib import ExitStack, asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer as LocalServer
from pydantic import ValidationError

from app.bedrock.call_log import LlmCallContext
from app.gov_uk_search.service import (
    GOV_UK_COLLECTION_MAX_DOCUMENTS,
    GOV_UK_SEARCH_MAX_COUNT,
    GOV_UK_SEARCH_MAX_TOOL_DOCUMENTS,
    GovUKContent,
    GovUKSearch,
    _assess_relevancy,
    _attachment_lines,
    _collection_lines,
    _download_pdf,
    _extract_pdf_pages,
    _search_filters,
    get_gov_uk_page_for_tool,
    get_gov_uk_pdf_for_tool,
    search_gov_uk_for_tool,
)
from app.gov_uk_search.tools import GET_GOV_UK_PAGE, GET_GOV_UK_PDF, SEARCH_GOV_UK

SERVICE = "app.gov_uk_search.service"
CONTEXT = LlmCallContext(purpose="gov_uk_relevancy", message_id=1)


def _args(**kwargs):
    return SEARCH_GOV_UK.parse(kwargs)


def _result(n: int) -> dict:
    return {"title": f"Page {n}", "description": f"About {n}", "link": f"/page-{n}"}


def test_no_filters_by_default():
    assert _search_filters(_args(queries=["tax"])) == []


def test_filters_include_date_range_and_content_purpose():
    filters = _search_filters(
        _args(
            queries=["x"],
            date_range={"from": "2026-01-01", "to": "2026-02-01"},
            content_purpose_supergroup=["news_and_communications"],
        )
    )
    assert filters == [
        ("public_timestamp", "from:2026-01-01,to:2026-02-01"),
        ("content_purpose_supergroup", "news_and_communications"),
    ]


def test_date_range_without_end_date_is_open_ended():
    filters = _search_filters(_args(queries=["x"], date_range={"from": "2026-07-01"}))

    assert filters == [("public_timestamp", "from:2026-07-01")]


@pytest.mark.parametrize(
    "date_range, message",
    [
        ({"from": "2026-09-29", "to": "2026-09-28"}, "'to' must not be before 'from'"),
        ({"from": "2026-09-31", "to": "2026-10-01"}, "date"),
        ({"from": "last week"}, "date"),
        ({"to": "2026-09-29"}, "Field required"),
    ],
    ids=["reversed", "nonexistent_date", "not_a_date", "missing_from"],
)
def test_invalid_date_ranges_are_rejected(date_range, message):
    with pytest.raises(ValidationError, match=message):
        _args(queries=["x"], date_range=date_range)


def test_single_day_date_range_is_allowed():
    filters = _search_filters(_args(queries=["x"], date_range={"from": "2026-09-29", "to": "2026-09-29"}))

    assert filters == [("public_timestamp", "from:2026-09-29,to:2026-09-29")]


@pytest.mark.parametrize(
    "tool_input",
    [{}, {"search_terms": ["tax"]}, {"queries": []}, {"queries": ["  "]}],
    ids=["missing", "wrong_key", "empty", "blank"],
)
def test_queries_are_required(tool_input):
    with pytest.raises(ValidationError, match="queries"):
        SEARCH_GOV_UK.parse(tool_input)


def test_queries_are_stripped_and_blanks_dropped():
    assert _args(queries=[" tax ", " "]).queries == ["tax"]


def test_ordering_arguments_are_not_accepted():
    args = _args(queries=["tax"], order_by="public_timestamp", descending_order=True)

    assert not hasattr(args, "order_by")
    assert not hasattr(args, "descending_order")


@asynccontextmanager
async def _no_session():
    yield MagicMock()


def _patches(fetch, extract, assess, already_used=frozenset(), events=None):
    """Patch the network, LLM and DB calls of search_gov_uk_for_tool. events, if given, records the call order."""
    events = events if events is not None else []

    async def insert_query(**kwargs):
        events.append("insert_query")
        return MagicMock(id=len([e for e in events if e == "insert_query"]))

    async def record(**kwargs):
        events.append("record_result")

    async def previously_used(db_session, message_id):
        events.append("read_previously_used")
        return set(already_used)

    insert_query_mock = AsyncMock(side_effect=insert_query)
    record_mock = AsyncMock(side_effect=record)
    return (
        [
            patch(f"{SERVICE}._fetch_search_results", fetch),
            patch(f"{SERVICE}.extract_content_from_gov_uk", extract),
            patch(f"{SERVICE}._assess_relevancy", side_effect=assess),
            patch(f"{SERVICE}._previously_used_urls", side_effect=previously_used),
            patch(f"{SERVICE}.async_db_session", _no_session),
            patch(f"{SERVICE}.LLMTable", MagicMock()),
            patch(f"{SERVICE}.BedrockHandler", MagicMock()),
            patch(f"{SERVICE}.DbOperations.insert_gov_uk_search_query", insert_query_mock),
            patch(f"{SERVICE}.record_search_result", record_mock),
        ],
        insert_query_mock,
        record_mock,
    )


async def _run(args, results_by_url_call, relevant_titles, already_used=frozenset(), extract=None, assess=None):
    fetch = AsyncMock(side_effect=results_by_url_call)
    extract = extract or AsyncMock(side_effect=lambda link: f"Body of {link}")

    async def default_assess(handler, query, pages, context):
        return [page["title"] in relevant_titles for page in pages], 7

    patches, insert_query, record = _patches(fetch, extract, assess or default_assess, already_used)
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        content, citations, failed = await search_gov_uk_for_tool(
            args, user_query="q", message_id=1, llm_internal_response_id=None, db_session=MagicMock()
        )
    fetch.insert_query = insert_query
    fetch.record = record
    fetch.failed = failed
    return content, citations, fetch


async def test_search_queries_are_recorded_even_when_the_call_could_not_be_logged():
    _, _, fetch = await _run(_args(queries=["a"]), [[_result(1)]], {"Page 1"})

    fetch.insert_query.assert_awaited_once()
    assert fetch.insert_query.call_args.kwargs["llm_internal_response_id"] is None


async def test_returns_relevant_pages_deduplicated_across_terms():
    content, citations, fetch = await _run(
        _args(queries=["a", "b"]), [[_result(1), _result(2)], [_result(2), _result(3)]], {"Page 1", "Page 3"}
    )

    assert fetch.await_count == 2
    assert citations == [
        {"docname": "Page 1", "docurl": "https://www.gov.uk/page-1"},
        {"docname": "Page 3", "docurl": "https://www.gov.uk/page-3"},
    ]
    assert content.startswith("<results-from-gov-uk-search>")
    assert "Body of https://www.gov.uk/page-3" in content


async def test_skips_pages_already_used_in_the_chat_when_removing_duplicates():
    content, citations, _ = await _run(
        _args(queries=["a"], remove_duplicates=True),
        [[_result(1), _result(2)]],
        {"Page 1", "Page 2"},
        already_used={"https://www.gov.uk/page-1"},
    )

    assert [c["docname"] for c in citations] == ["Page 2"]
    assert "1 matching page(s) were already provided" in content


async def test_returns_pages_already_used_by_default():
    content, citations, _ = await _run(
        _args(queries=["a"]),
        [[_result(1), _result(2)]],
        {"Page 1", "Page 2"},
        already_used={"https://www.gov.uk/page-1"},
    )

    assert [c["docname"] for c in citations] == ["Page 1", "Page 2"]
    assert "already provided" not in content


async def test_skips_results_that_are_not_gov_uk_pages():
    results = [
        _result(1),
        {"title": "NHS", "description": "", "link": "https://www.nhs.uk/conditions/flu/"},
        {"title": "Lookalike", "description": "", "link": "https://www.gov.uk.example.com/page"},
        {"title": "Bare", "description": "", "link": "page-2"},
    ]
    extract = AsyncMock(side_effect=lambda link: f"Body of {link}")
    relevant = {"Page 1", "NHS", "Lookalike", "Bare"}
    _, citations, _ = await _run(_args(queries=["a"]), [results], relevant, extract=extract)

    gov_uk_pages = ["https://www.gov.uk/page-1", "https://www.gov.uk/page-2"]
    assert [call.args[0] for call in extract.await_args_list] == gov_uk_pages
    assert [c["docurl"] for c in citations] == gov_uk_pages


async def test_absolute_gov_uk_links_are_normalised_for_deduplication():
    results = [_result(1), {"title": "Page 1 again", "description": "", "link": "https://gov.uk/page-1?tab=x"}]
    _, citations, _ = await _run(_args(queries=["a"]), [results], {"Page 1", "Page 1 again"})

    assert citations == [{"docname": "Page 1", "docurl": "https://www.gov.uk/page-1"}]


async def test_a_page_that_fails_to_fetch_falls_back_to_its_description():
    async def extract(link):
        if link.endswith("page-1"):
            raise RuntimeError("boom")
        return f"Body of {link}"

    content, citations, _ = await _run(
        _args(queries=["a"]), [[_result(1), _result(2)]], {"Page 1", "Page 2"}, extract=AsyncMock(side_effect=extract)
    )

    assert [c["docname"] for c in citations] == ["Page 1", "Page 2"]
    assert "About 1" in content
    assert "Body of https://www.gov.uk/page-2" in content


async def test_caps_documents_returned():
    results = [_result(n) for n in range(10)]
    _, citations, _ = await _run(_args(queries=["a"]), [results], {f"Page {n}" for n in range(10)})

    assert len(citations) == GOV_UK_SEARCH_MAX_TOOL_DOCUMENTS


async def test_no_relevant_pages():
    content, citations, _ = await _run(_args(queries=["a"]), [[_result(1)]], set())

    assert citations == []
    assert "No new relevant GOV.UK pages found" in content


async def test_terms_and_count_are_bounded():
    with patch(f"{SERVICE}.build_search_url", MagicMock(return_value="url")) as build:
        await _run(_args(queries=["a", "b", "c", "d", "a"], count=50), [[], [], []], set())

    assert [call.kwargs["query"] for call in build.call_args_list] == ["a", "b", "c"]
    assert all(call.kwargs["count"] == GOV_UK_SEARCH_MAX_COUNT for call in build.call_args_list)


async def test_search_urls_are_relevance_ranked():
    _, _, fetch = await _run(_args(queries=["a"]), [[]], set())

    assert "order=" not in fetch.call_args.args[0]


def _relevance_response(tool_input, stop_reason="tool_use"):
    from anthropic.types import Message, ToolUseBlock, Usage

    from app.bedrock.bedrock import BedrockMessage

    message = Message(
        id="msg",
        type="message",
        role="assistant",
        model="m",
        content=[ToolUseBlock(id="toolu_1", type="tool_use", name="evaluate_page_relevance", input=tool_input)],
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    return BedrockMessage(message, llm_internal_response_id=42)


def _pages(n: int) -> list[dict]:
    return [{"title": f"Title {i}", "description": f"Desc {i}", "full_content": f"Body {i}"} for i in range(n)]


async def test_relevancy_batches_pages_into_one_forced_tool_call():
    handler = MagicMock()
    evaluations = [
        {"page_index": 0, "reasoning": "on topic", "is_relevant": True},
        {"page_index": 1, "reasoning": "a login page", "is_relevant": False},
    ]
    handler.invoke_async = AsyncMock(return_value=_relevance_response({"evaluations": evaluations}))

    decisions, response_id = await _assess_relevancy(handler, "q", _pages(2), CONTEXT)

    assert decisions == [True, False]
    assert response_id == 42
    handler.invoke_async.assert_awaited_once()
    kwargs = handler.invoke_async.call_args.kwargs
    assert kwargs["context"] == CONTEXT
    assert kwargs["tool_choice"]["name"] == "evaluate_page_relevance"
    sent = handler.invoke_async.call_args.args[0][0]["content"]
    assert "<page-0>" in sent and "<page-1>" in sent and "Title 1" in sent


async def test_relevancy_excerpt_is_bounded_per_page():
    handler = MagicMock()
    handler.invoke_async = AsyncMock(return_value=_relevance_response({"evaluations": []}))
    pages = [{"title": "T", "description": "D", "full_content": "x" * 10_000}]

    await _assess_relevancy(handler, "q", pages, CONTEXT)

    sent = handler.invoke_async.call_args.args[0][0]["content"]
    assert "x" * 6000 in sent and "x" * 6001 not in sent


async def test_relevancy_page_missing_from_response_is_unchecked():
    handler = MagicMock()
    only_first = {"evaluations": [{"page_index": 0, "reasoning": "yes", "is_relevant": True}]}
    handler.invoke_async = AsyncMock(return_value=_relevance_response(only_first))

    with patch(f"{SERVICE}.logger") as log:
        decisions, response_id = await _assess_relevancy(handler, "q", _pages(2), CONTEXT)

    assert decisions == [True, None]
    assert response_id == 42
    assert "no decision for 1 of 2 pages" in log.warning.call_args.args[0]


async def test_relevancy_call_failure_leaves_every_page_unchecked():
    handler = MagicMock()
    handler.invoke_async = AsyncMock(side_effect=RuntimeError("down"))

    decisions, response_id = await _assess_relevancy(handler, "q", _pages(3), CONTEXT)

    assert decisions == [None] * 3
    assert response_id is None


async def test_relevancy_invalid_answer_leaves_every_page_unchecked_and_logs_an_error():
    handler = MagicMock()
    bad = {"evaluations": [{"page_index": 0, "reasoning": "?", "is_relevant": "perhaps"}]}
    handler.invoke_async = AsyncMock(return_value=_relevance_response(bad, stop_reason="max_tokens"))

    with patch(f"{SERVICE}.logger") as log:
        decisions, response_id = await _assess_relevancy(handler, "q", _pages(2), CONTEXT)

    assert decisions == [None] * 2
    assert response_id == 42
    assert "stop_reason='max_tokens'" in log.error.call_args.args[0]


async def test_candidates_are_assessed_in_batches():
    results = [_result(n) for n in range(5)]
    assess = AsyncMock(side_effect=lambda handler, query, pages, context: ([False] * len(pages), 7))
    with patch(f"{SERVICE}.GOV_UK_SEARCH_RELEVANCY_BATCH_SIZE", 2):
        await _run(_args(queries=["a"]), [results], set(), assess=assess)

    assert [len(call.args[2]) for call in assess.call_args_list] == [2, 2, 1]


async def test_every_page_is_recorded_with_its_query_and_relevancy_call():
    _, _, fetch = await _run(_args(queries=["a", "b"]), [[_result(1)], [_result(2)]], {"Page 2"})

    assert fetch.insert_query.await_count == 2
    rows = [call.kwargs for call in fetch.record.await_args_list]
    assert [(r["url"], r["gov_uk_search_query_id"], r["is_used"]) for r in rows] == [
        ("https://www.gov.uk/page-1", 1, False),
        ("https://www.gov.uk/page-2", 2, True),
    ]
    assert all(r["llm_internal_response_id"] == 7 for r in rows)


async def test_db_is_only_written_after_the_network_and_relevancy_calls():
    events = []

    async def fetch(url):
        events.append("search")
        return [_result(1)]

    async def extract(link):
        events.append("fetch_page")
        return "body"

    async def assess(handler, query, pages, context):
        events.append("relevancy")
        return [True] * len(pages), 7

    patches, _, _ = _patches(AsyncMock(side_effect=fetch), AsyncMock(side_effect=extract), assess, events=events)
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        db_session = MagicMock()
        await search_gov_uk_for_tool(
            _args(queries=["a"], remove_duplicates=True),
            user_query="q",
            message_id=1,
            llm_internal_response_id=None,
            db_session=db_session,
        )

    assert events == ["search", "read_previously_used", "fetch_page", "relevancy", "insert_query", "record_result"]


async def test_a_failed_relevancy_check_is_reported_as_an_error_not_as_no_results():
    async def assess(handler, query, pages, context):
        return [None] * len(pages), None

    content, citations, fetch = await _run(_args(queries=["a"]), [[_result(1), _result(2)]], set(), assess=assess)

    assert fetch.failed is True
    assert citations == []
    assert "relevance check failed" in content and "No new relevant" not in content
    assert content.startswith("<results-from-gov-uk-search>")
    rows = [call.kwargs for call in fetch.record.await_args_list]
    assert [(r["is_used"], r["llm_internal_response_id"]) for r in rows] == [(False, None), (False, None)]


async def test_a_partly_failed_relevancy_check_returns_the_checked_pages_with_a_note():
    results = [_result(n) for n in range(4)]

    async def assess(handler, query, pages, context):
        if pages[0]["title"] == "Page 0":
            return [True, False], 7
        return [None, None], None

    with patch(f"{SERVICE}.GOV_UK_SEARCH_RELEVANCY_BATCH_SIZE", 2):
        content, citations, fetch = await _run(_args(queries=["a"]), [results], set(), assess=assess)

    assert fetch.failed is False
    assert [c["docname"] for c in citations] == ["Page 0"]
    assert "2 result(s) could not be checked for relevance and were left out." in content


async def test_a_partly_failed_relevancy_check_with_nothing_relevant_says_so():
    results = [_result(n) for n in range(2)]

    async def assess(handler, query, pages, context):
        return ([False], 7) if pages[0]["title"] == "Page 0" else ([None], None)

    with patch(f"{SERVICE}.GOV_UK_SEARCH_RELEVANCY_BATCH_SIZE", 1):
        content, citations, fetch = await _run(_args(queries=["a"]), [results], set(), assess=assess)

    assert fetch.failed is False and citations == []
    assert "No new relevant GOV.UK pages found. 1 result(s) could not be checked" in content


async def test_no_results_is_not_a_failure():
    content, citations, fetch = await _run(_args(queries=["a"]), [[]], set())

    assert fetch.failed is False and citations == []
    assert "No new relevant GOV.UK pages found" in content


# --- get_gov_uk_page ---


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.gov.uk/guidance/style-guide", "https://www.gov.uk/guidance/style-guide"),
        ("http://gov.uk/government/news/x?foo=1#top", "https://www.gov.uk/government/news/x"),
        ("/government/news/x", "https://www.gov.uk/government/news/x"),
        ("  https://www.gov.uk/x  ", "https://www.gov.uk/x"),
    ],
)
def test_get_page_url_is_normalised(url, expected):
    assert GET_GOV_UK_PAGE.parse({"url": url}).url == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/x",
        "https://www.gov.uk.evil.com/x",
        "https://www.gov.uk@evil.com/x",
        "https://evil.com/?u=https://www.gov.uk/x",
        "https://www.gov.uk:8443/x",
        "ftp://www.gov.uk/x",
        "https://www.gov.uk/",
        "https://www.gov.uk",
        "not a url",
        "",
    ],
)
def test_get_page_rejects_anything_but_a_gov_uk_page(url):
    with pytest.raises(ValidationError, match="www.gov.uk"):
        GET_GOV_UK_PAGE.parse({"url": url})


async def test_get_page_returns_the_page_in_full():
    body = "# Style guide\n\nDocument type: guide\n\nWrite plainly."
    with patch(f"{SERVICE}.extract_content_from_gov_uk", AsyncMock(return_value=body)):
        content, citations, failed = await get_gov_uk_page_for_tool("https://www.gov.uk/style")

    assert not failed
    assert citations == [{"docname": "Style guide", "docurl": "https://www.gov.uk/style"}]
    assert content.startswith("<gov-uk-page>") and "Write plainly." in content


async def test_get_page_reports_a_page_it_could_not_fetch():
    with patch(f"{SERVICE}.extract_content_from_gov_uk", AsyncMock(return_value="")):
        content, citations, failed = await get_gov_uk_page_for_tool("https://www.gov.uk/missing")

    assert failed and citations == []
    assert "Couldn't get https://www.gov.uk/missing" in content


async def test_get_page_is_capped_by_its_own_limit_not_the_search_limit():
    body = "# Long page\n\n" + "x" * 5000
    with (
        patch(f"{SERVICE}.extract_content_from_gov_uk", AsyncMock(return_value=body)),
        patch(f"{SERVICE}.GOV_UK_SEARCH_MAX_DOCUMENT_CHARS", 100),
        patch(f"{SERVICE}.GOV_UK_PAGE_MAX_CHARS", 3000),
    ):
        content, _, _ = await get_gov_uk_page_for_tool("https://www.gov.uk/long")

    assert "x" * 100 in content and "x" * 2900 in content
    assert content.count("x") < 3000 and "characters truncated.]" in content
    assert "get_gov_uk_page" not in content


# --- attachments and get_gov_uk_pdf ---

ASSET = "https://assets.publishing.service.gov.uk/media/abc/My_report.pdf"


def _make_pdf(page_texts: list[str]) -> bytes:
    """A minimal text-only PDF with one page per string."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(page_texts)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>")
    font_id = 3 + 2 * len(page_texts)
    for i, text in enumerate(page_texts):
        stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


@pytest.mark.parametrize(
    "url, expected",
    [
        (ASSET, ASSET),
        ("http://assets.publishing.service.gov.uk/media/abc/My_report.pdf?x=1#p", ASSET),
    ],
)
def test_pdf_url_is_normalised(url, expected):
    assert GET_GOV_UK_PDF.parse({"url": url}).url == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://www.gov.uk/media/abc/x.pdf",
        "https://example.com/x.pdf",
        "https://assets.publishing.service.gov.uk.evil.com/x.pdf",
        "https://assets.publishing.service.gov.uk@evil.com/x.pdf",
        "https://assets.publishing.service.gov.uk:8443/x.pdf",
        "https://assets.publishing.service.gov.uk/",
        "/media/abc/x.pdf",
    ],
)
def test_pdf_rejects_anything_but_the_assets_host(url):
    with pytest.raises(ValidationError, match="assets.publishing.service.gov.uk"):
        GET_GOV_UK_PDF.parse({"url": url})


def test_pdf_start_page_must_be_positive():
    with pytest.raises(ValidationError, match="start_page"):
        GET_GOV_UK_PDF.parse({"url": ASSET, "start_page": 0})


def test_pdf_pages_are_read_from_start_page_within_the_limit():
    data = _make_pdf([f"Page {n} " + "word " * 20 for n in range(1, 6)])

    pages, total = _extract_pdf_pages(data, start_page=2, max_chars=250)

    assert total == 5
    assert [number for number, _ in pages] == [2, 3]
    assert pages[0][1].startswith("Page 2")


def test_pdf_start_page_is_cut_rather_than_dropped_when_over_the_limit():
    pages, _ = _extract_pdf_pages(_make_pdf(["x" * 500]), start_page=1, max_chars=100)

    assert len(pages) == 1 and len(pages[0][1]) == 100


async def test_pdf_tool_returns_text_and_says_how_to_continue():
    data = _make_pdf([f"Page {n} " + "word " * 20 for n in range(1, 4)])
    with (
        patch(f"{SERVICE}._download_pdf", AsyncMock(return_value=data)),
        patch(f"{SERVICE}.GOV_UK_PDF_MAX_CHARS", 250),
    ):
        content, citations, failed = await get_gov_uk_pdf_for_tool(ASSET, 1)

    assert not failed
    assert citations == [{"docname": "My report", "docurl": ASSET}]
    assert "[Page 1]" in content and "[Page 3]" not in content
    assert "Showing pages 1-2 of 3. To continue, call get_gov_uk_pdf with start_page=3." in content


async def test_pdf_tool_last_section_has_no_continue_hint():
    with patch(f"{SERVICE}._download_pdf", AsyncMock(return_value=_make_pdf(["Only page"]))):
        content, _, failed = await get_gov_uk_pdf_for_tool(ASSET, 1)

    assert not failed and "Showing pages 1-1 of 1." in content and "To continue" not in content


@pytest.mark.parametrize(
    "download, start_page, message",
    [
        (AsyncMock(side_effect=ValueError("file is too large")), 1, "file is too large"),
        (AsyncMock(side_effect=ValueError("file is not a PDF")), 1, "file is not a PDF"),
        (AsyncMock(side_effect=RuntimeError("boom")), 1, "damaged or password protected"),
        (AsyncMock(return_value=_make_pdf(["   "])), 1, "no readable text"),
        (AsyncMock(return_value=_make_pdf(["a", "b"])), 5, "past the end"),
    ],
    ids=["too_large", "not_pdf", "unexpected", "no_text", "past_end"],
)
async def test_pdf_tool_failures_are_reported(download, start_page, message):
    with patch(f"{SERVICE}._download_pdf", download):
        content, citations, failed = await get_gov_uk_pdf_for_tool(ASSET, start_page)

    assert failed and citations == [] and message in content


async def test_pdf_tool_reports_http_errors():
    response = httpx.Response(404, request=httpx.Request("GET", ASSET))
    error = httpx.HTTPStatusError("nope", request=response.request, response=response)
    with patch(f"{SERVICE}._download_pdf", AsyncMock(side_effect=error)):
        content, _, failed = await get_gov_uk_pdf_for_tool(ASSET, 1)

    assert failed and "HTTP 404" in content


def test_attachments_are_listed_with_addresses_and_how_to_read_them():
    lines = _attachment_lines(
        [
            {
                "title": "Statement (PDF)",
                "attachment_type": "file",
                "content_type": "application/pdf",
                "url": ASSET,
                "number_of_pages": 7,
            },
            {"title": "Statement (HTML)", "attachment_type": "html", "url": "/government/publications/x/y"},
            {
                "title": "Data",
                "attachment_type": "file",
                "content_type": "text/csv",
                "url": "https://assets.publishing.service.gov.uk/d.csv",
            },
            {"title": "Elsewhere", "attachment_type": "external", "url": "https://example.com"},
        ]
    )

    assert lines[0].startswith("Attachments (read HTML pages with get_gov_uk_page and PDFs with get_gov_uk_pdf)")
    assert lines[1:] == [
        f"- Statement (PDF) (PDF, 7 pages): {ASSET}",
        "- Statement (HTML) (HTML page): https://www.gov.uk/government/publications/x/y",
        "- Data (text/csv): https://assets.publishing.service.gov.uk/d.csv",
    ]


def test_no_attachments_adds_nothing():
    assert _attachment_lines(None) == [] and _attachment_lines([{"title": "x"}]) == []


def _collection_doc(n: int) -> dict:
    return {
        "content_id": f"id-{n}",
        "title": f"Report {n}",
        "web_url": f"https://www.gov.uk/government/publications/report-{n}",
        "public_updated_at": f"2025-0{n % 9 + 1}-01T10:00:00Z",
    }


def test_collection_documents_are_listed_by_group_with_addresses():
    lines = _collection_lines(
        [
            {"title": "Reports", "documents": ["id-1", "id-2"]},
            {"title": "Evidence", "documents": ["id-3", "missing"]},
            {"title": "Empty", "documents": ["missing"]},
        ],
        [_collection_doc(1), _collection_doc(2), _collection_doc(3)],
    )

    assert lines == [
        "Documents in this collection (read them with get_gov_uk_page):",
        "## Reports",
        "- Report 1 (updated 2025-02-01): https://www.gov.uk/government/publications/report-1",
        "- Report 2 (updated 2025-03-01): https://www.gov.uk/government/publications/report-2",
        "## Evidence",
        "- Report 3 (updated 2025-04-01): https://www.gov.uk/government/publications/report-3",
    ]


def test_collection_listing_is_capped():
    count = GOV_UK_COLLECTION_MAX_DOCUMENTS + 3
    lines = _collection_lines(
        [{"title": "All", "documents": [f"id-{n}" for n in range(count)]}], [_collection_doc(n) for n in range(count)]
    )

    assert sum(line.startswith("- ") for line in lines) == GOV_UK_COLLECTION_MAX_DOCUMENTS
    assert lines[-1] == "(3 more document(s) in this collection are not listed.)"


def test_no_collection_documents_adds_nothing():
    assert _collection_lines(None, None) == []
    assert _collection_lines([{"title": "Reports", "documents": ["id-1"]}], []) == []


def _mock_pdf_server(body: bytes, headers: dict | None = None):
    real_client = httpx.AsyncClient

    def client(**kwargs):
        handler = lambda request: httpx.Response(200, content=body, headers=headers or {})  # noqa: E731
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    return patch(f"{SERVICE}.httpx.AsyncClient", client)


async def test_pdf_download_returns_a_small_pdf():
    data = _make_pdf(["hello"])
    with _mock_pdf_server(data):
        assert await _download_pdf(ASSET) == data


async def test_pdf_download_refuses_a_file_declared_over_the_limit_before_reading_it():
    with (
        _mock_pdf_server(b"%PDF-" + b"x" * 10, {"content-length": "999999999"}),
        patch(f"{SERVICE}.GOV_UK_PDF_MAX_BYTES", 1000),
    ):
        with pytest.raises(ValueError, match="too large"):
            await _download_pdf(ASSET)


async def test_pdf_download_stops_a_file_that_outgrows_the_limit_without_declaring_its_size():
    body = b"%PDF-" + b"x" * 5000
    with _mock_pdf_server(body), patch(f"{SERVICE}.GOV_UK_PDF_MAX_BYTES", 1000):
        with pytest.raises(ValueError, match="too large"):
            await _download_pdf(ASSET)


async def test_pdf_download_refuses_a_file_that_is_not_a_pdf():
    with _mock_pdf_server(b"<html>not a pdf</html>"):
        with pytest.raises(ValueError, match="not a PDF"):
            await _download_pdf(ASSET)


@asynccontextmanager
async def _gov_uk_api(handler):
    """A local server standing in for the GOV.UK Content and Search APIs."""
    app = web.Application()
    app.router.add_get("/{tail:.*}", handler)
    async with LocalServer(app) as server:
        yield str(server.make_url("")).rstrip("/")


async def test_content_api_non_json_response_returns_nothing_so_the_html_fallback_runs():
    async def handler(request):
        return web.Response(text="<html>maintenance</html>", content_type="text/html")

    async with _gov_uk_api(handler) as base:
        with patch(f"{SERVICE}.CONTENT_URL", base):
            assert await GovUKContent.get_content("some/page") == {}


async def test_content_api_timeout_returns_nothing():
    async def handler(request):
        await asyncio.sleep(1)
        return web.json_response({"title": "late"})

    async with _gov_uk_api(handler) as base:
        with patch(f"{SERVICE}.CONTENT_URL", base), patch(f"{SERVICE}.GOV_UK_HTTP_TIMEOUT_SECONDS", 0.05):
            assert await GovUKContent.get_content("some/page") == {}


async def test_search_api_timeout_returns_no_results():
    async def handler(request):
        await asyncio.sleep(1)
        return web.json_response({"results": [_result(1)], "total": 1})

    async with _gov_uk_api(handler) as base:
        with patch(f"{SERVICE}.GOV_UK_HTTP_TIMEOUT_SECONDS", 0.05):
            assert await GovUKSearch.fetch(f"{base}/api/search.json") == {"results": [], "total": 0}
