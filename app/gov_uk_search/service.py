import asyncio
import io
import re
from typing import Any, Dict, List, Tuple
from urllib.parse import unquote, urlparse

import aiohttp
import httpx
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from pdfminer.pdfpage import PDFPage
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.bedrock import BedrockHandler, RunMode
from app.bedrock.call_log import LlmCallContext
from app.bedrock.tool_schema import ToolResponseError
from app.config import (
    GOV_UK_BASE_URL,
    GOV_UK_HTTP_TIMEOUT_SECONDS,
    GOV_UK_PAGE_MAX_CHARS,
    GOV_UK_PDF_MAX_BYTES,
    GOV_UK_PDF_MAX_CHARS,
    GOV_UK_SEARCH_DEFAULT_COUNT,
    GOV_UK_SEARCH_MAX_COUNT,
    GOV_UK_SEARCH_MAX_DOCUMENT_CHARS,
    GOV_UK_SEARCH_MAX_TOOL_DOCUMENTS,
    GOV_UK_SEARCH_MAX_TOOL_SEARCH_TERMS,
    GOV_UK_SEARCH_RELEVANCY_BATCH_SIZE,
    GOV_UK_SEARCH_RELEVANCY_EXCERPT_CHARS,
    LLM_DOCUMENT_RELEVANCY_MODEL,
)
from app.database.db_operations import DbOperations
from app.database.models import GovUkSearchResult, Message
from app.database.table import LLMTable, async_db_session
from app.gov_uk_search.constants import CONTENT_URL
from app.gov_uk_search.prompts import (
    SYSTEM_PROMPT_PAGE_RELEVANCE_EVALUATOR,
    TOOL_RESULT_NO_RELEVANT_PAGES,
    TOOL_RESULT_RELEVANCE_CHECK_FAILED,
    already_used_note,
    unchecked_note,
    wrap_search_tool_result,
)
from app.gov_uk_search.schemas import DocumentBlacklistStatus, NonRagDocument
from app.gov_uk_search.tools import PAGE_RELEVANCE, SearchGovUkArgs
from app.gov_uk_search.utils import build_search_url, normalise_gov_uk_url
from app.logs.logs_handler import logger


def _aiohttp_timeout() -> aiohttp.ClientTimeout:
    """The total timeout for one GOV.UK API request (aiohttp's own default is 5 minutes)."""
    return aiohttp.ClientTimeout(total=GOV_UK_HTTP_TIMEOUT_SECONDS)


class GovUKContent:
    @staticmethod
    async def get_content(path: str) -> Dict[str, Any]:
        """
        Get content from the GOV UK Content API.

        Args:
            path: The path of the content to retrieve (without leading slash)

        Returns:
            Dict containing the content data

        Example:
            >>> await GovUKContent.get_content("help/cookies")
        """
        url = f"{CONTENT_URL}/{path.lstrip('/')}"
        logger.debug(f"Fetching content from GOV UK Content API: {url}")

        try:
            async with aiohttp.ClientSession(timeout=_aiohttp_timeout()) as session:
                async with session.get(url) as response:
                    if response.status == 404:
                        logger.warning(f"Content not found at path: {path}")
                        return {}
                    if response.status != 200:
                        logger.error(f"Error fetching content: {response.status}")
                        return {}

                    return await response.json()
        except (aiohttp.ClientError, TimeoutError, ValueError) as e:
            # A timeout, connection error or non-JSON body: return nothing so callers fall back to the page HTML.
            logger.warning(f"GOV UK Content API request failed for {url}: {e!r}")
            return {}

    @staticmethod
    async def get_content_and_links(path: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        Get content and extract all internal links from it.

        Args:
            path: The path of the content to retrieve

        Returns:
            Tuple of (content_data, list_of_links)
        """
        content = await GovUKContent.get_content(path)
        if not content:
            return {}, []

        # Extract links from the content
        links = []

        # Add links from the content relationships
        for _, link_data in content.get("links", {}).items():
            if isinstance(link_data, list):
                for item in link_data:
                    if isinstance(item, dict) and "base_path" in item:
                        links.append(item["base_path"])

        return content, links


class GovUKSearch:
    @staticmethod
    async def search(
        query: str,
        count: int = GOV_UK_SEARCH_MAX_COUNT,
        order_by_field_name: str = "",
        descending_order: bool = False,
        start: int = 0,
        fields: list[str] | None = None,
        filter_by_field: list[tuple[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Query the GOV.UK Search API. No database access.

        Returns:
            The response JSON, or {"results": [], "total": 0} on any error.
        """
        url = build_search_url(
            query=query,
            count=count,
            order_by_field_name=order_by_field_name,
            descending_order=descending_order,
            start=start,
            fields=fields,
            filter_by_field=filter_by_field,
        )
        return await GovUKSearch.fetch(url)

    @staticmethod
    async def fetch(url: str) -> Dict[str, Any]:
        """GET a Search API URL built with `build_search_url`. No database access.

        Returns:
            The response JSON, or {"results": [], "total": 0} on any error.
        """
        empty_response = {"results": [], "total": 0}
        if len(url) > 2000:
            logger.warning(f"GOV UK Search API URL is very long ({len(url)} chars): {url[:200]}...")
        try:
            async with aiohttp.ClientSession(timeout=_aiohttp_timeout()) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"GOV UK Search API error {response.status} for {url}: {await response.text()}")
                        return empty_response
                    return await response.json()
        except Exception:
            logger.exception(f"GOV UK Search API request failed for {url}")
            return empty_response


async def process_documents(
    documents: List[Any], max_chars: int | None = None, can_fetch_full_page: bool = True
) -> str:
    """
    Process documents and wrap them for inclusion in the prompt.

    Args:
        documents: List of documents to process
        max_chars: Longest body to keep per document. Defaults to GOV_UK_SEARCH_MAX_DOCUMENT_CHARS.
        can_fetch_full_page: Whether to tell the model that get_gov_uk_page returns a truncated page in full.

    Returns:
        Formatted document content
    """
    max_chars = max_chars or GOV_UK_SEARCH_MAX_DOCUMENT_CHARS
    wrapped_document = ""
    for i, document in enumerate(documents, 1):
        body = document.body
        if len(body) > max_chars:
            logger.warning(f"GOV.UK document '{document.title}' is {len(body):,} chars, truncating to {max_chars:,}")
            omitted = len(body) - max_chars
            hint = " Call get_gov_uk_page to retrieve the full page." if can_fetch_full_page else ""
            body = f"{body[:max_chars]}... [{omitted:,} characters truncated.{hint}]"

        wrapped_document += f"<gov-uk-search-result-{i}>\n"
        wrapped_document += f"<document-title>{document.title}</document-title>\n"
        wrapped_document += f"<document-url>{document.url}</document-url>\n"
        wrapped_document += f"<document-body>\n{body}\n</document-body>\n"
        wrapped_document += f"</gov-uk-search-result-{i}>"
        if i < len(documents):
            wrapped_document += "\n"

    return wrapped_document


async def extract_content_with_httpx(url: str) -> str:
    """
    Extract content directly from the GOV UK website using httpx.
    Used as a fallback when the Content API doesn't work.

    Args:
        url: The GOV UK URL

    Returns:
        Extracted content as a string
    """
    try:
        async with httpx.AsyncClient(timeout=GOV_UK_HTTP_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract content
            result = []

            # Get title
            title_elem = soup.find("h1")
            if title_elem:
                result.append(f"# {title_elem.get_text().strip()}")
                result.append("")

            # Get metadata
            metadata_elems = soup.select(".app-c-publisher-metadata, .gem-c-metadata")
            if metadata_elems:
                for elem in metadata_elems:
                    meta_text = elem.get_text().strip()
                    if meta_text:
                        result.append(meta_text)
                result.append("")

            # Get main content
            content_elem = soup.select_one("#content, .govuk-grid-column-two-thirds")
            if content_elem:
                # Extract paragraphs and headings
                for elem in content_elem.find_all(["p", "h2", "h3", "h4", "li"]):
                    text = elem.get_text().strip()
                    if text:
                        if elem.name.startswith("h"):
                            # Add markdown heading format
                            level = int(elem.name[1])
                            result.append(f"{'#' * level} {text}")
                        else:
                            result.append(text)
                        result.append("")

            # Join all parts
            full_content = "\n".join(result)
            return full_content

    except (httpx.HTTPError, Exception) as e:
        logger.warning(f"Error fetching content with httpx: {str(e)}")
        return ""


def _attachment_lines(attachments: list[dict] | None) -> list[str]:
    """Lines listing a page's attachments with their addresses, so the model can read them.

    Args:
        attachments: The Content API's details.attachments.

    Returns:
        The lines to add to the page text (empty if there are no readable attachments).
    """
    lines = []
    for attachment in attachments or []:
        url = attachment.get("url")
        if not url or attachment.get("attachment_type") == "external":
            continue
        if url.startswith("/"):
            url = f"{GOV_UK_BASE_URL}{url}"
        kind = "HTML page" if attachment.get("attachment_type") == "html" else attachment.get("content_type") or "file"
        if kind == "application/pdf":
            kind = "PDF"
        pages = attachment.get("number_of_pages")
        detail = f"{kind}, {pages} pages" if pages else kind
        lines.append(f"- {attachment.get('title', 'Untitled')} ({detail}): {url}")
    if not lines:
        return []
    return [
        "Attachments (read HTML pages with get_gov_uk_page and PDFs with get_gov_uk_pdf):",
        *lines,
    ]


GOV_UK_COLLECTION_MAX_DOCUMENTS = 50


def _collection_lines(collection_groups: list[dict] | None, documents: list[dict] | None) -> list[str]:
    """Lines listing the documents in a document collection with their addresses, so the model can read them.

    Args:
        collection_groups: The Content API's details.collection_groups (titled groups of document content ids).
        documents: The Content API's links.documents (title, web_url and dates for each content id).

    Returns:
        The lines to add to the page text (empty if the page lists no documents). At most
        GOV_UK_COLLECTION_MAX_DOCUMENTS are listed, in the collection's own order.
    """
    by_content_id = {d["content_id"]: d for d in documents or [] if d.get("content_id")}
    lines = []
    listed = 0
    omitted = 0
    for group in collection_groups or []:
        group_lines = []
        for content_id in group.get("documents", []):
            document = by_content_id.get(content_id)
            if not document or not document.get("web_url"):
                continue
            if listed >= GOV_UK_COLLECTION_MAX_DOCUMENTS:
                omitted += 1
                continue
            published = (document.get("public_updated_at") or "")[:10]
            detail = f" (updated {published})" if published else ""
            group_lines.append(f"- {document.get('title', 'Untitled')}{detail}: {document['web_url']}")
            listed += 1
        if group_lines:
            lines.extend([f"## {group.get('title', 'Documents')}", *group_lines])
    if not lines:
        return []
    if omitted:
        lines.append(f"({omitted} more document(s) in this collection are not listed.)")
    return ["Documents in this collection (read them with get_gov_uk_page):", *lines]


async def extract_content_from_gov_uk(url: str) -> str:
    """
    Extract full content from GOV UK Content API for a given URL.
    Falls back to direct web scraping if the Content API fails.

    Args:
        url: The GOV UK URL

    Returns:
        Full content as a string
    """
    # Extract path from URL
    parsed_url = urlparse(url)
    path = parsed_url.path.lstrip("/")

    # Try Content API first
    content_data = await GovUKContent.get_content(path)

    if content_data:
        # Extract text content
        result = []

        # Get title
        title = content_data.get("title", "")
        if title:
            result.append(f"# {title}")
            result.append("")

        # Get document type
        document_type = content_data.get("document_type", "")
        if document_type:
            result.append(f"Document type: {document_type}")
            result.append("")

        # Get description
        description = content_data.get("description", "")
        if description:
            result.append(description)
            result.append("")

        details = content_data.get("details", {})

        attachment_lines = _attachment_lines(details.get("attachments"))
        if attachment_lines:
            result.extend(attachment_lines)
            result.append("")

        collection_lines = _collection_lines(
            details.get("collection_groups"), content_data.get("links", {}).get("documents")
        )
        if collection_lines:
            result.extend(collection_lines)
            result.append("")

        # Get body content

        # Handle different content structures
        if "body" in details:
            # HTML content - extract text
            body = details["body"]
            # Simple HTML tag removal (a more sophisticated HTML parser could be used)
            body = re.sub(r"<[^>]+>", " ", body)
            body = re.sub(r"\s+", " ", body).strip()
            result.append(body)

        elif "parts" in details:
            # Multi-part content
            for part in details["parts"]:
                part_title = part.get("title", "")
                part_body = part.get("body", "")

                if part_title:
                    result.append(f"## {part_title}")

                if part_body:
                    # Remove HTML tags
                    part_body = re.sub(r"<[^>]+>", " ", part_body)
                    part_body = re.sub(r"\s+", " ", part_body).strip()
                    result.append(part_body)

                result.append("")

        # Add publication date if available
        if "public_updated_at" in content_data:
            result.append(f"Last updated: {content_data['public_updated_at']}")

        # Join all content parts
        full_content = "\n".join(result)
        return full_content

    # Fallback to direct web scraping if Content API failed
    logger.warning(f"Content API failed for {url}, falling back to direct web scraping")
    content_from_httpx = await extract_content_with_httpx(url)

    if content_from_httpx:
        return content_from_httpx

    # Both methods failed
    logger.error(f"Failed to extract content for {url} using both Content API and httpx")
    return ""


async def record_search_result(
    m_user_id: int,
    llm_internal_response_id: int | None,
    gov_uk_search_query_id: int,
    url: str,
    content: str,
    is_used: bool,
    position: int,
    db_session: AsyncSession,
) -> None:
    """
    Record the search result in the database.
    """
    await DbOperations.insert_gov_uk_search_result(
        db_session=db_session,
        llm_internal_response_id=llm_internal_response_id,
        message_id=m_user_id,
        gov_uk_search_query_id=gov_uk_search_query_id,
        url=url,
        content=content,
        is_used=is_used,
        position=position,
    )


# --- search_gov_uk and get_gov_uk_page loop tools ---

GOV_UK_PAGE_TOOL_TAG = "gov-uk-page"
GOV_UK_PDF_TOOL_TAG = "gov-uk-pdf"
SEARCH_RESULT_FIELDS = ["title", "description", "link", "primary_publishing_organisation", "public_timestamp"]


def _search_filters(args: SearchGovUkArgs) -> list[tuple[str, Any]]:
    """Build the Search API filter (field, value) pairs for a tool call."""
    filter_by_field = []
    if args.date_range:
        date_filter = f"from:{args.date_range.from_}"
        if args.date_range.to:
            date_filter += f",to:{args.date_range.to}"
        filter_by_field.append(("public_timestamp", date_filter))
    for value in args.content_purpose_supergroup or []:
        filter_by_field.append(("content_purpose_supergroup", value))
    for value in args.content_purpose_subgroup or []:
        filter_by_field.append(("content_purpose_subgroup", value))
    if args.is_political is not None:
        filter_by_field.append(("is_political", args.is_political))
    return filter_by_field


async def _fetch_search_results(url: str) -> list[dict]:
    """GET a Search API URL. Returns its results, or [] on any error."""
    return (await GovUKSearch.fetch(url)).get("results", [])


def _full_link(link: str) -> str | None:
    """The canonical GOV.UK URL for a search result's link, or None if it isn't a GOV.UK page.

    The search index includes some external links (e.g. to nhs.uk), which are never fetched or cited.
    """
    if not link.startswith(("/", "http://", "https://")):
        link = f"/{link}"
    try:
        return normalise_gov_uk_url(link)
    except ValueError:
        return None


def _relevancy_excerpt(description: str, full_content: str) -> str:
    """The page text shown to the relevancy check: the description, then a bounded excerpt of the content."""
    if not full_content or full_content == description:
        return description
    truncated = full_content[:GOV_UK_SEARCH_RELEVANCY_EXCERPT_CHARS]
    if len(full_content) > GOV_UK_SEARCH_RELEVANCY_EXCERPT_CHARS:
        truncated += "... [content truncated]"
    return f"{description}\n\nContent excerpt:\n{truncated}"


async def _assess_relevancy(
    handler: BedrockHandler,
    relevancy_query: str,
    pages: list[dict],
    context: LlmCallContext,
) -> tuple[list[bool | None], int | None]:
    """Ask the relevancy model whether each of a batch of pages is relevant, in one call.

    Uses no DB session. A page gets None (unchecked) when the call fails, the response doesn't
    validate, or the response has no evaluation for it.

    Args:
        handler: The relevancy model's handler.
        relevancy_query: The user's query, with the search terms used.
        pages: The pages to assess, each with "title", "description" and "full_content".
        context: What the call is for, recorded with it.

    Returns:
        One decision per page, in order (True relevant, False not, None unchecked), and the logged
        call's llm_internal_response_id (None if the call itself failed).
    """
    page_blocks = [
        f"<page-{i}>\n"
        f"<page-title>{page['title']}</page-title>\n"
        f"<page-content>{_relevancy_excerpt(page['description'], page['full_content'])}</page-content>\n"
        f"</page-{i}>"
        for i, page in enumerate(pages)
    ]
    messages = [
        {
            "role": "user",
            "content": (
                f"<user-query>\n\n{relevancy_query}\n\n</user-query>\n\n"
                "<retrieved-gov-uk-pages>\n" + "\n".join(page_blocks) + "\n</retrieved-gov-uk-pages>"
            ),
        }
    ]
    try:
        response = await handler.invoke_async(
            messages,
            context=context,
            system=SYSTEM_PROMPT_PAGE_RELEVANCE_EVALUATOR,
            tools=[PAGE_RELEVANCE.spec],
            tool_choice=PAGE_RELEVANCE.forced_choice,
        )
    except Exception:
        logger.exception(f"GOV UK Search - relevancy check failed; none of the {len(pages)} pages could be checked")
        return [None] * len(pages), None

    try:
        evaluations = {e.page_index: e for e in PAGE_RELEVANCE.parse_response(response).evaluations}
    except (ToolResponseError, ValidationError):
        logger.error(
            "GOV UK Search - relevancy response didn't validate "
            f"(stop_reason={getattr(response, 'stop_reason', None)!r}); "
            f"none of the {len(pages)} pages could be checked",
            exc_info=True,
        )
        evaluations = {}

    decisions: list[bool | None] = []
    for i, page in enumerate(pages):
        evaluation = evaluations.get(i)
        decisions.append(evaluation.is_relevant if evaluation else None)
        logger.debug(
            f"GOV UK Search - '{page['title']}': is_relevant={decisions[-1]}, "
            f"reasoning={evaluation.reasoning if evaluation else 'No evaluation returned'}"
        )

    missing = sum(d is None for d in decisions)
    if evaluations and missing:
        logger.warning(f"GOV UK Search - relevancy check gave no decision for {missing} of {len(pages)} pages")
    return decisions, response.llm_internal_response_id


async def _previously_used_urls(db_session: AsyncSession, message_id: int) -> set[str]:
    """URLs of GOV.UK pages already used for this message's chat, this turn included."""
    chat_id = select(Message.chat_id).where(Message.id == message_id).scalar_subquery()
    stmt = (
        select(GovUkSearchResult.url)
        .join(Message, Message.id == GovUkSearchResult.message_id)
        .where(Message.chat_id == chat_id, GovUkSearchResult.is_used.is_(True))
        .distinct()
    )
    return set((await db_session.execute(stmt)).scalars().all())


async def search_gov_uk_for_tool(
    args: SearchGovUkArgs,
    user_query: str,
    message_id: int,
    llm_internal_response_id: int | None,
    db_session: AsyncSession,
) -> tuple[str, list[dict], bool]:
    """Search GOV.UK with the chat model's own terms and filters, and return the relevant pages.

    Holds no DB connection during the Search API, page fetch and relevance-check calls: with
    args.remove_duplicates, pages already used earlier in the chat are looked up first in their own
    short session, and db_session is only used at the end, to record each query
    (gov_uk_search_query) and each assessed page (gov_uk_search_result).

    Args:
        args: The model's queries and filters.
        user_query: This turn's user message text, for the relevancy check.
        message_id: This turn's user message id.
        llm_internal_response_id: The logged model call that requested the search.
        db_session: Session to record analytics in. Not touched until the network calls are done.

    Returns:
        The tool result text, the citations for the pages returned, and whether the search failed
        (the relevance check gave no decision for any candidate).
    """
    terms = list(dict.fromkeys(args.queries))[:GOV_UK_SEARCH_MAX_TOOL_SEARCH_TERMS]
    count = min(max(args.count or GOV_UK_SEARCH_DEFAULT_COUNT, 1), GOV_UK_SEARCH_MAX_COUNT)
    filter_by_field = _search_filters(args)
    urls = [
        build_search_url(
            query=term,
            count=count,
            start=0,
            fields=SEARCH_RESULT_FIELDS,
            filter_by_field=filter_by_field,
        )
        for term in terms
    ]
    result_sets = await asyncio.gather(*(_fetch_search_results(url) for url in urls))

    already_used = set()
    if args.remove_duplicates:
        async with async_db_session() as read_session:
            already_used = await _previously_used_urls(read_session, message_id)
    candidates = []
    seen_links = set(already_used)
    skipped = 0
    for query_index, results in enumerate(result_sets):
        for result in results:
            link = result.get("link")
            if not link:
                continue
            link = _full_link(link)
            if link is None:
                logger.info(f"GOV UK Search tool: skipping non-GOV.UK result {result.get('link')}")
                continue
            if link in already_used:
                skipped += 1
            if link in seen_links:
                continue
            seen_links.add(link)
            candidates.append(
                {
                    "title": result.get("title", ""),
                    "description": result.get("description", ""),
                    "link": link,
                    "query_index": query_index,
                    "position": len(candidates) + 1,
                }
            )

    contents = await asyncio.gather(
        *(extract_content_from_gov_uk(c["link"]) for c in candidates), return_exceptions=True
    )
    for candidate, content in zip(candidates, contents, strict=True):
        if isinstance(content, BaseException):
            if not isinstance(content, Exception):
                raise content
            logger.warning(f"GOV UK Search tool: couldn't fetch {candidate['link']}: {content!r}")
            content = ""
        candidate["full_content"] = content or candidate["description"]

    relevancy_handler = BedrockHandler(llm=LLMTable().get_by_model(LLM_DOCUMENT_RELEVANCY_MODEL), mode=RunMode.ASYNC)
    relevancy_query = f"{user_query}\n\n(Searched GOV.UK for: {'; '.join(terms)})"
    context = LlmCallContext(purpose="gov_uk_relevancy", message_id=message_id)
    batches = [
        candidates[i : i + GOV_UK_SEARCH_RELEVANCY_BATCH_SIZE]
        for i in range(0, len(candidates), GOV_UK_SEARCH_RELEVANCY_BATCH_SIZE)
    ]
    batch_results = await asyncio.gather(
        *(_assess_relevancy(relevancy_handler, relevancy_query, batch, context) for batch in batches)
    )
    for batch, (decisions, relevancy_response_id) in zip(batches, batch_results, strict=True):
        for candidate, decision in zip(batch, decisions, strict=True):
            candidate["decision"] = decision
            candidate["relevancy_response_id"] = relevancy_response_id

    relevant = [c for c in candidates if c["decision"]][:GOV_UK_SEARCH_MAX_TOOL_DOCUMENTS]
    used_links = {c["link"] for c in relevant}
    query_records = [
        await DbOperations.insert_gov_uk_search_query(
            db_session=db_session,
            llm_internal_response_id=llm_internal_response_id,
            message_id=message_id,
            query=url,
        )
        for url in urls
    ]
    for candidate in candidates:
        await record_search_result(
            m_user_id=message_id,
            llm_internal_response_id=candidate["relevancy_response_id"],
            gov_uk_search_query_id=query_records[candidate["query_index"]].id,
            url=candidate["link"],
            content=candidate["full_content"],
            is_used=candidate["link"] in used_links,
            position=candidate["position"],
            db_session=db_session,
        )

    unchecked = sum(c["decision"] is None for c in candidates)
    logger.info(
        f"GOV UK Search tool: {len(candidates)} candidates, {len(relevant)} relevant, {skipped} already used, "
        f"{unchecked} unchecked"
    )
    if candidates and unchecked == len(candidates):
        return wrap_search_tool_result(TOOL_RESULT_RELEVANCE_CHECK_FAILED), [], True

    notes = []
    if unchecked:
        notes.append(unchecked_note(unchecked))
    if skipped:
        notes.append(already_used_note(skipped))
    if not relevant:
        return wrap_search_tool_result(" ".join([TOOL_RESULT_NO_RELEVANT_PAGES, *notes])), [], False

    documents = [
        NonRagDocument(title=c["title"], url=c["link"], body=c["full_content"], status=DocumentBlacklistStatus.OK)
        for c in relevant
    ]
    wrapped = await process_documents(documents)
    citations = [{"docname": c["title"], "docurl": c["link"]} for c in relevant]
    return wrap_search_tool_result("\n".join([wrapped, *notes])), citations, False


async def get_gov_uk_page_for_tool(url: str) -> tuple[str, list[dict], bool]:
    """Fetch one GOV.UK page in full.

    Args:
        url: A canonical GOV.UK page URL (see normalise_gov_uk_url).

    Returns:
        The tool result text, citations for the page (empty if it couldn't be fetched), and whether
        the fetch failed.
    """
    content = await extract_content_from_gov_uk(url)
    if not content:
        message = f"Couldn't get {url}. Check that it is the address of a published GOV.UK page."
        return f"<{GOV_UK_PAGE_TOOL_TAG}>\n{message}\n</{GOV_UK_PAGE_TOOL_TAG}>", [], True

    first_line = content.split("\n", 1)[0]
    title = first_line.removeprefix("# ").strip() if first_line.startswith("# ") else url
    page = NonRagDocument(title=title, url=url, body=content, status=DocumentBlacklistStatus.OK)
    wrapped = await process_documents([page], max_chars=GOV_UK_PAGE_MAX_CHARS, can_fetch_full_page=False)
    return f"<{GOV_UK_PAGE_TOOL_TAG}>\n{wrapped}\n</{GOV_UK_PAGE_TOOL_TAG}>", [{"docname": title, "docurl": url}], False


async def _download_pdf(url: str) -> bytes:
    """Download a PDF without following redirects, refusing anything over GOV_UK_PDF_MAX_BYTES.

    Raises:
        httpx.HTTPError: The request failed or the response was an error status.
        ValueError: The file is too large or isn't a PDF.
    """
    async with httpx.AsyncClient(timeout=GOV_UK_HTTP_TIMEOUT_SECONDS, follow_redirects=False) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            if int(response.headers.get("content-length") or 0) > GOV_UK_PDF_MAX_BYTES:
                raise ValueError("file is too large")
            buffer = io.BytesIO()
            async for chunk in response.aiter_bytes():
                if buffer.tell() + len(chunk) > GOV_UK_PDF_MAX_BYTES:
                    raise ValueError("file is too large")
                buffer.write(chunk)
    data = buffer.getvalue()
    if not data.startswith(b"%PDF"):
        raise ValueError("file is not a PDF")
    return data


def _extract_pdf_pages(data: bytes, start_page: int, max_chars: int) -> tuple[list[tuple[int, str]], int]:
    """Read a PDF's text page by page from start_page, stopping before max_chars would be exceeded.

    Always returns at least the first page requested (cut to max_chars if it alone is longer).

    Args:
        data: The PDF file.
        start_page: The 1-based page to start from.
        max_chars: Most characters of text to return.

    Returns:
        (page number, text) pairs, and the PDF's total page count.
    """
    total_pages = sum(1 for _ in PDFPage.get_pages(io.BytesIO(data)))
    pages: list[tuple[int, str]] = []
    if start_page > total_pages:  # pdfminer treats an empty page_numbers as "every page"
        return pages, total_pages
    used = 0
    layouts = extract_pages(io.BytesIO(data), page_numbers=range(start_page - 1, total_pages))
    for number, layout in enumerate(layouts, start=start_page):
        text = "".join(element.get_text() for element in layout if isinstance(element, LTTextContainer)).strip()
        if pages and used + len(text) > max_chars:
            break
        pages.append((number, text[:max_chars]))
        used += len(text)
    return pages, total_pages


async def get_gov_uk_pdf_for_tool(url: str, start_page: int) -> tuple[str, list[dict], bool]:
    """Read the text of a PDF attached to a GOV.UK page, a section at a time.

    Args:
        url: A canonical GOV.UK assets URL (see normalise_gov_uk_asset_url).
        start_page: The 1-based page to start from.

    Returns:
        The tool result text, citations for the PDF (empty on failure), and whether it failed.
    """

    def failure(message: str) -> tuple[str, list[dict], bool]:
        return f"<{GOV_UK_PDF_TOOL_TAG}>\n{message}\n</{GOV_UK_PDF_TOOL_TAG}>", [], True

    try:
        data = await _download_pdf(url)
        pages, total_pages = await asyncio.to_thread(_extract_pdf_pages, data, start_page, GOV_UK_PDF_MAX_CHARS)
    except httpx.HTTPStatusError as e:
        logger.warning(f"GOV UK PDF - {url} returned {e.response.status_code}")
        status = e.response.status_code
        return failure(f"Couldn't get {url} (HTTP {status}). Use the address listed on the GOV.UK page.")
    except ValueError as e:
        return failure(f"Couldn't read {url}: {e}.")
    except Exception:
        logger.exception(f"GOV UK PDF - failed to read {url}")
        return failure(f"Couldn't read {url}. The file may be damaged or password protected.")

    if not pages:
        return failure(f"{url} has {total_pages} page(s); start_page {start_page} is past the end.")
    if not any(text for _, text in pages):
        return failure(f"{url} has no readable text; it may be a scanned image.")

    last_page = pages[-1][0]
    filename = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    title = re.sub(r"\s+", " ", filename.removesuffix(".pdf").replace("_", " ")).strip() or url
    body = "\n\n".join(f"[Page {number}]\n{text}" for number, text in pages)
    note = f"Showing pages {pages[0][0]}-{last_page} of {total_pages}."
    if last_page < total_pages:
        note += f" To continue, call get_gov_uk_pdf with start_page={last_page + 1}."
    wrapped = (
        f"<document-title>{title}</document-title>\n<document-url>{url}</document-url>\n"
        f"<document-body>\n{body}\n</document-body>\n<pages>{note}</pages>"
    )
    return f"<{GOV_UK_PDF_TOOL_TAG}>\n{wrapped}\n</{GOV_UK_PDF_TOOL_TAG}>", [{"docname": title, "docurl": url}], False
