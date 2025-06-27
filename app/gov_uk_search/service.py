import asyncio
import re
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import aiohttp
import httpx
from bs4 import BeautifulSoup
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.bedrock import BedrockHandler, RunMode
from app.bedrock.bedrock import Result
from app.bedrock.tools_use import (
    DOCUMENT_RELEVANCE_ASSESSMENT,
    DOWNLOAD_URLS,
    NAME_USE_GOV_UK_SEARCH_ASSESSMENT,
    PROPERTY_NAME_USE_GOV_UK_SEARCH_ASSESSMENT,
    SEARCH_API_SEARCH_TERMS,
    USE_GOV_UK_SEARCH_ASSESSMENT,
)
from app.chat.exceptions import ChatNotFoundError, LLMNotFoundError
from app.chat.schemas import ChatCreateMessageInput, RoleEnum
from app.config import (
    GOV_UK_SEARCH_MAX_COUNT,
    LLM_DEFAULT_MODEL,
    LLM_DOCUMENT_RELEVANCY_MODEL,
    LLM_GOV_UK_SEARCH_FOLLOWUP_ASSESMENT,
)
from app.database.db_operations import DbOperations
from app.database.models import Chat, GovUkSearchResult, Message, UseGovUkSearchDecision
from app.database.table import LLMTable
from app.gov_uk_search.constants import CONTENT_URL
from app.gov_uk_search.schemas import DocumentBlacklistStatus, NonRagDocument, SearchCost
from app.gov_uk_search.utils import build_search_url
from app.logs.logs_handler import logger


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

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 404:
                    logger.warning(f"Content not found at path: {path}")
                    return {}
                if response.status != 200:
                    logger.error(f"Error fetching content: {response.status}")
                    return {}

                return await response.json()

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
    async def simple_search(
        query: str,
        db_session: AsyncSession,
        count: int = GOV_UK_SEARCH_MAX_COUNT,
        order_by_field_name: str = "",
        descending_order: bool = False,
        start: int = 0,
        fields: list[str] | None = None,
        filter_by_field: list[tuple[str, Any]] | None = None,
        llm_internal_response_id_query: int | None = None,
        message_id: int | None = None,
    ) -> Tuple[Dict[str, Any], int]:
        get_url = build_search_url(
            query=query,
            count=count,
            order_by_field_name=order_by_field_name,
            descending_order=descending_order,
            start=start,
            fields=fields,
            filter_by_field=filter_by_field,
        )

        logger.debug(f"GOV UK Search API URL: {get_url}")

        # Add additional validation logging
        if len(get_url) > 2000:
            logger.warning(f"GOV UK Search API URL is very long ({len(get_url)} chars): {get_url[:200]}...")

        # Log the query parameters being used
        logger.debug(
            f"GOV UK Search API params - query: '{query}', count: {count}, order: "
            f"'{order_by_field_name}', descending: {descending_order}, fields: {fields}, filters: {filter_by_field}"
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(get_url) as response:
                # Log the response status and headers for debugging
                logger.debug(f"GOV UK Search API response status: {response.status}")
                logger.debug(f"GOV UK Search API response headers: {response.headers}")

                # Check if the response is successful
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"GOV UK Search API error: {response.status} - {error_text}")
                    logger.error(f"Failed GOV UK Search URL request: {get_url}")
                    # Return an empty response with a default query ID
                    empty_response = {"results": [], "total": 0}

                    # Log the LLM-generated query passed on to the GOV UK Search API
                    gov_uk_search_query = await DbOperations.insert_gov_uk_search_query(
                        db_session=db_session,
                        llm_internal_response_id=llm_internal_response_id_query,
                        message_id=message_id,
                        query=get_url,
                    )

                    # logger.info(f"Queries built using tool use (function calling): {query}")

                    return empty_response, gov_uk_search_query

                # Try to parse the response as JSON
                try:
                    response_data = await response.json()
                    logger.debug(f"GOV UK Search API response: {response_data}")
                except aiohttp.client_exceptions.ContentTypeError as e:
                    # If the response is not JSON, log the error and return an empty response
                    error_text = await response.text()
                    logger.error(f"GOV UK Search API content type error: {e} - Response text: {error_text[:500]}")
                    response_data = {"results": [], "total": 0}

                # Log the LLM-generated query passed on to the GOV UK Search API
                gov_uk_search_query = await DbOperations.insert_gov_uk_search_query(
                    db_session=db_session,
                    llm_internal_response_id=llm_internal_response_id_query,
                    message_id=message_id,
                    query=get_url,
                )

                # logger.info(f"Queries built using tool use (function calling): {query}")

                return response_data, gov_uk_search_query


async def get_search_documents(
    llm: BedrockHandler, query: str, m_user_id: int, db_session: AsyncSession
) -> Tuple[List[Any], List[Dict[str, str]], Dict[str, Any]]:
    """
    Get documents using GOV UK Search API.

    Args:
        llm: LLM for web browsing
        query: User query
        m_user_id: User ID

    Returns:
        Tuple of documents, citations, and cost information
    """
    # Get relevant documents using the new streamlined function
    documents, citations, search_cost = await get_relevant_documents_from_gov_uk_search(
        llm=llm, role=RoleEnum.user, query=query, m_user_id=m_user_id, db_session=db_session
    )

    # Process documents and filter blacklisted ones
    valid_documents = []
    valid_citations = []
    blacklisted_documents = []

    for document in documents:
        if document.status == DocumentBlacklistStatus.BLACKLISTED:
            blacklisted_documents.append(document)
            continue
        valid_documents.append(document)
        valid_citations.append({"docname": document.title, "docurl": document.url})

    # Log document access results
    # logger.info("Document access results:")
    # logger.info("Allowed documents: %s", [doc.title for doc in valid_documents])
    # logger.info("Blacklisted documents: %s", [doc.title for doc in blacklisted_documents])

    # Create cost information dictionary
    cost_info = {
        "total_cost": float(search_cost.total_cost),
        "search_tool_cost": float(search_cost.search_tool_cost),
        "relevancy_assessment_cost": float(search_cost.relevancy_assessment_cost),
        "relevancy_assessment_count": search_cost.relevancy_assessment_count,
    }

    return valid_documents, valid_citations, cost_info


# async def get_web_documents(llm: BedrockHandler, query: str) -> Tuple[List[Any], List[Dict[str, str]]]:
#     """
#     Get documents from URLs mentioned in the user query.

#     Args:
#         llm: LLM for web browsing
#         query: User query

#     Returns:
#         Tuple of documents and citations
#     """
#     # Extract URLs from user prompt
#     document_urls = await extract_urls_from_user_prompt(llm=llm, role=RoleEnum.user, query=query)
#     logger.debug(f"URLs found in the user query: {document_urls}")

#     # Get documents from URLs
#     documents = await WebBrowserService.get_documents(urls=document_urls)

#     # Process documents and filter blacklisted ones
#     valid_documents = []
#     citations = []
#     blacklisted_documents = []

#     for document in documents:
#         if document.status == DocumentBlacklistStatus.BLACKLISTED:
#             blacklisted_documents.append(document)
#             continue
#         valid_documents.append(document)
#         citations.append({"docname": document.title, "docurl": document.url})

#     return valid_documents, citations


async def process_documents(documents: List[Any]) -> str:
    """
    Process documents and wrap them for inclusion in the prompt.

    Args:
        documents: List of documents to process

    Returns:
        Formatted document content
    """
    wrapped_document = ""
    for i, document in enumerate(documents, 1):
        wrapped_document += f"<gov-uk-search-result-{i}>\n"
        wrapped_document += f"<document-title>{document.title}</document-title>\n"
        wrapped_document += f"<document-url>{document.url}</document-url>\n"
        wrapped_document += f"<document-body>\n{document.body}\n</document-body>\n"
        wrapped_document += f"</gov-uk-search-result-{i}>"
        if i < len(documents):
            wrapped_document += "\n"

    return wrapped_document


async def enhance_user_prompt(
    chat: Chat,
    input_data: ChatCreateMessageInput,
    m_user_id: int,
    db_session: AsyncSession,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Enhance a user prompt with relevant documents from multiple sources.

    Args:
        chat: The chat context
        input_data: User input and configuration
        m_user_id: User ID for document access

    Returns:
        DocumentResult containing wrapped document content and citations

    Raises:
        ChatNotFoundError: If chat doesn't exist
        LLMNotFoundError: If required LLM models aren't available
    """
    if not chat:
        raise ChatNotFoundError("Chat not found")

    # Get LLM models
    llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)

    if not llm_obj:
        raise LLMNotFoundError(f"LLM not found with name: {LLM_DEFAULT_MODEL}")

    # Initialize LLM handler
    llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

    # Initialize result containers
    all_documents = []
    all_citations = []
    total_cost = 0.0

    # Run document retrieval tasks
    if input_data.use_gov_uk_search_api:
        documents, citations, cost_info = await get_search_documents(
            llm=llm, query=input_data.query, m_user_id=m_user_id, db_session=db_session
        )
        all_documents.extend(documents)
        all_citations.extend(citations)
        total_cost += cost_info["total_cost"]

    # if input_data.enable_web_browsing:
    #     web_documents, web_citations = await get_web_documents(llm=llm, query=input_data.query)
    #     all_documents.extend(web_documents)
    #     all_citations.extend(web_citations)

    # Process documents for inclusion in prompt
    wrapped_documents = await process_documents(all_documents)

    return wrapped_documents, all_citations


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
        # logger.info(f"Fetching content via httpx from: {url}")
        async with httpx.AsyncClient(timeout=20.0) as client:
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
        # logger.info(f"Successfully fetched content from GOV UK Content API for: {path}")
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

        # Get body content
        details = content_data.get("details", {})

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
        # logger.info(f"Successfully extracted content using httpx for: {url}")
        return content_from_httpx

    # Both methods failed
    logger.error(f"Failed to extract content for {url} using both Content API and httpx")
    return ""


async def is_document_relevant(
    llm: BedrockHandler, role: str, query: str, non_rag_document: str, non_rag_document_title: str
) -> Result:
    messages = [
        {
            "role": role,
            "content": (
                f"<user-query>\n{query}\n</user-query>\n\n"
                f"<retrieved-gov-uk-page>\n"
                f"<retrieved-gov-uk-page-title>{non_rag_document_title}</retrieved-gov-uk-page-title>\n"
                f"<retrieved-gov-uk-page-content>{non_rag_document}</retrieved-gov-uk-page-content>\n"
                f"</retrieved-gov-uk-page>"
            ),
        }
    ]
    logger.debug(f"messages: {messages}")

    llm_res = await llm.invoke_async(
        messages=messages,
        tools=DOCUMENT_RELEVANCE_ASSESSMENT["tools"],
    )

    return llm_res


async def get_relevant_documents_from_gov_uk_search(
    llm: BedrockHandler, role: str, query: str, m_user_id: int, db_session: AsyncSession
) -> Tuple[List[NonRagDocument], List[Dict[str, str]], SearchCost]:
    """
    Main entry point for the GOV UK Search feature.
    Returns a tuple of (relevant_documents, citations, search_cost)
    """
    search_cost = SearchCost()

    # Step 1: Get search terms and parameters from LLM
    search_terms, llm_response_id, search_params, llm_cost = await get_search_queries(llm, role, query, db_session)
    search_cost.search_tool_cost = llm_cost
    search_cost.total_cost += llm_cost

    # logger.info(f"GOV UK Search - LLM suggested search terms: {search_terms}")
    # logger.info(f"GOV UK Search - LLM suggested search parameters: {search_params}")

    # Step 2: Execute searches and get relevant documents
    relevant_documents, irrelevant_documents, relevancy_cost = await execute_searches(
        role=role,
        query=query,
        search_terms=search_terms,
        search_params=search_params,
        m_user_id=m_user_id,
        llm_internal_response_id_query=llm_response_id,
        db_session=db_session,
    )

    # search_cost.relevancy_assessment_cost = relevancy_cost
    # search_cost.relevancy_assessment_count = len(relevant_documents) + len(irrelevant_documents)
    # search_cost.total_cost += relevancy_cost

    # if search_cost.relevancy_assessment_count > 0:
    #     mean_cost = search_cost.relevancy_assessment_cost / search_cost.relevancy_assessment_count
    # else:
    #     mean_cost = Decimal(0)

    # Log the final results
    # logger.info(f"GOV UK Search - Relevant documents: {[doc.title for doc in relevant_documents]}")
    # logger.info(f"GOV UK Search - Irrelevant documents: {[doc['title'] for doc in irrelevant_documents]}")
    # logger.info(f"GOV UK Search - Total cost: {search_cost.total_cost}")
    # logger.info(f"GOV UK Search - Search tool cost: {search_cost.search_tool_cost}")
    # logger.info(
    #     f"GOV UK Search - Relevancy assessment: {search_cost.relevancy_assessment_count} "
    #     f"documents at mean cost {mean_cost}"
    # )

    # Step 3: Generate citations for relevant documents
    citations = []

    if relevant_documents:
        citations = [{"title": doc.title, "url": doc.url} for doc in relevant_documents]

    return relevant_documents, citations, search_cost


async def get_search_queries(
    llm: BedrockHandler, role: str, query: str, db_session: AsyncSession
) -> Tuple[List[str], int, Dict[str, Any], Decimal]:
    """
    Get search terms and parameters from LLM for GOV UK Search API.
    Returns (search_terms, llm_response_id, search_params, cost)
    """
    # Prepare message for the LLM
    message = [
        {
            "role": role,
            "content": (
                "Get documents that best match the query below using the GOV UK Search API. "
                "\n\nYou can provide search terms, date ranges, content purpose filters, and other parameters. "
                "The GOV UK Search API uses keyword-based search. "
                "Therefore use a variety of synonyms in your queries to improve your chances of getting a good result. "
                "\n\nYou can provide an empty query to get results ordered by popularity. "
                "For example, when asked about recent announcements "
                "you can use an empty query to get results ordered by popularity. "
                "Do not create search terms like 'government announcements' or 'recent announcements'. "
                "\n\nOnly use the call_gov_uk_search_api tool. "
                f"\n\nTodays date is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
            )
            + "\n\n"
            + query,
        }
    ]

    # Ask the LLM to return a call_gov_uk_search_api function call object
    llm_response = await llm.invoke_async(message, tools=SEARCH_API_SEARCH_TERMS["tools"])

    # Record the LLM transaction in the database
    llm_internal_response = await DbOperations.insert_llm_internal_response_id_query(
        db_session=db_session,
        web_browsing_llm=llm.llm,
        content=llm_response.content[0].text,
        tokens_in=llm_response.usage.input_tokens,
        tokens_out=llm_response.usage.output_tokens,
        completion_cost=llm_response.usage.input_tokens * llm.llm.input_cost_per_token
        + llm_response.usage.output_tokens * llm.llm.output_cost_per_token,
    )
    llm_response_id = llm_internal_response.id

    # Calculate cost
    cost = Decimal(
        llm_response.usage.input_tokens * llm.llm.input_cost_per_token
        + llm_response.usage.output_tokens * llm.llm.output_cost_per_token
    )

    # Parse search terms and parameters
    search_terms = []
    search_params = {}

    try:
        llm_response_dict = llm_response.dict()
        content_items = llm_response_dict.get("content", [])

        for item in content_items:
            if item.get("type") == "tool_use":
                input_data = item.get("input", {})

                # Get search terms
                terms = input_data.get("search_terms", [])
                search_terms.extend(terms)

                # Extract additional search parameters
                param_mappings = [
                    "date_range",
                    "content_purpose_supergroup",
                    "content_purpose_subgroup",
                    "is_political",
                    "order_by",
                    "descending_order",
                    "count",
                ]

                for param in param_mappings:
                    if param in input_data:
                        search_params[param] = input_data[param]

                # Log the LLM's tool usage details
                # logger.info(f"GOV UK Search Tool Usage - Search terms: {terms}")
                # logger.info(
                #     "GOV UK Search Tool Usage - Search parameters: "
                #     f"{json.dumps({k: v for k, v in input_data.items() if k != 'search_terms'})}"
                # )
    except Exception as e:
        logger.exception(f"Error parsing LLM response for search terms: {e}")

    # Deduplicate search terms
    search_terms = list(set(search_terms))

    return search_terms, llm_response_id, search_params, cost


async def execute_searches(
    role: str,
    query: str,
    search_terms: List[str],
    search_params: Dict[str, Any],
    m_user_id: int,
    llm_internal_response_id_query: int,
    db_session: AsyncSession,
) -> Tuple[List[NonRagDocument], List[Dict[str, str]], Decimal]:
    """
    Execute GOV UK searches and assess document relevancy.
    Returns (relevant_documents, irrelevant_documents, total_relevancy_cost)
    """
    gov_uk_search = GovUKSearch()
    search_results = await perform_searches(
        gov_uk_search=gov_uk_search,
        search_terms=search_terms,
        search_params=search_params,
        llm_internal_response_id_query=llm_internal_response_id_query,
        message_id=m_user_id,
        db_session=db_session,
    )

    # Collect all documents from search results
    all_documents = []
    seen_links = set()  # Track seen links to deduplicate

    for result_set in search_results:
        results = result_set[0].get("results", [])
        query_id = result_set[1].id

        for result in results:
            title = result.get("title", "")
            description = result.get("description", "")
            link = result.get("link", "")

            if not link:
                continue

            # Hydrate links that don't have the full domain
            if link.startswith("/"):
                link = f"https://www.gov.uk{link}"
            elif not (link.startswith("http://") or link.startswith("https://")):
                link = f"https://www.gov.uk/{link}"

            # Skip if we've already seen this link
            if link in seen_links:
                continue

            seen_links.add(link)
            all_documents.append(
                {
                    "title": title,
                    "description": description,
                    "link": link,
                    "query_id": query_id,
                    "position": len(all_documents) + 1,  # Use position in list as position in results
                }
            )

    # logger.info(f"Found {len(all_documents)} unique documents to fetch content for")

    # Fetch full content for all documents in parallel before relevancy assessment
    content_fetch_tasks = []

    if all_documents:
        async with asyncio.TaskGroup() as tg:
            for doc in all_documents:
                content_fetch_tasks.append(
                    {"doc": doc, "task": tg.create_task(extract_content_from_gov_uk(doc["link"]))}
                )

        # Add full content to documents
        for i, task_info in enumerate(content_fetch_tasks):
            doc = task_info["doc"]
            full_content = task_info["task"].result()

            # If full content fetch failed, use the description as fallback
            if not full_content:
                logger.warning(f"Failed to fetch full content for {doc['link']}, using description as fallback")
                full_content = doc["description"]

            # Add full content to the document
            all_documents[i]["full_content"] = full_content

            # logger.info(f"Fetched content for document: {doc['title']} (content length: {len(full_content)})")

    # Now assess relevancy with full content
    relevancy_tasks = []

    async with asyncio.TaskGroup() as tg:
        for doc in all_documents:
            relevancy_tasks.append(
                {
                    "doc": doc,
                    "task": tg.create_task(
                        assess_document_relevancy(
                            role=role,
                            query=query,
                            title=doc["title"],
                            description=doc["description"],
                            full_content=doc["full_content"],
                            db_session=db_session,
                        )
                    ),
                }
            )

    # Process completed tasks
    relevant_documents = []
    irrelevant_documents = []
    total_relevancy_cost = Decimal(0)

    for task_info in relevancy_tasks:
        doc = task_info["doc"]
        is_relevant, relevancy_cost, llm_internal_response_id = task_info["task"].result()

        total_relevancy_cost += relevancy_cost

        # Record result in database
        await record_search_result(
            m_user_id=m_user_id,
            llm_internal_response_id=llm_internal_response_id,
            gov_uk_search_query_id=doc["query_id"],
            url=doc["link"],
            content=doc["full_content"],  # Store full content in database
            is_used=is_relevant,
            position=doc["position"],
            db_session=db_session,
        )

        # Add to relevant or irrelevant documents list
        if is_relevant:
            # Create NonRagDocument with full content
            relevant_documents.append(
                NonRagDocument(
                    title=doc["title"], url=doc["link"], body=doc["full_content"], status=DocumentBlacklistStatus.OK
                )
            )
            # logger.info(f"Document deemed relevant: {doc['title']}")
        else:
            irrelevant_documents.append({"title": doc["title"], "url": doc["link"]})
            logger.debug(f"Document deemed not relevant: {doc['title']}")

    return relevant_documents, irrelevant_documents, total_relevancy_cost


async def perform_searches(
    gov_uk_search: GovUKSearch,
    search_terms: List[str],
    search_params: Dict[str, Any],
    llm_internal_response_id_query: int,
    message_id: int,
    db_session: AsyncSession,
) -> List[Tuple[Dict[str, Any], Any]]:
    """
    Perform the actual GOV UK searches based on search terms and parameters.
    Returns a list of (search_result, query_record) tuples.
    """
    search_tasks = []
    search_queries = []  # Store the query terms in order
    results = []

    # Prepare filter parameters
    filter_by_field = []
    if search_params:
        # Handle date range filter
        if "date_range" in search_params:
            date_range = search_params["date_range"]
            if "from" in date_range and "to" in date_range:
                filter_by_field.append(("public_timestamp", f"from:{date_range['from']},to:{date_range['to']}"))

        # Handle content purpose filters
        for filter_type in ["content_purpose_supergroup", "content_purpose_subgroup"]:
            if filter_type in search_params:
                for value in search_params[filter_type]:
                    filter_by_field.append((filter_type, value))

        # Handle is_political filter
        if "is_political" in search_params:
            filter_by_field.append(("is_political", search_params["is_political"]))

    # Set order parameters
    order_by_field_name = ""
    descending_order = False
    if search_params and "order_by" in search_params:
        order_by_field_name = search_params["order_by"]
        descending_order = search_params.get("descending_order", False)
    elif not search_terms or (len(search_terms) == 1 and not search_terms[0]):
        order_by_field_name = "popularity"
        descending_order = True

    # Get count parameter or use default
    count = search_params.get("count", 10) if search_params else 10

    # Standard fields to retrieve
    fields = ["title", "description", "link", "primary_publishing_organisation", "public_timestamp"]

    # If we have no valid search terms, use empty query which defaults to popularity ordering
    valid_search_terms = [term for term in search_terms if term]
    if not valid_search_terms:
        # Build and log the search URL for empty query
        # url = gov_uk_search.build_search_url(
        #     query="", count=count, descending_order=True, fields=fields, filter_by_field=filter_by_field
        # )
        # logger.info(f"GOV UK Search API Call - Empty query URL: {url}")

        search_queries.append("empty query")
        search_task = gov_uk_search.simple_search(
            query="",
            db_session=db_session,
            count=count,
            llm_internal_response_id_query=llm_internal_response_id_query,
            message_id=message_id,
            fields=fields,
            filter_by_field=filter_by_field,
            descending_order=True,
        )
        search_tasks.append(search_task)
    else:
        # Create search tasks for each valid term
        async with asyncio.TaskGroup() as tg:
            for term in valid_search_terms:
                # # Build and log the search URL for each term
                # url = gov_uk_search.build_search_url(
                #     query=term,
                #     count=count,
                #     order_by_field_name=order_by_field_name,
                #     descending_order=descending_order,
                #     fields=fields,
                #     filter_by_field=filter_by_field,
                # )
                # # logger.info(f"GOV UK Search API Call - Query term: '{term}', URL: {url}")

                search_queries.append(term)
                task = tg.create_task(
                    gov_uk_search.simple_search(
                        query=term,
                        db_session=db_session,
                        count=count,
                        llm_internal_response_id_query=llm_internal_response_id_query,
                        message_id=message_id,
                        fields=fields,
                        filter_by_field=filter_by_field,
                        order_by_field_name=order_by_field_name,
                        descending_order=descending_order,
                    )
                )
                search_tasks.append(task)

    # Execute search tasks
    for _, task in enumerate(search_tasks):
        if isinstance(task, asyncio.Task):
            result = task.result()
        else:
            result = await task
        results.append(result)

        # Log the document titles returned by each query
        # document_titles = [doc.get("title", "No title") for doc in result[0].get("results", [])]
        # query_term = search_queries[i]
        # logger.info(
        #     f"GOV UK Search Results - Query: '{query_term}', Found {len(document_titles)} "
        #     f"documents: {document_titles}, Search params: {fields=}, {filter_by_field=}, "
        #     f"{order_by_field_name=}, {descending_order=}"
        # )

    return results


async def assess_document_relevancy(
    role: str, query: str, title: str, description: str, full_content: str, db_session: AsyncSession
) -> Tuple[bool, Decimal, int]:
    """
    Assess if a document is relevant to the query using its full content.

    Args:
        role: Role for the LLM message
        query: User query
        title: Document title
        description: Document description
        full_content: Full document content

    Returns:
        Tuple of (is_relevant, cost, llm_response_id)
    """
    # Create a truncated version of the full content to avoid token limits
    # First use the description as a summary, then add as much of the full content as reasonable
    content_for_assessment = description

    # If full content is different from description, add a sample of it
    if full_content != description and len(full_content) > 0:
        # Limit content to ~6000 characters to avoid excessive token usage
        max_content_chars = 6000
        truncated_content = full_content[:max_content_chars]
        if len(full_content) > max_content_chars:
            truncated_content += "... [content truncated]"

        content_for_assessment = f"{description}\n\nContent excerpt:\n{truncated_content}"

    # Get LLM for document relevancy assessment
    llm_obj = LLMTable().get_by_model(LLM_DOCUMENT_RELEVANCY_MODEL)
    llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)
    messages = [
        {
            "role": role,
            "content": (
                f"<user-query>\n\n{query}\n\n</user-query>\n\n"
                f"<retrieved-gov-uk-page>\n\n"
                f"<retrieved-gov-uk-page-title>{title}</retrieved-gov-uk-page-title>\n\n"
                f"<retrieved-gov-uk-page-content>{content_for_assessment}</retrieved-gov-uk-page-content>\n\n"
                f"</retrieved-gov-uk-page>"
            ),
        }
    ]

    try:
        document_relevancy = await llm.invoke_async(
            messages=messages,
            tools=DOCUMENT_RELEVANCE_ASSESSMENT["tools"],
        )

        # Record the LLM transaction

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=llm.llm,
            content=document_relevancy.content[0].text,
            tokens_in=document_relevancy.usage.input_tokens,
            tokens_out=document_relevancy.usage.output_tokens,
            completion_cost=document_relevancy.usage.input_tokens * llm.llm.input_cost_per_token
            + document_relevancy.usage.output_tokens * llm.llm.output_cost_per_token,
        )
        llm_response_id = response.id

        # Calculate cost
        cost = Decimal(
            document_relevancy.usage.input_tokens * llm.llm.input_cost_per_token
            + document_relevancy.usage.output_tokens * llm.llm.output_cost_per_token
        )

        # Determine if document is relevant
        document_relevancy_dict = document_relevancy.dict()
        content = document_relevancy_dict.get("content", [])
        is_relevant_values = [
            item.get("input", {}).get("is_relevant", False) for item in content if item.get("type") == "tool_use"
        ]

        is_relevant = False
        if is_relevant_values:
            value = is_relevant_values[0]
            is_relevant = value == "True" or value is True

        return is_relevant, cost, llm_response_id

    except Exception as e:
        logger.exception(f"Error assessing document relevancy: {e}")
        return False, Decimal(0), 0


async def record_search_result(
    m_user_id: int,
    llm_internal_response_id: int,
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


async def extract_urls_from_user_prompt(llm: BedrockHandler, role: str, query: str) -> List[str]:
    """
    Extract URLs from the user prompt using the LLM.
    """
    message = [
        {
            "role": role,
            "content": ("Get documents whose URLs are listed in the query. Only use the download_urls tool")
            + "\n\n"
            + query,
        }
    ]

    # Ask the LLM to return a download_urls function call object
    try:
        llm_response = await llm.invoke_async(message, tools=DOWNLOAD_URLS["tools"])

        # Extract URLs from response
        urls = []
        llm_response_dict = llm_response.dict()
        content_items = llm_response_dict.get("content", [])

        for item in content_items:
            if item.get("type") == "tool_use":
                urls.extend(item.get("input", {}).get("urls", []))

        # Process and deduplicate URLs
        processed_urls = set()
        for url in urls:
            # Hydrate URLs that don't have the full domain
            if url.startswith("/"):
                url = f"https://www.gov.uk{url}"
            elif not (url.startswith("http://") or url.startswith("https://")):
                # Check if it might be a gov.uk URL (without protocol)
                if "gov.uk" in url:
                    url = f"https://{url}"
                else:
                    url = f"https://www.gov.uk/{url}"

            processed_urls.add(url)

        return list(processed_urls)

    except Exception as e:
        logger.exception(f"Error extracting URLs from prompt: {e}")
        return []


async def assess_if_next_message_should_use_gov_uk_search(
    messages: list[Message], new_user_message_content: str, new_user_message_id: int, db_session: AsyncSession
) -> bool:
    """ """
    llm_obj = LLMTable().get_by_model(LLM_GOV_UK_SEARCH_FOLLOWUP_ASSESMENT)
    llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

    # Get citations for all messages
    message_ids = [message.id for message in messages]
    stmt = (
        select(GovUkSearchResult.message_id, GovUkSearchResult.url, GovUkSearchResult.content)
        .where(GovUkSearchResult.message_id.in_(message_ids))
        .where(GovUkSearchResult.is_used)
        .order_by(GovUkSearchResult.message_id, GovUkSearchResult.position)
    )
    result = await db_session.execute(stmt)
    citations_rows = result.fetchall()

    # Group citations by message_id
    citations_by_message = {}
    for row in citations_rows:
        message_id = row.message_id
        if message_id not in citations_by_message:
            citations_by_message[message_id] = []
        citations_by_message[message_id].append({"url": row.url, "content": row.content})

    messages_formatted = []
    for message in messages:
        citations = citations_by_message.get(message.id, [])
        citations_text = ""
        if citations:
            citations_text = f"\n<gov-uk-search-citations>{citations}</gov-uk-search-citations>"
        messages_formatted.append({"role": message.role, "content": f"{message.content}{citations_text}"})
    messages_formatted.append({"role": "user", "content": new_user_message_content})

    try:
        assessment = await llm.invoke_async(
            system=(
                "You have been included mid-conversation. "
                f"Your task is to use the tool '{NAME_USE_GOV_UK_SEARCH_ASSESSMENT}' "
                "to determine if searching GOV.UK for additional information is recommended. "
                "\nSearch results may have previously been retrieved in the conversation. "
                "Pay attention to whether the latest user query requires an additional search "
                "of GOV.UK, or if the existing search results are sufficient. "
                "Assume the answer to be 'false' unless the user explicitly "
                "asks for more search results in the LAST message."
                "\nPay most attention to the LAST message from the user and what the user wants. "
                "\n<example-1>If the user's latest query relates to the content that has already been "
                "retrieved, the answer should be 'false'.</example-1>"
                "\n<example-2>If the user is asking about content that is unlikely to be available on GOV.UK, "
                "the answer should be 'false'.</example-2>"
                "\n<example-3>If the user is asking for additional content from GOV.UK and it was not "
                "previously retrieved, the answer should be 'true'.</example-3>"
            ),
            messages=messages_formatted,
            tools=[USE_GOV_UK_SEARCH_ASSESSMENT],
            tool_choice={"type": "tool", "name": NAME_USE_GOV_UK_SEARCH_ASSESSMENT},
        )

        # logger.info(f"LLM Assessment: {assessment}")
        tool_blocks = [block for block in assessment.content if block.type == "tool_use"]
        # logger.info(f"Tool blocks: {tool_blocks}")
        result = tool_blocks[0].input[PROPERTY_NAME_USE_GOV_UK_SEARCH_ASSESSMENT]
        # logger.info(f"Result: {result}")
        # logger.info(f"Result type: {type(result)}")

        # Record the LLM transaction
        llm_internal_response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=llm.llm,
            content=str(result),
            tokens_in=assessment.usage.input_tokens,
            tokens_out=assessment.usage.output_tokens,
            completion_cost=assessment.usage.input_tokens * llm.llm.input_cost_per_token
            + assessment.usage.output_tokens * llm.llm.output_cost_per_token,
        )

        # Record the decision
        stmt = insert(UseGovUkSearchDecision).values(
            llm_internal_response_id=llm_internal_response.id,
            message_id=new_user_message_id,
            decision=result,
        )
        await db_session.execute(stmt)

        if result is True:
            return True
        if result is False:
            return False
        raise ValueError(
            "Could not parse LLM response. "
            f"Expected boolean True or False, got '{result}' (type: {type(result).__name__})"
        )

    except Exception as e:
        logger.exception(f"Error determining whether to use GOV.UK Search: {e}")
        return False
