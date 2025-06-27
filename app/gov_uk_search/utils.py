import re
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from bs4 import BeautifulSoup

from app.gov_uk_search.constants import SEARCH_URL
from app.logs.logs_handler import logger

# GovUKContent will be imported from client.py within extract_content_from_gov_uk
# from app.gov_uk_search.client import GovUKContent


def build_search_url(
    query: str,
    count: int = 10,
    order_by_field_name: str = "",
    descending_order: bool = False,
    start: int = 0,
    fields: list[str] | None = None,
    filter_by_field: list[tuple[str, Any]] | None = None,
) -> str:
    """
    Build a simplified GOV UK Search API URL with only the parameters that are actually used.

    Args:
        query: Search query string
        count: Number of results to return (1-50)
        order_by_field_name: Field to order by (e.g., 'popularity', 'public_timestamp')
        descending_order: Whether to sort in descending order
        start: Starting index for pagination
        fields: List of fields to return in results
        filter_by_field: List of (field_name, value) tuples for filtering

    Returns:
        Complete URL for the GOV UK Search API
    """
    # Base URL with encoded query
    url = f"{SEARCH_URL}?q={quote(query)}"

    # Add count parameter
    if count and count > 0:
        url += f"&count={min(count, 50)}"  # Cap at 50 to avoid API limits

    # Add ordering (skip if requesting relevance since it's the default)
    if order_by_field_name and order_by_field_name.lower() != "relevance":
        if descending_order:
            url += f"&order=-{order_by_field_name}"
        else:
            url += f"&order={order_by_field_name}"

    # Add start parameter for pagination
    if start and start > 0:
        url += f"&start={start}"

    # Add fields to return
    if fields:
        for field in fields:
            if field:  # Skip None/empty fields
                url += f"&fields={quote(field)}"

    # Add filters
    if filter_by_field:
        for field_name, value in filter_by_field:
            if field_name and value is not None:
                # Handle different value types appropriately
                if isinstance(value, bool):
                    url += f"&filter_{field_name}={str(value).lower()}"
                else:
                    url += f"&filter_{field_name}={quote(str(value))}"

    return url


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
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        result = []

        title_elem = soup.find("h1")
        if title_elem:
            result.append(f"# {title_elem.get_text().strip()}")
            result.append("")

        metadata_elems = soup.select(".app-c-publisher-metadata, .gem-c-metadata")
        if metadata_elems:
            for elem in metadata_elems:
                meta_text = elem.get_text().strip()
                if meta_text:
                    result.append(meta_text)
            result.append("")

        content_elem = soup.select_one("#content, .govuk-grid-column-two-thirds")
        if content_elem:
            for elem in content_elem.find_all(["p", "h2", "h3", "h4", "li"]):
                text = elem.get_text().strip()
                if text:
                    if elem.name.startswith("h"):
                        level = int(elem.name[1])
                        result.append(f"{'#' * level} {text}")
                    else:
                        result.append(text)
                    result.append("")

        return "\n".join(result)

    except (httpx.HTTPError, Exception) as e:
        logger.warning(f"Error fetching content with httpx for {url}: {str(e)}")
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
    from app.gov_uk_search.service import GovUKContent  # Delayed import

    parsed_url = urlparse(url)
    path = parsed_url.path.lstrip("/")

    content_data = await GovUKContent.get_content(path)

    if content_data:
        logger.info(f"Successfully fetched content from GOV UK Content API for: {path}")
        result = []
        title = content_data.get("title", "")
        if title:
            result.append(f"# {title}")
            result.append("")

        document_type = content_data.get("document_type", "")
        if document_type:
            result.append(f"Document type: {document_type}")
            result.append("")

        description = content_data.get("description", "")
        if description:
            result.append(description)
            result.append("")

        details = content_data.get("details", {})
        if "body" in details:
            body = details["body"]
            body = re.sub(r"<[^>]+>", " ", body)
            body = re.sub(r"\s+", " ", body).strip()
            result.append(body)
        elif "parts" in details:
            for part in details["parts"]:
                part_title = part.get("title", "")
                part_body = part.get("body", "")
                if part_title:
                    result.append(f"## {part_title}")
                if part_body:
                    part_body = re.sub(r"<[^>]+>", " ", part_body)
                    part_body = re.sub(r"\s+", " ", part_body).strip()
                    result.append(part_body)
                result.append("")

        if "public_updated_at" in content_data:
            result.append(f"Last updated: {content_data['public_updated_at']}")

        return "\n".join(result)

    logger.warning(f"Content API failed for {url}, falling back to direct web scraping.")
    content_from_httpx = await extract_content_with_httpx(url)
    if content_from_httpx:
        logger.info(f"Successfully extracted content using httpx for: {url}")
        return content_from_httpx

    logger.error(f"Failed to extract content for {url} using both Content API and httpx.")
    return ""
