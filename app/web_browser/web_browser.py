import asyncio
from typing import List, Optional
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup

from app.config import BLACKLISTED_URLS, GOV_UK_BASE_URL, WEB_BROWSING_TIMEOUT, WHITELISTED_URLS
from app.gov_uk_search.schemas import DocumentBlacklistStatus, NonRagDocument
from app.gov_uk_search.service import GovUKContent

gov_uk_base_url_parsed = urlparse(GOV_UK_BASE_URL)

timeout = aiohttp.ClientTimeout(total=WEB_BROWSING_TIMEOUT)


class WebBrowserService:
    @staticmethod
    def strip_url(url: str) -> str:
        o = urlparse(url)
        if o.netloc == gov_uk_base_url_parsed.netloc or o.netloc == "":
            stripped_url = f"{GOV_UK_BASE_URL}{o.path}"
        else:
            # external links will be fetched using a simple https://host/path
            # pattern while #anchor or ?q=queries are stripped to prevent
            # likelihood of PII leaking (it is not 100% foolproof)
            stripped_url = f"{o.scheme}://{o.netloc}{o.path}"

        return stripped_url

    @staticmethod
    async def is_blacklisted(url: str) -> bool:
        # implement blacklisting here
        o = urlparse(url)
        # only allow URLs with www.gov.uk domain
        if o.netloc == "":
            return False
        if o.netloc != gov_uk_base_url_parsed.netloc:
            return True
        found_match = [matched_url for matched_url in BLACKLISTED_URLS if url.startswith(matched_url)]
        if len(found_match) > 0:
            return True
        return False

    @staticmethod
    async def is_whitelisted(url: str) -> bool:
        found_match = [matched_url for matched_url in WHITELISTED_URLS if url.startswith(matched_url)]
        if len(found_match) > 0:
            return True
        return False

    @staticmethod
    async def get_document_from_content_api(url: str) -> Optional[NonRagDocument]:
        """
        Get document content using the GOV UK Content API.

        Args:
            url: The URL to fetch content for

        Returns:
            NonRagDocument if successful, None if not
        """
        # Extract path from URL
        o = urlparse(url)
        path = o.path.lstrip("/")

        # Get content from API
        content = await GovUKContent.get_content(path)
        if not content:
            return None

        # Extract title and body
        title = content.get("title", "")
        details = content.get("details", {})
        body = details.get("body", "")

        # If we got HTML content, parse it to extract text
        if isinstance(body, str) and body.strip().startswith("<"):
            soup = BeautifulSoup(body, "html.parser")
            body = soup.get_text(separator="\n", strip=True)

        return NonRagDocument(url=url, title=title, body=body, status=DocumentBlacklistStatus.OK)

    @staticmethod
    async def get_documents(urls: List[str], use_content_api: bool = True) -> List[NonRagDocument]:
        """
        Get documents from a list of URLs.

        Args:
            urls: List of URLs to fetch
            use_content_api: Whether to try the Content API first for GOV.UK URLs

        Returns:
            List of NonRagDocument objects
        """
        documents = []
        tasks = []

        async with asyncio.TaskGroup() as tg:
            for url in urls:
                if await WebBrowserService.is_blacklisted(url):
                    documents.append(
                        NonRagDocument(
                            url=url,
                            title="",
                            body="",
                            status=DocumentBlacklistStatus.BLACKLISTED,
                        )
                    )
                    continue

                if use_content_api and gov_uk_base_url_parsed.netloc in url:
                    # Try Content API first for GOV.UK URLs
                    doc = await WebBrowserService.get_document_from_content_api(url)
                    if doc is not None:
                        documents.append(doc)
                        continue

                # Fall back to web scraping if Content API fails or for non-GOV.UK URLs
                tasks.append(tg.create_task(WebBrowserService._get_document_from_web(url)))

        # Add results from web scraping
        for task in tasks:
            result = task.result()
            if result:
                documents.append(result)

        return documents

    @staticmethod
    async def _get_document_from_web(url: str) -> Optional[NonRagDocument]:
        """
        Get document content by scraping the web page.

        Args:
            url: The URL to fetch

        Returns:
            NonRagDocument if successful, None if not
        """
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return None

                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    # Get title
                    title = soup.title.string if soup.title else ""

                    # Get main content
                    main = soup.find("main")
                    if main:
                        body = main.get_text(separator="\n", strip=True)
                    else:
                        body = soup.get_text(separator="\n", strip=True)

                    return NonRagDocument(url=url, title=title, body=body, status=DocumentBlacklistStatus.OK)
        except Exception:
            return None
