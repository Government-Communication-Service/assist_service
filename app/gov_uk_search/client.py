from typing import Any, Dict, List, Tuple

import aiohttp

from app.gov_uk_search.constants import CONTENT_URL
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
