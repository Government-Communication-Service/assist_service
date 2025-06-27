import logging

import pytest

from app.web_browser.web_browser import WebBrowserService

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_strip_url():
    url = "https://www.gov.uk/some_page?q=secret"
    assert WebBrowserService.strip_url(url=url) == "https://www.gov.uk/some_page"


@pytest.mark.asyncio
async def test_is_blacklisted():
    url_ok = "https://www.gov.uk/some_page"
    url_blacklisted = "https://www.example.com/some_page"
    assert await WebBrowserService.is_blacklisted(url=url_ok) is False
    assert await WebBrowserService.is_blacklisted(url=url_blacklisted) is True


# @patch("app.web_browser.web_browser.aiohttp.ClientSession")
# @pytest.mark.asyncio
# async def test_get_html_document(mock: MagicMock):
#     url = "https://www.gov.uk/some_page?q=secret"
#     response_text = "<title>Text Document</title><body>Test Body</body>"
#     response_json = json.dumps(response_text)

#     session = MagicMock()
#     session.get.return_value.__aenter__.return_value.status = 200
#     session.get.return_value.__aenter__.return_value.text.return_value = response_json

#     mock.return_value.__aenter__.return_value = session

#     response = await WebBrowserService.get_html_document(session=session, url=url)

#     assert response.url == WebBrowserService.strip_url(url)
#     assert response.title == "Text Document"
#     assert response.body == "Test Body"
#     assert response.status == DocumentBlacklistStatus.OK


# @patch("app.web_browser.web_browser.aiohttp.ClientSession")
# @pytest.mark.asyncio
# async def test_get_documents(mock: MagicMock):
#     url = "https://www.gov.uk/some_page?q=secret"
#     response_text = "<title>Text Document</title><body>Test Body</body>"
#     response_json = json.dumps(response_text)
#     document = NonRagDocument(
#         url=WebBrowserService.strip_url(url),
#         title="Text Document",
#         body="Test Body",
#         status=DocumentBlacklistStatus.OK,
#     )
#     session = MagicMock()
#     session.get.return_value.__aenter__.return_value.status = 200
#     session.get.return_value.__aenter__.return_value.text.return_value = response_json

#     mock.return_value.__aenter__.return_value = session

#     response = await WebBrowserService.get_documents(urls=[url])

#     assert response[0].url == WebBrowserService.strip_url(url)
#     assert response[0].title == document.title
#     assert response[0].body == document.body
#     assert response[0].status == document.status


# @pytest.mark.asyncio
# async def test_get_documents_blacklisted():
#     url = "https://www.gov.uk/publications/some_page?q=secret"

#     response = await WebBrowserService.get_documents(urls=[url])

#     assert response[0].url == url
#     assert response[0].title == ""
#     assert response[0].body == ""
#     assert response[0].status == DocumentBlacklistStatus.BLACKLISTED


# @pytest.mark.asyncio
# async def test_get_documents_whitelisted():
#     url = "https://www.gov.uk/browse/benefits"

#     response = await WebBrowserService.get_documents(urls=[url])

#     assert response[0].url == url
#     assert "Benefits" in response[0].title
#     assert len(response[0].body) > 0
#     assert response[0].status == DocumentBlacklistStatus.OK
