from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, RedirectResponse

router = APIRouter()


@router.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    """
    Returns robots.txt content to prevent web crawlers from indexing the API.

    Returns:
        str: robots.txt content disallowing all crawlers
    """
    return "User-agent: *\nDisallow: /"


@router.get("/", include_in_schema=False)
def root():
    """
    Redirects root URL to API documentation.

    Returns:
        RedirectResponse: Redirect to /docs endpoint
    """
    return RedirectResponse(url="/docs")
