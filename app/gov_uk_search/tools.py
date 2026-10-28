from datetime import date
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from app.bedrock.tool_schema import REMOVE_DUPLICATES_DESCRIPTION, LenientList, ToolArgs, ToolDef
from app.config import GOV_UK_SEARCH_DEFAULT_COUNT, GOV_UK_SEARCH_MAX_COUNT, GOV_UK_SEARCH_MAX_TOOL_SEARCH_TERMS
from app.gov_uk_search.utils import GOV_UK_ASSETS_HOST, normalise_gov_uk_asset_url, normalise_gov_uk_url

ContentPurposeSupergroup = Literal[
    "other",
    "news_and_communications",
    "services",
    "guidance_and_regulation",
    "policy_and_engagement",
    "research_and_statistics",
    "transparency",
]

ContentPurposeSubgroup = Literal[
    "updates_and_alerts",
    "news",
    "decisions",
    "speeches_and_statements",
    "transactions",
    "regulation",
    "guidance",
    "business_support",
    "policy",
    "consultations",
    "research",
    "statistics",
    "transparency_data",
    "freedom_of_information_releases",
    "incidents",
    "calls_for_evidence",
]


class DateRange(ToolArgs):
    """Public timestamp range, inclusive of both dates."""

    model_config = ConfigDict(populate_by_name=True)

    from_: date = Field(alias="from", description="First date included, YYYY-MM-DD")
    to: date | None = Field(default=None, description="Last date included, YYYY-MM-DD. Leave empty for no end date.")

    @model_validator(mode="after")
    def _check_order(self) -> "DateRange":
        if self.to is not None and self.to < self.from_:
            raise ValueError("'to' must not be before 'from'")
        return self


class SearchGovUkArgs(ToolArgs):
    """Search GOV.UK, the UK government's website, using the GOV.UK Search API. Returns the full text of
    the most relevant pages found, with their titles and URLs. Use it for government policy,
    announcements, news, guidance, services, statistics and anything else published on GOV.UK,
    especially recent information.
    The search is keyword-based: use short queries, and synonyms across queries rather than repeating
    words. Results are ranked by relevance, so older pages can outrank new ones: for the latest or most recent
    version of something, set date_range to recent pages. Content
    purpose filters narrow results a lot; use at most one of the two, and only when clearly needed."""

    queries: LenientList[str] = Field(
        min_length=1,
        description=(
            f"1 to {GOV_UK_SEARCH_MAX_TOOL_SEARCH_TERMS} short, meaningfully different keyword queries, "
            "most obvious first."
        ),
    )
    count: int | None = Field(
        default=None,
        description=(
            f"Results to fetch per query, 1 to {GOV_UK_SEARCH_MAX_COUNT}. Default {GOV_UK_SEARCH_DEFAULT_COUNT}."
        ),
    )
    date_range: DateRange | None = Field(default=None, description="Only pages published in this date range.")
    content_purpose_supergroup: LenientList[ContentPurposeSupergroup] | None = Field(
        default=None, description="Filter by top-level content purpose."
    )
    content_purpose_subgroup: LenientList[ContentPurposeSubgroup] | None = Field(
        default=None, description="Filter by content purpose subgroup."
    )
    is_political: bool | None = Field(default=None, description="Filter by whether the content is political.")
    remove_duplicates: bool = Field(default=False, description=REMOVE_DUPLICATES_DESCRIPTION)

    @field_validator("queries")
    @classmethod
    def _require_non_blank_terms(cls, terms: list[str]) -> list[str]:
        """Strip terms and drop blanks; at least one must remain."""
        stripped = [term.strip() for term in terms if term.strip()]
        if not stripped:
            raise ValueError("at least one non-blank query is required")
        return stripped


SEARCH_GOV_UK = ToolDef("search_gov_uk", SearchGovUkArgs, strict=False)


class PageEvaluation(ToolArgs):
    """One page's evaluation."""

    page_index: int = Field(description="The index of the page being evaluated, matching its <page-N> tag.")
    reasoning: str = Field(description="Brief explanation of why the page is or isn't highly relevant to the query.")
    is_relevant: bool = Field(description="Whether the page is highly relevant to the query")


class PageRelevanceArgs(ToolArgs):
    """Evaluate whether each of a list of GOV.UK pages is highly relevant to the user's query"""

    evaluations: LenientList[PageEvaluation] = Field(description="One evaluation per page provided, in any order.")


PAGE_RELEVANCE = ToolDef("evaluate_page_relevance", PageRelevanceArgs)


class GetGovUkPageArgs(ToolArgs):
    """Get the full text of one GOV.UK page. Use it to read a page in full, for example a page found by
    search_gov_uk whose content was cut short, an HTML attachment listed on a page, or a GOV.UK link the
    user has given you. It can't read PDFs: use get_gov_uk_pdf for those."""

    url: str = Field(description="The page's GOV.UK URL (https://www.gov.uk/...) or path (/...).")

    @field_validator("url")
    @classmethod
    def _require_gov_uk_page(cls, url: str) -> str:
        """Normalise to a https://www.gov.uk URL; reject anything that isn't a GOV.UK page."""
        return normalise_gov_uk_url(url)


GET_GOV_UK_PAGE = ToolDef("get_gov_uk_page", GetGovUkPageArgs, strict=False)


class GetGovUkPdfArgs(ToolArgs):
    """Read the text of a PDF attached to a GOV.UK page. Use it for PDF attachments listed on a page,
    when the page itself doesn't contain the content you need. Long PDFs are returned a section at a time:
    the result says which pages it covers and how to continue."""

    url: str = Field(description=f"The PDF's address, on {GOV_UK_ASSETS_HOST}, as listed on the GOV.UK page.")
    start_page: int = Field(default=1, ge=1, description="The page to start from. Default 1.")

    @field_validator("url")
    @classmethod
    def _require_gov_uk_asset(cls, url: str) -> str:
        """Normalise to a https URL on the GOV.UK assets host; reject anything else."""
        return normalise_gov_uk_asset_url(url)


GET_GOV_UK_PDF = ToolDef("get_gov_uk_pdf", GetGovUkPdfArgs, strict=False)
