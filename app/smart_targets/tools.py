from datetime import date

from pydantic import Field

from app.bedrock.tool_schema import LenientList, ToolArgs, ToolDef
from app.config import CAMPAIGN_BENCHMARKS_MAX_TOOL_METRICS
from app.smart_targets.constants import CAMPAIGN_SPEND_FILTER, CAMPAIGN_START_DATE_FILTER
from app.smart_targets.schemas import SelectedFilter, SelectedFilters


class DateRange(ToolArgs):
    """A date range; leave either end out for an open range."""

    from_date: date | None = Field(None, description="Earliest date, YYYY-MM-DD")
    to_date: date | None = Field(None, description="Latest date, YYYY-MM-DD")


class SpendRange(ToolArgs):
    """A spend range in pounds; leave either end out for an open range."""

    min_spend: float | None = Field(None, description="Minimum spend in pounds")
    max_spend: float | None = Field(None, description="Maximum spend in pounds")


class SmartTargetsMetricRequest(ToolArgs):
    """One metric to look up, with the filters that pick the campaigns to compare against."""

    metric_name: str = Field(description="A campaign benchmarks metric name, exactly as listed in the system prompt")
    option_filters: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Option filters, keyed by filter name, each with one or more of its values, all exactly as listed in "
            'the system prompt, e.g. {"gov_audience": ["youth-16-24"], "region": ["london-only"]}. Campaigns '
            "matching any of a filter's values are included. Leave out filters the conversation doesn't call for."
        ),
    )
    campaign_start_date: DateRange | None = Field(None, description="Only campaigns that started in this range")
    campaign_spend: SpendRange | None = Field(None, description="Only campaigns whose spend is in this range")

    def to_selected_filters(self) -> SelectedFilters:
        """The requested filters in the form the service queries and prunes."""
        filters = [SelectedFilter(name=name, values=values) for name, values in self.option_filters.items() if values]
        if self.campaign_start_date and (self.campaign_start_date.from_date or self.campaign_start_date.to_date):
            dates = {}
            if self.campaign_start_date.from_date:
                dates["min_date"] = self.campaign_start_date.from_date.isoformat()
            if self.campaign_start_date.to_date:
                dates["max_date"] = self.campaign_start_date.to_date.isoformat()
            filters.append(SelectedFilter(name=CAMPAIGN_START_DATE_FILTER, values=dates))
        if self.campaign_spend and (
            self.campaign_spend.min_spend is not None or self.campaign_spend.max_spend is not None
        ):
            spend = {}
            if self.campaign_spend.min_spend is not None:
                spend["min_value"] = self.campaign_spend.min_spend
            if self.campaign_spend.max_spend is not None:
                spend["max_value"] = self.campaign_spend.max_spend
            filters.append(SelectedFilter(name=CAMPAIGN_SPEND_FILTER, values=spend))
        return SelectedFilters(filters=filters)


class QuerySmartTargetsArgs(ToolArgs):
    """Get benchmark statistics about past government communications campaigns from the campaign benchmarks
    tool (the Benchmark Database). For each requested metric (e.g. impressions, click-through rate,
    awareness) it applies your filters and returns aggregate statistics for the matching campaigns: median,
    25th and 75th percentiles, 2.5th and 97.5th percentiles (the range of the middle 95% of data points;
    minimum and maximum if there are fewer than 40), and the counts of campaigns and measurements. Not every
    metric has every filter or filter value; the result says which of yours were ignored. Filters are
    loosened automatically if too few campaigns match, and the result lists the filters actually applied.
    The statistics describe the whole filtered selection, so to compare (e.g. across years or audiences)
    request the same metric several times with different filters. If the count of measurements differs from
    the count of campaigns, the metric was measured across several channels. Use it to set realistic
    expectations for a campaign's performance."""

    metrics: LenientList[SmartTargetsMetricRequest] = Field(
        min_length=1,
        max_length=CAMPAIGN_BENCHMARKS_MAX_TOOL_METRICS,
        description=f"1 to {CAMPAIGN_BENCHMARKS_MAX_TOOL_METRICS} metrics to look up",
    )


QUERY_CAMPAIGN_BENCHMARKS = ToolDef("query_campaign_benchmarks", QuerySmartTargetsArgs, strict=False)
