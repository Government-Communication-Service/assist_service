# ruff: noqa: E501

from typing import Any

from pydantic import Field

from app.bedrock.tool_schema import LenientList, ToolArgs, ToolDef
from app.config import SMART_TARGETS_MAX_TOOL_METRICS

SELECT_FILTERS_TOOL_NAME = "select_filters_for_smart_targets_tool"


def generate_select_filters_tool_schema(filters_data: dict[str, Any]) -> dict[str, Any]:
    """Generate the SELECT_FILTERS_TOOL schema dynamically based on filter data"""
    properties = {
        "reasoning": {
            "description": "A scratchpad to reason through your response before selecting filters.",
            "type": "string",
        }
    }

    for filter_info in filters_data.get("filters", []):
        filter_name = filter_info["name"]
        filter_type = filter_info["type"]

        if filter_type == "categorical":
            # For categorical filters, create array of enum values
            options = [option["name"] for option in filter_info["options"]]
            properties[filter_name] = {
                "type": "array",
                "items": {"type": "string", "enum": options},
                "description": f"Select one or more {filter_name} options",
            }

        elif filter_type == "date":
            # For date filters, create min/max date range
            properties[filter_name] = {
                "type": "object",
                "properties": {
                    "min_date": {
                        "type": "string",
                        "format": "date",
                        "description": f"Minimum {filter_name} (YYYY-MM-DD), must be >= {filter_info['min']}",
                    },
                    "max_date": {
                        "type": "string",
                        "format": "date",
                        "description": f"Maximum {filter_name} (YYYY-MM-DD), must be <= {filter_info['max']}",
                    },
                },
                "description": f"Filter by {filter_name} range",
            }

        elif filter_type == "continuous":
            # For continuous filters, create min/max value range
            properties[filter_name] = {
                "type": "object",
                "properties": {
                    "min_value": {
                        "type": "number",
                        "minimum": filter_info["min"],
                        "description": f"Minimum {filter_name}, must be >= {filter_info['min']}",
                    },
                    "max_value": {
                        "type": "number",
                        "maximum": filter_info["max"],
                        "description": f"Maximum {filter_name}, must be <= {filter_info['max']}",
                    },
                },
                "description": f"Filter by {filter_name} range",
            }

    return {
        "name": SELECT_FILTERS_TOOL_NAME,
        "description": "A tool used to retrieve information about a given CampaignMetric, drawing on a central database of government campaigns data. This tool is used to select filters that should be applied when retrieving this data. For categorical filters, select multiple option names. For date/continuous filters, specify ranges.",
        "input_schema": {"type": "object", "properties": properties},
    }


class SmartTargetsMetricRequest(ToolArgs):
    """One metric to look up."""

    metric_name: str = Field(description="A Smart Targets metric name, exactly as listed in the system prompt")
    context_for_filters: str = Field(
        description=(
            "Plain-language description of the campaigns to compare against, used to choose filters: "
            "audiences, channels, dates, spend, campaign type and any other constraints from the conversation"
        )
    )


class QuerySmartTargetsArgs(ToolArgs):
    """Get benchmark statistics about past government communications campaigns from the Smart Targets
    tool (the Benchmark Database). For each requested metric (e.g. impressions, click-through rate,
    awareness) it chooses filters from your context_for_filters and returns aggregate statistics for the
    matching campaigns: median, 25th and 75th percentiles, 95% confidence interval, and the counts of
    campaigns and measurements. Filters are loosened automatically if too few campaigns match. The
    statistics describe the whole filtered selection, so to compare (e.g. across years or audiences)
    request the same metric several times with different context_for_filters. If the count of
    measurements differs from the count of campaigns, the metric was measured across several channels.
    Use it to set realistic expectations for a campaign's performance."""

    metrics: LenientList[SmartTargetsMetricRequest] = Field(
        description=f"1 to {SMART_TARGETS_MAX_TOOL_METRICS} metrics to look up"
    )


QUERY_SMART_TARGETS = ToolDef("query_smart_targets", QuerySmartTargetsArgs, strict=False)
