# ruff: noqa: E501

from typing import Any

from app.smart_targets.prompts import SMART_TARGETS_TOOL_DEFINITION

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


SELECT_METRICS_TOOL_NAME = "select_campaign_metrics"
SELECT_METRICS_TOOL = {
    "name": SELECT_METRICS_TOOL_NAME,
    "description": f"A tool used to retrieve information about one or more CampaignMetrics from the Smart Targets tool. Select multiple metrics if the user's question would benefit from data about different types of measurements (e.g., both impressions and click-through rates, or awareness and understanding metrics). {SMART_TARGETS_TOOL_DEFINITION}",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "description": "A scratchpad to reason through your response before answering.",
                "type": "string",
            },
            "selected_metrics": {
                "description": "An array of selected metrics with context for each",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric_name": {
                            "description": "The 'name' of the campaign metric you wish to select.",
                            "type": "string",
                        },
                        "context_for_filters": {
                            "description": "Relevant information extracted from the chat that will help with filter selection for this specific metric (e.g., specific audiences, time periods, channels, or other constraints mentioned by the user).",
                            "type": "string",
                        },
                    },
                    "required": ["metric_name", "context_for_filters"],
                },
                "minItems": 0,
            },
        },
        "required": ["selected_metrics"],
    },
}
