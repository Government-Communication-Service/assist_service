import asyncio
from logging import getLogger
from uuid import UUID

from anthropic.types import ToolUseBlock
from httpx import AsyncClient, HTTPError, HTTPStatusError

from app.bedrock.bedrock import BedrockHandler, RunMode
from app.chat.utils import prepare_message_objects_for_llm
from app.database.models import Message
from app.smart_targets.constants import (
    URL_SMART_TARGETS_FILTERS,
    URL_SMART_TARGETS_HEALTHCHECK,
    URL_SMART_TARGETS_METRICS,
    URL_SMART_TARGETS_SUMMARY,
)
from app.smart_targets.exceptions import GetSmartTargetsMetricsError, SmartTargetsConnectionError
from app.smart_targets.prompts import (
    SYSTEM_PROMPT_SELECT_FILTERS_DYNAMIC,
    USER_PROMPT_SELECT_FILTERS,
    USER_PROMPT_SELECT_METRIC,
    get_system_prompt_select_filters_with_context,
    get_system_prompt_select_metrics,
)
from app.smart_targets.schemas import (
    SelectedFilter,
    SelectedFilters,
)
from app.smart_targets.tools import (
    SELECT_FILTERS_TOOL_NAME,
    SELECT_METRICS_TOOL,
    SELECT_METRICS_TOOL_NAME,
    generate_select_filters_tool_schema,
)

logger = getLogger(__name__)


class SmartTargetsService:
    def __init__(self):
        self.async_client: AsyncClient = AsyncClient()
        self.bedrock_handler: BedrockHandler = BedrockHandler(mode=RunMode.ASYNC)

    async def verify_connection(self):
        try:
            r = await self.async_client.get(URL_SMART_TARGETS_HEALTHCHECK)
            r.raise_for_status()
            logger.info("Succesfully connected to GCS Data API")
            return
        except Exception as e:
            m = "Failed to connect with the GCS Data API"
            logger.error(f"{m}: {e}")
            raise SmartTargetsConnectionError("Failed to verify connection with the GCS Data API") from e

    def _wrap_chat_messages(self, messages: list[Message]) -> str:
        # It's important to use this function so that the appropriate message
        # content is actually sent to the LLM. This prevents us blowing up the
        # context.
        messages_prep: list[dict] = prepare_message_objects_for_llm(messages)
        all_but_final_messages = messages_prep[0:-1]
        last_message = messages_prep[-1]

        user_message = "<ChatMessages>\n"
        for m in all_but_final_messages:
            user_message += f"<{m['role']}>{m['content']}</{m['role']}>\n"
        user_message += f"<MostRecentMessage>{last_message['content']}</MostRecentMessage>\n"
        user_message += "</ChatMessages>"
        return user_message

    async def get_available_metrics(self) -> dict:
        try:
            r = await self.async_client.get(URL_SMART_TARGETS_METRICS)
            r.raise_for_status()
        except HTTPError as e:
            raise GetSmartTargetsMetricsError("Failed to get metrics from the Smart Targets tool") from e

        # Encode the result in a dict with name keys
        # This is so we can give the LLM an English name to choose from when making a selection
        # We think this will be helpful because LLMs are trained on English words and so
        # are less likely to make mistakes when using English words
        # After the LLM makes it's selection, we retrieve the UUID from this dict

        # Make sure the LLM has the option to select none of the metrics
        metrics = {"no-metric-selected": {"id": "0", "name": "no-metric-selected", "uuid": None}}
        for i, m in enumerate(r.json()["metrics"]):
            shifted_index = i + 1
            name = m["name"]
            metrics[name] = {"id": shifted_index, "name": name, "uuid": m["uuid"]}
        return metrics

    def _wrap_metrics(self, available_metrics: dict) -> str:
        wrapped_metrics = "<CampaignMetrics>\n"
        for name, m in available_metrics.items():
            metric_id = m["id"]
            wrapped_metrics += f"<metric {metric_id}>name={name}</metric {metric_id}\n>"
        wrapped_metrics += "</CampaignMetrics>"
        return wrapped_metrics

    async def _select_metrics(self, messages: list[Message], available_metrics: dict = None) -> list[dict]:
        wrapped_metrics = self._wrap_metrics(available_metrics)
        system = get_system_prompt_select_metrics(wrapped_metrics)
        r = await self.bedrock_handler.invoke_async(
            system=system,
            messages=[
                {"role": "user", "content": f"{USER_PROMPT_SELECT_METRIC}\n{self._wrap_chat_messages(messages)}"}
            ],
            tools=[SELECT_METRICS_TOOL],
            tool_choice={"type": "tool", "name": SELECT_METRICS_TOOL_NAME},
        )
        for content in r.content:
            if content.type == "tool_use":
                selected_metrics_data = content.input["selected_metrics"]
                result = []
                for metric_data in selected_metrics_data:
                    selected_metric_name = metric_data["metric_name"]
                    context_for_filters = metric_data["context_for_filters"]
                    if selected_metric_name in available_metrics:
                        selected_metric = available_metrics[selected_metric_name]
                        try:
                            selected_metric_uuid = selected_metric["uuid"]
                            parsed_selected_metric_uuid = UUID(selected_metric_uuid)
                            result.append(
                                {
                                    "uuid": parsed_selected_metric_uuid,
                                    "name": selected_metric_name,
                                    "context_for_filters": context_for_filters,
                                }
                            )
                        except Exception:
                            logger.exception(
                                "Could not append selected metric: "
                                f"{selected_metric=}, {selected_metric_name}, "
                                f"{selected_metric_uuid=}, {context_for_filters=}"
                            )
                            continue
                logger.info(f"Selected Smart Targets metrics: {result}")
                return result
        raise Exception("No tool use was detected.")

    async def _get_filters_for_selected_metric(self, metric_uuid: UUID) -> dict:
        url = URL_SMART_TARGETS_FILTERS.format(metric_uuid=metric_uuid)
        r = await self.async_client.get(url)
        return r.json()

    def _wrap_filters(self, available_filters: dict) -> str:
        wrapped_filters = "<CampaignFilters>\n"
        for name, f in available_filters.items():
            filter_id = f["id"]
            wrapped_filters += f"<filter {filter_id}>name={name}</filter {filter_id}\n>"
        wrapped_filters += "</CampaignFilters>"
        return wrapped_filters

    async def _select_filters(
        self,
        messages: list[Message],
        raw_filters_data: dict,
        metric_name: str | None = None,
        extracted_context: str | None = None,
    ) -> SelectedFilters:
        user_message: str = f"{USER_PROMPT_SELECT_FILTERS}\n{self._wrap_chat_messages(messages)}"
        tool = generate_select_filters_tool_schema(raw_filters_data)

        # Use context-aware system prompt if context is provided
        if metric_name and extracted_context:
            system_prompt = get_system_prompt_select_filters_with_context(metric_name, extracted_context)
        else:
            system_prompt = SYSTEM_PROMPT_SELECT_FILTERS_DYNAMIC

        r = await self.bedrock_handler.invoke_async(
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            tools=[tool],
            tool_choice={"type": "tool", "name": SELECT_FILTERS_TOOL_NAME},
        )

        for content in r.content:
            if isinstance(content, ToolUseBlock):
                if isinstance(content.input, dict):
                    return SelectedFilters(
                        reasoning=content.input["reasoning"],
                        filters=[
                            SelectedFilter(name=k, values=v) for k, v in content.input.items() if k != "reasoning"
                        ],
                    )

        raise Exception("No tool use was detected.")

    async def _prune_selected_filters(
        self, selected_filters: SelectedFilters, raw_filters_data: dict
    ) -> SelectedFilters:
        """Remove the least important filter based on priority order.

        Priority (most to least important):
        1. campaign_spend (continuous) - most important
        2. campaign_start_date (date) - most important
        3. campaign_end_date (date) - most important
        4. Categorical filters - least important (removed first)

        Returns a new SelectedFilters object with one filter removed.
        """
        if not selected_filters.filters:
            return selected_filters

        # Build a mapping of filter names to their types
        filter_types = {}
        for filter_info in raw_filters_data.get("filters", []):
            filter_types[filter_info["name"]] = filter_info["type"]

        # Prioritised order of filters to remove
        order_to_remove = [
            "mission",
            "region",
            "umbrella_campaign",
            "maturity",
            "gov_audience",
            "standardised_objective",
            "channel",
        ]

        # Categorize filters by type
        categorical_filters = []
        date_filters = []
        continuous_filters = []

        for f in selected_filters.filters:
            filter_type = filter_types.get(f.name)
            if filter_type == "categorical":
                categorical_filters.append(f)
            elif filter_type == "date":
                date_filters.append(f)
            elif filter_type == "continuous":
                continuous_filters.append(f)

        # Remove filters in order of least importance (categorical first)
        filter_to_remove = None

        # First, remove categorical filters (least important) in priority order
        if categorical_filters:
            # Find the first filter in order_to_remove that exists in categorical_filters
            categorical_filter_names = {f.name for f in categorical_filters}
            for filter_name in order_to_remove:
                if filter_name in categorical_filter_names:
                    filter_to_remove = next(f for f in categorical_filters if f.name == filter_name)
                    break
            # If no filter matches the priority list, fall back to removing any categorical filter
            if filter_to_remove is None:
                filter_to_remove = categorical_filters[0]
        # Then campaign_end_date
        elif any(f.name == "campaign_end_date" for f in date_filters):
            filter_to_remove = next(f for f in date_filters if f.name == "campaign_end_date")
        # Then campaign_start_date
        elif any(f.name == "campaign_start_date" for f in date_filters):
            filter_to_remove = next(f for f in date_filters if f.name == "campaign_start_date")
        # Finally campaign_spend (most important, removed last)
        elif continuous_filters:
            filter_to_remove = continuous_filters[0]

        if filter_to_remove:
            logger.info(f"Pruning filter: {filter_to_remove.name}")
            new_filters = [f for f in selected_filters.filters if f != filter_to_remove]
            return SelectedFilters(reasoning=selected_filters.reasoning, filters=new_filters)

        return selected_filters

    def _convert_tool_results_to_query_params(self, selected_filters: SelectedFilters, raw_filters_data: dict) -> dict:
        """Convert LLM tool call results to API query parameters with UUIDs"""
        query_params = {}

        # Create lookup dictionaries for each filter type
        filter_lookups = {}
        for filter_info in raw_filters_data.get("filters", []):
            filter_name = filter_info["name"]
            filter_type = filter_info["type"]

            if filter_type == "categorical":
                # Create name -> UUID mapping for categorical filters
                filter_lookups[filter_name] = {
                    option["name"]: option["uuid"] for option in filter_info.get("options", [])
                }

        # Process tool results
        for selected_filter in selected_filters.filters:
            # Find the corresponding filter in raw data
            filter_info = None
            for f in raw_filters_data.get("filters", []):
                if f["name"] == selected_filter.name:
                    filter_info = f
                    break

            if not filter_info:
                continue

            filter_type = filter_info["type"]

            if filter_type == "categorical":
                # Convert selected option names to UUIDs
                if isinstance(selected_filter.values, list) and selected_filter.values:
                    uuids = []
                    for option_name in selected_filter.values:
                        if option_name in filter_lookups[selected_filter.name]:
                            uuids.append(filter_lookups[selected_filter.name][option_name])

                    if uuids:
                        # Map filter name to API parameter name
                        param_name = f"{selected_filter.name}_uuids"
                        query_params[param_name] = uuids

            elif filter_type == "date":
                # Convert date range to API parameters
                if isinstance(selected_filter.values, dict):
                    # Map filter names to API parameter names
                    if selected_filter.name == "campaign_start_date":
                        if "min_date" in selected_filter.values:
                            query_params["start_date_from"] = selected_filter.values["min_date"]
                        if "max_date" in selected_filter.values:
                            query_params["start_date_to"] = selected_filter.values["max_date"]
                    elif selected_filter.name == "campaign_end_date":
                        if "min_date" in selected_filter.values:
                            query_params["end_date_from"] = selected_filter.values["min_date"]
                        if "max_date" in selected_filter.values:
                            query_params["end_date_to"] = selected_filter.values["max_date"]

            elif filter_type == "continuous":
                # Convert continuous range to API parameters
                if isinstance(selected_filter.values, dict):
                    if selected_filter.name == "campaign_spend":
                        if "min_value" in selected_filter.values:
                            query_params["campaign_spend_min"] = selected_filter.values["min_value"]
                        if "max_value" in selected_filter.values:
                            query_params["campaign_spend_max"] = selected_filter.values["max_value"]

        return query_params

    async def _get_summary_for_metric(self, metric_uuid: UUID, query_params: dict) -> dict:
        """Retrieve measurements from the API with query parameters"""
        url = URL_SMART_TARGETS_SUMMARY.format(metric_uuid=metric_uuid)

        try:
            r = await self.async_client.get(url, params=query_params)
            r.raise_for_status()
            return r.json()
        except HTTPStatusError as e:
            raise Exception("Failed to get measurements from the Smart Targets API") from e

    def _format_filters(self, selected_filters: SelectedFilters) -> str | None:
        filters_display = "None"
        if selected_filters.filters:
            filter_parts = []
            for filter_obj in selected_filters.filters:
                if isinstance(filter_obj.values, list):
                    # Categorical filter
                    values_str = ", ".join(filter_obj.values)
                    filter_parts.append(f"{filter_obj.name}: [{values_str}]")
                elif isinstance(filter_obj.values, dict):
                    if "date" in filter_obj.name:
                        min_val = filter_obj.values.get("min_date", "")
                        max_val = filter_obj.values.get("max_date", "")
                    elif "spend" in filter_obj.name:
                        min_val = filter_obj.values.get("min_value", "")
                        max_val = filter_obj.values.get("max_value", "")
                    filter_parts.append(f"{filter_obj.name}: [{min_val} to {max_val}]")
            filters_display = "; ".join(filter_parts) if filter_parts else "None"
        return filters_display

    def _format_measurements_response(
        self,
        summary_data: dict,
        selected_metric_name: str | None = None,
        selected_filters: SelectedFilters | None = None,
    ) -> str:
        """Format measurements API response into readable markdown format"""
        metric_info = summary_data.get("metric", {})
        stats = summary_data.get("summary_statistics", {})
        unit = metric_info.get("unit", "")
        if selected_filters:
            filters_display = self._format_filters(selected_filters)
        else:
            filters_display = "None"

        result = [
            (
                f"Campaign Performance Statistics ({selected_metric_name}, unit={unit}), "
                f"filters_applied={{{filters_display}}}:"
            )
        ]

        # Add campaign and measurement counts
        n_campaigns = stats.get("n_campaigns", "N/A")
        n_measurements = stats.get("n_measurements", "N/A")
        result.append(f"- Number of campaigns: {n_campaigns}")
        result.append(f"- Number of data points: {n_measurements}")

        # Add quartiles
        lower_q = stats.get("lower_quartile", "N/A")
        median = stats.get("median", "N/A")
        upper_q = stats.get("upper_quartile", "N/A")
        if lower_q != "N/A":
            result.append(f"- Lower quartile: {lower_q} {unit}")
        if median != "N/A":
            result.append(f"- Median: {median}")
        if upper_q != "N/A":
            result.append(f"- Upper quartile: {upper_q} {unit}")

        # Add confidence interval
        ci = stats.get("confidence_interval_95", [])
        if ci and len(ci) >= 2:
            ci_low = ci[0]
            ci_high = ci[1]
            result.append(f"- 95% confidence interval: {ci_low} to {ci_high} {unit}")

        return "\n".join(result)

    async def _process_single_metric(
        self, messages: list[Message], metric_info: dict
    ) -> tuple[str, str, str, str] | str:
        """Process a single metric and return formatted results"""
        metric_uuid = metric_info["uuid"]
        metric_name = metric_info["name"]
        extracted_context = metric_info["context_for_filters"]

        logger.debug(f"Processing metric: {metric_name}")
        logger.debug(f"Extracted context for {metric_name}: {extracted_context}")

        try:
            # Get filters for this specific metric
            raw_filters_data: dict = await self._get_filters_for_selected_metric(metric_uuid)

            # Select filters with the extracted context
            selected_filters = await self._select_filters(
                messages, raw_filters_data, metric_name=metric_name, extracted_context=extracted_context
            )

            # Iteratively prune filters if campaign count is too low
            current_filters = selected_filters
            summary_data = None
            min_campaigns_threshold = 20

            while True:
                # Convert current filters to query parameters
                query_params: dict = self._convert_tool_results_to_query_params(current_filters, raw_filters_data)

                # Get measurements with current filters
                summary_data = await self._get_summary_for_metric(metric_uuid, query_params)

                # Check campaign count
                n_campaigns = summary_data.get("summary_statistics", {}).get("n_campaigns", 0)

                logger.info(
                    f"Metric {metric_name}: n_campaigns={n_campaigns}, filters_count={len(current_filters.filters)}"
                )

                # If we have enough campaigns or no more filters to remove, stop
                if n_campaigns >= min_campaigns_threshold or not current_filters.filters:
                    break

                # Prune one filter and try again
                current_filters = await self._prune_selected_filters(current_filters, raw_filters_data)

            # Format response for this metric with final pruned filters
            formatted_response = self._format_measurements_response(
                summary_data,
                selected_metric_name=metric_name,
                selected_filters=current_filters,
            )

            citation = summary_data["url"]

            filters_display = self._format_filters(current_filters)

            return formatted_response, citation, metric_name, filters_display

        except Exception as e:
            logger.error(f"Error processing metric {metric_name}: {str(e)}")
            return f"ERROR processing metric '{metric_name}': {str(e)}"

    def _create_context_string(self, selected_metrics: list[dict], all_results: list) -> str:
        """Create properly formatted context string for LLM consumption"""
        if len(all_results) == 1:
            result = all_results[0]
            return result[0]
        context_parts = []
        for result in all_results:
            metric_name = result[2]
            metric_data = result[0]
            context_parts.append(f"{metric_name} Summary Statistics:\n\n{metric_data}")

        return "\n\n".join(context_parts)

    async def use_smart_targets_tool(self, messages: list[Message]) -> tuple[str, list[dict[str, str]]] | None:
        """Process multiple metrics in parallel and return combined results"""
        available_metrics: dict = await self.get_available_metrics()
        selected_metrics: list[dict] = await self._select_metrics(messages, available_metrics)

        if not selected_metrics:
            logger.debug("No metrics were selected for analysis.")
            return None

        # Process all metrics in parallel using asyncio.gather
        metric_tasks = [self._process_single_metric(messages, metric_info) for metric_info in selected_metrics]

        all_results = await asyncio.gather(*metric_tasks)

        # Create formatted context string using new method
        final_response = self._create_context_string(selected_metrics, all_results)
        citations = [
            {"docname": f"Smart Targets dashboard: metric={result[2]}; filters={result[3]}", "docurl": result[1]}
            for result in all_results
        ]
        logger.debug(f"Smart Targets context string: {final_response}")
        logger.info(f"Citations: {citations}")
        return final_response, citations
