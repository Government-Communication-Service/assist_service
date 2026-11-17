import asyncio
import math
import time
from logging import getLogger
from uuid import UUID

from httpx import AsyncClient, HTTPError, HTTPStatusError

from app.config import CAMPAIGN_BENCHMARKS_FILTER_OPTIONS_CACHE_SECONDS
from app.smart_targets.constants import (
    FILTER_OPTIONS_RETRY_BACKOFF_SECONDS,
    URL_CAMPAIGN_BENCHMARKS_FILTERS,
    URL_CAMPAIGN_BENCHMARKS_HEALTHCHECK,
    URL_CAMPAIGN_BENCHMARKS_METRICS,
    URL_CAMPAIGN_BENCHMARKS_SUMMARY,
)
from app.smart_targets.exceptions import GetSmartTargetsMetricsError, SmartTargetsConnectionError
from app.smart_targets.schemas import SelectedFilter, SelectedFilters

logger = getLogger(__name__)

# (fetched_at monotonic seconds, options) for get_filter_options, and its running background refresh
_filter_options_cache: tuple[float, dict] | None = None
_filter_options_refresh: asyncio.Task | None = None
# Monotonic seconds of the last failed get_filter_options fetch, cleared by a successful one
_filter_options_last_failure: float | None = None


def _filter_options_in_backoff() -> bool:
    return (
        _filter_options_last_failure is not None
        and time.monotonic() - _filter_options_last_failure < FILTER_OPTIONS_RETRY_BACKOFF_SECONDS
    )


def _sig_figs(value, figures: int = 3):
    """A number rounded to significant figures without scientific notation; anything else unchanged."""
    if isinstance(value, bool) or not isinstance(value, int | float) or value == 0 or not math.isfinite(value):
        return value
    rounded = round(value, figures - 1 - math.floor(math.log10(abs(value))))
    return int(rounded) if rounded == int(rounded) else rounded


def format_filter_options(filter_options: dict) -> list[str]:
    """One "name: values" line per filter, from get_filter_options or _metric_filter_options."""
    lines = []
    for name, f in filter_options.items():
        if f["type"] == "categorical":
            lines.append(f"{name}: {', '.join(f['options'])}")
        elif f["type"] == "continuous":
            lines.append(f"{name}: {round(f['min']):,} to {round(f['max']):,}")
        else:
            lines.append(f"{name}: {f['min']} to {f['max']}")
    return lines


def _validated_metric_filter_options(raw_filters_data: dict) -> dict:
    """_metric_filter_options, checked so that format_filter_options can render every entry.

    Raises:
        ValueError: A filter has the wrong shape, e.g. a missing range end or a non-numeric spend.
    """
    try:
        options = _metric_filter_options(raw_filters_data)
    except (AttributeError, KeyError, TypeError) as e:
        raise ValueError(f"Malformed filters data: {e!r}") from e
    for name, f in options.items():
        if f["type"] == "categorical":
            if not all(isinstance(o, str) for o in f["options"]):
                raise ValueError(f"Filter {name} has a non-string option name")
        elif f["type"] == "continuous":
            if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in (f["min"], f["max"])):
                raise ValueError(f"Filter {name} has a non-numeric range")
        elif not all(isinstance(v, str) for v in (f["min"], f["max"])):
            raise ValueError(f"Filter {name} has a non-string range")
    return options


def _metric_filter_options(raw_filters_data: dict) -> dict:
    """One metric's filters in the get_filter_options shape."""
    options = {}
    for f in sorted(raw_filters_data.get("filters", []), key=lambda f: f["name"]):
        if f["type"] == "categorical":
            options[f["name"]] = {"type": "categorical", "options": sorted(o["name"] for o in f.get("options", []))}
        else:
            options[f["name"]] = {"type": f["type"], "min": f["min"], "max": f["max"]}
    return options


class SmartTargetsService:
    def __init__(self):
        self.async_client: AsyncClient = AsyncClient()

    async def verify_connection(self):
        try:
            r = await self.async_client.get(URL_CAMPAIGN_BENCHMARKS_HEALTHCHECK)
            r.raise_for_status()
            logger.info("Succesfully connected to GCS Data API")
            return
        except Exception as e:
            m = "Failed to connect with the GCS Data API"
            logger.error(f"{m}: {e}")
            raise SmartTargetsConnectionError("Failed to verify connection with the GCS Data API") from e

    async def get_available_metrics(self) -> dict:
        try:
            r = await self.async_client.get(URL_CAMPAIGN_BENCHMARKS_METRICS)
            r.raise_for_status()
        except HTTPError as e:
            raise GetSmartTargetsMetricsError("Failed to get metrics from the Smart Targets tool") from e

        # Keyed by name so the model can request metrics by name; the UUID is looked up afterwards.
        metrics = {"no-metric-selected": {"id": "0", "name": "no-metric-selected", "uuid": None}}
        for i, m in enumerate(r.json()["metrics"]):
            shifted_index = i + 1
            name = m["name"]
            metrics[name] = {"id": shifted_index, "name": name, "uuid": m["uuid"]}
        return metrics

    async def _get_filters_for_selected_metric(self, metric_uuid: UUID) -> dict:
        url = URL_CAMPAIGN_BENCHMARKS_FILTERS.format(metric_uuid=metric_uuid)
        r = await self.async_client.get(url)
        r.raise_for_status()
        return r.json()

    async def get_filter_options(self) -> dict:
        """Every filter across all metrics, from a process-wide cache.

        Once the cache is older than CAMPAIGN_BENCHMARKS_FILTER_OPTIONS_CACHE_SECONDS it is still returned, and
        a background refresh is started; only the first call in a process waits for the fetch. After a failed
        fetch, none is tried again for FILTER_OPTIONS_RETRY_BACKOFF_SECONDS.

        Returns:
            Filter name -> {"type", "options"} for categorical filters (sorted option names) or
            {"type", "min", "max"} for date and continuous filters (the widest range over all metrics).

        Raises:
            GetSmartTargetsMetricsError: Nothing is cached and the filters couldn't be fetched, now or recently.
        """
        global _filter_options_refresh
        if _filter_options_cache is None:
            if _filter_options_in_backoff():
                raise GetSmartTargetsMetricsError("Smart Targets filter options failed recently; not retrying yet")
            return await self._refresh_filter_options()
        fetched_at, options = _filter_options_cache
        stale = time.monotonic() - fetched_at >= CAMPAIGN_BENCHMARKS_FILTER_OPTIONS_CACHE_SECONDS
        idle = _filter_options_refresh is None or _filter_options_refresh.done()
        if stale and idle and not _filter_options_in_backoff():
            _filter_options_refresh = asyncio.create_task(self._refresh_filter_options_in_background())
        return options

    async def _refresh_filter_options_in_background(self) -> None:
        try:
            await self._refresh_filter_options()
        except GetSmartTargetsMetricsError as e:
            logger.error(f"Failed to refresh Smart Targets filter options; keeping the cached ones. Error: {e}")

    async def _refresh_filter_options(self) -> dict:
        """Fetch the filter options into the cache, recording a failure for the retry backoff.

        Raises:
            GetSmartTargetsMetricsError: The fetch failed, whatever the cause.
        """
        global _filter_options_cache, _filter_options_last_failure
        try:
            options = await self._fetch_filter_options()
        except Exception as e:
            _filter_options_last_failure = time.monotonic()
            if isinstance(e, GetSmartTargetsMetricsError):
                raise
            raise GetSmartTargetsMetricsError("Failed to get filter options from the Smart Targets tool") from e
        _filter_options_cache = (time.monotonic(), options)
        _filter_options_last_failure = None
        return options

    async def _fetch_filter_options(self) -> dict:
        """Every metric's filters combined, skipping any metric whose filters can't be fetched or read.

        Raises:
            GetSmartTargetsMetricsError: No metric's filters could be fetched.
        """
        metrics = await self.get_available_metrics()
        uuids = [m["uuid"] for m in metrics.values() if m["uuid"]]
        results = await asyncio.gather(
            *(self._get_filters_for_selected_metric(u) for u in uuids), return_exceptions=True
        )

        metric_options: list[tuple[str, dict]] = []
        for uuid, result in zip(uuids, results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result
                logger.warning(f"Skipping Smart Targets metric {uuid} in the filter options: {result!r}")
                continue
            try:
                metric_options.append((uuid, _validated_metric_filter_options(result)))
            except ValueError as e:
                logger.warning(f"Skipping Smart Targets metric {uuid} in the filter options: {e!r}")
        if uuids and not metric_options:
            raise GetSmartTargetsMetricsError("Failed to get any metric's filters from the Smart Targets tool")

        combined: dict[str, dict] = {}
        for uuid, options in metric_options:
            for name, f in options.items():
                entry = combined.get(name)
                if entry is None:
                    combined[name] = {**f, "options": set(f["options"])} if f["type"] == "categorical" else {**f}
                elif entry["type"] != f["type"]:
                    logger.warning(f"Smart Targets metric {uuid}: filter {name} is {f['type']}, not {entry['type']}")
                elif f["type"] == "categorical":
                    entry["options"].update(f["options"])
                else:
                    entry["min"] = min(entry["min"], f["min"])
                    entry["max"] = max(entry["max"], f["max"])
        return {
            name: {**entry, "options": sorted(entry["options"])} if entry["type"] == "categorical" else entry
            for name, entry in sorted(combined.items())
        }

    def _check_filters(
        self, selected_filters: SelectedFilters, raw_filters_data: dict, metric_name: str
    ) -> tuple[SelectedFilters, list[str]]:
        """Drop filters and option values this metric doesn't have.

        Returns:
            The filters to apply, and a note for each filter or value that was dropped.
        """
        available = {f["name"]: f for f in raw_filters_data.get("filters", [])}
        kept: list[SelectedFilter] = []
        notes: list[str] = []
        for f in selected_filters.filters:
            filter_info = available.get(f.name)
            if filter_info is None:
                notes.append(f"{f.name} is not a filter for {metric_name}")
                continue
            if filter_info["type"] != "categorical":
                if isinstance(f.values, dict):
                    kept.append(f)
                continue
            if not isinstance(f.values, list):
                notes.append(f"{f.name} needs a list of values")
                continue
            option_names = {o["name"] for o in filter_info.get("options", [])}
            values = [v for v in f.values if v in option_names]
            missing = [v for v in f.values if v not in option_names]
            if missing:
                notes.append(f"{f.name} has no {', '.join(missing)} for {metric_name}")
            if values:
                kept.append(SelectedFilter(name=f.name, values=values))
        return SelectedFilters(filters=kept), notes

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
            return SelectedFilters(filters=new_filters)

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
        url = URL_CAMPAIGN_BENCHMARKS_SUMMARY.format(metric_uuid=metric_uuid)

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
        known_unit = unit if unit and unit != "unknown" else ""
        unit_suffix = f" {known_unit}" if known_unit else ""
        unit_display = f", unit={known_unit}" if known_unit else ""
        if selected_filters:
            filters_display = self._format_filters(selected_filters)
        else:
            filters_display = "None"

        result = [
            (
                f"Campaign Performance Statistics ({selected_metric_name}{unit_display}), "
                f"filters_applied={{{filters_display}}}:"
            )
        ]

        # Add campaign and measurement counts
        n_campaigns = stats.get("n_campaigns", "N/A")
        n_measurements = stats.get("n_measurements", "N/A")
        result.append(f"- Number of campaigns: {n_campaigns}")
        result.append(f"- Number of data points: {n_measurements}")

        # Add quartiles
        lower_q = _sig_figs(stats.get("lower_quartile", "N/A"))
        median = _sig_figs(stats.get("median", "N/A"))
        upper_q = _sig_figs(stats.get("upper_quartile", "N/A"))
        if lower_q != "N/A":
            result.append(f"- Lower quartile: {lower_q}{unit_suffix}")
        if median != "N/A":
            result.append(f"- Median: {median}{unit_suffix}")
        if upper_q != "N/A":
            result.append(f"- Upper quartile: {upper_q}{unit_suffix}")

        # Add the spread of the data; the API returns the 2.5th and 97.5th percentiles (min and max below 40 points)
        spread = stats.get("confidence_interval_95", [])
        if spread and len(spread) >= 2:
            spread_low = _sig_figs(spread[0])
            spread_high = _sig_figs(spread[1])
            if isinstance(n_measurements, int) and n_measurements < 40:
                label = "Minimum to maximum (too few data points for percentiles)"
            else:
                label = "2.5th to 97.5th percentile range (middle 95% of data points)"
            result.append(f"- {label}: {spread_low} to {spread_high}{unit_suffix}")

        return "\n".join(result)

    async def _process_single_metric(self, metric_info: dict) -> tuple[str, str, str, str] | str:
        """Query one metric with the requested filters, loosening them if too few campaigns match.

        Args:
            metric_info: The metric's "uuid" and "name", and the requested "filters" (SelectedFilters).

        Returns:
            (formatted statistics, dashboard URL, metric name, filters applied), or an error string.
        """
        metric_uuid = metric_info["uuid"]
        metric_name = metric_info["name"]

        try:
            raw_filters_data: dict = await self._get_filters_for_selected_metric(metric_uuid)
            selected_filters, ignored = self._check_filters(metric_info["filters"], raw_filters_data, metric_name)

            # Iteratively prune filters if campaign count is too low.
            # Each pass removes at most one filter, so one query per filter plus the unfiltered one is enough.
            current_filters = selected_filters
            summary_data = None
            min_campaigns_threshold = 20
            max_queries = len(current_filters.filters) + 1

            for query_number in range(1, max_queries + 1):
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

                if query_number == max_queries:
                    logger.error(
                        f"Metric {metric_name}: filter pruning hit its bound of {max_queries} queries "
                        f"with filters {[f.name for f in current_filters.filters]}"
                    )
                    break

                # Prune one filter and try again; stop if none of the remaining filters can be pruned
                pruned_filters = await self._prune_selected_filters(current_filters, raw_filters_data)
                if pruned_filters.filters == current_filters.filters:
                    logger.warning(
                        f"Metric {metric_name}: no prunable filter left in {[f.name for f in current_filters.filters]}"
                    )
                    break
                current_filters = pruned_filters

            # Format response for this metric with final pruned filters
            formatted_response = self._format_measurements_response(
                summary_data,
                selected_metric_name=metric_name,
                selected_filters=current_filters,
            )
            applied_names = {f.name for f in current_filters.filters}
            dropped = [f.name for f in selected_filters.filters if f.name not in applied_names]
            if dropped:
                formatted_response += (
                    f"\n- Filters dropped because fewer than {min_campaigns_threshold} campaigns matched: "
                    f"{', '.join(dropped)}"
                )
            if ignored:
                available = format_filter_options(_metric_filter_options(raw_filters_data))
                formatted_response += (
                    f"\n- Requested filters ignored: {'; '.join(ignored)}"
                    f"\n- Filters available for {metric_name}:" + "".join(f"\n  - {line}" for line in available)
                )

            citation = summary_data["url"]

            filters_display = self._format_filters(current_filters)

            return formatted_response, citation, metric_name, filters_display

        except Exception as e:
            logger.error(f"Error processing metric {metric_name}: {str(e)}")
            return f"ERROR processing metric '{metric_name}': {str(e)}"

    async def query_metrics_for_tool(
        self, requests: list[tuple[str, SelectedFilters]]
    ) -> tuple[str, list[dict[str, str]]]:
        """Look up the chat model's requested metrics with the filters it chose.

        Args:
            requests: (metric name, filters) pairs; the tool schema caps how many.

        Returns:
            The tool result text, and a source dict (docname, docurl) per metric returned.
        """
        available_metrics = await self.get_available_metrics()
        metric_infos = []
        unknown = []
        for metric_name, filters in requests:
            metric = available_metrics.get(metric_name)
            if metric is None or not metric["uuid"]:
                unknown.append(metric_name)
                continue
            metric_infos.append({"uuid": UUID(metric["uuid"]), "name": metric_name, "filters": filters})

        results = await asyncio.gather(*(self._process_single_metric(info) for info in metric_infos))

        parts = []
        sources = []
        for info, result in zip(metric_infos, results, strict=True):
            if isinstance(result, str):
                parts.append(f"{info['name']}: the data could not be retrieved.")
                continue
            formatted, url, metric_name, filters_display = result
            parts.append(f"{metric_name} summary statistics:\n\n{formatted}")
            sources.append(
                {"docname": f"Smart Targets dashboard: metric={metric_name}; filters={filters_display}", "docurl": url}
            )
        if unknown:
            names = ", ".join(name for name in available_metrics if available_metrics[name]["uuid"])
            parts.append(f"Unknown metric(s): {', '.join(unknown)}. Available metrics: {names}")
        return (
            "<results-from-campaign-benchmarks-tool>\n"
            + "\n\n".join(parts)
            + "\n</results-from-campaign-benchmarks-tool>",
            sources,
        )
