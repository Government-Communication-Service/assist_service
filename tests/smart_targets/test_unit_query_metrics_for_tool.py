"""Unit tests for SmartTargetsService.query_metrics_for_tool (API calls mocked)."""

from unittest.mock import AsyncMock
from uuid import uuid4

from app.smart_targets.schemas import SelectedFilter, SelectedFilters
from app.smart_targets.service import SmartTargetsService

METRICS = {
    "no-metric-selected": {"id": "0", "name": "no-metric-selected", "uuid": None},
    "Impressions": {"id": 1, "name": "Impressions", "uuid": str(uuid4())},
    "Click-through rate": {"id": 2, "name": "Click-through rate", "uuid": str(uuid4())},
}

NO_FILTERS = SelectedFilters(filters=[])


def _service(process_results):
    service = SmartTargetsService()
    service.get_available_metrics = AsyncMock(return_value=METRICS)
    service._process_single_metric = AsyncMock(side_effect=process_results)
    return service


async def test_returns_statistics_and_sources_per_metric():
    service = _service([("median 5", "https://st/1", "Impressions", "audience=young")])

    filters = SelectedFilters(filters=[SelectedFilter(name="gov_audience", values=["youth-16-24"])])

    content, sources = await service.query_metrics_for_tool([("Impressions", filters)])

    assert "Impressions summary statistics" in content
    assert "median 5" in content
    assert sources == [
        {"docname": "Smart Targets dashboard: metric=Impressions; filters=audience=young", "docurl": "https://st/1"}
    ]
    passed_info = service._process_single_metric.await_args.args[0]
    assert passed_info["filters"] == filters


async def test_unknown_metric_lists_available_names():
    service = _service([])

    content, sources = await service.query_metrics_for_tool([("Nope", NO_FILTERS), ("no-metric-selected", NO_FILTERS)])

    assert sources == []
    assert "Unknown metric(s): Nope, no-metric-selected" in content
    assert "Available metrics: Impressions, Click-through rate" in content


async def test_failed_metric_is_reported_without_exception_detail():
    service = _service(["ERROR processing metric 'Impressions': secret-host refused"])

    content, sources = await service.query_metrics_for_tool([("Impressions", NO_FILTERS)])

    assert "Impressions: the data could not be retrieved." in content
    assert "secret-host" not in content
    assert sources == []
