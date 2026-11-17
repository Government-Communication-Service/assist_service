"""Unit tests for campaign benchmarks filters: tool args, per-metric checks and the filter options cache."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import ValidationError

import app.smart_targets.service as service_module
from app.config import CAMPAIGN_BENCHMARKS_MAX_TOOL_METRICS
from app.smart_targets.exceptions import GetSmartTargetsMetricsError
from app.smart_targets.schemas import SelectedFilter, SelectedFilters
from app.smart_targets.service import SmartTargetsService
from app.smart_targets.tools import QUERY_CAMPAIGN_BENCHMARKS

METRICS = {
    "no-metric-selected": {"id": "0", "name": "no-metric-selected", "uuid": None},
    "ctr": {"id": 1, "name": "ctr", "uuid": "u1"},
    "impressions": {"id": 2, "name": "impressions", "uuid": "u2"},
}
FILTERS_BY_METRIC = {
    "u1": {
        "filters": [
            {"name": "region", "type": "categorical", "options": [{"name": "wales", "uuid": "r2"}]},
            {"name": "campaign_spend", "type": "continuous", "min": 500.0, "max": 1000.0},
        ]
    },
    "u2": {
        "filters": [
            {"name": "region", "type": "categorical", "options": [{"name": "london-only", "uuid": "r1"}]},
            {"name": "channel", "type": "categorical", "options": [{"name": "tv", "uuid": "c1"}]},
            {"name": "campaign_spend", "type": "continuous", "min": 100.0, "max": 800.0},
        ]
    },
}


@pytest.fixture(autouse=True)
def _reset_filter_options_cache():
    service_module._filter_options_cache = None
    service_module._filter_options_refresh = None
    service_module._filter_options_last_failure = None
    yield
    service_module._filter_options_cache = None
    service_module._filter_options_refresh = None
    service_module._filter_options_last_failure = None


def _service() -> SmartTargetsService:
    service = SmartTargetsService()
    service.get_available_metrics = AsyncMock(return_value=METRICS)
    service._get_filters_for_selected_metric = AsyncMock(side_effect=lambda uuid: FILTERS_BY_METRIC[uuid])
    return service


def test_tool_args_become_selected_filters():
    args = QUERY_CAMPAIGN_BENCHMARKS.parse(
        {
            "metrics": [
                {
                    "metric_name": "ctr",
                    "option_filters": {"region": ["wales"], "channel": []},
                    "campaign_start_date": {"from_date": "2023-01-01"},
                    "campaign_spend": {"max_spend": 500000},
                }
            ]
        }
    )

    assert args.metrics[0].to_selected_filters() == SelectedFilters(
        filters=[
            SelectedFilter(name="region", values=["wales"]),
            SelectedFilter(name="campaign_start_date", values={"min_date": "2023-01-01"}),
            SelectedFilter(name="campaign_spend", values={"max_value": 500000}),
        ]
    )


def test_metric_without_filters_has_none():
    args = QUERY_CAMPAIGN_BENCHMARKS.parse({"metrics": [{"metric_name": "ctr", "campaign_start_date": {}}]})

    assert args.metrics[0].to_selected_filters() == SelectedFilters(filters=[])


def test_invalid_date_is_rejected():
    with pytest.raises(ValidationError):
        QUERY_CAMPAIGN_BENCHMARKS.parse(
            {"metrics": [{"metric_name": "ctr", "campaign_start_date": {"from_date": "last year"}}]}
        )


def _metrics(n):
    return {"metrics": [{"metric_name": "ctr"} for _ in range(n)]}


def test_metric_count_is_capped_by_the_schema():
    assert len(QUERY_CAMPAIGN_BENCHMARKS.parse(_metrics(CAMPAIGN_BENCHMARKS_MAX_TOOL_METRICS)).metrics) == (
        CAMPAIGN_BENCHMARKS_MAX_TOOL_METRICS
    )
    for n in (0, CAMPAIGN_BENCHMARKS_MAX_TOOL_METRICS + 1):
        with pytest.raises(ValidationError):
            QUERY_CAMPAIGN_BENCHMARKS.parse(_metrics(n))


def test_metric_count_limits_are_in_the_schema_sent_to_the_model():
    schema = QUERY_CAMPAIGN_BENCHMARKS.spec["input_schema"]["properties"]["metrics"]
    assert schema["minItems"] == 1
    assert schema["maxItems"] == CAMPAIGN_BENCHMARKS_MAX_TOOL_METRICS


def test_check_filters_drops_what_the_metric_lacks():
    selected = SelectedFilters(
        filters=[
            SelectedFilter(name="region", values=["london-only", "wales"]),
            SelectedFilter(name="channel", values=["tv"]),
            SelectedFilter(name="campaign_spend", values={"min_value": 600}),
        ]
    )

    kept, notes = SmartTargetsService()._check_filters(selected, FILTERS_BY_METRIC["u1"], "ctr")

    assert kept == SelectedFilters(
        filters=[
            SelectedFilter(name="region", values=["wales"]),
            SelectedFilter(name="campaign_spend", values={"min_value": 600}),
        ]
    )
    assert notes == ["region has no london-only for ctr", "channel is not a filter for ctr"]


def test_check_filters_drops_a_filter_with_no_valid_values():
    selected = SelectedFilters(filters=[SelectedFilter(name="region", values=["scotland"])])

    kept, notes = SmartTargetsService()._check_filters(selected, FILTERS_BY_METRIC["u1"], "ctr")

    assert kept.filters == []
    assert notes == ["region has no scotland for ctr"]


async def test_filter_options_combine_all_metrics():
    options = await _service().get_filter_options()

    assert options == {
        "campaign_spend": {"type": "continuous", "min": 100.0, "max": 1000.0},
        "channel": {"type": "categorical", "options": ["tv"]},
        "region": {"type": "categorical", "options": ["london-only", "wales"]},
    }


async def test_filter_options_are_cached():
    service = _service()
    await service.get_filter_options()
    service._get_filters_for_selected_metric.reset_mock()

    await service.get_filter_options()

    service._get_filters_for_selected_metric.assert_not_awaited()


async def test_stale_filter_options_are_returned_while_refreshing():
    service = _service()
    old = await service.get_filter_options()
    service._get_filters_for_selected_metric = AsyncMock(
        side_effect=lambda uuid: {"filters": [{"name": "maturity", "type": "categorical", "options": []}]}
    )

    with patch("app.smart_targets.service.CAMPAIGN_BENCHMARKS_FILTER_OPTIONS_CACHE_SECONDS", 0):
        assert await service.get_filter_options() == old
        await service_module._filter_options_refresh

    assert list(service_module._filter_options_cache[1]) == ["maturity"]


async def test_failed_background_refresh_keeps_cached_options():
    service = _service()
    old = await service.get_filter_options()
    service.get_available_metrics = AsyncMock(side_effect=GetSmartTargetsMetricsError("down"))

    with patch("app.smart_targets.service.CAMPAIGN_BENCHMARKS_FILTER_OPTIONS_CACHE_SECONDS", 0):
        assert await service.get_filter_options() == old
        await asyncio.wait_for(service_module._filter_options_refresh, 1)

    assert service_module._filter_options_cache[1] == old


async def test_filter_options_failure_without_cache_raises():
    service = _service()
    service.get_available_metrics = AsyncMock(side_effect=GetSmartTargetsMetricsError("down"))

    with pytest.raises(GetSmartTargetsMetricsError):
        await service.get_filter_options()


async def test_filter_options_skip_a_metric_whose_filters_fail():
    service = _service()

    def filters(uuid):
        if uuid == "u1":
            raise httpx.HTTPStatusError("boom", request=httpx.Request("GET", "http://x"), response=httpx.Response(500))
        return FILTERS_BY_METRIC[uuid]

    service._get_filters_for_selected_metric = AsyncMock(side_effect=filters)

    options = await service.get_filter_options()

    assert options["region"]["options"] == ["london-only"]
    assert options["campaign_spend"] == {"type": "continuous", "min": 100.0, "max": 800.0}


@pytest.mark.parametrize(
    "bad_filters",
    [
        {"filters": [{"name": "region", "options": []}]},
        {"filters": [{"name": "campaign_spend", "type": "continuous", "min": None, "max": 5.0}]},
        {"filters": [{"name": "region", "type": "categorical", "options": [{"name": 3, "uuid": "x"}]}]},
        ["not", "a", "dict"],
    ],
)
async def test_filter_options_skip_a_metric_with_malformed_filters(bad_filters):
    service = _service()
    service._get_filters_for_selected_metric = AsyncMock(
        side_effect=lambda uuid: bad_filters if uuid == "u1" else FILTERS_BY_METRIC[uuid]
    )

    options = await service.get_filter_options()

    assert set(options) == {"campaign_spend", "channel", "region"}
    assert options["campaign_spend"]["max"] == 800.0


async def test_filter_options_skip_a_filter_whose_type_differs_between_metrics():
    service = _service()
    service._get_filters_for_selected_metric = AsyncMock(
        side_effect=lambda uuid: {"filters": [{"name": "campaign_start_date", "type": "date", "min": "a", "max": "b"}]}
        if uuid == "u1"
        else {"filters": [{"name": "campaign_start_date", "type": "continuous", "min": 1, "max": 2}]}
    )

    options = await service.get_filter_options()

    assert options == {"campaign_start_date": {"type": "date", "min": "a", "max": "b"}}


async def test_filter_options_raise_when_every_metric_fails():
    service = _service()
    service._get_filters_for_selected_metric = AsyncMock(side_effect=ValueError("not json"))

    with pytest.raises(GetSmartTargetsMetricsError):
        await service.get_filter_options()


async def test_unexpected_metrics_error_becomes_smart_targets_error():
    service = _service()
    service.get_available_metrics = AsyncMock(side_effect=KeyError("metrics"))

    with pytest.raises(GetSmartTargetsMetricsError):
        await service.get_filter_options()


async def test_filter_options_are_not_fetched_again_during_backoff():
    service = _service()
    service.get_available_metrics = AsyncMock(side_effect=GetSmartTargetsMetricsError("down"))
    with pytest.raises(GetSmartTargetsMetricsError):
        await service.get_filter_options()

    with pytest.raises(GetSmartTargetsMetricsError):
        await service.get_filter_options()
    assert service.get_available_metrics.await_count == 1

    with patch("app.smart_targets.service.FILTER_OPTIONS_RETRY_BACKOFF_SECONDS", 0):
        with pytest.raises(GetSmartTargetsMetricsError):
            await service.get_filter_options()
    assert service.get_available_metrics.await_count == 2


async def test_stale_filter_options_are_not_refreshed_during_backoff():
    service = _service()
    await service.get_filter_options()
    service.get_available_metrics = AsyncMock(side_effect=GetSmartTargetsMetricsError("down"))

    with patch("app.smart_targets.service.CAMPAIGN_BENCHMARKS_FILTER_OPTIONS_CACHE_SECONDS", 0):
        await service.get_filter_options()
        await asyncio.wait_for(service_module._filter_options_refresh, 1)
        await service.get_filter_options()

    assert service_module._filter_options_refresh.done()
    assert service.get_available_metrics.await_count == 1
