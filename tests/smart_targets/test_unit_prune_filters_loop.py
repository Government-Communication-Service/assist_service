"""Unit tests for the filter-pruning loop in SmartTargetsService._process_single_metric (API and LLM calls mocked)."""

from unittest.mock import AsyncMock, MagicMock, patch

from app.smart_targets.schemas import SelectedFilter, SelectedFilters
from app.smart_targets.service import SmartTargetsService

RAW_FILTERS = {
    "filters": [
        {"name": "region", "type": "categorical", "options": [{"name": "London", "uuid": "r1"}]},
        {"name": "channel", "type": "categorical", "options": [{"name": "TV", "uuid": "c1"}]},
        {"name": "campaign_start_date", "type": "date"},
        {"name": "other_date", "type": "date"},
    ]
}
METRIC_INFO = {"uuid": "m1", "name": "Impressions", "context_for_filters": "x"}


def _summary(n_campaigns):
    return {"summary_statistics": {"n_campaigns": n_campaigns}, "url": "https://st/1"}


def _service(selected, summaries):
    with patch("app.smart_targets.service.LLMTable", MagicMock()), patch("app.smart_targets.service.BedrockHandler"):
        service = SmartTargetsService()
    service._get_filters_for_selected_metric = AsyncMock(return_value=RAW_FILTERS)
    service._select_filters = AsyncMock(return_value=SelectedFilters(reasoning="r", filters=selected))
    service._get_summary_for_metric = AsyncMock(side_effect=summaries)
    return service


async def test_unknown_filters_are_dropped_before_querying():
    selected = [SelectedFilter(name="made_up", values=["a"]), SelectedFilter(name="region", values=["London"])]
    service = _service(selected, lambda *_: _summary(0))

    result = await service._process_single_metric([], METRIC_INFO)

    # region, then no filters: made_up never gets a query of its own
    assert service._get_summary_for_metric.await_count == 2
    assert isinstance(result, tuple)
    assert "made_up" not in result[3]


async def test_unprunable_filter_stops_the_loop():
    selected = [SelectedFilter(name="other_date", values={"min_date": "2024-01-01"})]
    service = _service(selected, lambda *_: _summary(0))

    with patch("app.smart_targets.service.logger") as logger:
        result = await service._process_single_metric([], METRIC_INFO)

    assert service._get_summary_for_metric.await_count == 1
    assert isinstance(result, tuple)
    logger.warning.assert_called_once()
    logger.error.assert_not_called()


async def test_loop_is_bounded_when_pruning_never_empties_filters():
    region = SelectedFilter(name="region", values=["London"])
    channel = SelectedFilter(name="channel", values=["TV"])
    service = _service([region, channel], lambda *_: _summary(0))
    # A broken prune that keeps swapping one filter for the other, so the list never empties or repeats
    service._prune_selected_filters = AsyncMock(
        side_effect=lambda filters, raw: SelectedFilters(
            reasoning="r", filters=[channel] if filters.filters[0] == region else [region]
        )
    )

    with patch("app.smart_targets.service.logger") as logger:
        result = await service._process_single_metric([], METRIC_INFO)

    assert service._get_summary_for_metric.await_count == 3
    assert isinstance(result, tuple)
    logger.error.assert_called_once()


async def test_stops_once_enough_campaigns():
    selected = [SelectedFilter(name="region", values=["London"]), SelectedFilter(name="channel", values=["TV"])]
    service = _service(selected, [_summary(5), _summary(25)])

    result = await service._process_single_metric([], METRIC_INFO)

    assert service._get_summary_for_metric.await_count == 2
    # region is pruned before channel
    assert "channel" in result[3]
    assert "region" not in result[3]
