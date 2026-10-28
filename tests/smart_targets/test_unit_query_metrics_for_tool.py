"""Unit tests for SmartTargetsService.query_metrics_for_tool (API and LLM calls mocked)."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.bedrock.call_log import LlmCallContext
from app.smart_targets.service import CAMPAIGN_BENCHMARKS_MAX_TOOL_METRICS, SmartTargetsService

METRICS = {
    "no-metric-selected": {"id": "0", "name": "no-metric-selected", "uuid": None},
    "Impressions": {"id": 1, "name": "Impressions", "uuid": str(uuid4())},
    "Click-through rate": {"id": 2, "name": "Click-through rate", "uuid": str(uuid4())},
}


def _service(process_results):
    with patch("app.smart_targets.service.LLMTable", MagicMock()), patch("app.smart_targets.service.BedrockHandler"):
        service = SmartTargetsService()
    service.get_available_metrics = AsyncMock(return_value=METRICS)
    service._process_single_metric = AsyncMock(side_effect=process_results)
    return service


async def test_returns_statistics_and_sources_per_metric():
    service = _service([("median 5", "https://st/1", "Impressions", "audience=young")])

    content, sources = await service.query_metrics_for_tool([("Impressions", "young people")], messages=[])

    assert "Impressions summary statistics" in content
    assert "median 5" in content
    assert sources == [
        {"docname": "Smart Targets dashboard: metric=Impressions; filters=audience=young", "docurl": "https://st/1"}
    ]
    passed_info = service._process_single_metric.await_args.args[1]
    assert passed_info["context_for_filters"] == "young people"


async def test_llm_call_context_is_passed_to_each_metric():
    service = _service([("median 5", "https://st/1", "Impressions", "audience=young")])
    context = LlmCallContext(purpose="smart_targets_select_filters", chat_id=1, message_id=2)

    await service.query_metrics_for_tool([("Impressions", "young people")], messages=[], context=context)

    assert service._process_single_metric.await_args.args[2] == context


async def test_unknown_metric_lists_available_names():
    service = _service([])

    content, sources = await service.query_metrics_for_tool([("Nope", "x"), ("no-metric-selected", "x")], messages=[])

    assert sources == []
    assert "Unknown metric(s): Nope, no-metric-selected" in content
    assert "Available metrics: Impressions, Click-through rate" in content


async def test_failed_metric_is_reported_without_exception_detail():
    service = _service(["ERROR processing metric 'Impressions': secret-host refused"])

    content, sources = await service.query_metrics_for_tool([("Impressions", "x")], messages=[])

    assert "Impressions: the data could not be retrieved." in content
    assert "secret-host" not in content
    assert sources == []


async def test_metric_count_is_capped():
    service = _service(lambda messages, info, context=None: ("s", "u", info["name"], "f"))

    await service.query_metrics_for_tool([("Impressions", str(i)) for i in range(10)], messages=[])

    assert service._process_single_metric.await_count == CAMPAIGN_BENCHMARKS_MAX_TOOL_METRICS
