"""Unit tests for SmartTargetsService._format_measurements_response (no API or LLM calls)."""

from unittest.mock import MagicMock, patch

from app.smart_targets.service import SmartTargetsService


def _format(n_measurements):
    with patch("app.smart_targets.service.LLMTable", MagicMock()), patch("app.smart_targets.service.BedrockHandler"):
        service = SmartTargetsService()
    summary = {
        "metric": {"unit": "percentage"},
        "summary_statistics": {
            "n_campaigns": n_measurements,
            "n_measurements": n_measurements,
            "median": 50,
            "lower_quartile": 40,
            "upper_quartile": 60,
            "confidence_interval_95": [10, 90],
        },
    }
    return service._format_measurements_response(summary, selected_metric_name="Impressions")


def test_spread_is_labelled_as_percentile_range():
    text = _format(100)
    assert "2.5th to 97.5th percentile range" in text
    assert "10 to 90 percentage" in text
    assert "confidence interval" not in text


def test_spread_is_labelled_as_min_max_below_40_data_points():
    text = _format(39)
    assert "Minimum to maximum" in text
    assert "percentile range" not in text
