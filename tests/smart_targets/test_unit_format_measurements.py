"""Unit tests for SmartTargetsService._format_measurements_response (no API calls)."""

import pytest

from app.smart_targets.service import SmartTargetsService, _sig_figs


def _format(n_measurements, unit="percentage"):
    service = SmartTargetsService()
    summary = {
        "metric": {"unit": unit},
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


def test_unknown_unit_is_left_out():
    text = _format(100, unit="unknown")
    assert "unknown" not in text
    assert "- Median: 50\n" in text
    assert "- Lower quartile: 40\n" in text


def test_known_unit_is_shown():
    text = _format(100)
    assert "(Impressions, unit=percentage)" in text
    assert "- Median: 50 percentage" in text


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.8299999999999998, 0.83),
        (0.0012000000000000001, 0.0012),
        (0.1382872975591805, 0.138),
        (1234567, 1230000),
        (37366422.0, 37400000),
        (50, 50),
        (0, 0),
        ("N/A", "N/A"),
    ],
)
def test_sig_figs(value, expected):
    assert _sig_figs(value) == expected
    assert "e" not in str(_sig_figs(value)).lower().replace("n/a", "")


def test_statistics_are_rounded():
    service = SmartTargetsService()
    summary = {
        "metric": {"unit": "unknown"},
        "summary_statistics": {
            "n_campaigns": 354,
            "n_measurements": 354,
            "median": 0.1382872975591805,
            "lower_quartile": 0.30999999999999994,
            "upper_quartile": 0.56,
            "confidence_interval_95": [0.0012000000000000001, 0.8299999999999998],
        },
    }
    text = service._format_measurements_response(summary, selected_metric_name="ctr")
    assert "- Median: 0.138\n" in text
    assert "- Lower quartile: 0.31\n" in text
    assert ": 0.0012 to 0.83" in text
