"""
Tests for the chat title evaluation script.

Run with:
    uv run --no-project tests/evals/chat_titles/test_evaluate_chat_titles.py
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx",
#   "pydantic-settings",
#   "rich",
#   "typer",
#   "pytest",
# ]
# ///

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

# Allow importing the eval script by adding its directory to sys.path
_this = Path(__file__).resolve()
_repo = next(p for p in _this.parents if (p / "scripts" / "chat_titles").is_dir())
sys.path.insert(0, str(_repo / "scripts" / "chat_titles"))

from evaluate_chat_titles import (  # noqa: E402
    TitleResult,
    check_title,
    evaluate_titles,
    render_json_output,
    render_markdown,
)

# ---------------------------------------------------------------------------
# check_title
# ---------------------------------------------------------------------------


class TestCheckTitle:
    def test_good_title_passes(self):
        assert check_title("Budget writing assistance") == []

    def test_sentence_case_with_acronym_passes(self):
        assert check_title("OASIS plan for NHS targets") == []

    def test_first_word_lowercase_warns(self):
        issues = check_title("budget writing assistance")
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"
        assert "sentence case" in issues[0]["label"].lower()

    def test_mid_word_capitalised_warns(self):
        issues = check_title("Budget Writing assistance")
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"

    def test_trailing_full_stop_errors(self):
        issues = check_title("Budget writing assistance.")
        assert any(i["label"] == "Trailing full stop" for i in issues)
        assert any(i["severity"] == "error" for i in issues)

    def test_quite_long_warns(self):
        title = " ".join(["word"] * 8)
        issues = check_title(title)
        assert any("Quite long" in i["label"] for i in issues)
        assert all(i["severity"] == "warning" for i in issues)

    def test_too_long_errors(self):
        title = " ".join(["word"] * 12)
        issues = check_title(title)
        assert any("Too long" in i["label"] for i in issues)
        assert any(i["severity"] == "error" for i in issues)

    def test_refusal_detected(self):
        issues = check_title("Unable to generate title")
        labels = [i["label"] for i in issues]
        assert "Refusal" in labels

    def test_multiple_issues(self):
        # Lowercase first word + trailing full stop + refusal
        issues = check_title("unable to generate title.")
        labels = {i["label"] for i in issues}
        assert "Trailing full stop" in labels
        assert "Refusal" in labels
        assert any("sentence case" in label.lower() for label in labels)

    def test_single_word_passes(self):
        assert check_title("Help") == []

    def test_single_word_lowercase_warns(self):
        issues = check_title("help")
        assert len(issues) == 1
        assert "sentence case" in issues[0]["label"].lower()


# ---------------------------------------------------------------------------
# evaluate_titles
# ---------------------------------------------------------------------------


def _make_response(status_code: int = 200, title: str = "Good title") -> httpx.Response:
    """Build a fake httpx.Response with the given status and JSON body."""
    return httpx.Response(
        status_code=status_code,
        json={"title": title},
        request=httpx.Request("PUT", "http://fake"),
    )


class TestEvaluateTitles:
    def test_successful_response(self):
        mock_response = _make_response(title="Budget writing assistance")

        with patch("evaluate_chat_titles.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = asyncio.run(evaluate_titles(["Help me write"], "http://fake/title", {"Auth-Token": "t"}))

        assert len(results) == 1
        assert results[0]["title"] == "Budget writing assistance"
        assert results[0]["passed"] is True
        assert results[0]["error"] is None
        assert isinstance(results[0]["duration_ms"], float)

    def test_http_error_response(self):
        mock_response = _make_response(status_code=500, title="")

        with patch("evaluate_chat_titles.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = asyncio.run(evaluate_titles(["Help"], "http://fake/title", {}))

        assert results[0]["error"] == "HTTP 500"
        assert results[0]["passed"] is False
        assert results[0]["title"] is None

    def test_exception_during_request(self):
        with patch("evaluate_chat_titles.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = asyncio.run(evaluate_titles(["Help"], "http://fake/title", {}))

        assert results[0]["error"] == "ConnectError"
        assert results[0]["passed"] is False
        assert isinstance(results[0]["duration_ms"], float)

    def test_multiple_prompts(self):
        responses = [
            _make_response(title="Good title"),
            _make_response(status_code=422, title=""),
            _make_response(title="Another good title"),
        ]
        call_count = 0

        async def fake_put(*_args, **_kwargs):
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        with patch("evaluate_chat_titles.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(side_effect=fake_put)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = asyncio.run(evaluate_titles(["a", "b", "c"], "http://fake/title", {}))

        assert len(results) == 3
        assert results[0]["index"] == 1
        assert results[1]["index"] == 2
        assert results[2]["index"] == 3
        assert results[0]["passed"] is True
        assert results[1]["error"] == "HTTP 422"
        assert results[2]["passed"] is True

    def test_title_with_quality_issues(self):
        mock_response = _make_response(title="unable to generate title.")

        with patch("evaluate_chat_titles.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = asyncio.run(evaluate_titles(["Hi"], "http://fake/title", {}))

        assert results[0]["passed"] is False
        assert results[0]["title"] == "unable to generate title."
        assert len(results[0]["issues"]) >= 2

    def test_duration_is_recorded(self):
        mock_response = _make_response(title="Fine")

        with patch("evaluate_chat_titles.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.put = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = asyncio.run(evaluate_titles(["Hi"], "http://fake/title", {}))

        assert results[0]["duration_ms"] is not None
        assert results[0]["duration_ms"] >= 0


# ---------------------------------------------------------------------------
# render_json_output
# ---------------------------------------------------------------------------


class TestRenderJsonOutput:
    def _make_results(self) -> list[TitleResult]:
        return [
            {
                "index": 1,
                "prompt": "Help",
                "title": "Help with comms",
                "error": None,
                "issues": [],
                "passed": True,
                "duration_ms": 100.0,
            },
            {
                "index": 2,
                "prompt": "Draft something",
                "title": None,
                "error": "HTTP 500",
                "issues": [],
                "passed": False,
                "duration_ms": 50.0,
            },
        ]

    def test_json_structure(self, capsys):
        render_json_output(self._make_results())
        output = json.loads(capsys.readouterr().out)
        assert output["total"] == 2
        assert output["passed"] == 1
        assert output["errors"] == 1
        assert output["quality_failures"] == 0
        assert len(output["results"]) == 2

    def test_json_includes_duration(self, capsys):
        render_json_output(self._make_results())
        output = json.loads(capsys.readouterr().out)
        assert output["results"][0]["duration_ms"] == 100.0


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def _make_results(self) -> list[TitleResult]:
        return [
            {
                "index": 1,
                "prompt": "Help",
                "title": "Help with comms",
                "error": None,
                "issues": [],
                "passed": True,
                "duration_ms": 1234.0,
            },
            {
                "index": 2,
                "prompt": "Prompt with | pipe",
                "title": "Title with | pipe",
                "error": None,
                "issues": [{"label": "Possibly not sentence case", "severity": "warning"}],
                "passed": False,
                "duration_ms": 2345.0,
            },
        ]

    def test_markdown_table_header(self, capsys):
        render_markdown(self._make_results(), Path("test.json"))
        output = capsys.readouterr().out
        assert "| # | Prompt | Title | Duration (ms) | Issues | Passed |" in output

    def test_markdown_contains_data(self, capsys):
        render_markdown(self._make_results(), Path("test.json"))
        output = capsys.readouterr().out
        assert "Help with comms" in output
        assert "1234" in output

    def test_markdown_escapes_pipes(self, capsys):
        render_markdown(self._make_results(), Path("test.json"))
        output = capsys.readouterr().out
        assert "Prompt with \\| pipe" in output
        assert "Title with \\| pipe" in output

    def test_markdown_summary_line(self, capsys):
        render_markdown(self._make_results(), Path("test.json"))
        output = capsys.readouterr().out
        assert "**Summary**" in output
        assert "1 passed" in output
        assert "Average duration: 1790 ms" in output

    def test_markdown_shows_issues(self, capsys):
        render_markdown(self._make_results(), Path("test.json"))
        output = capsys.readouterr().out
        assert "Possibly not sentence case" in output

    def test_markdown_title_from_filename(self, capsys):
        render_markdown(self._make_results(), Path("/some/path/my_prompts.json"))
        output = capsys.readouterr().out
        assert "my_prompts.json" in output


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v", "--noconftest", "--rootdir", str(_this.parent), "-c", "/dev/null"]))
