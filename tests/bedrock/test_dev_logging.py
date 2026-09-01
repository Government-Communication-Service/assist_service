"""Tests for the dev-only full-invocation file logging helpers."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.bedrock import dev_logging


def make_response(content, usage=None):
    return SimpleNamespace(
        content=content,
        usage=usage or SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0),
    )


def make_block(**model_dump_fields):
    block = MagicMock()
    block.model_dump.return_value = model_dump_fields
    return block


class TestLogInvocationToFile:
    def test_returns_none_when_logging_disabled(self, monkeypatch):
        monkeypatch.setattr(dev_logging, "LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH", "")
        path = dev_logging.log_invocation_to_file("model", "system", [{"role": "user", "content": "hi"}])
        assert path is None

    def test_writes_request_payload_to_file(self, monkeypatch, tmp_path):
        base = tmp_path / "invocation.json"
        monkeypatch.setattr(dev_logging, "LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH", str(base))

        path = dev_logging.log_invocation_to_file(
            "claude-haiku",
            "system prompt",
            [{"role": "user", "content": "hello"}],
            extra={"tools": [{"name": "some_tool"}]},
        )

        payload = json.loads(Path(path).read_text())
        assert payload["model"] == "claude-haiku"
        assert payload["system"] == "system prompt"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["extra"] == {"tools": [{"name": "some_tool"}]}

    def test_omits_extra_key_when_not_provided(self, monkeypatch, tmp_path):
        base = tmp_path / "invocation.json"
        monkeypatch.setattr(dev_logging, "LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH", str(base))

        path = dev_logging.log_invocation_to_file("model", None, [])

        payload = json.loads(Path(path).read_text())
        assert "extra" not in payload

    def test_each_call_gets_its_own_numbered_file(self, monkeypatch, tmp_path):
        base = tmp_path / "invocation.json"
        monkeypatch.setattr(dev_logging, "LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH", str(base))

        path_a = dev_logging.log_invocation_to_file("model", None, [])
        path_b = dev_logging.log_invocation_to_file("model", None, [])

        assert path_a != path_b
        assert Path(path_a).exists()
        assert Path(path_b).exists()

    def test_creates_parent_directory_if_missing(self, monkeypatch, tmp_path):
        base = tmp_path / "nested" / "dir" / "invocation.json"
        monkeypatch.setattr(dev_logging, "LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH", str(base))

        path = dev_logging.log_invocation_to_file("model", None, [])

        assert Path(path).exists()


class TestLogResponseToFile:
    def test_noop_when_path_is_none(self):
        # Should not raise even though response.usage/content would fail if accessed.
        dev_logging.log_response_to_file(response=object(), path=None)

    def test_noop_when_file_does_not_exist(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        response = make_response(content=[])
        # Should not raise.
        dev_logging.log_response_to_file(response, str(missing))

    def test_appends_usage_and_response_to_existing_file(self, tmp_path):
        path = tmp_path / "invocation_1.json"
        path.write_text(json.dumps({"model": "m", "system": "s", "messages": []}))

        response = make_response(
            content=[make_block(type="text", text="hello")],
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                cache_read_input_tokens=5,
                cache_creation_input_tokens=0,
                cache_creation=None,
            ),
        )

        dev_logging.log_response_to_file(response, str(path))

        payload = json.loads(path.read_text())
        assert payload["usage"] == {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 5,
            "cache_creation_input_tokens": 0,
            "cache_creation": None,
        }
        assert payload["response"] == [{"type": "text", "text": "hello"}]

    def test_preserves_tool_use_block_input(self, tmp_path):
        """Regression: tool_use blocks must keep their full input, not just .text."""
        path = tmp_path / "invocation_1.json"
        path.write_text(json.dumps({"model": "m", "system": "s", "messages": []}))

        response = make_response(
            content=[
                make_block(
                    type="tool_use",
                    name="evaluate_index_relevance",
                    input={"requires_index": True, "reasoning": "because"},
                )
            ]
        )

        dev_logging.log_response_to_file(response, str(path))

        payload = json.loads(path.read_text())
        assert payload["response"] == [
            {
                "type": "tool_use",
                "name": "evaluate_index_relevance",
                "input": {"requires_index": True, "reasoning": "because"},
            }
        ]
