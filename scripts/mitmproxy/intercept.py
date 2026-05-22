"""
mitmproxy addon that intercepts requests matching patterns in mocks.json and
returns configured mock responses. Everything else is forwarded to the real API.

mocks.json keys are "<METHOD> <path-regex>" strings.
"""

import json
import re
from pathlib import Path

from mitmproxy import http

MOCKS_FILE = Path(__file__).parent / "mocks.json"

_rules: list[tuple[str, re.Pattern, dict]] = []
_mtime: float = 0.0


def _reload_if_changed() -> None:
    global _rules, _mtime
    current_mtime = MOCKS_FILE.stat().st_mtime
    if current_mtime == _mtime:
        return
    try:
        raw = json.loads(MOCKS_FILE.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"[mitmproxy] mocks.json is invalid JSON: {e}") from e
    rules = []
    for pattern, config in raw.items():
        method, path_pattern = pattern.split(" ", 1)
        rules.append((method.upper(), re.compile(f"^{path_pattern}$"), config))
    _rules = rules
    _mtime = current_mtime


def request(flow: http.HTTPFlow) -> None:
    _reload_if_changed()
    method = flow.request.method.upper()
    path = flow.request.path

    for rule_method, path_re, config in _rules:
        if rule_method != method:
            continue
        if not path_re.match(path):
            continue

        status = config.get("status", 200)
        body = config.get("body", {})
        headers = config.get("headers", {})

        response_body = json.dumps(body).encode()
        response_headers = {"Content-Type": "application/json", **headers}

        flow.response = http.Response.make(
            status,
            response_body,
            response_headers,
        )
        return
