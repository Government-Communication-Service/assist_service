#!/usr/bin/env python3
# ruff: noqa: E501
"""Pretty-print a full LLM invocation log as collapsible HTML.

Usage:
    python scripts/pretty_print_invocation.py <path-to-log.json> [output.html]

If output path is omitted, writes to invocation.html in the same directory as the input.
"""

import json
import re
import sys
from pathlib import Path

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LLM Invocation</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.2rem; color: #555; margin-bottom: 1.5rem; }}
  details {{ border: 1px solid #ddd; border-radius: 6px; margin-bottom: 1rem; }}
  summary {{ padding: 0.6rem 1rem; cursor: pointer; font-weight: 600; background: #f5f5f5; border-radius: 5px; user-select: none; }}
  details[open] summary {{ border-bottom: 1px solid #ddd; border-radius: 5px 5px 0 0; }}
  .role-system summary {{ background: #eef2ff; }}
  .role-user   summary {{ background: #f0fdf4; }}
  .role-assistant summary {{ background: #fff7ed; }}
  .role-dynamic summary {{ background: #faf5ff; }}
  pre {{ margin: 0; padding: 1rem; white-space: pre-wrap; word-break: break-word; font-size: 0.85rem; line-height: 1.6; }}
  .meta {{ font-size: 0.8rem; color: #888; padding: 0.4rem 1rem 0.8rem; }}
  .usage {{ background: #f9f9f9; border: 1px solid #ddd; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; display: flex; flex-wrap: wrap; gap: 1rem; }}
  .usage-item {{ display: flex; flex-direction: column; }}
  .usage-label {{ font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }}
  .usage-value {{ font-weight: 600; font-size: 1rem; }}
  .usage-value.cached {{ color: #6366f1; }}
  .usage-value.created {{ color: #f59e0b; }}
  .toggle-btn {{ font-size: 0.8rem; padding: 0.3rem 0.8rem; border: 1px solid #ccc; border-radius: 4px; background: #fff; cursor: pointer; margin-bottom: 1rem; }}
  .toggle-btn:hover {{ background: #f5f5f5; }}
</style>
<script>
  function toggleAll() {{
    const all = document.querySelectorAll('details');
    const anyOpen = Array.from(all).some(d => d.open);
    all.forEach(d => d.open = !anyOpen);
    document.getElementById('toggle-btn').textContent = anyOpen ? 'Expand all' : 'Collapse all';
  }}
</script>
</head>
<body>
<h1>LLM Invocation &mdash; <code>{model}</code></h1>
<button id="toggle-btn" class="toggle-btn" onclick="toggleAll()">Expand all</button>
{sections}
</body>
</html>
"""


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_section(label: str, text: str, css_class: str = "", extra_meta: str = "") -> str:
    meta = f'<div class="meta">{extra_meta}</div>' if extra_meta else ""
    return f'<details class="{css_class}"><summary>{escape(label)}</summary>{meta}<pre>{escape(text)}</pre></details>'


def render_system(system) -> str:
    if isinstance(system, str):
        return render_section("System prompt", system, "role-system")
    if isinstance(system, list):
        parts = []
        for i, block in enumerate(system):
            text = block.get("text", json.dumps(block, indent=2))
            cached = block.get("cache_control") is not None
            label = f"System block {i + 1}" + (" — \U0001f4be cached" if cached else "")
            css = "role-system" if i == 0 else "role-dynamic"
            parts.append(render_section(label, text, css))
        return "\n".join(parts)
    return ""


def render_usage(usage: dict) -> str:
    def item(label, value, css=""):
        return f'<div class="usage-item"><span class="usage-label">{label}</span><span class="usage-value {css}">{value if value is not None else "—"}</span></div>'

    input_tokens = usage.get("input_tokens") or 0
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_created = usage.get("cache_creation_input_tokens") or 0
    total_input = input_tokens + cache_read + cache_created

    parts = [
        item("Total input tokens", total_input),
        item("Input tokens", input_tokens),
        item("Output tokens", usage.get("output_tokens")),
        item("Cache read", cache_read, "cached"),
        item("Cache created", cache_created, "created"),
    ]
    cache_creation = usage.get("cache_creation")
    if isinstance(cache_creation, dict):
        for k, v in cache_creation.items():
            parts.append(item(k.replace("_", " "), v))
    return f'<div class="usage">{"".join(parts)}</div>'


def render_user_content(text: str, depth: int = 0) -> str:
    """Render user message content, turning top-level XML tags into collapsible sections."""
    indent = f"margin-left:{depth * 1}rem"
    parts = []
    last = 0
    for m in re.finditer(r"<([a-zA-Z][a-zA-Z0-9_-]*)(\s[^>]*)?>(.+?)</\1>", text, re.DOTALL):
        before = text[last : m.start()].strip()
        if before:
            parts.append(f'<pre style="{indent}">{escape(before)}</pre>')
        tag, _, body = m.group(1), m.group(2), m.group(3)
        parts.append(
            f'<details style="{indent}"><summary style="background:#e0f2fe;padding:0.4rem 0.8rem;border-radius:4px;cursor:pointer;font-weight:600">'
            f"&lt;{escape(tag)}&gt;</summary>"
            f"{render_user_content(body.strip(), depth + 1)}</details>"
        )
        last = m.end()
    remainder = text[last:].strip()
    if remainder:
        parts.append(f'<pre style="{indent}">{escape(remainder)}</pre>')
    return "\n".join(parts) if parts else f'<pre style="{indent}">{escape(text)}</pre>'


def render_messages(messages: list) -> str:
    parts = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n\n".join(
                block.get("text", json.dumps(block, indent=2)) for block in content if isinstance(block, dict)
            )
        content = str(content)
        label = f"Message {i + 1} — {role}"
        css = f"role-{role}" if role in ("user", "assistant") else ""
        if role == "user":
            inner = render_user_content(content)
            parts.append(f'<details class="{css}"><summary>{escape(label)}</summary>{inner}</details>')
        else:
            parts.append(render_section(label, content, css))
    return "\n".join(parts)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".html")

    with open(input_path) as f:
        data = json.load(f)

    model = data.get("model", "unknown")
    system = data.get("system")
    messages = data.get("messages", [])
    usage = data.get("usage")
    extra = data.get("extra")

    sections = []
    if usage:
        sections.append(render_usage(usage))
    if system:
        sections.append(render_system(system))
    if extra:
        sections.append(render_section("Extra kwargs (tools, tool_choice, …)", json.dumps(extra, indent=2)))
    sections.append(render_messages(messages))
    response = data.get("response")
    if response:
        text = "\n\n".join(b.get("text", "") for b in response if b.get("text"))
        sections.append(render_section("Response", text, "role-assistant"))

    html = HTML_TEMPLATE.format(model=escape(model), sections="\n".join(sections))
    output_path.write_text(html)
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
