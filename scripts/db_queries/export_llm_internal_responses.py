#!/usr/bin/env python3
# ruff: noqa: E501
"""Export llm_internal_response rows in a given time range as browsable HTML.

Connects to a copilot-api Postgres database over an SSH bastion (same connection pattern as
scripts/db_queries/db-connect.sh). There's no direct FK from llm_internal_response to chat or
message, so this is a standalone, time-range-scoped query rather than something bundled into
export_chat.py — pulling every row in a wide window there was unusable, so pick a tight
start/end after you've looked at the conversation (export_chat.py prints a suggested range
based on the chat's own message timestamps).

The page has a text search box and a start/end datetime filter for narrowing further within the
fetched rows, pre-filled with the range you queried.

Writes to /tmp/llm_internal_responses_<start>_<end>.html by default and opens it in a browser
(use --no-open to skip, or -o to pick a different path).

Usage:
    python scripts/db_queries/export_llm_internal_responses.py <bastion-host> <env> <start> <end> [options]

Example:
    python scripts/db_queries/export_llm_internal_responses.py my-bastion dev \\
        "2026-09-15 14:00:00" "2026-09-15 14:05:00"
"""

import argparse
import ast
import json
import re
import sys
import webbrowser
from pathlib import Path

from psql_client import DB_NAMES, escape, format_number, iso_ts, run_psql_json

LLM_INTERNAL_SQL_TEMPLATE = """
SELECT r.id, r.created_at::text AS created_at, r.llm_id, l.model, l.provider,
       r.tokens_in, r.tokens_out, r.cache_read_tokens, r.cache_write_tokens,
       r.completion_cost::text AS completion_cost,
       length(r.content) AS content_len,
       r.content
FROM llm_internal_response r
LEFT JOIN llm l ON l.id = r.llm_id
WHERE r.created_at BETWEEN :'start_ts' AND :'end_ts'
ORDER BY r.created_at
"""


# ---- content parsing -------------------------------------------------------
#
# llm_internal_response.content is never JSON. Depending on the call site it's one of:
#   - plain prose text (e.g. a compaction summary)
#   - str(python_dict_or_bool), e.g. "{'is_relevant': True}" or "True"
#   - str(list_of_anthropic_sdk_blocks), e.g.
#     "[TextBlock(text='...', type='text'), ToolUseBlock(id='...', input={...}, name='...', type='tool_use', ...)]"
# The last form contains constructor-call syntax (ToolUseBlock(...)), which ast.literal_eval
# rejects outright, so we parse the whole thing with ast.parse(mode="eval") and only literal_eval
# each keyword's *value* subtree (those are always plain literals: str/dict/list/bool/None).


def parse_content(content: str | None) -> dict:
    """Best-effort parse of a content string into {"kind": "blocks"|"literal"|"text", "value": ...}."""
    if not content:
        return {"kind": "text", "value": content or ""}
    try:
        tree = ast.parse(content, mode="eval")
    except SyntaxError:
        return {"kind": "text", "value": content}

    body = tree.body
    if isinstance(body, ast.List) and body.elts and all(isinstance(e, ast.Call) for e in body.elts):
        blocks = []
        try:
            for call in body.elts:
                if not isinstance(call.func, ast.Name):
                    raise ValueError("not a simple constructor call")
                fields = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}
                fields["_block_type"] = call.func.id
                blocks.append(fields)
        except (ValueError, TypeError):
            return {"kind": "text", "value": content}
        return {"kind": "blocks", "value": blocks}

    try:
        literal = ast.literal_eval(body)
    except (ValueError, TypeError):
        return {"kind": "text", "value": content}
    return {"kind": "literal", "value": literal}


def extract_tool_names(parsed: dict) -> list[str]:
    if parsed["kind"] != "blocks":
        return []
    return [b["name"] for b in parsed["value"] if b.get("type") == "tool_use" and b.get("name")]


def render_content_body(parsed: dict) -> str:
    if parsed["kind"] == "text":
        return f"<pre>{escape(parsed['value'])}</pre>"
    if parsed["kind"] == "literal":
        return f"<pre>{escape(json.dumps(parsed['value'], indent=2, default=str))}</pre>"

    parts = []
    for block in parsed["value"]:
        btype = block.get("type", block.get("_block_type", "block"))
        if btype == "tool_use":
            pretty = json.dumps(block.get("input"), indent=2, default=str)
            parts.append(
                f'<div class="block-label">tool_use: {escape(block.get("name", "?"))}</div><pre>{escape(pretty)}</pre>'
            )
        elif btype == "text":
            text = block.get("text", "")
            if text:
                parts.append(f"<pre>{escape(text)}</pre>")
        else:
            dump = {k: v for k, v in block.items() if k != "_block_type"}
            parts.append(
                f'<div class="block-label">{escape(btype)}</div><pre>{escape(json.dumps(dump, indent=2, default=str))}</pre>'
            )
    return "\n".join(parts) if parts else f"<pre>{escape(str(parsed['value']))}</pre>"


# ---- HTML rendering -------------------------------------------------------

PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>llm_internal_response export — {start_ts} to {end_ts}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.3rem; margin-bottom: 0.3rem; }}
  .header {{ border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin-bottom: 1.5rem; background: #fafafa; }}
  .header .meta {{ font-size: 0.85rem; color: #666; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; margin-bottom: 1rem; position: sticky; top: 0; background: #fff; padding: 0.6rem 0; border-bottom: 1px solid #eee; z-index: 10; }}
  .controls input[type="text"] {{ flex: 1; min-width: 200px; padding: 0.4rem 0.6rem; border: 1px solid #ccc; border-radius: 4px; font-size: 0.9rem; }}
  .controls input[type="datetime-local"] {{ padding: 0.35rem 0.5rem; border: 1px solid #ccc; border-radius: 4px; font-size: 0.85rem; }}
  .controls label {{ font-size: 0.8rem; color: #666; }}
  .toggle-btn {{ font-size: 0.8rem; padding: 0.35rem 0.8rem; border: 1px solid #ccc; border-radius: 4px; background: #fff; cursor: pointer; }}
  .toggle-btn:hover {{ background: #f5f5f5; }}
  details {{ border: 1px solid #ddd; border-radius: 6px; margin-bottom: 0.6rem; }}
  summary {{ padding: 0.5rem 0.9rem; cursor: pointer; background: #f5f5f5; border-radius: 5px; user-select: none; display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: baseline; }}
  details[open] > summary {{ border-bottom: 1px solid #ddd; border-radius: 5px 5px 0 0; }}
  .idx {{ color: #999; font-size: 0.8rem; }}
  .model-badge {{ font-weight: 600; font-size: 0.85rem; }}
  .ts {{ font-size: 0.8rem; color: #666; font-family: monospace; }}
  .stats {{ font-size: 0.78rem; color: #888; font-family: monospace; }}
  .tool-name {{ margin-left: auto; font-weight: 700; text-align: right; }}
  .body {{ padding: 0.4rem 0.9rem 0.7rem; }}
  .block-label {{ font-size: 0.78rem; color: #6366f1; font-weight: 600; padding: 0.5rem 0 0.2rem; }}
  pre {{ margin: 0; padding: 0.8rem; white-space: pre-wrap; word-break: break-word; font-size: 0.85rem; line-height: 1.6; }}
  .entry {{ margin-bottom: 0.6rem; }}
  .empty-note {{ color: #888; font-style: italic; padding: 1rem 0; }}
  .count {{ color: #888; font-weight: normal; }}
</style>
<script>
  function applyFilters() {{
    const q = document.getElementById('search-box').value.toLowerCase();
    const startVal = document.getElementById('start-ts').value;
    const endVal = document.getElementById('end-ts').value;
    document.querySelectorAll('.entry').forEach(function(entry) {{
      const ts = entry.dataset.ts || '';
      let visible = true;
      if (q && !entry.textContent.toLowerCase().includes(q)) visible = false;
      if (visible && startVal && ts < startVal) visible = false;
      if (visible && endVal && ts > endVal) visible = false;
      entry.style.display = visible ? '' : 'none';
    }});
  }}
  function toggleAll() {{
    const all = document.querySelectorAll('.entries details');
    const anyOpen = Array.from(all).some(d => d.open);
    all.forEach(d => d.open = !anyOpen);
  }}
</script>
</head>
<body>
<div class="header">
  <h1>llm_internal_response export</h1>
  <div class="meta">queried range: {start_ts} &ndash; {end_ts}</div>
</div>
<div class="controls">
  <input type="text" id="search-box" placeholder="Search visible text..." oninput="applyFilters()">
  <label>From <input type="datetime-local" id="start-ts" step="1" value="{start_ts_iso}" oninput="applyFilters()"></label>
  <label>To <input type="datetime-local" id="end-ts" step="1" value="{end_ts_iso}" oninput="applyFilters()"></label>
  <button class="toggle-btn" onclick="toggleAll()">Expand/collapse all</button>
</div>
<h2>Rows <span class="count">({row_count})</span></h2>
<div class="entries">
{entries}
</div>
</body>
</html>
"""


def render_entry(entry: dict, idx: int) -> str:
    stats = (
        f"llm_id={entry['llm_id']} tokens_in={format_number(entry['tokens_in'])} "
        f"tokens_out={format_number(entry['tokens_out'])} "
        f"cache_read={format_number(entry['cache_read_tokens'])} cache_write={format_number(entry['cache_write_tokens'])} "
        f"cost={entry['completion_cost']} len(content)={format_number(entry['content_len'])}"
    )
    model_label = entry.get("model") or f"llm_id={entry['llm_id']}"
    parsed = parse_content(entry.get("content"))
    tool_names = ", ".join(extract_tool_names(parsed))
    tool_name_html = f'<span class="tool-name">{escape(tool_names)}</span>' if tool_names else ""
    return f"""\
<div class="entry" data-ts="{escape(iso_ts(entry["created_at"]))}">
  <details>
    <summary>
      <span class="idx">#{idx}</span>
      <span class="model-badge">{escape(model_label)}</span>
      <span class="ts">{escape(entry["created_at"])}</span>
      <span class="stats">{escape(stats)}</span>
      {tool_name_html}
    </summary>
    <div class="body">{render_content_body(parsed)}</div>
  </details>
</div>"""


def build_html(start_ts: str, end_ts: str, entries: list[dict]) -> str:
    entries_html = (
        "\n".join(render_entry(e, i + 1) for i, e in enumerate(entries))
        if entries
        else '<div class="empty-note">No llm_internal_response rows found in this time range.</div>'
    )
    return PAGE_TEMPLATE.format(
        start_ts=escape(start_ts),
        end_ts=escape(end_ts),
        start_ts_iso=escape(iso_ts(start_ts)),
        end_ts_iso=escape(iso_ts(end_ts)),
        row_count=len(entries),
        entries=entries_html,
    )


def slugify_ts(ts: str) -> str:
    return re.sub(r"[^0-9]", "", ts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bastion_host", help="SSH host/alias for the bastion (as passed to `ssh`)")
    parser.add_argument("env", choices=sorted(DB_NAMES), help="Target environment")
    parser.add_argument(
        "start", help="Range start, anything Postgres can cast to a timestamp, e.g. '2026-09-15 14:00:00'"
    )
    parser.add_argument("end", help="Range end, same format as start")
    parser.add_argument("--postgres-user", default="postgres", help="Postgres user (default: postgres)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: /tmp/llm_internal_responses_<start>_<end>.html)",
    )
    parser.add_argument("--no-open", action="store_true", help="Don't open the HTML file in a browser when done")
    args = parser.parse_args()

    db = DB_NAMES[args.env]

    print(
        f"Fetching llm_internal_response rows between {args.start!r} and {args.end!r} in {args.env}...", file=sys.stderr
    )
    entries = run_psql_json(
        args.bastion_host,
        db,
        args.postgres_user,
        LLM_INTERNAL_SQL_TEMPLATE,
        {"start_ts": args.start, "end_ts": args.end},
    )
    print(f"Fetched {len(entries)} rows", file=sys.stderr)

    output_path = args.output or Path(
        f"/tmp/llm_internal_responses_{slugify_ts(args.start)}_{slugify_ts(args.end)}.html"
    )
    output_path.write_text(build_html(args.start, args.end, entries))
    print(f"Written to {output_path}", file=sys.stderr)

    if not args.no_open:
        webbrowser.open(f"file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
