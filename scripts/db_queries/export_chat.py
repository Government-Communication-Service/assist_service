#!/usr/bin/env python3
# ruff: noqa: E501
"""Export a chat's full conversation as browsable HTML.

Connects to a copilot-api Postgres database over an SSH bastion (same connection pattern as
scripts/db_queries/db-connect.sh) and renders every `message` row for a chat (matched by uuid
prefix), joined with the `chat` and `user` tables for context. Token/length stats are visible in
the collapsed summary line; full content / content_enhanced_with_rag / summary text is nested
inside collapsible sections.

The page has a text search box (matches against the rendered text of an entry) and a start/end
datetime filter (matches against each entry's created_at).

To look at llm_internal_response rows from around the same time as this conversation, use
scripts/db_queries/export_llm_internal_responses.py with an explicit time range — that's a
separate script/query because the two tables have no direct FK link, and pulling every
llm_internal_response row in a wide time window is usually too much to page through blindly.

Writes to /tmp/chat_<uuid-prefix>.html by default and opens it in a browser (use --no-open to
skip, or -o to pick a different path).

Usage:
    python scripts/db_queries/export_chat.py <bastion-host> <env> <uuid-prefix> [options]

Example:
    python scripts/db_queries/export_chat.py my-bastion dev 3f2a9c
"""

import argparse
import sys
import webbrowser
from pathlib import Path

from psql_client import DB_NAMES, escape, format_number, iso_ts, run_psql_json

CHAT_LOOKUP_SQL = """
SELECT c.id, c.uuid::text AS uuid, c.title, c.created_at::text AS created_at, c.user_id, c.use_case_id,
       c.use_rag, c.use_gov_uk_search_api, c.use_smart_targets, c.favourite, c.share, c.share_private, c.from_open_chat,
       u.uuid::text AS user_uuid, u.job_title, u.region, u.sector, u.organisation, u.grade, u.communicator_role,
       u.created_at::text AS user_created_at
FROM chat c
JOIN "user" u ON u.id = c.user_id
WHERE c.uuid::text LIKE :'uuid_prefix' || '%'
ORDER BY c.created_at
"""

MESSAGES_SQL_TEMPLATE = """
SELECT id, created_at::text AS created_at, role, tokens, cache_read_tokens, cache_write_tokens,
       length(content) AS content_len,
       length(content_enhanced_with_rag) AS enhanced_len,
       length(summary) AS summary_len,
       content, content_enhanced_with_rag, summary
FROM message
WHERE chat_id = {chat_id}
ORDER BY created_at
"""


def resolve_chat(bastion_host: str, db: str, user: str, uuid_prefix: str) -> dict:
    rows = run_psql_json(bastion_host, db, user, CHAT_LOOKUP_SQL, {"uuid_prefix": uuid_prefix})
    if not rows:
        print(f"No chat found with uuid starting '{uuid_prefix}'", file=sys.stderr)
        sys.exit(1)
    if len(rows) > 1:
        print(f"{len(rows)} chats match uuid prefix '{uuid_prefix}', be more specific:", file=sys.stderr)
        for row in rows:
            print(
                f"  id={row['id']} uuid={row['uuid']} title={row['title']!r} created_at={row['created_at']}",
                file=sys.stderr,
            )
        sys.exit(1)
    return rows[0]


# ---- HTML rendering -------------------------------------------------------

PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Chat export — {title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.3rem; margin-bottom: 0.3rem; }}
  .header {{ border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin-bottom: 1.5rem; background: #fafafa; }}
  .header .meta {{ font-size: 0.85rem; color: #666; margin-bottom: 0.4rem; }}
  .flags {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
  .flag {{ display: flex; align-items: center; gap: 0.35rem; border: 1px solid #ddd; border-radius: 4px; padding: 0.2rem 0.6rem; font-size: 0.78rem; background: #fff; }}
  .flag-label {{ color: #666; }}
  .flag-value {{ font-weight: 600; }}
  .flag-on .flag-value {{ color: #16a34a; }}
  .flag-off .flag-value {{ color: #9ca3af; }}
  .flag-na .flag-value {{ color: #9ca3af; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; margin-bottom: 1rem; position: sticky; top: 0; background: #fff; padding: 0.6rem 0; border-bottom: 1px solid #eee; z-index: 10; }}
  .controls input[type="text"] {{ flex: 1; min-width: 200px; padding: 0.4rem 0.6rem; border: 1px solid #ccc; border-radius: 4px; font-size: 0.9rem; }}
  .controls input[type="datetime-local"] {{ padding: 0.35rem 0.5rem; border: 1px solid #ccc; border-radius: 4px; font-size: 0.85rem; }}
  .controls label {{ font-size: 0.8rem; color: #666; }}
  .toggle-btn {{ font-size: 0.8rem; padding: 0.35rem 0.8rem; border: 1px solid #ccc; border-radius: 4px; background: #fff; cursor: pointer; }}
  .toggle-btn:hover {{ background: #f5f5f5; }}
  details {{ border: 1px solid #ddd; border-radius: 6px; margin-bottom: 0.6rem; }}
  summary {{ padding: 0.5rem 0.9rem; cursor: pointer; background: #f5f5f5; border-radius: 5px; user-select: none; display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: baseline; }}
  details[open] > summary {{ border-bottom: 1px solid #ddd; border-radius: 5px 5px 0 0; }}
  .role-system summary {{ background: #eef2ff; }}
  .role-user summary {{ background: #f0fdf4; }}
  .role-assistant summary {{ background: #fff7ed; }}
  .role-dynamic summary {{ background: #faf5ff; }}
  .idx {{ color: #999; font-size: 0.8rem; }}
  .role-badge {{ font-weight: 600; font-size: 0.85rem; text-transform: uppercase; }}
  .ts {{ font-size: 0.8rem; color: #666; font-family: monospace; }}
  .stats {{ font-size: 0.78rem; color: #888; font-family: monospace; }}
  .body {{ padding: 0.4rem 0.9rem 0.7rem; }}
  .body details {{ margin-top: 0.5rem; }}
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
{header}
<div class="controls">
  <input type="text" id="search-box" placeholder="Search visible text..." oninput="applyFilters()">
  <label>From <input type="datetime-local" id="start-ts" step="1" oninput="applyFilters()"></label>
  <label>To <input type="datetime-local" id="end-ts" step="1" oninput="applyFilters()"></label>
  <button class="toggle-btn" onclick="toggleAll()">Expand/collapse all</button>
</div>
<h2>Conversation <span class="count">({message_count})</span></h2>
<div class="entries">
{message_entries}
</div>
</body>
</html>
"""


def render_flag(label: str, value) -> str:
    if value is None:
        css, text = "flag-na", "n/a"
    elif value:
        css, text = "flag-on", "on"
    else:
        css, text = "flag-off", "off"
    return f'<span class="flag {css}"><span class="flag-label">{escape(label)}</span><span class="flag-value">{text}</span></span>'


def render_header(chat: dict) -> str:
    flags = "".join(
        render_flag(label, chat.get(key))
        for label, key in [
            ("use_rag", "use_rag"),
            ("gov_uk_search", "use_gov_uk_search_api"),
            ("smart_targets", "use_smart_targets"),
            ("favourite", "favourite"),
            ("share", "share"),
            ("share_private", "share_private"),
            ("from_open_chat", "from_open_chat"),
        ]
    )
    user_meta = " &middot; ".join(
        f"{label}={escape(chat.get(key))}"
        for label, key in [
            ("job_title", "job_title"),
            ("region", "region"),
            ("sector", "sector"),
            ("organisation", "organisation"),
            ("grade", "grade"),
            ("communicator_role", "communicator_role"),
        ]
    )
    return f"""\
<div class="header">
  <h1>{escape(chat["title"])}</h1>
  <div class="meta">uuid=<code>{escape(chat["uuid"])}</code> &middot; id={chat["id"]} &middot; use_case_id={chat["use_case_id"]} &middot; created_at={escape(chat["created_at"])}</div>
  <div class="meta">user uuid=<code>{escape(chat["user_uuid"])}</code> (id={chat["user_id"]}, joined {escape(chat["user_created_at"])}) &middot; {user_meta}</div>
  <div class="flags">{flags}</div>
</div>"""


def render_message_entry(msg: dict, idx: int) -> str:
    role = msg.get("role") or "unknown"
    css = f"role-{role}" if role in ("user", "assistant", "system") else "role-dynamic"
    stats = (
        f"tokens={format_number(msg['tokens'])} cache_read={format_number(msg['cache_read_tokens'])} "
        f"cache_write={format_number(msg['cache_write_tokens'])} "
        f"len(content)={format_number(msg['content_len'])} len(rag)={format_number(msg['enhanced_len'])} "
        f"len(summary)={format_number(msg['summary_len'])}"
    )
    body_sections = [
        ("content", msg.get("content")),
        ("content_enhanced_with_rag", msg.get("content_enhanced_with_rag")),
        ("summary", msg.get("summary")),
    ]
    body_html = "".join(
        f"<details><summary>{label}</summary><pre>{escape(text)}</pre></details>" for label, text in body_sections
    )
    return f"""\
<div class="entry {css}" data-ts="{escape(iso_ts(msg["created_at"]))}">
  <details>
    <summary>
      <span class="idx">#{idx}</span>
      <span class="role-badge">{escape(role)}</span>
      <span class="ts">{escape(msg["created_at"])}</span>
      <span class="stats">{escape(stats)}</span>
    </summary>
    <div class="body">{body_html}</div>
  </details>
</div>"""


def build_html(chat: dict, messages: list[dict]) -> str:
    message_html = (
        "\n".join(render_message_entry(m, i + 1) for i, m in enumerate(messages))
        if messages
        else '<div class="empty-note">No messages found for this chat.</div>'
    )
    return PAGE_TEMPLATE.format(
        title=escape(chat["title"]),
        header=render_header(chat),
        message_count=len(messages),
        message_entries=message_html,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bastion_host", help="SSH host/alias for the bastion (as passed to `ssh`)")
    parser.add_argument("env", choices=sorted(DB_NAMES), help="Target environment")
    parser.add_argument("uuid_prefix", help="Full chat uuid, or just the beginning of it")
    parser.add_argument("--postgres-user", default="postgres", help="Postgres user (default: postgres)")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output HTML path (default: /tmp/chat_<uuid_prefix>.html)"
    )
    parser.add_argument("--no-open", action="store_true", help="Don't open the HTML file in a browser when done")
    args = parser.parse_args()

    db = DB_NAMES[args.env]

    print(f"Looking up chat with uuid prefix '{args.uuid_prefix}' in {args.env}...", file=sys.stderr)
    chat = resolve_chat(args.bastion_host, db, args.postgres_user, args.uuid_prefix)
    print(f"Matched chat id={chat['id']} uuid={chat['uuid']} title={chat['title']!r}", file=sys.stderr)

    messages = run_psql_json(
        args.bastion_host, db, args.postgres_user, MESSAGES_SQL_TEMPLATE.format(chat_id=int(chat["id"]))
    )
    print(f"Fetched {len(messages)} messages", file=sys.stderr)
    if messages:
        print(
            f"Conversation spans {messages[0]['created_at']} to {messages[-1]['created_at']} "
            f"— use that range with export_llm_internal_responses.py if you need background LLM calls",
            file=sys.stderr,
        )

    output_path = args.output or Path(f"/tmp/chat_{args.uuid_prefix}.html")
    output_path.write_text(build_html(chat, messages))
    print(f"Written to {output_path}", file=sys.stderr)

    if not args.no_open:
        webbrowser.open(f"file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
