"""
Evaluate chat title generation across a range of prompts via the Assist API.

Quality criteria checked on each title:
    - Sentence case: only the first word capitalised (acronyms excepted)
    - No trailing full stop
    - Not a refusal or error message
    - Concise: ideally ≤6 words (warning), definitely ≤10 words (error)

The API must be running at the specified base URL (default: localhost:5312).
Install uv at https://docs.astral.sh/uv/getting-started/installation/

Usage:
    # Pretty output (default):
    uv run scripts/chat_titles/evaluate_chat_titles.py

    # JSON output:
    uv run scripts/chat_titles/evaluate_chat_titles.py --json

    # Markdown table output:
    uv run scripts/chat_titles/evaluate_chat_titles.py --md

    # Custom prompts file:
    uv run scripts/chat_titles/evaluate_chat_titles.py \\
        --prompts-file scripts/chat_titles/prompts.json

    # Auth via CLI flags:
    uv run scripts/chat_titles/evaluate_chat_titles.py \\
        --user-uuid abc123 \\
        --chat-uuid def456 \\
        --auth-token mytoken \\
        --session-auth mysession

    # Auth via environment (or .env / .env.titles):
    export USER_UUID=abc123
    export CHAT_UUID=def456
    export AUTH_TOKEN=mytoken
    export SESSION_AUTH=mysession
    uv run scripts/chat_titles/evaluate_chat_titles.py

Module usage:
    import asyncio
    from scripts.chat_titles.evaluate_chat_titles import evaluate_titles, check_title

    results = asyncio.run(evaluate_titles(prompts, url, headers))
    for r in results:
        print(r["title"], r["issues"])
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx",
#   "pydantic-settings",
#   "rich",
#   "typer",
# ]
# ///

# ---------------------------------------------------------------------------
# region Imports and constants
# ---------------------------------------------------------------------------

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import httpx
import typer
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DEFAULT_PROMPTS_FILE = Path(__file__).parent / "title_evaluation_prompts.json"

REFUSAL_PHRASES = (
    "title bot",
    "bot",
    "unable to",
    "generate title",
    "no document attached",
    "no document provided",
    "insufficient information",
    "cannot generate",
    "i don't see any attached document",
    "document content unclear",
    "provide more information",
    "provide more details",
    "missing document",
)

# endregion
# ---------------------------------------------------------------------------
# region Data types
# ---------------------------------------------------------------------------


class Issue(TypedDict):
    label: str
    severity: Literal["warning", "error"]


class TitleResult(TypedDict):
    index: int
    prompt: str
    title: str | None
    error: str | None
    issues: list[Issue]
    passed: bool
    duration_ms: float | None


# endregion
# ---------------------------------------------------------------------------
# region Core module logic
# ---------------------------------------------------------------------------


def check_title(title: str) -> list[Issue]:
    """Check a generated title against quality criteria.

    Returns a list of Issue dicts, each with a human-readable label and
    a severity of 'warning' or 'error'. An empty list means the title passes.

    Example:
    >>> check_title("The impact of the new policy on education")
    []
    >>> check_title("The Impact of the New Policy on Education")
    [{'label': 'Possibly not sentence case', 'severity': 'warning'}]
    """
    issues: list[Issue] = []
    words = title.split()

    if words and not words[0][0].isupper():
        issues.append({"label": "Possibly not sentence case", "severity": "warning"})
    elif len(words) > 1:
        # A word is "title-cased" if it starts uppercase but isn't an acronym (all-caps).
        def is_acronym(w: str) -> bool:
            return all(c.isupper() for c in w if c.isalpha())

        has_title_cased_word = any(w[0].isupper() and not is_acronym(w) for w in words[1:] if w and w[0].isalpha())
        if has_title_cased_word:
            issues.append({"label": "Possibly not sentence case", "severity": "warning"})

    word_count = len(words)
    if word_count > 10:
        issues.append({"label": f"Too long ({word_count} words)", "severity": "error"})
    elif word_count > 6:
        issues.append({"label": f"Quite long ({word_count} words)", "severity": "warning"})

    if title.endswith("."):
        issues.append({"label": "Trailing full stop", "severity": "error"})

    if any(phrase in title.lower() for phrase in REFUSAL_PHRASES):
        issues.append({"label": "Refusal", "severity": "error"})

    return issues


async def evaluate_titles(
    prompts: list[str],
    url: str,
    headers: dict[str, str],
) -> list[TitleResult]:
    """Evaluate chat title generation for a list of prompts.

    Fires all requests concurrently and returns a list of TitleResult dicts.

    Parameters:
    - prompts: list of human query strings to send to the API
    - url: full URL of the chat title endpoint, e.g. http://localhost:5312/v1/chats/users/abc123/chats/abc123/title
    - headers: dict of HTTP headers to include in each request (must include auth headers)

    Example:
    results = asyncio.run(evaluate_titles(
        prompts=["Help", "Write a press release about the budget."],
        url="http://localhost:5312/v1/chats/users/abc123/chats/def456/title",
        headers={
            "Auth-Token": "mytoken",
            "User-Key-UUID": "abc123",
            "Session-Auth": "mysession",
            "Content-Type": "application/json",
        }
    ))
    """

    async def fetch(client: httpx.AsyncClient, prompt: str) -> tuple[httpx.Response | Exception, float]:
        start = time.perf_counter()
        try:
            resp = await client.put(url, headers=headers, json={"query": prompt, "use_rag": False})
        except Exception as exc:
            return exc, (time.perf_counter() - start) * 1000
        return resp, (time.perf_counter() - start) * 1000

    # Fire all requests concurrently, collecting results and durations
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10)) as client:
        raw = await asyncio.gather(*[fetch(client, p) for p in prompts])

    results: list[TitleResult] = []
    for i, (prompt, (response, duration_ms)) in enumerate(zip(prompts, raw, strict=True), start=1):
        if isinstance(response, Exception):
            results.append(
                {
                    "index": i,
                    "prompt": prompt,
                    "title": None,
                    "error": type(response).__name__,
                    "issues": [],
                    "passed": False,
                    "duration_ms": duration_ms,
                }
            )
        elif not response.is_success:
            results.append(
                {
                    "index": i,
                    "prompt": prompt,
                    "title": None,
                    "error": f"HTTP {response.status_code}",
                    "issues": [],
                    "passed": False,
                    "duration_ms": duration_ms,
                }
            )
        else:
            title = response.json().get("title", "")
            issues = check_title(title)
            results.append(
                {
                    "index": i,
                    "prompt": prompt,
                    "title": title,
                    "error": None,
                    "issues": issues,
                    "passed": len(issues) == 0,
                    "duration_ms": duration_ms,
                }
            )

    return results


# endregion
# ---------------------------------------------------------------------------
# region Rendering
# ---------------------------------------------------------------------------

_SEVERITY_COLOUR: dict[str, str] = {"warning": "yellow", "error": "red"}


def _issue_colour(issue: Issue) -> str:
    return _SEVERITY_COLOUR.get(issue["severity"], "white")


def _title_colour(issues: list[Issue]) -> str:
    if any(i["severity"] == "error" for i in issues):
        return "red"
    if any(i["severity"] == "warning" for i in issues):
        return "yellow"
    return "green"


def render_pretty(results: list[TitleResult], prompts_file: Path) -> None:
    console = Console()
    passed = sum(1 for r in results if r["passed"])
    errors = sum(1 for r in results if r["error"])
    quality_failures = len(results) - passed - errors

    intro = Text.assemble(
        ("Evaluating the chat title endpoint across ", "white"),
        (f"{len(results)} prompts", "bold cyan"),
        (f" from {prompts_file.name}.\n\n", "white"),
        ("Each generated title is checked against these criteria:\n", "white"),
        ("  ✓  ", "green"),
        ("Sentence case", "bold"),
        " — only the first word capitalised\n",
        ("  ✓  ", "green"),
        ("No trailing full stop\n", "bold"),
        ("  ✓  ", "green"),
        ("Not a refusal", "bold"),
        " — title contains usable content\n",
        ("  ✓  ", "green"),
        ("Concise", "bold"),
        " — ideally ≤6 words (warning at 7+, error at 11+)",
    )
    console.print(
        Panel(
            intro,
            title="[bold cyan]Chat Title Evaluation[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    table = Table(
        box=box.ROUNDED,
        show_lines=True,
        title="[bold]Results[/bold]",
        title_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Prompt", style="white", max_width=50)
    table.add_column("Generated Title", max_width=40)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Issues", max_width=28)

    for r in results:
        idx = str(r["index"])
        dur = f"{r['duration_ms']:.0f} ms" if r["duration_ms"] is not None else "—"
        if r["error"]:
            table.add_row(idx, r["prompt"], f"[red]{r['error']}[/red]", dur, "")
        else:
            tc = _title_colour(r["issues"])
            if r["issues"]:
                issue_str = ", ".join(
                    f"[{_issue_colour(iss)}]{iss['label']}[/{_issue_colour(iss)}]" for iss in r["issues"]
                )
            else:
                issue_str = "[green]✓[/green]"
            table.add_row(idx, r["prompt"], f"[{tc}]{r['title']}[/{tc}]", dur, issue_str)

    console.print()
    console.print(table)
    console.print(
        f"\n[dim]Evaluated {len(results)} prompts — "
        f"[green]{passed} passed[/green], "
        f"[yellow]{quality_failures} quality issue(s)[/yellow], "
        f"[red]{errors} error(s)[/red][/dim]"
    )


def render_json_output(results: list[TitleResult]) -> None:
    passed = sum(1 for r in results if r["passed"])
    errors = sum(1 for r in results if r["error"])
    output = {
        "total": len(results),
        "passed": passed,
        "quality_failures": len(results) - passed - errors,
        "errors": errors,
        "results": list(results),
    }
    print(json.dumps(output, indent=2))


def render_markdown(results: list[TitleResult], prompts_file: Path) -> None:
    passed = sum(1 for r in results if r["passed"])
    errors = sum(1 for r in results if r["error"])
    quality_failures = len(results) - passed - errors
    durations = [r["duration_ms"] for r in results if r["duration_ms"] is not None]
    avg_dur = sum(durations) / len(durations) if durations else 0

    lines: list[str] = []
    lines.append(f"# Chat title evaluation — {prompts_file.name}")
    lines.append("")
    lines.append("| # | Prompt | Title | Duration (ms) | Issues | Passed |")
    lines.append("|--:|--------|-------|-------------:|--------|:------:|")

    for r in results:
        idx = r["index"]
        prompt = r["prompt"].replace("|", "\\|")
        title = (r["title"] or r["error"] or "—").replace("|", "\\|")
        dur = f"{r['duration_ms']:.0f}" if r["duration_ms"] is not None else "—"
        issues = ", ".join(i["label"] for i in r["issues"]) if r["issues"] else "—"
        passed_mark = "Yes" if r["passed"] else "No"
        lines.append(f"| {idx} | {prompt} | {title} | {dur} | {issues} | {passed_mark} |")

    lines.append("")
    lines.append(
        f"**Summary**: {len(results)} prompts — {passed} passed, "
        f"{quality_failures} quality issue(s), {errors} error(s). "
        f"Average duration: {avg_dur:.0f} ms."
    )
    print("\n".join(lines))


# endregion
# ---------------------------------------------------------------------------
# region Settings + CLI
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    user_uuid: SecretStr
    chat_uuid: SecretStr
    auth_token: SecretStr
    session_auth: SecretStr
    base_url: str = "http://localhost:5312"

    model_config = {"env_file": [".env", ".env.titles"], "extra": "ignore"}


app = typer.Typer()


@app.command()
def main(
    user_uuid: Annotated[str | None, typer.Option(help="UUID of the user.")] = None,
    chat_uuid: Annotated[str | None, typer.Option(help="UUID of the chat.")] = None,
    auth_token: Annotated[str | None, typer.Option(help="Auth-Token header value.")] = None,
    session_auth: Annotated[str | None, typer.Option(help="Session-Auth header value.")] = None,
    base_url: Annotated[str | None, typer.Option(help="Base URL of the API (default: http://localhost:5312).")] = None,
    prompts_file: Annotated[
        Path, typer.Option(help="Path to a JSON file containing a list of prompts.")
    ] = DEFAULT_PROMPTS_FILE,
    output_json: Annotated[bool, typer.Option("--json", help="Output results as JSON.")] = False,
    output_md: Annotated[bool, typer.Option("--md", help="Output results as a markdown table.")] = False,
) -> None:
    """Evaluate chat title generation across a range of prompts via the Assist API.

    Auth parameters can be provided in three ways (in order of precedence):

    \b
    1. CLI flags        --user-uuid, --chat-uuid, --auth-token, --session-auth
    2. Environment      USER_UUID, CHAT_UUID, AUTH_TOKEN, SESSION_AUTH
    3. .env files       .env or .env.titles in the project root
    """
    try:
        prompts: list[str] = json.loads(prompts_file.read_text())
    except FileNotFoundError:
        typer.echo(f"Prompts file not found: {prompts_file}", err=True)
        sys.exit(1)
    except json.JSONDecodeError as e:
        typer.echo(f"Invalid JSON in prompts file: {e}", err=True)
        sys.exit(1)

    raw = {
        "user_uuid": user_uuid,
        "chat_uuid": chat_uuid,
        "auth_token": auth_token,
        "session_auth": session_auth,
        "base_url": base_url,
    }
    settings = Settings(**{k: v for k, v in raw.items() if v is not None})

    resolved_user_uuid = settings.user_uuid.get_secret_value()
    resolved_chat_uuid = settings.chat_uuid.get_secret_value()
    url = f"{settings.base_url}/v1/chats/users/{resolved_user_uuid}/chats/{resolved_chat_uuid}/title"
    headers = {
        "Auth-Token": settings.auth_token.get_secret_value(),
        "User-Key-UUID": resolved_user_uuid,
        "Session-Auth": settings.session_auth.get_secret_value(),
        "Content-Type": "application/json",
    }

    if output_json:
        results = asyncio.run(evaluate_titles(prompts, url, headers))
        render_json_output(results)
    elif output_md:
        results = asyncio.run(evaluate_titles(prompts, url, headers))
        render_markdown(results, prompts_file)
    else:
        console = Console()
        with console.status("[cyan]Evaluating prompts...[/cyan]", spinner="dots"):
            results = asyncio.run(evaluate_titles(prompts, url, headers))
        render_pretty(results, prompts_file)

    if any(not r["passed"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    app()
