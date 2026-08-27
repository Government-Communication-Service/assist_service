import itertools
import json
import os
from pathlib import Path

from app.config import LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH

_call_counter = itertools.count(1)


def log_invocation_to_file(
    model: str, system: str | list | None, messages: list, extra: dict | None = None
) -> str | None:
    """Write the full Bedrock request to its own file for local dev inspection.

    Do not set LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH in production - it writes to file for every
    invocation.

    Each call gets its own numbered file (rather than overwriting one shared path) so that
    concurrent calls - e.g. a background compaction call firing moments after a reply - don't
    clobber each other. Returns the path written, to be passed to log_response_to_file, or None
    if logging is disabled.

    extra captures any additional kwargs passed to the API call (tools, tool_choice, etc.).
    """
    if not LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH:
        return None
    base_path = Path(LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH)
    os.makedirs(base_path.parent, exist_ok=True)
    path = base_path.with_name(f"{base_path.stem}_{next(_call_counter)}{base_path.suffix}")
    payload = {"model": model, "system": system, "messages": messages}
    if extra:
        payload["extra"] = extra
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return str(path)


def log_response_to_file(response, path: str | None) -> None:
    """Append token usage and response content from the Bedrock response to the invocation log file."""
    if not path:
        return
    try:
        with open(path) as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return
    usage = response.usage
    payload["usage"] = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
        "cache_creation": getattr(usage, "cache_creation", None),
    }
    payload["response"] = [{"type": block.type, "text": getattr(block, "text", None)} for block in response.content]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
