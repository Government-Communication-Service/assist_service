import json
import os

from app.config import LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH


def log_invocation_to_file(model: str, system: str | list | None, messages: list, extra: dict | None = None) -> None:
    """Write the full Bedrock request to a file for local dev inspection.

    Do not set LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH in production - it writes to file for every
    invocation.

    extra captures any additional kwargs passed to the API call (tools, tool_choice, etc.).
    """
    if not LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH:
        return
    os.makedirs(os.path.dirname(LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH), exist_ok=True)
    payload = {"model": model, "system": system, "messages": messages}
    if extra:
        payload["extra"] = extra
    with open(LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def log_response_to_file(response) -> None:
    """Append token usage and response content from the Bedrock response to the invocation log file."""
    if not LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH:
        return
    try:
        with open(LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH) as f:
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
    with open(LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)
