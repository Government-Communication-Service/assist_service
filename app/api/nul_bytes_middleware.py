import json
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# JSON can carry a NUL character only as this escape; raw control characters are invalid JSON.
NUL_ESCAPE = b"\\u0000"
NUL_ERROR_MESSAGE = "Text must not contain NUL characters"


def _find_nul_locations(value: Any, location: tuple) -> list[tuple]:
    if isinstance(value, str):
        return [location] if "\x00" in value else []
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if "\x00" in key:
                found.append((*location, key))
            found.extend(_find_nul_locations(item, (*location, key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_nul_locations(item, (*location, index)))
    return found


def _is_json_request(scope: Scope) -> bool:
    for name, value in scope.get("headers", []):
        if name == b"content-type":
            media_type = value.split(b";")[0].strip().lower()
            return media_type == b"application/json" or media_type.endswith(b"+json")
    return False


class RejectNulBytesMiddleware:
    """Returns 422 for JSON request bodies containing a NUL character in any string, which Postgres can't store."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _is_json_request(scope):
            await self.app(scope, receive, send)
            return

        chunks = []
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)
        body = b"".join(chunks)

        if NUL_ESCAPE in body:
            try:
                nul_locations = _find_nul_locations(json.loads(body), ("body",))
            except (ValueError, RecursionError):
                # Invalid or too deeply nested: let the app reject it as it would any unparseable body.
                nul_locations = []
            if nul_locations:
                errors = [{"type": "value_error", "loc": list(loc), "msg": NUL_ERROR_MESSAGE} for loc in nul_locations]
                await JSONResponse(status_code=422, content={"detail": errors})(scope, receive, send)
                return

        body_sent = False

        async def replay_receive() -> Message:
            nonlocal body_sent
            if body_sent:
                return await receive()
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay_receive, send)
