"""Minimal Copilot API client for the evaluation runner: users, sessions, uploads and streamed chat."""

import json
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class EvalUser:
    uuid: str
    details: dict
    session_auth: str | None = None


@dataclass
class TurnResult:
    http_status: int | None = None
    chat_uuid: str | None = None
    assistant_message_uuid: str | None = None
    response_text: str = ""
    citations: str | None = None
    sources: str | None = None
    statuses: list[dict] = field(default_factory=list)
    error: dict | str | None = None
    time_to_first_token_s: float | None = None
    total_time_s: float | None = None


class EvalClient:
    """Thin wrapper over the chat API. Streamed responses are concatenated JSON objects, not SSE."""

    def __init__(self, base_url: str, auth_secret_key: str, frontend_url: str, timeout_s: float):
        self._http = httpx.Client(base_url=base_url.rstrip("/") + "/v1", timeout=httpx.Timeout(timeout_s, connect=15))
        self._auth_secret_key = auth_secret_key
        self._frontend_url = frontend_url

    def _headers(self, user: EvalUser) -> dict[str, str]:
        headers = {
            "Auth-Token": self._auth_secret_key,
            "User-Key-UUID": user.uuid,
            # Browser identity headers, needed to pass the AWS WAF common rule set on deployed envs.
            "User-Agent": _BROWSER_USER_AGENT,
            "Origin": self._frontend_url,
            "Referer": f"{self._frontend_url}/",
            "Accept": "application/json, text/plain, */*",
        }
        if user.session_auth:
            headers["Session-Auth"] = user.session_auth
        return headers

    def start_session(self, user: EvalUser) -> None:
        """Create the user if absent and open an auth session, storing its token on the user."""
        response = self._http.post("/users", headers=self._headers(user), json={"uuid": user.uuid, **user.details})
        if response.status_code not in (200, 201, 409):
            raise RuntimeError(f"Creating user {user.uuid} failed: {response.status_code} {response.text[:200]}")
        response = self._http.post("/auth-sessions", headers=self._headers(user))
        response.raise_for_status()
        user.session_auth = response.json()["Session-Auth"]

    def upload_document(self, user: EvalUser, path: Path) -> str:
        """Upload a document and return its uuid."""
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        response = self._http.post(
            f"/users/{user.uuid}/documents",
            headers=self._headers(user),
            files={"file": (path.name, path.read_bytes(), mime)},
        )
        response.raise_for_status()
        return response.json()["document_uuid"]

    def stream_chat(self, user: EvalUser, body: dict, chat_uuid: str | None = None) -> TurnResult:
        """Send one chat turn and consume the streamed response.

        Args:
            user: The user sending the turn.
            body: Request body (query, flags, document_uuids).
            chat_uuid: None to create a chat (first turn), otherwise the chat to continue.

        Returns:
            The parsed result. Failures are reported in `http_status` and `error`, not raised.
        """
        result = TurnResult(chat_uuid=chat_uuid)
        started = time.perf_counter()
        try:
            if chat_uuid is None:
                request = self._http.build_request(
                    "POST", f"/chats/users/{user.uuid}/stream", headers=self._headers(user), json=body
                )
            else:
                request = self._http.build_request(
                    "PUT", f"/chats/users/{user.uuid}/chats/{chat_uuid}/stream", headers=self._headers(user), json=body
                )
            response = self._http.send(request, stream=True)
            try:
                result.http_status = response.status_code
                if response.status_code != 200:
                    result.error = response.read().decode(errors="replace")[:1000]
                else:
                    self._consume_stream(response, result, started)
            finally:
                response.close()
        except httpx.HTTPError as e:
            result.error = f"{type(e).__name__}: {e}"
        result.total_time_s = round(time.perf_counter() - started, 3)
        return result

    @staticmethod
    def _consume_stream(response: httpx.Response, result: TurnResult, started: float) -> None:
        decoder = json.JSONDecoder()
        buffer = ""
        for chunk in response.iter_text():
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                try:
                    event, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break  # Incomplete object; wait for more data.
                buffer = buffer[end:]
                _apply_event(event, result, time.perf_counter() - started)
        if buffer.strip():
            result.error = result.error or f"Unparsed trailing stream data: {buffer[:200]}"
        if result.error is None and not result.response_text:
            result.error = "Stream ended without any message content"


def _apply_event(event: dict, result: TurnResult, elapsed_s: float) -> None:
    if "error_code" in event:
        result.error = {"error_code": event["error_code"], "error_message": event.get("error_message")}
    elif "status" in event:
        result.statuses.append({**event["status"], "t_s": round(elapsed_s, 3)})
    elif "message_streamed" in event:
        message = event["message_streamed"]
        result.chat_uuid = result.chat_uuid or event.get("uuid")
        result.assistant_message_uuid = message.get("uuid")
        result.response_text = message.get("content") or ""
        result.citations = message.get("citations")
        result.sources = message.get("sources")
        if result.time_to_first_token_s is None and result.response_text:
            result.time_to_first_token_s = round(elapsed_s, 3)
