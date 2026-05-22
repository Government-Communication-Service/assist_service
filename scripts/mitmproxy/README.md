# mitmproxy dev interceptor

Sits between the frontend and the API at `:5312`, intercepting requests that match rules in `mocks.json` and forwarding everything else to the real API (shifted to `:5313`).

Useful for simulating error states or specific payloads without touching the API code.

## Starting and stopping

```bash
make down   # stop normal stack, start proxy stack
make up-proxy
make down-proxy    # stop proxy stack
make up            # return to normal stack
```

The underlying image is reused - no rebuild needed.

To watch traffic passing through:

```bash
make logs-proxy
```

## Adding a mock

Edit `mocks.json`. Keys are `"METHOD /path/regex"` — the path portion is matched as a regex against the full request path. Values define the response.

```json
{
  "GET /v1/chats/users/.*/chats/.*/messages": {
    "status": 503,
    "body": { "error": "upstream unavailable" },
    "headers": {
      "Retry-After": "30"
    }
  }
}
```

`status`, `body`, and `headers` are all optional (defaulting to `200`, `{}`, and `{}` respectively).

Changes to `mocks.json` are picked up automatically on the next request — no restart needed.

## How matching works

Rules are tested in order. The first match wins. Non-matching requests are forwarded transparently to the real API.

Path patterns are anchored (`^...$`), so `/v1/chats` will not match `/v1/chats/users/...` — use `.*` to match subpaths.

## Verifying the proxy with sausage_haiku.sh

`sausage_haiku.sh` sends a real `POST /v1/chats/users/{user_uuid}` request asking for a haiku about sausages and prints the LLM reply. It's a handy end-to-end smoke test for checking whether a request is being intercepted or reaching the real API.

**Without the proxy** (normal stack via `make up`): the request hits the real API and you get a genuine haiku.

**With the proxy and no matching mock**: the request passes through transparently — same real haiku.

**With the proxy and a matching mock**: the request is intercepted and you get your mocked response instead. To mock the haiku endpoint, add a rule to `mocks.json` for `POST /v1/chats/users/[^/]+` and run the script to confirm it's intercepted:

```json
{
  "POST /v1/chats/users/[^/]+": {
    "status": 200,
    "body": {
      "uuid": "mock",
      "created_at": "2024-01-01T00:00:00",
      "updated_at": "2024-01-01T00:00:00",
      "title": "Mock chat",
      "from_open_chat": false,
      "message": {
        "uuid": "mock",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "content": "Your mock response here",
        "role": "assistant"
      }
    }
  }
}
```

```bash
./scripts/mitmproxy/sausage_haiku.sh
# Your mock response here
```

If the proxy is running but the mock isn't matching, check `make logs-proxy` to see whether the request is appearing and which path it's hitting.

## Files

| File | Purpose |
|---|---|
| `mocks.json` | Mock rules — edit this to add/change intercepts |
| `intercept.py` | mitmproxy addon — loads and applies the rules |
| `sausage_haiku.sh` | Smoke test script — sends a real request through the proxy |
