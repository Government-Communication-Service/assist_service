import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from app.api.nul_bytes_middleware import RejectNulBytesMiddleware

pytestmark = [pytest.mark.unit]


@pytest.fixture
async def client():
    app = FastAPI()
    app.add_middleware(RejectNulBytesMiddleware)

    @app.post("/echo")
    async def echo(request: Request):
        return {"body": (await request.body()).decode()}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as cl:
        yield cl


async def test_rejects_nul_with_field_locations(client):
    response = await client.post("/echo", json={"name": "a\x00b", "items": [{"ok": "x"}, {"bad\x00key": "\x00"}]})

    assert response.status_code == 422
    assert [error["loc"] for error in response.json()["detail"]] == [
        ["body", "name"],
        ["body", "items", 1, "bad\x00key"],
        ["body", "items", 1, "bad\x00key"],
    ]


async def test_passes_literal_backslash_u0000_text(client):
    response = await client.post("/echo", json={"name": "\\u0000"})

    assert response.status_code == 200
    assert response.json()["body"] == '{"name":"\\\\u0000"}'


async def test_passes_body_without_nul_unchanged(client):
    response = await client.post("/echo", json={"name": "Campaign"})

    assert response.status_code == 200
    assert response.json()["body"] == '{"name":"Campaign"}'


async def test_ignores_non_json_bodies(client):
    response = await client.post("/echo", content=b'{"name": "a\\u0000b"}', headers={"content-type": "text/plain"})

    assert response.status_code == 200


async def test_passes_invalid_json_through(client):
    response = await client.post("/echo", content=b'{"name": "a\\u0000', headers={"content-type": "application/json"})

    assert response.status_code == 200


async def test_deeply_nested_body_with_nul_escape_is_not_a_500(client):
    body = b"[" * 5000 + b'"\\u0000"' + b"]" * 5000

    response = await client.post("/echo", content=body, headers={"content-type": "application/json"})

    assert response.status_code == 200
