import logging

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.bedrock.bedrock_types import BedrockError, BedrockErrorType
from app.database.database_exception import DatabaseError, DatabaseExceptionErrorCode
from app.exceptions.app_error import AppError
from app.exceptions.handlers import register_exception_handlers

pytestmark = [pytest.mark.unit]


class ThingNotFoundError(AppError):
    status_code = 404
    detail = "Thing not found"
    expected = True


class ThingBrokeError(AppError):
    pass


class ThingCodeError(AppError):
    def __init__(self, code: int):
        super().__init__(f"code={code}")
        self.code = code

    @property
    def status_code(self) -> int:
        return self.code

    @property
    def expected(self) -> bool:
        return self.code < 500


class ThingCustomBodyError(AppError):
    status_code = 401
    expected = True

    def content(self) -> dict:
        return {"error": "THING_ERROR", "ids": ["a", "b"]}


@pytest.fixture
async def client():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/not-found")
    async def not_found():
        raise ThingNotFoundError("thing 123 missing")

    @app.get("/broke")
    async def broke():
        raise ThingBrokeError("thing 123 broke")

    @app.get("/code/{code}")
    async def by_code(code: int):
        raise ThingCodeError(code)

    @app.get("/custom")
    async def custom():
        raise ThingCustomBodyError()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as cl:
        yield cl


def error_records(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno >= logging.ERROR]


async def test_expected_error_uses_class_mapping_and_logs_at_info(client, caplog):
    with caplog.at_level(logging.INFO):
        response = await client.get("/not-found")

    assert response.status_code == 404
    assert response.json() == {"detail": "Thing not found"}
    assert error_records(caplog) == []
    assert "ThingNotFoundError in endpoint GET /not-found" in caplog.text


async def test_unexpected_error_defaults_to_500_and_logs_at_error(client, caplog):
    response = await client.get("/broke")

    assert response.status_code == 500
    assert response.json() == {"detail": "An internal error occurred."}
    [record] = error_records(caplog)
    assert record.exc_info[0] is ThingBrokeError


@pytest.mark.parametrize(("code", "logged_as_error"), [(404, False), (503, True)])
async def test_mapping_can_depend_on_instance(client, caplog, code, logged_as_error):
    response = await client.get(f"/code/{code}")

    assert response.status_code == code
    assert bool(error_records(caplog)) is logged_as_error


async def test_content_can_be_overridden(client):
    response = await client.get("/custom")

    assert response.status_code == 401
    assert response.json() == {"error": "THING_ERROR", "ids": ["a", "b"]}


@pytest.mark.parametrize(
    ("code", "status_code", "content", "expected"),
    [
        (DatabaseExceptionErrorCode.GET_BY_UUID_ERROR, 404, {"detail": "Record not found"}, True),
        (DatabaseExceptionErrorCode.USE_CASE_NOT_UNDER_THIS_THEME_ERROR, 404, {"detail": "db message"}, True),
        (DatabaseExceptionErrorCode.UPDATE_ERROR, 500, {"detail": "An internal error occurred."}, False),
    ],
)
def test_database_error_mapping_depends_on_code(code, status_code, content, expected):
    exc = DatabaseError(code=code, message="db message")

    assert (exc.status_code, exc.content(), exc.expected) == (status_code, content, expected)


@pytest.mark.parametrize(
    ("error_type", "status_message"),
    [
        (BedrockErrorType.INPUT_TOO_LONG, "Input is too long"),
        (BedrockErrorType.TIMEOUT, "AWS Bedrock call timed out"),
        (None, "AWS Bedrock call failed"),
    ],
)
def test_bedrock_error_body_omits_exception_text(error_type, status_message):
    exc = BedrockError("AWS Bedrock call failed, last exception: secret internals", error_type)

    assert exc.content() == {
        "status": "failed",
        "error_code": "BEDROCK_SERVICE_ERROR",
        "status_message": status_message,
    }
