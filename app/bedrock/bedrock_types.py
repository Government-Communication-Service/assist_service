# ruff: noqa: A005
from enum import Enum, auto
from typing import Optional

from anthropic import AnthropicBedrock, AsyncAnthropicBedrock, Timeout

from app.config import BEDROCK_API_CONNECT_TIMEOUT_SECONDS, BEDROCK_API_READ_TIMEOUT_SECONDS
from app.exceptions.app_error import AppError


class BedrockErrorType(str, Enum):
    INPUT_TOO_LONG = auto()
    BEDROCK_AGENT_EXCEPTION = auto()
    TIMEOUT = auto()


class BedrockError(AppError):
    """Base class for all Bedrock exceptions. `app/bedrock/retry.py` logs the failure before raising."""

    status_code = 503
    expected = True

    def __init__(self, msg: str, error_type: Optional[BedrockErrorType] = None):
        super().__init__(msg)
        self.error_type = error_type

    @property
    def detail(self) -> str:
        if self.error_type == BedrockErrorType.INPUT_TOO_LONG:
            return "Input is too long"
        if self.error_type == BedrockErrorType.TIMEOUT:
            return "AWS Bedrock call timed out"
        return "AWS Bedrock call failed"

    def content(self) -> dict:
        return {"status": "failed", "error_code": "BEDROCK_SERVICE_ERROR", "status_message": self.detail}


class AnthropicBedrockProvider:
    """
    Provides a single instance AnthropicBedrock.
    It creates a new instance on first call and returns existing client on subsequent calls
    """

    __clients = {}

    @classmethod
    def get(cls, aws_region: str) -> AnthropicBedrock:
        if cls.__clients.get(aws_region) is None:
            current_client = AnthropicBedrock(
                aws_region=aws_region,
                max_retries=0,
                timeout=Timeout(timeout=BEDROCK_API_READ_TIMEOUT_SECONDS, connect=BEDROCK_API_CONNECT_TIMEOUT_SECONDS),
            )
            cls.__clients[aws_region] = current_client

        return cls.__clients[aws_region]


class AsyncAnthropicBedrockProvider:
    """
    Provides a single instance of AsyncAnthropicBedrock.
    It creates a new instance on first call and returns existing client on subsequent calls

    """

    __clients = {}

    @classmethod
    def get(cls, aws_region: str) -> AsyncAnthropicBedrock:
        if cls.__clients.get(aws_region) is None:
            current_client = AsyncAnthropicBedrock(
                aws_region=aws_region,
                max_retries=0,
                timeout=Timeout(timeout=BEDROCK_API_READ_TIMEOUT_SECONDS, connect=BEDROCK_API_CONNECT_TIMEOUT_SECONDS),
            )
            cls.__clients[aws_region] = current_client

        return cls.__clients[aws_region]
