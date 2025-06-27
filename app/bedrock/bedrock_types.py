# ruff: noqa: A005
from enum import Enum, auto
from typing import Optional

import httpx
from anthropic import AnthropicBedrock, AsyncAnthropicBedrock

BEDROCK_API_READ_TIMEOUT_SECS = 115
BEDROCK_API_CONNECT_TIMEOUT_SECS = 3


class BedrockErrorType(str, Enum):
    INPUT_TOO_LONG = auto()
    BEDROCK_AGENT_EXCEPTION = auto()


class BedrockError(Exception):
    """Base class for all Bedrock exceptions."""

    def __init__(self, msg: str, error_type: Optional[BedrockErrorType] = None):
        super().__init__(msg)
        self.error_type = error_type


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
                timeout=httpx.Timeout(timeout=BEDROCK_API_READ_TIMEOUT_SECS, connect=BEDROCK_API_CONNECT_TIMEOUT_SECS),
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
                timeout=httpx.Timeout(timeout=BEDROCK_API_READ_TIMEOUT_SECS, connect=BEDROCK_API_CONNECT_TIMEOUT_SECS),
            )
            cls.__clients[aws_region] = current_client

        return cls.__clients[aws_region]
