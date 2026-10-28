import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import replace
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic.types.message import Message as AnthropicMessage
from pydantic import BaseModel

from app.bedrock.bedrock_stream import StreamDone, StreamEvent, stream_message_events
from app.bedrock.bedrock_types import AnthropicBedrockProvider, AsyncAnthropicBedrockProvider
from app.bedrock.call_log import LlmCallContext, log_llm_call
from app.bedrock.dev_logging import log_invocation_to_file, log_response_to_file
from app.bedrock.retry import handle_region_failover_with_retries, stream_with_region_failover
from app.bedrock.schemas import LLMResponse, LLMTransaction
from app.bedrock.service import extract_cache_tokens, llm_transaction
from app.config import AWS_BEDROCK_REGION1, LLM_DEFAULT_MODEL
from app.database.models import LLM
from app.database.table import LLMTable

logger = logging.getLogger(__name__)


def llm_get_default_model() -> LLM:
    """
    Get the LLM model object from the database where model name is specified by LLM_DEFAULT_MODEL config parameter.
    """
    return LLMTable().get_by_model(LLM_DEFAULT_MODEL)


class RunMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"


class Content(BaseModel):
    text: str


class ContentToolUse(BaseModel):
    input: Dict[str, Any]
    type: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ToolResult(BaseModel):
    content: List[Content | ContentToolUse]
    completion_cost: float
    input_tokens: int
    output_tokens: int


class BedrockMessage(AnthropicMessage):
    llm_internal_response_id: int | None = None

    def __init__(self, message: AnthropicMessage, llm_internal_response_id: int | None = None):
        super().__init__(**message.model_dump(), llm_internal_response_id=llm_internal_response_id)


class BedrockHandler:
    __CROSS_REGION_INFERENCE_MODELS = {
        AWS_BEDROCK_REGION1: [
            "anthropic.claude-haiku-5-5",
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            "anthropic.claude-sonnet-5-5",
            "anthropic.claude-sonnet-5",
            "anthropic.claude-sonnet-4-6",
            "anthropic.claude-opus-4-6-v1",
            "anthropic.claude-opus-4-5-20251101-v1:0",
            "anthropic.claude-opus-4-1-20250805-v1:0",
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "anthropic.claude-opus-4-20250514-v1:0",
            "anthropic.claude-sonnet-4-20250514-v1:0",
            "anthropic.claude-3-7-sonnet-20250219-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "meta.llama3-1-70b-instruct-v1:0",
            "meta.llama3-1-8b-instruct-v1:0",
            "meta.llama3-2-11b-instruct-v1:0",
            "meta.llama3-2-1b-instruct-v1:0",
            "meta.llama3-2-3b-instruct-v1:0",
            "meta.llama3-2-90b-instruct-v1:0",
        ],
    }
    """
    Supported regions and models for cross-region inference
    See https://eu-central-1.console.aws.amazon.com/bedrock/home?region=eu-central-1#/cross-region-inference
    """

    __AWS_REGION_GEO_PREFIX = AWS_BEDROCK_REGION1.split("-")[0]

    def __init__(
        self,
        max_tokens=None,
        llm: Optional[LLM] = None,
        mode: RunMode = RunMode.SYNC,
        system: str | list | None = None,
    ):
        """
        Initializes the `BedrockHandler` instance with the following parameters:

        Args:
            max_tokens (int, optional): Maximum number of tokens for the model's response.
            Defaults to the model's internal `max_tokens` if not provided.
            llm (LLM, optional): An instance of the `LLM` model. If not provided, a default model is assigned.
            mode (RunMode, optional): Indicates the operation mode of the handler, defaults to `RunMode.SYNC`.
            If not specified, a synchronous client (`AnthropicBedrock`) is initialized.
            system (str, optional): Optional system configuration for the model.

        The constructor also checks if the model is supported for cross region inference in the specified AWS region
        and sets up cross-region inference.

        """
        # assign default llm model if no LLM model is provided
        if llm is None:
            logger.error(
                "BedrockHandler instantiated without an llm argument — falling back to default model %s. "
                "Pass llm= explicitly to use the intended model.",
                LLM_DEFAULT_MODEL,
            )
        self.llm = llm if llm else llm_get_default_model()

        if not max_tokens:
            max_tokens = self.llm.max_tokens

        if mode == RunMode.SYNC:
            self.client = AnthropicBedrockProvider.get(AWS_BEDROCK_REGION1)
        elif mode == RunMode.ASYNC:
            self.async_client = AsyncAnthropicBedrockProvider.get(AWS_BEDROCK_REGION1)
        else:
            raise ValueError(f"Invalid Bedrock client mode: {mode}")

        # check if the model is supported for cross_region_inference
        # if so, use the model with cross region inference mode.
        cross_region_inference_models = self.__CROSS_REGION_INFERENCE_MODELS.get(AWS_BEDROCK_REGION1, [])
        final_model_id = (
            f"{self.__AWS_REGION_GEO_PREFIX}.{self.llm.model}"
            if self.llm.model in cross_region_inference_models
            else self.llm.model
        )

        self.model = final_model_id
        self.config = {
            "model": self.model,
            "max_tokens": max_tokens,
        }
        if system:
            self.config["system"] = system

    def format_content_for_chat_title(self, content: str) -> List[dict]:
        """
        Formats the provided content into a list of messages required by AnthropicBedrock API.

        This method takes a string `content` and formats it into a list with one dictionary, where the dictionary
        represents a message with a 'role' of 'user' and the 'content'
        being the input string.

        Args:
            content (str): The content to be formatted into a chat message.

        Returns:
            List[dict]: A list with a single dictionary with keys 'role' and 'content'.

        """
        messages = []
        title_message = {"role": "user", "content": content}
        messages.append(title_message)

        return messages

    def format_response(self, response: ToolResult | AnthropicMessage) -> LLMTransaction:
        cache_read_tokens, cache_write_tokens = extract_cache_tokens(response.usage)
        payload = LLMResponse(
            content=response.content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        return llm_transaction(self.llm, payload)

    async def _invoke_async(self, messages, **data) -> AnthropicMessage:
        logger.debug("LLM _invoke_async started")
        config = self.config | data
        logger.debug(f"Messages sent to LLM: {messages}")
        extra = {k: v for k, v in config.items() if k not in ("model", "max_tokens", "system")}
        log_path = log_invocation_to_file(self.model, config.get("system"), messages, extra=extra or None)
        response = await self.async_client.messages.create(messages=messages, **config)
        log_response_to_file(response, log_path)
        logger.debug("LLM _invoke_async completed")
        return response

    @handle_region_failover_with_retries
    async def _invoke_async_with_failover(self, messages, **data) -> AnthropicMessage:
        """One Messages API call, retried in the other region on failure. Never touches the DB."""
        return await self._invoke_async(messages, **data)

    async def invoke_async(self, messages, context: LlmCallContext | None = None, **data) -> BedrockMessage:
        """Invoke the Bedrock API asynchronously with region failover, and log the call.

        If a context is given, the call is logged to llm_internal_response after it succeeds,
        outside the failover retries and on its own session (see log_llm_call), so a logging
        failure can neither re-run the model call nor fail it. Without one, it isn't logged.

        Args:
            messages: Request messages.
            context: What the call was for, recorded on the log row. Passing one is what turns logging on.
            **data: Request params overriding the handler config (system, tools, tool_choice...).

        Returns:
            The response, with llm_internal_response_id set to the log row (None if it wasn't logged).
        """
        response = await self._invoke_async_with_failover(messages, **data)
        llm_internal_response_id = await log_llm_call(self.llm, response, context) if context is not None else None
        return BedrockMessage(response, llm_internal_response_id=llm_internal_response_id)

    async def _invoke_async_with_call_cost_details(self, messages, **data) -> LLMTransaction:
        config = self.config | data
        response = await self.async_client.messages.create(messages=messages, **config)
        return self.format_response(response)

    @handle_region_failover_with_retries
    async def invoke_async_with_call_cost_details(self, messages, **data) -> LLMTransaction:
        return await self._invoke_async_with_call_cost_details(messages, **data)

    async def stream_events(
        self, messages: list[dict], context: LlmCallContext | None = None, **data
    ) -> AsyncIterator[StreamEvent]:
        """Stream one call as events, with region failover before the first event, and log the call.

        If a context is given, the call is logged when the response completes, on its own session
        (see log_llm_call), and the row's id is set on the final StreamDone. Without one, it isn't logged.

        Args:
            messages: Request messages.
            context: What the call was for, recorded on the log row. Passing one is what turns logging on.
            **data: Request params overriding the handler config (system, tools, tool_choice...).

        Returns:
            An async iterator of StreamStarted, StreamText and a final StreamDone.
        """
        config = self.config | data
        events = stream_with_region_failover(self, stream_message_events, messages=messages, **config)
        async with aclosing(events):
            async for event in events:
                if isinstance(event, StreamDone) and context is not None:
                    llm_internal_response_id = await log_llm_call(self.llm, event.message, context)
                    event = replace(event, llm_internal_response_id=llm_internal_response_id)
                yield event
