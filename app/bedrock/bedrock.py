import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from anthropic.types import MessageParam
from anthropic.types.message import Message as AnthropicMessage
from pydantic import BaseModel

from app.bedrock.bedrock_stream import BedrockStreamInput, bedrock_stream
from app.bedrock.bedrock_types import AnthropicBedrockProvider, AsyncAnthropicBedrockProvider
from app.bedrock.retry import handle_region_failover_with_retries, with_region_failover_for_streaming
from app.bedrock.schemas import LLMResponse, LLMTransaction
from app.bedrock.service import llm_transaction
from app.config import AWS_BEDROCK_REGION1, LLM_DEFAULT_MODEL
from app.database.models import LLM, Message
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


class Result(BaseModel):
    content: List[Content | ContentToolUse]
    completion_cost: float
    usage: Usage


class ToolResult(BaseModel):
    content: List[Content | ContentToolUse]
    completion_cost: float
    input_tokens: int
    output_tokens: int


class BedrockHandler:
    __CROSS_REGION_INFERENCE_MODELS = {
        AWS_BEDROCK_REGION1: [
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
        system=None,
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

    def format_response(self, response: [Result | ToolResult | Message]) -> LLMTransaction:
        if isinstance(response, ToolResult):
            payload = LLMResponse(
                content=response.content,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        elif isinstance(response, Result):
            payload = LLMResponse(
                content=response.content,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        else:
            payload = LLMResponse(
                content=response.content,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        return llm_transaction(self.llm, payload)

    def _format_chat_title_response(self, response: AnthropicMessage) -> LLMTransaction:
        result = LLMResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return llm_transaction(self.llm, result)

    async def _invoke_async(self, messages, **data) -> Result:
        logger.debug("LLM _invoke_async started")
        config = self.config | data
        logger.debug(f"Messages sent to LLM: {messages}")
        response = await self.async_client.messages.create(messages=messages, **config)
        logger.debug("LLM _invoke_async completed")
        return response

    @handle_region_failover_with_retries
    async def invoke_async(self, messages, **data) -> Result:
        return await self._invoke_async(messages, **data)

    async def _invoke_async_with_call_cost_details(self, messages, **data) -> LLMTransaction:
        config = self.config | data
        response = await self.async_client.messages.create(messages=messages, **config)

        return self.format_response(response)

    @handle_region_failover_with_retries
    async def invoke_async_with_call_cost_details(self, messages, **data) -> LLMTransaction:
        return await self._invoke_async_with_call_cost_details(messages, **data)

    async def _create_chat_title(self, messages: List[MessageParam]):
        """
        Create a chat title using the LLM model configured in the init with the
        Returns a LLMTransaction object encapsulating the title and the cost of generating it.
        """

        response = await self.async_client.messages.create(messages=messages, **self.config)
        logger.debug(f"Messages sent to LLM {messages}")
        return self._format_chat_title_response(response)

    @handle_region_failover_with_retries
    async def create_chat_title(self, messages: List[MessageParam]) -> LLMTransaction:
        return await self._create_chat_title(messages)

    def _stream(
        self,
        messages,
        user_message: Message = None,
        system: str = None,
        parse_data=None,
        **data,
    ):
        config = self.config | data

        bedrock_stream_input = BedrockStreamInput(
            async_client=self.async_client,
            messages=messages,
            max_tokens=config["max_tokens"],
            model=config["model"],
            user_message=user_message,
            system=system,
            parse_data=parse_data,
            **data,
        )
        return bedrock_stream(bedrock_stream_input)

    def stream(
        self,
        messages,
        on_error: Callable[[Exception], str],
        user_message: Message = None,
        system: str = None,
        parse_data=None,
        **data,
    ):
        return with_region_failover_for_streaming(
            self, self._stream, on_error, messages, user_message, system, parse_data, **data
        )
