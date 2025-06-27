import json
import logging
from typing import Any, List, Optional

from anthropic import AsyncAnthropicBedrock
from anthropic.types import MessageParam
from pydantic import BaseModel

from app.bedrock.american_word_swap import PARTIAL_WORD_PATTERN, replace_american_words
from app.database.models import Message

logger = logging.getLogger(__name__)


class BedrockStreamInput(BaseModel):
    async_client: AsyncAnthropicBedrock
    messages: List[MessageParam]
    max_tokens: int
    model: str
    parse_data: Optional[Any] = None
    user_message: Optional[Message] = None
    system: str = None
    on_complete: Any = None

    class Config:
        arbitrary_types_allowed = True


async def bedrock_stream(bedrock_stream_input: BedrockStreamInput):
    full_message = ""
    logger.info("Calling bedrock_stream")
    async with bedrock_stream_input.async_client.messages.stream(
        max_tokens=bedrock_stream_input.max_tokens,
        messages=bedrock_stream_input.messages,
        model=bedrock_stream_input.model,
        system=bedrock_stream_input.system,
    ) as stream:
        remaining_word = ""
        async for text in stream.text_stream:
            text = text or ""
            text = remaining_word + text

            """handling partial word splits across chunks."""

            has_partial_word = PARTIAL_WORD_PATTERN.search(text)
            if has_partial_word:
                split_index = has_partial_word.start()
                remaining_word = text[split_index:]  # Save the incomplete word in remaining_word
                text = text[:split_index]  # Process only complete words
            else:
                # reset remaining_word
                remaining_word = ""

            text = replace_american_words(text)
            full_message += text

            if bedrock_stream_input.parse_data:
                citations = getattr(bedrock_stream_input.user_message, "citation", None)
                parsed_data = json.dumps(
                    bedrock_stream_input.parse_data(full_message, citations),
                    indent=4,
                )

                yield f"{parsed_data}"
            else:
                yield text

        # Handle remaining word at end of stream
        if remaining_word:
            # Add remaining word to full message
            text = replace_american_words(remaining_word)
            full_message += text

            # Process through the same pipeline as other chunks
            if bedrock_stream_input.parse_data:
                citations = getattr(bedrock_stream_input.user_message, "citation", None)
                parsed_data = json.dumps(
                    bedrock_stream_input.parse_data(full_message, citations),
                    indent=4,
                )
                yield f"{parsed_data}"
            else:
                yield text

    response = await stream.get_final_message()
    logger.debug("Final response is: %s", response)

    logger.info("Completed bedrock_stream")

    if bedrock_stream_input.on_complete:
        bedrock_stream_input.on_complete(response)
