# ruff: noqa: E501, F841

import json
import logging

import pytest
from anthropic import AnthropicBedrock

from app.api.endpoints import ENDPOINTS

logger = logging.getLogger(__name__)


async def get_final_streamed_content(
    async_client, method: str, endpoint: str, headers: dict, json_payload: dict
) -> str:
    """Connects to a streaming endpoint, reads the full non-standard SSE response,
    parses the last JSON object, and returns the content of 'message_streamed'."""
    try:
        async with async_client.stream(method, endpoint, json=json_payload, headers=headers) as response:
            response.raise_for_status()  # Check for HTTP errors first
            if "text/event-stream" not in response.headers.get("content-type", ""):
                logger.warning(f"Expected text/event-stream, got {response.headers.get('content-type')}")
                # Continue anyway as the stream format seems non-standard

            # Read the entire response body
            full_response_bytes = await response.aread()
            full_response_text = full_response_bytes.decode("utf-8")
            logger.info(f"Full response text received:\n{full_response_text}")

            last_message_streamed_content = None

            # Attempt to parse the full response text as potentially concatenated JSON
            try:
                json_objects = []
                decoder = json.JSONDecoder()
                pos = 0
                trimmed_text = full_response_text.strip()  # Process trimmed text
                while pos < len(trimmed_text):
                    try:
                        obj, pos = decoder.raw_decode(trimmed_text, pos)
                        json_objects.append(obj)
                    except json.JSONDecodeError:
                        # Skip whitespace or invalid chars between objects
                        remaining_text = trimmed_text[pos:]
                        non_space = remaining_text.lstrip()
                        if not non_space:
                            break  # End of string
                        skipped_len = len(remaining_text) - len(non_space)
                        pos += skipped_len
                        # If after skipping whitespace we still can't decode, skip one char
                        if pos < len(trimmed_text):
                            try:
                                decoder.raw_decode(trimmed_text, pos)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Skipping unexpected character '{trimmed_text[pos]}' at index {pos} after whitespace."
                                )
                                pos += 1

                if not json_objects:
                    raise ValueError("No valid JSON objects found in the response text.")

                # Assume the last JSON object contains the final state
                final_chunk_data = json_objects[-1]

                if "message_streamed" in final_chunk_data and isinstance(final_chunk_data["message_streamed"], dict):
                    message_content = final_chunk_data["message_streamed"].get("content")
                    if message_content is not None:
                        last_message_streamed_content = message_content
                    else:
                        raise ValueError("Final JSON object has 'message_streamed' but missing 'content' key.")
                else:
                    raise ValueError("Final JSON object does not contain a valid 'message_streamed' key.")

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed processing response JSON: {e}\nFull Text: {full_response_text}")
                raise ValueError(f"Failed processing response JSON: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error processing response text: {e}")
                raise RuntimeError(f"Unexpected error processing response text: {e}") from e

            if last_message_streamed_content is None:
                raise ValueError("Stream finished but no valid 'message_streamed' content was captured.")

            return last_message_streamed_content

    except Exception as e:
        # Catch httpx errors or errors from raise_for_status
        logger.error(f"Streaming request failed: {e}")
        raise RuntimeError(f"Streaming request failed: {e}") from e


class LLMEvaluator:
    def __init__(self, evaluation_criterion: str, allowed_scores: list[dict] | None = None):
        if allowed_scores is None:
            # Default value assigned here if none provided, avoiding mutable default argument.
            allowed_scores = [{"score": 0, "description": "False"}, {"score": 1, "description": "True"}]
        self.client = AnthropicBedrock(
            aws_region="us-west-2",
        )
        self.allowed_scores = allowed_scores
        self.system_prompt = f"""The assistant is LlmResponseEvaluator.
        LlmResponseEvaluator is given a response from an LLM, labelled as llm-response-to-evaluate.
        LlmResponseEvaluator is also given a single evaluation criterion:
        <evaluation-criterion>\n{evaluation_criterion}\n</evaluation-criterion>\n\n
        LlmResponseEvaluator reads the evaluation-criterion to understand how it should score the llm-response-to-evaluate.
        LlmResponseEvaluator uses the 'thinking' property of the EvaluationTool to think about the it's response before providing a score.
        LlmResponseEvaluator uses the 'score' property of the EvaluationTool to score the response.
        Now connecting LlmResponseEvaluator to the llm-response-to-evaluate.
        """
        self.tool_definitions = [
            {
                "name": "EvaluationTool",
                "description": "Evaluate a response based on the evaluation criteria.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "thinking": {
                            "type": "string",
                            "description": "Use this area to think about the response before providing a score.",
                        },
                        "score": {
                            "type": "number",
                            "description": (
                                "The score for the response. Allowed values are:\n"
                                + "\n".join([f"{s['score']}: {s['description']}" for s in self.allowed_scores])
                            ),
                            "enum": [s["score"] for s in self.allowed_scores],
                        },
                    },
                    "required": ["thinking", "score"],
                },
            }
        ]

    def evaluate(self, response: str) -> float:
        llm_response = self.client.messages.create(
            model="anthropic.claude-3-5-haiku-20241022-v1:0",
            system=self.system_prompt,
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": (f"<llm-response-to-evaluate>\n{response}\n</llm-response-to-evaluate>\n\n"),
                },
            ],
            tools=self.tool_definitions,
        )
        for chunk in llm_response.content:
            if chunk.type == "text":
                logger.info(f"Chunk: {chunk.text}")
            elif chunk.type == "tool_use":
                logger.info(f"Tool use: {chunk.input}")
                score = chunk.input["score"]
                thinking = chunk.input["thinking"]
        return score, thinking


@pytest.mark.asyncio
async def test_gov_uk_search_evals_test_1_hmrc_letter(default_headers, async_client, user_id):
    prompt = """I am writing a letter to be sent to people operating small businesses in the UK. The aim of the letter is to encourage the recipient to fill out a self assessment to return to HMRC.

Below I have provided the contents of my letter draft. Please can you check that the information matches the information on gov.uk?

<letter-draft>
HM Revenue & Customs Self Assessment PO Box 4000 Cardiff CF14 8HR

Reference: UTR XX XX XX XX X

Mr/Ms [Name] [Address Line 1] [Address Line 2] [Postcode]

Dear Mr/Ms [Name],

Complete your Self Assessment tax return for 2024-25

Our records show that you need to complete a Self Assessment tax return for the tax year 6 April 2024 to 5 April 2025.

You must send your tax return by: • 30 October 2025 if you send a paper form • 30 January 2026 if you file online

Why you need to complete a return [You received income from self-employment/You are a company director/You receive rental income] and need to declare this to HMRC.

What you need to do

If you haven't already, register for Self Assessment at www.gov.uk/register-for-self-assessment
Keep records of your income and expenses
Complete and submit your tax return by the deadline
Pay any tax you owe by 30 January 2026
If you miss the deadline, you'll have to pay a penalty of £100, even if you don't owe any tax. Additional penalties apply for further delays.

Need help? • Visit www.gov.uk/self-assessment-tax-returns • Call us on 0300 200 331 • Speak to an accountant or tax adviser

Yours sincerely,

[Name] HM Revenue & Customs
</letter-draft>
"""

    api_class = ENDPOINTS()
    endpoint = api_class.create_chat_stream(user_uuid=user_id)
    payload = {
        "query": prompt,
        "use_rag": False,
        "use_gov_uk_search_api": True,
        "enable_web_browsing": False,
    }

    try:
        # Call the helper function to get the final content
        response = await get_final_streamed_content(
            async_client=async_client, method="POST", endpoint=endpoint, headers=default_headers, json_payload=payload
        )
        logger.info(f"Final response content captured by helper: {response[:200]}...")
    except (ValueError, RuntimeError) as e:
        pytest.fail(f"Failed to get streamed content: {e}")

    llm_evaluator_dates = LLMEvaluator(
        evaluation_criterion="The response should identify the errors in dates (deadlines are 31 October and 31 January, not 30)."
    )
    score, thinking = llm_evaluator_dates.evaluate(response)
    assert score == 1, (
        f"The LLM Evaluator determined that Assist did not correctly identify that the dates are incorrect. "
        f"Assist's response: {response}"
        f"LLM Evaluator's score: {score}, Thinking: {thinking}"
    )

    llm_evaluator_phone_number = LLMEvaluator(
        evaluation_criterion="The response should identify the errors in the phone number (it should be 0300 200 3310)."
    )
    score, thinking = llm_evaluator_phone_number.evaluate(response)
    assert score == 1, (
        f"The LLM Evaluator determined that Assist did not correctly identify that the phone number is incorrect. "
        f"Assist's response: {response}"
        f"LLM Evaluator's score: {score}, Thinking: {thinking}"
    )

    pass  # Test currently ends here
