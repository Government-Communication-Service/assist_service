import asyncio
import logging
import random
import time
from asyncio import iscoroutinefunction
from functools import wraps
from typing import TYPE_CHECKING

import anthropic
import httpx

from app.bedrock.bedrock_types import (
    AnthropicBedrockProvider,
    AsyncAnthropicBedrockProvider,
    BedrockError,
    BedrockErrorType,
)
from app.config import (
    AWS_BEDROCK_REGION1,
    AWS_BEDROCK_REGION2,
    AWS_BEDROCK_REGIONS_MAX_RETRIES,
    BEDROCK_RETRY_BACKOFF_BASE_SECONDS,
    BEDROCK_RETRY_BACKOFF_MAX_SECONDS,
    BEDROCK_RETRY_BUDGET_SECONDS,
    STREAM_FIRST_CHUNK_TIMEOUT,
)

if TYPE_CHECKING:
    from app.bedrock.bedrock import BedrockHandler

AWS_BEDROCK_REGIONS = (AWS_BEDROCK_REGION1, AWS_BEDROCK_REGION2)

logger = logging.getLogger()


def switch_region(current_region: str, ex: Exception) -> str:
    """
    Switches the AWS region in case of a failure, logging the region change and returning the new region.

    Args:
        current_region (str): The current AWS region.
        ex (Exception): The exception raised during the API call.

    Returns:
        str: The new region to switch to.
    """
    idx_current_region = AWS_BEDROCK_REGIONS.index(current_region)
    # swap region
    idx_current_region ^= 1
    new_region = AWS_BEDROCK_REGIONS[idx_current_region]
    logger.warning(f"Error in bedrock handler: {ex}, swapping from {current_region} to {new_region}")
    return new_region


def _is_retryable(ex: Exception) -> bool:
    """Whether a failed call is worth retrying: throttling (429), server errors (5xx), connection errors
    and timeouts. Anything else, such as a bad request or an auth error, fails the same way every time."""
    if isinstance(ex, anthropic.APIStatusError):
        return ex.status_code == 429 or ex.status_code >= 500
    # APITimeoutError is an APIConnectionError
    return isinstance(ex, (anthropic.APIConnectionError, httpx.TransportError, TimeoutError))


def _backoff_seconds(attempt: int) -> float:
    """Full-jitter exponential backoff before retry number `attempt + 1`."""
    return random.uniform(0, min(BEDROCK_RETRY_BACKOFF_MAX_SECONDS, BEDROCK_RETRY_BACKOFF_BASE_SECONDS * 2**attempt))


def _retry_delay(ex: Exception, attempt: int, deadline: float) -> float | None:
    """The backoff before the next attempt, or None if the call should give up now."""
    if not _is_retryable(ex) or attempt >= AWS_BEDROCK_REGIONS_MAX_RETRIES:
        return None
    delay = _backoff_seconds(attempt)
    if time.monotonic() + delay >= deadline:
        return None
    return delay


def _final_error(ex: Exception, attempts: int, region: str) -> BedrockError:
    """Log the failure with its stack and turn it into a BedrockError carrying the error type callers check."""
    logger.error(
        "AWS Bedrock call failed after %d attempt(s) (region: %s): %s: %s",
        attempts,
        region,
        type(ex).__name__,
        ex,
        exc_info=ex,
    )
    if _is_input_too_long(ex):
        return BedrockError("Input is too long", BedrockErrorType.INPUT_TOO_LONG)
    if isinstance(ex, (anthropic.APITimeoutError, httpx.TimeoutException, TimeoutError)):
        return BedrockError(f"AWS Bedrock call timed out, last exception: {ex!r}", BedrockErrorType.TIMEOUT)
    return BedrockError(f"AWS Bedrock call failed, last exception: {ex}")


def handle_region_failover_with_retries(func):
    """
    Decorator that retries a non-stream AWS Bedrock call in the other region on a transient failure.

    Only throttling, 5xx, connection errors and timeouts are retried, up to `AWS_BEDROCK_REGIONS_MAX_RETRIES`
    times, alternating between the regions in AWS_BEDROCK_REGIONS with jittered exponential backoff. The whole
    call, retries included, is bounded by `BEDROCK_RETRY_BUDGET_SECONDS`, which is less than the request timeout.

    Raises:
        BedrockError: If the call fails with a non-retryable error, runs out of retries or runs out of time.
            error_type is INPUT_TOO_LONG or TIMEOUT when the last error was one of those.
    """

    @wraps(func)
    def wrapper(bedrock_handler: "BedrockHandler", *args, **kwargs):
        deadline = time.monotonic() + BEDROCK_RETRY_BUDGET_SECONDS
        attempt = 0
        while True:
            region = bedrock_handler.client.aws_region
            try:
                return func(bedrock_handler, *args, **kwargs)
            except Exception as e:
                delay = _retry_delay(e, attempt, deadline)
                if delay is None:
                    raise _final_error(e, attempt + 1, region) from e
                bedrock_handler.client = AnthropicBedrockProvider.get(switch_region(region, e))
            time.sleep(delay)
            attempt += 1

    @wraps(func)
    async def async_wrapper(bedrock_handler: "BedrockHandler", *args, **kwargs):
        deadline = time.monotonic() + BEDROCK_RETRY_BUDGET_SECONDS
        attempt = 0
        while True:
            region = bedrock_handler.async_client.aws_region
            try:
                return await asyncio.wait_for(
                    func(bedrock_handler, *args, **kwargs), timeout=deadline - time.monotonic()
                )
            except Exception as e:
                delay = _retry_delay(e, attempt, deadline)
                if delay is None:
                    raise _final_error(e, attempt + 1, region) from e
                bedrock_handler.async_client = AsyncAnthropicBedrockProvider.get(switch_region(region, e))
            await asyncio.sleep(delay)
            attempt += 1

    # check which wrapper to use
    return async_wrapper if iscoroutinefunction(func) else wrapper


def _is_input_too_long(ex: Exception) -> bool:
    """Whether a Bedrock error says the request exceeds the model's context window."""
    ex_msg = str(ex)
    return "Input is too long for requested model." in ex_msg or (
        "input length and `max_tokens` exceed context limit" in ex_msg
    )


async def stream_with_region_failover(bedrock_handler: "BedrockHandler", stream_func, **kwargs):
    """Run one streaming call, retrying in the other region until its first event arrives.

    Once the first event has been yielded the call is committed to its region: later errors
    are raised, not retried, so a caller never sees a response twice.

    Args:
        bedrock_handler: Handler whose `async_client` is used and swapped on failover.
        stream_func: Async generator function taking the client as its first argument.
        **kwargs: Passed to stream_func.

    Yields:
        The events from stream_func.

    Raises:
        BedrockError: INPUT_TOO_LONG if the request exceeds the context window; TIMEOUT or a
            generic error if every attempt failed before its first event.
        Exception: Whatever stream_func raised after its first event.
    """
    retries = 0
    last_ex = None
    while retries <= AWS_BEDROCK_REGIONS_MAX_RETRIES:
        region = bedrock_handler.async_client.aws_region
        async_gen = stream_func(bedrock_handler.async_client, **kwargs).__aiter__()
        try:
            start_time = time.monotonic()
            try:
                first_event = await asyncio.wait_for(async_gen.__anext__(), timeout=STREAM_FIRST_CHUNK_TIMEOUT)
            except asyncio.TimeoutError as err:
                logger.warning(
                    "Stream first event timeout after %.2fs (region: %s)", time.monotonic() - start_time, region
                )
                raise BedrockError(
                    f"Stream timed out waiting for first event after {STREAM_FIRST_CHUNK_TIMEOUT}s",
                    BedrockErrorType.TIMEOUT,
                ) from err
        except Exception as e:
            try:
                await async_gen.aclose()
            except Exception as close_error:
                logger.warning("Failed to close stream after an error: %s", close_error)
            if _is_input_too_long(e):
                logger.error(f"Input is too long: {type(e)}: {e}")
                raise BedrockError("Input is too long", BedrockErrorType.INPUT_TOO_LONG) from e
            new_region = switch_region(region, e)
            bedrock_handler.async_client = AsyncAnthropicBedrockProvider.get(new_region)
            retries += 1
            last_ex = e
            continue

        logger.info("Stream first event received in %.2fs (region: %s)", time.monotonic() - start_time, region)
        try:
            yield first_event
            async for event in async_gen:
                yield event
        finally:
            await async_gen.aclose()
        return

    logger.error("AWS Bedrock call failed, last exception: %s:%s", type(last_ex), last_ex)
    raise BedrockError(f"AWS Bedrock call failed, last exception: {last_ex}") from last_ex
