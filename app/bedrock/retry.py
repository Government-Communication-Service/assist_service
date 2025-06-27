import logging
from asyncio import iscoroutinefunction
from functools import wraps
from typing import TYPE_CHECKING, Callable

from app.bedrock.bedrock_types import (
    AnthropicBedrockProvider,
    AsyncAnthropicBedrockProvider,
    BedrockError,
    BedrockErrorType,
)
from app.config import AWS_BEDROCK_REGION1, AWS_BEDROCK_REGION2

if TYPE_CHECKING:
    from app.bedrock.bedrock import BedrockHandler

AWS_BEDROCK_REGIONS = (AWS_BEDROCK_REGION1, AWS_BEDROCK_REGION2)
AWS_BEDROCK_REGIONS_MAX_RETRIES: int = 3

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


def handle_region_failover_with_retries(func):
    """
    Decorator that handles region failover with retries when making AWS Bedrock API calls.
    It retries the operation by switching between AWS regions. The failover occurs between two AWS regions
    defined  in variable AWS_BEDROCK_REGIONS. The decorated function will be retried up to the specified
    `AWS_BEDROCK_REGIONS_MAX_RETRIES` count, using the alternate region for each retry.

    Args:
        max_retries (int): The maximum number of retry attempts for a failed operation. Default is 3.

    Returns:
        A decorated function that retries the operation with failover between AWS regions.

    Raises:
        BedrockError: If the operation fails after the specified number of retries.

    """

    @wraps(func)
    def wrapper(bedrock_handler: "BedrockHandler", *args, **kwargs):
        """
        Wrapper function for retrying the decorated function with region failover.
        It changes bedrock_handler's client to a new client with the new AWS region

        Args:
            bedrock_handler: The handler for AWS Bedrock API calls.
            *args: Positional arguments passed to the decorated function.
            **kwargs: Keyword arguments passed to the decorated function.

        Returns:
            The result of the decorated function if successful.

        Raises:
            BedrockError: If the operation fails after the specified number of retries.
        """

        retries = 0
        # reference to last exception in case retry attempts failed
        ex = None
        while retries < AWS_BEDROCK_REGIONS_MAX_RETRIES:
            try:
                return func(bedrock_handler, *args, **kwargs)
            except Exception as e:
                new_region = switch_region(bedrock_handler.client.aws_region, e)
                # swap client for new region
                bedrock_handler.client = AnthropicBedrockProvider.get(new_region)
                retries += 1
                ex = e

        logger.error("AWS Bedrock call failed, last exception: %s", ex)
        raise BedrockError(f"AWS Bedrock call failed, last exception: {str(ex)}") from ex

    @wraps(func)
    async def async_wrapper(bedrock_handler: "BedrockHandler", *args, **kwargs):
        """
        Asynchronous wrapper function for retrying the decorated function with region failover.
        It changes bedrock_handler's async_client to a new client with the new AWS region

        Args:
            bedrock_handler: The handler for AWS Bedrock API calls.
            *args: Positional arguments passed to the decorated function.
            **kwargs: Keyword arguments passed to the decorated function.

        Returns:
            The result of the decorated function if successful.

        Raises:
            Exception: If the operation fails after the specified number of retries.
        """

        retries = 0
        # reference to last exception in case retry attempts failed
        ex = None
        while retries < AWS_BEDROCK_REGIONS_MAX_RETRIES:
            try:
                return await func(bedrock_handler, *args, **kwargs)
            except Exception as e:
                new_region = switch_region(bedrock_handler.async_client.aws_region, e)
                bedrock_handler.async_client = AsyncAnthropicBedrockProvider.get(new_region)
                retries += 1
                ex = e

        logger.exception("AWS Bedrock call failed, last exception: %s", ex)
        raise BedrockError(f"AWS Bedrock call failed, last exception: {str(ex)}") from ex

    # check which wrapper to use
    return async_wrapper if iscoroutinefunction(func) else wrapper


async def with_region_failover_for_streaming(
    bedrock_handler: "BedrockHandler",
    func,
    on_error: Callable[[Exception], str],
    *args,
    **kwargs,
):
    """
    Decorator that handles region failover with retries when making AWS Bedrock API calls.
    It retries the operation by switching between AWS regions. The failover occurs between two AWS regions
    defined  in variable AWS_BEDROCK_REGIONS. The decorated function will be retried up to the specified
    `AWS_BEDROCK_REGIONS_MAX_RETRIES` count, using the alternate region for each retry.

    If streaming encounters an error after it has started, on_error function is used to generate a custom error json msg

    Args:
        bedrock_handler: The handler for AWS Bedrock API calls.
        func: The streaming Generator function that generates json strings
        on_error: The function taking an exception and generating a json string,
        used when an error encountered after the streaming has started.
        *args: Positional arguments passed to the decorated function.
        **kwargs: Keyword arguments passed to the decorated function.

    Raises:
        BedrockError: If the operation fails after the specified number of retries.

    Returns:
        The result of the streaming function if no errors
    """

    retries = 0
    # reference to last exception in case retry attempts failed
    ex = None
    while retries < AWS_BEDROCK_REGIONS_MAX_RETRIES:
        streaming_started = False
        try:
            async for item in func(*args, **kwargs):
                yield item
                streaming_started = True
            # this is to exit while loop when streaming is completed.
            return
        except Exception as e:
            ex = e
            ex_msg = f"{e}"
            # check if error is due to large input, which can't be retried.
            if (
                isinstance(ex, ValueError)
                and "validationException" in ex_msg
                and (
                    "Input is too long for requested model." in ex_msg
                    or "input length and `max_tokens` exceed context limit" in ex_msg
                )
            ):
                error_msg = "Input is too long"
                logger.error(f"{error_msg}: {type(ex)}: {ex}")
                yield on_error(BedrockError(error_msg, BedrockErrorType.INPUT_TOO_LONG))
                return
            # if error happened at the beginning of streaming, then switch over region
            if not streaming_started:
                new_region = switch_region(bedrock_handler.async_client.aws_region, e)
                # swap client for new region
                bedrock_handler.async_client = AsyncAnthropicBedrockProvider.get(new_region)
                retries += 1
            else:
                # stream a custom error message payload if exception happened midway through streaming
                yield on_error(e)
                logger.error("AWS Bedrock error through streaming, exception: %s", ex)
                return

    # all attempts failed
    logger.error("AWS Bedrock call failed, last exception: %s:%s", type(ex), ex)
    yield on_error(ex)
