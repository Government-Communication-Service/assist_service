import asyncio
import os

import httpx
from anthropic import AsyncAnthropicBedrock, AsyncClient, AsyncStream
from dotenv import load_dotenv

BEDROCK_API_READ_TIMEOUT_SECS = 115
BEDROCK_API_CONNECT_TIMEOUT_SECS = 3

load_dotenv()
client: AsyncClient = AsyncAnthropicBedrock(
    aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_region="us-west-2",
    timeout=httpx.Timeout(timeout=BEDROCK_API_READ_TIMEOUT_SECS, connect=BEDROCK_API_CONNECT_TIMEOUT_SECS),
    max_retries=0,
)


async def ping():
    stream: AsyncStream = await client.messages.create(
        model="us.anthropic.claude-opus-4-20250514-v1:0",
        max_tokens=32000,
        messages=[{"role": "user", "content": "Write a poem with 12 verses."}],
        stream=True,
    )
    async for chunk in stream:
        print(chunk)


asyncio.run(ping())
