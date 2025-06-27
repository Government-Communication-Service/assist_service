import logging
from unittest.mock import patch
from uuid import UUID

import pytest

from app.chat.schemas import MessageDefaults
from app.chat.service import chat_system_prompt
from app.database.table import async_db_session

logger = logging.getLogger(__name__)


pytestmark = [
    pytest.mark.messages,
    pytest.mark.unit,
]


def test_message_defaults():
    message = MessageDefaults(chat_id=1, auth_session_id=2, llm_id=3)
    assert isinstance(UUID(message.uuid, version=4), UUID)
    assert message.chat_id == 1
    assert message.auth_session_id == 2
    assert not message.interrupted
    assert message.llm_id == 3
    assert message.tokens == 0


@patch("app.chat.schemas.uuid4")
def test_auto_generated_uuid(mock_uuid4):
    mock_uuid4.side_effect = [
        UUID("123e4567-e89b-12d3-a456-426614174000"),
        UUID("223e4567-e89b-12d3-a456-426614174000"),
    ]

    message1 = MessageDefaults(chat_id=1, auth_session_id=2, llm_id=3)
    message2 = MessageDefaults(chat_id=1, auth_session_id=2, llm_id=3)

    assert message1.uuid != message2.uuid
    assert message1.uuid == "123e4567-e89b-12d3-a456-426614174000"
    assert message2.uuid == "223e4567-e89b-12d3-a456-426614174000"


@pytest.mark.asyncio
async def test_return_type():
    async with async_db_session() as db_session:
        response = await chat_system_prompt(db_session)
        logging.info(f"Generated system prompt: {response}")
    assert isinstance(response, str)
