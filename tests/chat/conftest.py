"""
Fixtures for chat-related tests.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional

import pytest
from sqlalchemy import insert

from app.database.models import Chat, Message
from app.database.table import async_db_session


@pytest.fixture
async def old_chat_factory(user, auth_session):
    """
    Factory fixture that creates old chats with configurable parameters.

    This fixture returns a callable that can be used to create one or more old chats
    with various configurations. It's designed to support both simple single-chat tests
    and complex scale testing scenarios (e.g., creating 100+ old chats).

    Args:
        user: User fixture (from tests/conftest.py)
        auth_session: AuthSession fixture (from tests/conftest.py)

    Returns:
        async function that creates old chats based on provided parameters

    Example:
        async def test_cleanup(old_chat_factory):
            # Create a single old chat with 3 messages
            chat = await old_chat_factory(days_old=400, num_messages=3)

            # Create another old chat with specific configuration
            chat2 = await old_chat_factory(
                days_old=500,
                chat_title="Custom Title",
                use_rag=True
            )
    """
    created_chats = []  # Track created chats for reference

    async def _create_old_chat(
        days_old: int = 400,
        num_messages: int = 1,
        chat_title: str = "Test Chat for Cleanup",
        message_content: str = "This is old message content that should be deleted",
        include_recent_message: bool = False,
        use_case_id: Optional[int] = None,
        use_rag: bool = False,
        use_gov_uk_search_api: bool = False,
    ):
        """
        Creates an old chat with messages.

        Args:
            days_old: How many days in the past to create the chat (default 400, > 365 triggers cleanup)
            num_messages: Number of old messages to create in the chat (default 1)
            chat_title: Title for the chat
            message_content: Base content for messages (numbered if num_messages > 1)
            include_recent_message: If True, adds a recent (non-old) message to the chat
            use_case_id: Optional use case ID to associate with the chat
            use_rag: Whether the chat uses RAG functionality
            use_gov_uk_search_api: Whether the chat uses Gov.UK Search API

        Returns:
            dict containing:
                - chat_id: Database ID of the created chat
                - chat_uuid: UUID of the created chat
                - message_ids: List of database IDs for old messages
                - recent_message_id: Database ID of recent message (if include_recent_message=True)
                - old_date: The datetime used for old content
                - user_id: ID of the user who owns the chat
        """
        old_date = datetime.now() - timedelta(days=days_old)

        async with async_db_session() as db_session:
            # Create the chat
            chat_uuid = uuid.uuid4()
            chat_stmt = (
                insert(Chat)
                .values(
                    uuid=chat_uuid,
                    user_id=user.id,
                    title=chat_title,
                    from_open_chat=True,
                    use_rag=use_rag,
                    use_gov_uk_search_api=use_gov_uk_search_api,
                    created_at=old_date,
                    use_case_id=use_case_id,
                )
                .returning(Chat.id)
            )
            chat_result = await db_session.execute(chat_stmt)
            chat_id = chat_result.scalar()

            # Create old messages
            message_ids = []
            for i in range(num_messages):
                # Number messages if there are multiple
                content = f"{message_content} #{i + 1}" if num_messages > 1 else message_content

                message_stmt = (
                    insert(Message)
                    .values(
                        chat_id=chat_id,
                        content=content,
                        content_enhanced_with_rag=content,
                        role="user",
                        tokens=10,
                        auth_session_id=auth_session.id,
                        interrupted=False,
                        created_at=old_date,
                        llm_id=1,
                    )
                    .returning(Message.id)
                )
                message_result = await db_session.execute(message_stmt)
                message_ids.append(message_result.scalar())

            # Optionally add a recent message (for testing mixed old/new scenarios)
            recent_message_id = None
            if include_recent_message:
                recent_stmt = (
                    insert(Message)
                    .values(
                        chat_id=chat_id,
                        content="This is recent message content that should be preserved",
                        content_enhanced_with_rag="This is recent message content that should be preserved",
                        role="user",
                        tokens=10,
                        auth_session_id=auth_session.id,
                        interrupted=False,
                        created_at=datetime.now(),
                        llm_id=1,
                    )
                    .returning(Message.id)
                )
                recent_result = await db_session.execute(recent_stmt)
                recent_message_id = recent_result.scalar()

            await db_session.commit()

            result = {
                "chat_id": chat_id,
                "chat_uuid": chat_uuid,
                "message_ids": message_ids,
                "recent_message_id": recent_message_id,
                "old_date": old_date,
                "user_id": user.id,
            }
            created_chats.append(result)
            return result

    yield _create_old_chat


@pytest.fixture
async def old_chat_with_single_message(old_chat_factory):
    """
    Convenience fixture for the common case: single old chat with one old message.

    This is pre-configured for the most common test scenario where you need
    a chat with exactly one old message that will be cleaned up.

    Returns:
        dict with chat_id, chat_uuid, message_ids (list with 1 ID), old_date, user_id

    Example:
        async def test_cleanup(old_chat_with_single_message):
            chat = old_chat_with_single_message
            message_id = chat['message_ids'][0]
            # Test cleanup logic
    """
    return await old_chat_factory(
        days_old=400,
        num_messages=1,
        chat_title="Test Chat for Cleanup",
        message_content="This is old message content that should be deleted",
    )


@pytest.fixture
async def old_chat_with_mixed_messages(old_chat_factory):
    """
    Convenience fixture for mixed old/new message scenario.

    Creates a chat with both old messages (that should be cleaned) and
    a recent message (that should be preserved). Useful for testing that
    cleanup only affects old content and doesn't mark chats as deleted
    when they still have recent messages.

    Returns:
        dict with chat_id, chat_uuid, message_ids (old messages),
        recent_message_id, old_date, user_id

    Example:
        async def test_mixed_cleanup(old_chat_with_mixed_messages):
            chat = old_chat_with_mixed_messages
            old_message_id = chat['message_ids'][0]
            recent_message_id = chat['recent_message_id']
            # Test that only old message is cleaned
    """
    return await old_chat_factory(
        days_old=400,
        num_messages=1,
        chat_title="Test Chat with Mixed Messages",
        message_content="This is old message content that should be deleted",
        include_recent_message=True,
    )


@pytest.fixture
async def multiple_old_chats_factory(old_chat_factory):
    """
    Factory for creating multiple old chats at once.

    This fixture is designed for scale testing scenarios where you need to create
    many old chats (e.g., 100+) to test performance and bulk operations.

    Returns:
        async function that creates N old chats with the same configuration

    Example:
        async def test_bulk_cleanup(multiple_old_chats_factory):
            # Create 100 old chats for scale testing
            chats = await multiple_old_chats_factory(
                count=100,
                days_old=400,
                num_messages=2
            )

            # Test bulk cleanup performance
            assert len(chats) == 100
    """

    async def _create_multiple(count: int = 3, **kwargs):
        """
        Creates multiple old chats with the same configuration.

        Args:
            count: Number of chats to create
            **kwargs: Additional parameters passed to old_chat_factory
                     (days_old, num_messages, chat_title, etc.)

        Returns:
            list of chat info dicts, one per created chat
        """
        chats = []
        base_title = kwargs.get("chat_title", "Old Test Chat")

        for i in range(count):
            # Customize title for each chat to make them distinguishable
            kwargs["chat_title"] = f"{base_title} #{i + 1}"
            chat = await old_chat_factory(**kwargs)
            chats.append(chat)

        return chats

    return _create_multiple
