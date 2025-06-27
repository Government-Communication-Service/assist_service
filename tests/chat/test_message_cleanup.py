import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import insert, select

from app.api import ENDPOINTS
from app.chat.constants import DELETION_NOTICE
from app.chat.schemas import MessageCleanupResponse
from app.database.models import Chat, Message
from app.database.table import async_db_session

api = ENDPOINTS()

pytestmark = [
    pytest.mark.chat,
    pytest.mark.unit,
]


@pytest.mark.asyncio
async def test_delete_expired_message_content_removes_old_content_and_marks_chat_deleted(
    async_client, async_http_requester
):
    """
    Test that messages older than 1 year have their content replaced with deletion notice
    and that chats are marked as deleted when ALL messages in the chat are deleted.
    """

    old_date = datetime.now() - timedelta(days=400)  # More than 1 year ago
    test_content = "This is old message content that should be deleted"

    # Create a new chat and insert old message
    async with async_db_session() as db_session:
        # Create a new chat
        chat_uuid = uuid.uuid4()
        chat_stmt = (
            insert(Chat)
            .values(
                uuid=chat_uuid,
                user_id=1,  # Dummy user ID
                title="Test Chat for Cleanup",
                from_open_chat=True,
                use_rag=False,
                use_gov_uk_search_api=False,
                created_at=old_date,
            )
            .returning(Chat.id)
        )

        chat_result = await db_session.execute(chat_stmt)
        chat_id = chat_result.scalar()

        # Insert old message - this will be the ONLY message in the chat
        message_stmt = (
            insert(Message)
            .values(
                chat_id=chat_id,
                content=test_content,
                content_enhanced_with_rag=test_content,
                role="user",
                tokens=10,
                auth_session_id=1,  # Dummy auth session
                interrupted=False,
                created_at=old_date,
                llm_id=1,  # Dummy LLM ID
            )
            .returning(Message.id)
        )

        message_result = await db_session.execute(message_stmt)
        message_id = message_result.scalar()

    # Verify the test message was actually created with old date (separate session)
    async with async_db_session() as db_session:
        verify_stmt = select(Message).where(Message.id == message_id)
        verify_result = await db_session.execute(verify_stmt)
        created_message: Message = verify_result.scalar()

        assert created_message is not None, "Test message was not created"
        assert created_message.content == test_content, "Test message content doesn't match"
        assert created_message.content_enhanced_with_rag == test_content, "Test message content doesn't match"
        assert created_message.created_at == old_date, "Test message created_at date is not old enough"
        assert created_message.deleted_at is None, "Test message should not be deleted initially"

    # Call the cleanup endpoint
    cleanup_url = api.build_url(api.CHAT_CLEANUP_EXPIRED_CONTENT)
    response = await async_http_requester(
        "test_delete_expired_message_content_removes_old_content_and_marks_chat_deleted",
        async_client.delete,
        cleanup_url,
    )

    # Verify response
    cleanup_response = MessageCleanupResponse(**response)
    assert cleanup_response.status == "success"
    assert cleanup_response.cleaned_count >= 1  # At least our test message was cleaned

    # Verify the message content was replaced with deletion notice (separate session)
    async with async_db_session() as db_session:
        stmt = select(Message).where(Message.id == message_id)
        result = await db_session.execute(stmt)
        updated_message: Message = result.scalar()

        assert updated_message.content == DELETION_NOTICE  # Content should be replaced with notice
        assert updated_message.content_enhanced_with_rag == DELETION_NOTICE  # Content should be replaced with notice
        assert updated_message.deleted_at is not None  # deleted_at should be set

        # Verify the associated chat is also marked as deleted (since ALL messages are now deleted)
        chat_stmt = select(Chat).where(Chat.id == chat_id)
        chat_result = await db_session.execute(chat_stmt)
        updated_chat: Chat = chat_result.scalar()

        assert updated_chat.deleted_at is not None  # Chat should be marked as deleted
        assert updated_chat.title is not None  # But title should be preserved


@pytest.mark.asyncio
async def test_delete_expired_message_content_preserves_chat_with_recent_messages(async_client, async_http_requester):
    """
    Test that chats with mixed old/new messages only have old content cleaned but chat remains active.
    This verifies that chats are only marked as deleted when ALL messages are deleted.
    Creates a chat with a very old message, then adds a new message with current datetime,
    and verifies that only the old message content is removed while the chat stays active.
    """

    old_date = datetime.now() - timedelta(days=400)  # More than 1 year ago
    recent_date = datetime.now()  # Current datetime
    old_content = "This is old message content that should be deleted"
    recent_content = "This is recent message content that should be preserved"

    # Create a new chat and insert both old and recent messages
    async with async_db_session() as db_session:
        # Create a new chat
        chat_uuid = uuid.uuid4()
        chat_stmt = (
            insert(Chat)
            .values(
                uuid=chat_uuid,
                user_id=1,  # Dummy user ID
                title="Test Chat with Mixed Messages",
                from_open_chat=True,
                use_rag=False,
                use_gov_uk_search_api=False,
                created_at=old_date,  # Chat created long ago
            )
            .returning(Chat.id)
        )

        chat_result = await db_session.execute(chat_stmt)
        chat_id = chat_result.scalar()

        # Insert old message first
        old_stmt = (
            insert(Message)
            .values(
                chat_id=chat_id,
                content=old_content,
                content_enhanced_with_rag=old_content,
                role="user",
                tokens=10,
                auth_session_id=1,
                interrupted=False,
                created_at=old_date,
                llm_id=1,
            )
            .returning(Message.id)
        )

        old_result = await db_session.execute(old_stmt)
        old_message_id = old_result.scalar()

        # Insert recent message (simulating someone adding a new message to an old chat)
        recent_stmt = (
            insert(Message)
            .values(
                chat_id=chat_id,
                content=recent_content,
                content_enhanced_with_rag=recent_content,
                role="user",
                tokens=10,
                auth_session_id=1,
                interrupted=False,
                created_at=recent_date,
                llm_id=1,
            )
            .returning(Message.id)
        )

        recent_result = await db_session.execute(recent_stmt)
        recent_message_id = recent_result.scalar()

    # Call the cleanup endpoint
    cleanup_url = api.build_url(api.CHAT_CLEANUP_EXPIRED_CONTENT)
    response = await async_http_requester(
        "test_delete_expired_message_content_preserves_chat_with_recent_messages",
        async_client.delete,
        cleanup_url,
    )

    # Verify response
    cleanup_response = MessageCleanupResponse(**response)
    assert cleanup_response.status == "success"
    assert cleanup_response.cleaned_count >= 1  # At least our old message was cleaned

    # Verify the results (separate session)
    async with async_db_session() as db_session:
        # Check old message was cleaned
        old_stmt = select(Message).where(Message.id == old_message_id)
        old_result = await db_session.execute(old_stmt)
        old_message: Message = old_result.scalar()

        assert old_message.content == DELETION_NOTICE
        assert old_message.content_enhanced_with_rag == DELETION_NOTICE
        assert old_message.deleted_at is not None

        # Check recent message was preserved
        recent_stmt = select(Message).where(Message.id == recent_message_id)
        recent_result = await db_session.execute(recent_stmt)
        recent_message: Message = recent_result.scalar()

        assert recent_message.content == recent_content
        assert recent_message.content_enhanced_with_rag == recent_content
        assert recent_message.deleted_at is None  # Should not be deleted

        # Verify the chat is NOT marked as deleted (since not ALL messages are deleted)
        chat_stmt = select(Chat).where(Chat.id == chat_id)
        chat_result = await db_session.execute(chat_stmt)
        updated_chat: Chat = chat_result.scalar()

        assert updated_chat.deleted_at is None  # Chat should NOT be marked as deleted
        assert updated_chat.title is not None  # Title should be preserved
