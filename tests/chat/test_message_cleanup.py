from logging import getLogger
from time import perf_counter

import pytest
from sqlalchemy import select

from app.api.endpoints import ENDPOINTS
from app.chat.constants import DELETION_NOTICE
from app.chat.schemas import MessageCleanupResponse
from app.database.models import Chat, Message
from app.database.table import async_db_session

logger = getLogger(__name__)
api = ENDPOINTS()

pytestmark = [
    pytest.mark.chat,
    pytest.mark.unit,
]


@pytest.mark.asyncio
async def test_delete_expired_message_content_removes_old_content_and_marks_chat_deleted(
    async_client, async_http_requester, old_chat_with_single_message
):
    """
    Test that messages older than 1 year have their content replaced with deletion notice
    and that chats are marked as deleted when ALL messages in the chat are deleted.
    """
    # Use the fixture to get the old chat data
    chat = old_chat_with_single_message
    chat_id = chat["chat_id"]
    message_id = chat["message_ids"][0]
    old_date = chat["old_date"]
    test_content = "This is old message content that should be deleted"

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
async def test_delete_expired_message_content_preserves_chat_with_recent_messages(
    async_client, async_http_requester, old_chat_with_mixed_messages
):
    """
    Test that chats with mixed old/new messages only have old content cleaned but chat remains active.
    This verifies that chats are only marked as deleted when ALL messages are deleted.
    Creates a chat with a very old message, then adds a new message with current datetime,
    and verifies that only the old message content is removed while the chat stays active.
    """
    # Use the fixture to get the chat with mixed messages
    chat = old_chat_with_mixed_messages
    chat_id = chat["chat_id"]
    old_message_id = chat["message_ids"][0]
    recent_message_id = chat["recent_message_id"]
    recent_content = "This is recent message content that should be preserved"

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


@pytest.mark.skip(reason="Created as a once-off test")
async def test_deletion_does_not_block_message_requests(
    multiple_old_chats_factory, user_id, async_client, async_http_requester
):
    """
    Test that message cleanup does not block concurrent chat creation requests.

    This test creates 1000 old chats, measures baseline chat creation time,
    runs cleanup in background, and then measures if a concurrent chat creation
    is blocked by the cleanup operation.
    """
    # Step 1: Create 1000 old chats with 2 messages each to increase cleanup workload
    logger.info("Creating 5,000 old chats for cleanup testing...")
    creation_start = perf_counter()
    old_chats = await multiple_old_chats_factory(
        count=5000,
        days_old=400,  # > 365 days, will be cleaned
        num_messages=4,
        chat_title="Old Chat for Blocking Test",
    )
    creation_time = perf_counter() - creation_start
    logger.info(f"Created {len(old_chats)} old chats in {creation_time:.2f}s")

    async def start_chat():
        """Helper to create a new chat via API"""
        url = api.chats(user_uuid=user_id)
        response = await async_http_requester(
            "test_deletion_does_not_block_message_requests",
            async_client.post,
            url,
            json={"query": "Say Hi"},
        )
        return response

    async def cleanup_via_endpoint():
        """Helper to call cleanup endpoint (proper transaction handling)"""
        cleanup_url = api.build_url(api.CHAT_CLEANUP_EXPIRED_CONTENT)
        response = await async_http_requester(
            "test_deletion_does_not_block_message_requests_cleanup",
            async_client.delete,
            cleanup_url,
        )
        return response

    # Step 2: Baseline - Create a chat and measure time (before any cleanup)
    logger.info("Measuring baseline chat creation time...")
    baseline_start = perf_counter()
    baseline_response = await start_chat()
    baseline_time = perf_counter() - baseline_start
    logger.info(f"Baseline chat creation time: {baseline_time:.4f}s")
    assert baseline_response is not None, "Baseline chat creation failed"

    # Step 3: Start cleanup in background and measure its duration
    logger.info("Starting cleanup operation in background...")
    import asyncio

    cleanup_start = perf_counter()
    cleanup_task = asyncio.create_task(cleanup_via_endpoint())

    # Step 4: Immediately try to create a new chat while cleanup is running
    # Give cleanup a tiny head start to ensure it's actually running
    await asyncio.sleep(0.01)

    logger.info("Creating chat concurrently with cleanup operation...")
    concurrent_start = perf_counter()
    concurrent_response = await start_chat()
    concurrent_time = perf_counter() - concurrent_start
    logger.info(f"Concurrent chat creation time: {concurrent_time:.4f}s")

    # Check if cleanup is still running (if not, chat was blocked)
    cleanup_still_running = not cleanup_task.done()

    # Step 5: Wait for cleanup to complete and measure total cleanup time
    cleanup_response = await cleanup_task
    cleanup_total_time = perf_counter() - cleanup_start
    logger.info(f"Cleanup operation completed in {cleanup_total_time:.2f}s")

    # Verify cleanup worked
    cleanup_result = MessageCleanupResponse(**cleanup_response)
    assert cleanup_result.status == "success"
    logger.info(f"Cleanup processed {cleanup_result.cleaned_count} messages")

    # Step 6: Analysis - Determine if cleanup blocked chat creation
    slowdown_ratio = concurrent_time / baseline_time
    logger.info(f"\n{'=' * 60}")
    logger.info("BLOCKING ANALYSIS:")
    logger.info(f"  Baseline chat creation:    {baseline_time:.4f}s")
    logger.info(f"  Concurrent chat creation:  {concurrent_time:.4f}s")
    logger.info(f"  Cleanup duration:          {cleanup_total_time:.2f}s")
    logger.info(f"  Slowdown ratio:            {slowdown_ratio:.2f}x")
    logger.info(f"  Chat completed before cleanup finished: {'YES' if cleanup_still_running else 'NO'}")
    logger.info(f"{'=' * 60}\n")

    if slowdown_ratio > 2.0:
        logger.warning(f"⚠️  BLOCKING DETECTED: Concurrent chat creation was {slowdown_ratio:.2f}x slower than baseline")
        logger.warning("   This suggests the cleanup operation may be blocking new chat requests")
    elif slowdown_ratio > 1.5:
        logger.info(f"⚡ MINOR IMPACT: Concurrent chat creation was {slowdown_ratio:.2f}x slower than baseline")
        logger.info("   Some impact detected but not significant blocking")
    else:
        logger.info(f"✓  NO BLOCKING: Concurrent chat creation was only {slowdown_ratio:.2f}x baseline time")
        logger.info("   Cleanup does not appear to block new chat requests")

    if cleanup_still_running:
        logger.info("✓  CONCURRENT EXECUTION CONFIRMED: Chat completed while cleanup was still running")
        logger.info("   This proves the cleanup operation does not block new chat requests")

    # Assertions
    assert concurrent_response is not None, "Concurrent chat creation failed"

    # Optional: Assert that blocking is not excessive (allows for some database contention)
    # If this fails, it indicates a serious blocking issue
    assert slowdown_ratio < 2.0, (
        f"Severe blocking detected: concurrent chat creation was {slowdown_ratio:.2f}x slower than baseline. "
        "This indicates the cleanup operation is significantly blocking new requests."
    )
