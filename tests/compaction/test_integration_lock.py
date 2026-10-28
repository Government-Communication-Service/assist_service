"""Integration tests for the DB-backed compaction lock.

The interesting behaviour here is a real SQL predicate (`ON CONFLICT ... WHERE compaction_lock
= false OR locked_at < stale_cutoff`) — a mocked session can't evaluate that, so this exercises
it against the real test database. No LLM calls are made.

Deliberately does not use the shared `db_session` test fixture: acquire/release each open
their own short-lived, independently-committing session (see acquire_compaction_lock's
docstring for why), which is incompatible with that fixture's one-transaction-per-test model —
a write made through it is never actually committed until the test itself tears down, long
after a separately-opened session would need to see it.
"""

import logging
from datetime import datetime, timedelta

import pytest
from sqlalchemy import update

from app.compaction.service import acquire_compaction_lock, is_compaction_locked, release_compaction_lock
from app.config import settings
from app.database.models import Chat, ChatCompactionLock, User
from app.database.table import async_db_session

logger = logging.getLogger(__name__)


@pytest.fixture
async def chat_id():
    async with async_db_session() as session:
        user = User()
        session.add(user)
        await session.flush()

        chat = Chat(user_id=user.id, from_open_chat=False)
        session.add(chat)
        await session.flush()
        return chat.id


@pytest.mark.asyncio
async def test_first_acquire_succeeds(chat_id):
    assert await acquire_compaction_lock(chat_id) is True


@pytest.mark.asyncio
async def test_second_acquire_fails_while_held(chat_id):
    assert await acquire_compaction_lock(chat_id) is True
    assert await acquire_compaction_lock(chat_id) is False


@pytest.mark.asyncio
async def test_reacquire_after_release_succeeds(chat_id):
    assert await acquire_compaction_lock(chat_id) is True
    await release_compaction_lock(chat_id)
    assert await acquire_compaction_lock(chat_id) is True


@pytest.mark.asyncio
async def test_stale_lock_can_be_reacquired_without_being_released(chat_id):
    """A worker crash never calls release — the staleness check is the only recovery."""
    assert await acquire_compaction_lock(chat_id) is True

    stale_at = datetime.now() - timedelta(minutes=settings.compaction_lock_stale_after_minutes + 1)
    async with async_db_session() as session:
        await session.execute(
            update(ChatCompactionLock).where(ChatCompactionLock.chat_id == chat_id).values(locked_at=stale_at)
        )

    assert await acquire_compaction_lock(chat_id) is True


@pytest.mark.asyncio
async def test_a_recently_held_lock_is_not_treated_as_stale(chat_id):
    assert await acquire_compaction_lock(chat_id) is True
    assert await acquire_compaction_lock(chat_id) is False


@pytest.mark.asyncio
async def test_peek_is_false_with_no_lock_row(chat_id):
    async with async_db_session() as session:
        assert await is_compaction_locked(chat_id, session) is False


@pytest.mark.asyncio
async def test_peek_is_true_while_held(chat_id):
    assert await acquire_compaction_lock(chat_id) is True
    async with async_db_session() as session:
        assert await is_compaction_locked(chat_id, session) is True


@pytest.mark.asyncio
async def test_peek_is_false_after_release(chat_id):
    assert await acquire_compaction_lock(chat_id) is True
    await release_compaction_lock(chat_id)
    async with async_db_session() as session:
        assert await is_compaction_locked(chat_id, session) is False


@pytest.mark.asyncio
async def test_peek_is_false_for_a_stale_lock(chat_id):
    """A worker crash never calls release - the peek must agree with acquire's staleness check."""
    assert await acquire_compaction_lock(chat_id) is True

    stale_at = datetime.now() - timedelta(minutes=settings.compaction_lock_stale_after_minutes + 1)
    async with async_db_session() as session:
        await session.execute(
            update(ChatCompactionLock).where(ChatCompactionLock.chat_id == chat_id).values(locked_at=stale_at)
        )

    async with async_db_session() as session:
        assert await is_compaction_locked(chat_id, session) is False
