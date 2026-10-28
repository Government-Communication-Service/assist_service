"""
Tests for loading a chat's full message history (GET .../chats/{chat_uuid}/messages).
"""

import re
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import event, insert, select

from app.chat.service import chat_get_messages
from app.database.models import Chat, ChatDocumentMapping, Document, DocumentUserMapping, Message
from app.database.table import AsyncEngineProvider, async_db_session

pytestmark = [pytest.mark.chat, pytest.mark.messages]

NUM_MESSAGES = 6
NUM_DOCUMENTS = 4


@pytest.fixture
async def chat_with_messages_and_documents(user, auth_session):
    """A chat with several messages and several attached documents, inserted directly."""
    base_time = datetime.now() - timedelta(hours=1)

    async with async_db_session() as db_session:
        chat_id = (
            await db_session.execute(
                insert(Chat)
                .values(uuid=uuid.uuid4(), user_id=user.id, title="Chat with documents", from_open_chat=True)
                .returning(Chat.id)
            )
        ).scalar()

        message_uuids = []
        for i in range(NUM_MESSAGES):
            message_uuid = uuid.uuid4()
            await db_session.execute(
                insert(Message).values(
                    uuid=message_uuid,
                    chat_id=chat_id,
                    content=f"message {i}",
                    content_enhanced_with_rag=f"message {i} with retrieved document text",
                    role="user" if i % 2 == 0 else "assistant",
                    tokens=10,
                    auth_session_id=auth_session.id,
                    interrupted=False,
                    created_at=base_time + timedelta(minutes=i),
                    llm_id=1,
                )
            )
            message_uuids.append(message_uuid)

        document_names = []
        for i in range(NUM_DOCUMENTS):
            document_uuid = uuid.uuid4()
            name = f"document {i}.docx"
            created_at = base_time + timedelta(minutes=i)
            document_id = (
                await db_session.execute(
                    insert(Document)
                    .values(uuid=document_uuid, name=name, is_central=False, created_at=created_at)
                    .returning(Document.id)
                )
            ).scalar()
            await db_session.execute(
                insert(DocumentUserMapping).values(
                    document_id=document_id, user_id=user.id, auth_session_id=auth_session.id
                )
            )
            await db_session.execute(insert(ChatDocumentMapping).values(chat_id=chat_id, document_uuid=document_uuid))
            document_names.append(name)

        chat = (await db_session.execute(select(Chat).where(Chat.id == chat_id))).scalar_one()

    return {"chat": chat, "message_uuids": message_uuids, "document_names": document_names}


@pytest.fixture
def executed_statements():
    """Records the SQL of every statement executed while the test runs."""
    statements = []
    sync_engine = AsyncEngineProvider.get().sync_engine

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement.lower())

    event.listen(sync_engine, "before_cursor_execute", record)
    yield statements
    event.remove(sync_engine, "before_cursor_execute", record)


def _tables_in(sql: str) -> set[str]:
    return set(re.findall(r'(?:from|join)\s+"?(\w+)"?', sql))


async def test_chat_get_messages_returns_all_messages_and_documents(chat_with_messages_and_documents):
    chat_response = await chat_get_messages(chat_with_messages_and_documents["chat"])

    assert [m.uuid for m in chat_response.messages] == chat_with_messages_and_documents["message_uuids"]
    assert [m.content for m in chat_response.messages] == [f"message {i}" for i in range(NUM_MESSAGES)]
    assert [d.name for d in chat_response.documents] == chat_with_messages_and_documents["document_names"]
    assert all(d.created_at is not None for d in chat_response.documents)


async def test_chat_get_messages_does_not_join_messages_to_documents(
    chat_with_messages_and_documents, executed_statements
):
    """
    Joining messages and documents in one query returns one row per message x document pair,
    each repeating the full message text. On large chats that exhausted production worker memory.
    """
    await chat_get_messages(chat_with_messages_and_documents["chat"])

    message_statements = [sql for sql in executed_statements if "message" in _tables_in(sql)]
    assert message_statements, "expected the messages to be loaded"
    for sql in message_statements:
        assert "chat_document_mapping" not in _tables_in(sql)
        assert "content_enhanced_with_rag" not in sql
        assert "summary" not in sql
