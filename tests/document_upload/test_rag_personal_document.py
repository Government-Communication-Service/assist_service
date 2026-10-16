import asyncio
import datetime
import json
import logging

import pytest
from sqlalchemy.future import select

from app.api.endpoints import ENDPOINTS
from app.database.models import (
    Document,
    DocumentUserMapping,
    Message,
    MessageDocumentChunkMapping,
    MessageSearchIndexMapping,
    RewrittenQuery,
    User,
)
from app.database.table import async_db_session

logger = logging.getLogger()

pytestmark = pytest.mark.usefixtures("mock_llm_response")

SEARCH_DOCUMENTS = "search_uploaded_documents"
SEARCH_CENTRAL_GUIDANCE = "search_central_guidance"


def _offered_tools(request: dict) -> set[str]:
    return {tool["name"] for tool in request.get("tools", [])}


def _search(*queries: str) -> list[tuple[str, dict]]:
    return [(SEARCH_DOCUMENTS, {"queries": list(queries)})]


@pytest.mark.asyncio
async def test_use_personal_document_in_rag_process(
    async_client,
    user_id,
    async_http_requester,
    file_uploader,
    mock_chat_model,
):
    """
    Uploads a user document, starts a chat referencing it, and has the model search it: the
    search is recorded, the document is cited and its expiry is extended.
    """
    mock_chat_model.script(_search("machine learning", "how it works"), "Machine learning is…")

    api = ENDPOINTS()
    file_path = "tests/resources/Introduction_to_Machine_Learning_with_Python.pdf"
    response = await file_uploader(file_path, "test_delete_file")
    document_uuid = response["document_uuid"]
    logger.info(f"Uploaded document_uuid: {document_uuid}")
    # allow time for document to be indexed before referencing in chat message (longer for large PDF)
    await asyncio.sleep(20)
    async with async_db_session() as db_session1:
        execute = await db_session1.execute(select(User).filter_by(uuid=user_id))
        user = execute.scalar()

        execute = await db_session1.execute(select(Document).filter_by(uuid=document_uuid))
        document = execute.scalar()
        result = await db_session1.execute(
            select(DocumentUserMapping).filter_by(document_id=document.id, user_id=user.id)
        )
        first_expiry_date = result.scalar().expired_at

    url = api.chats(user_uuid=user_id)
    response_data = await async_http_requester(
        "user_document_referenced_in_chat_used_in_llm_rag_process",
        async_client.post,
        url,
        json={
            "query": "What is machine learning and how does it work?",
            "document_uuids": [document_uuid],
        },
    )

    # The tool list is constant per deploy (cache stability); use_rag only gates central guidance
    # at call time, not whether it's offered.
    offered = _offered_tools(mock_chat_model.requests[0])
    assert {SEARCH_DOCUMENTS, SEARCH_CENTRAL_GUIDANCE} <= offered

    message_uuid = response_data["message"]["uuid"]
    async with async_db_session() as db_session2:
        result = await db_session2.execute(
            select(DocumentUserMapping).filter_by(document_id=document.id, user_id=user.id)
        )
        assert result.scalar().expired_at > first_expiry_date

        result = await db_session2.execute(select(Message).filter_by(uuid=message_uuid))
        user_message_id = result.scalar().parent_message_id

        result = await db_session2.execute(select(MessageDocumentChunkMapping).filter_by(message_id=user_message_id))
        assert len(result.all()) > 0

        result = await db_session2.execute(select(RewrittenQuery).filter_by(message_id=user_message_id))
        assert sorted(q.content for (q,) in result.all()) == ["how it works", "machine learning"]

        # Only the index the model searched is recorded.
        result = await db_session2.execute(select(MessageSearchIndexMapping).filter_by(message_id=user_message_id))
        assert len(result.all()) == 1

    assert response_data["message"]["citation"] == (
        '[{"docname": "Introduction_to_Machine_Learning_with_Python.pdf", "docurl": null}]'
    )


async def test_request_to_use_non_owned_file(async_client, user_id, async_http_requester):
    # Fetch chat endpoint URL and make another request with the uploaded document
    api = ENDPOINTS()
    url = api.chats(user_uuid=user_id)
    response = await async_http_requester(
        "test_request_to_use_non_owned_file",
        async_client.post,
        url,
        response_code=401,
        json={
            "query": "what is DNA",
            "document_uuids": ["1239a6a0-aae7-4e72-9cc8-5ec9e120e08a", "2239a6a0-aae7-4e72-9cc8-5ec9e120e08a"],
        },
    )
    assert response == {
        "error": "DOCUMENT_ACCESS_ERROR",
        "documents_uuids": ["1239a6a0-aae7-4e72-9cc8-5ec9e120e08a", "2239a6a0-aae7-4e72-9cc8-5ec9e120e08a"],
    }


@pytest.mark.asyncio
async def test_save_referenced_documents_in_chat(
    async_client, user_id, async_http_requester, file_uploader, mock_chat_model
):
    """
    Checks documents referenced in a new chat stay searchable for subsequent chat messages
    and documents are returned when a chat is retrieved through api.
    Checks document user mapping expiry time and last used time are updated.
    """
    mock_chat_model.script(_search("DNA"), "DNA is…", _search("DNA structure"), "More about DNA…")

    date_test_start = datetime.datetime.now()
    api = ENDPOINTS()
    file_path = "tests/resources/DNA_Topics_UK.docx"
    response = await file_uploader(file_path, "test_save_document_uuids_file1")
    url = api.chats(user_uuid=user_id)
    document_uuid1 = response["document_uuid"]

    await asyncio.sleep(5)

    chat_response = await async_http_requester(
        "test_save_document_uuids_in_message",
        async_client.post,
        url,
        json={"query": "what is DNA", "document_uuids": [document_uuid1]},
    )

    chat_uuid = chat_response["uuid"]

    # send another message to check documents are used for subsequent messages.
    add_msg_url = api.get_chat_item(user_uuid=user_id, chat_uuid=chat_uuid)
    await async_http_requester(
        "test_save_document_uuids_in_message",
        async_client.put,
        add_msg_url,
        json={"query": "explain more about DNA"},
    )

    # The follow-up message didn't re-send document_uuids; the chat's documents are still searchable.
    assert SEARCH_DOCUMENTS in _offered_tools(mock_chat_model.requests[2])

    url = api.get_chat_messages(user_uuid=user_id, chat_uuid=chat_uuid)
    chat_history = await async_http_requester("test_save_document_uuids_in_message", async_client.get, url)

    user_msg1 = chat_history["messages"][0]
    ai_msg1 = chat_history["messages"][1]

    user_msg2 = chat_history["messages"][2]
    ai_msg2 = chat_history["messages"][3]

    # assert user message1
    assert user_msg1["content"] == "what is DNA"
    assert user_msg1["role"] == "user"
    assert user_msg1["citation"] == '[{"docname": "DNA_Topics_UK.docx", "docurl": null}]'

    # assert ai message1
    assert ai_msg1["content"] is not None
    assert ai_msg1["role"] == "assistant"
    assert ai_msg1["citation"] == '[{"docname": "DNA_Topics_UK.docx", "docurl": null}]'

    # assert user message2
    assert user_msg2["content"] == "explain more about DNA"
    assert user_msg2["role"] == "user"
    assert '{"docname": "DNA_Topics_UK.docx", "docurl": null}' in user_msg2["citation"]

    # assert ai message2
    assert ai_msg2["content"] is not None
    assert ai_msg2["role"] == "assistant"
    assert '{"docname": "DNA_Topics_UK.docx", "docurl": null}' in ai_msg2["citation"]

    # assert chat document
    assert len(chat_history["documents"]) == 1
    document1 = chat_history["documents"][0]
    assert document1["uuid"] == document_uuid1
    assert document1["name"] == "DNA_Topics_UK.docx"
    assert document1["created_at"] is not None
    assert document1["expired_at"] > document1["created_at"]
    assert document1["deleted_at"] is None
    last_used_date_str = document1["last_used"]
    last_used_date = datetime.datetime.fromisoformat(last_used_date_str)
    assert last_used_date > date_test_start

    # assert last_used returned in user documents api
    user_documents_api_endpoint = api.user_documents(user_id)
    response = await async_http_requester(
        "list_user_documents",
        async_client.get,
        user_documents_api_endpoint,
    )

    user_documents = response["user_documents"]
    document1 = [doc for doc in user_documents if doc["uuid"] == document_uuid1][0]
    last_used_date_str = document1["last_used"]
    last_used_date = datetime.datetime.fromisoformat(last_used_date_str)
    assert last_used_date > date_test_start


@pytest.mark.asyncio
async def test_deleted_docs_not_used_in_chat(
    async_client, user_id, async_http_requester, file_uploader, mock_chat_model
):
    """
    Checks deleted documents are no longer searched in later messages. The document search
    tool stays offered (the tool list is constant per deploy) even once every attached
    document is deleted, but nothing is cited from it.
    """
    mock_chat_model.script(
        _search("DNA"), "DNA is…", _search("DNA structure"), "More about DNA…", "Without the documents…"
    )

    api = ENDPOINTS()
    file_path = "tests/resources/DNA_Topics_UK.docx"
    response = await file_uploader(file_path, "test_save_document_uuids_file1")
    url = api.chats(user_uuid=user_id)
    document_uuid1 = response["document_uuid"]

    file_path2 = "tests/resources/DNA_Topics_UK2.docx"
    response = await file_uploader(file_path2, "test_save_document_uuids_file2")
    document_uuid2 = response["document_uuid"]

    await asyncio.sleep(10)

    chat_response = await async_http_requester(
        "test_save_document_uuids_in_message",
        async_client.post,
        url,
        json={"query": "what is DNA", "document_uuids": [document_uuid1, document_uuid2]},
    )
    chat_uuid = chat_response["uuid"]

    # delete  document
    api_endpoint = api.user_document(user_id, document_uuid1)

    logger.info("Deleting document %s", document_uuid1)

    # delete document
    await async_http_requester(
        "delete_document",
        async_client.delete,
        api_endpoint,
        response_code=200,
    )

    # send another message to check document2 is used for subsequent messages.
    add_msg_url = api.get_chat_item(user_uuid=user_id, chat_uuid=chat_uuid)
    await async_http_requester(
        "test_save_document_uuids_in_message",
        async_client.put,
        add_msg_url,
        json={"query": "explain more about DNA"},
    )

    url = api.get_chat_messages(user_uuid=user_id, chat_uuid=chat_uuid)
    chat_history = await async_http_requester("test_save_document_uuids_in_message", async_client.get, url)

    user_msg1 = chat_history["messages"][0]
    ai_msg1 = chat_history["messages"][1]

    user_msg2 = chat_history["messages"][2]
    ai_msg2 = chat_history["messages"][3]

    # assert user message1
    assert user_msg1["content"] == "what is DNA"
    assert user_msg1["role"] == "user"
    assert {d["docname"] for d in json.loads(user_msg1["citation"])} == {"DNA_Topics_UK.docx", "DNA_Topics_UK2.docx"}

    # assert ai message1
    assert ai_msg1["content"] is not None
    assert ai_msg1["role"] == "assistant"
    assert {d["docname"] for d in json.loads(ai_msg1["citation"])} == {"DNA_Topics_UK.docx", "DNA_Topics_UK2.docx"}

    # assert user message2: only the undeleted document was searched
    assert user_msg2["content"] == "explain more about DNA"
    assert user_msg2["role"] == "user"
    assert json.loads(user_msg2["citation"]) == [{"docname": "DNA_Topics_UK2.docx", "docurl": None}]

    # assert ai message2
    assert ai_msg2["content"] is not None
    assert ai_msg2["role"] == "assistant"
    assert '{"docname": "DNA_Topics_UK2.docx", "docurl": null}' in ai_msg2["citation"]

    # assert chat document
    assert len(chat_history["documents"]) == 2
    document1 = chat_history["documents"][0]
    assert document1["uuid"] == document_uuid1
    assert document1["name"] == "DNA_Topics_UK.docx"
    assert document1["created_at"] is not None
    assert document1["expired_at"] > document1["created_at"]
    assert document1["deleted_at"] is not None

    document2 = chat_history["documents"][1]
    assert document2["uuid"] == document_uuid2
    assert document2["name"] == "DNA_Topics_UK2.docx"
    assert document2["created_at"] is not None
    assert document2["expired_at"] > document2["created_at"]
    assert document2["deleted_at"] is None

    # delete  document2
    api_endpoint = api.user_document(user_id, document_uuid2)

    logger.info("Deleting document %s", document_uuid2)

    # delete document2
    await async_http_requester(
        "delete_document",
        async_client.delete,
        api_endpoint,
        response_code=200,
    )

    # send another message to check citations
    add_msg_url = api.get_chat_item(user_uuid=user_id, chat_uuid=chat_uuid)
    await async_http_requester(
        "test_save_document_uuids_in_message",
        async_client.put,
        add_msg_url,
        json={"query": "explain more about DNA please"},
    )

    chat_history = await async_http_requester("test_save_document_uuids_in_message", async_client.get, url)

    user_msg3 = chat_history["messages"][4]
    ai_msg3 = chat_history["messages"][5]

    # Still offered (constant tool list), but with every document deleted there's nothing to cite.
    assert SEARCH_DOCUMENTS in _offered_tools(mock_chat_model.requests[4])
    assert user_msg3["content"] == "explain more about DNA please"
    assert user_msg3["role"] == "user"
    assert not json.loads(user_msg3["citation"])

    assert ai_msg3["content"] is not None
    assert ai_msg3["role"] == "assistant"
    assert not json.loads(ai_msg3["citation"])

    # check both docs marked as deleted.
    assert len(chat_history["documents"]) == 2
    document1 = chat_history["documents"][0]
    assert document1["uuid"] == document_uuid1
    assert document1["name"] == "DNA_Topics_UK.docx"
    assert document1["created_at"] is not None
    assert document1["expired_at"] > document1["created_at"]
    assert document1["deleted_at"] is not None

    document2 = chat_history["documents"][1]
    assert document2["uuid"] == document_uuid2
    assert document2["name"] == "DNA_Topics_UK2.docx"
    assert document2["created_at"] is not None
    assert document2["expired_at"] > document2["created_at"]
    assert document2["deleted_at"] is not None
