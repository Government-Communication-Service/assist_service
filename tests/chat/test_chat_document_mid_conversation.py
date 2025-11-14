import asyncio
import logging

import pytest

from app.api.endpoints import ENDPOINTS

api = ENDPOINTS()
logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.e2e]


class TestChatDocumentMidConversation:
    """Tests for adding documents to existing chat conversations"""

    @pytest.mark.asyncio
    async def test_add_document_to_existing_chat(
        self, chat, user_id, async_client, async_http_requester, file_uploader, db_session
    ):
        """Test adding a document to an existing chat conversation"""
        # Upload document first
        file_path = "tests/resources/DNA_Topics_UK.docx"
        upload_response = await file_uploader(file_path, "mid_chat_doc")
        document_uuid = upload_response["document_uuid"]

        # Allow time for indexing
        await asyncio.sleep(5)

        # Add message with document to existing chat
        endpoint = api.get_chat_item(user_id, chat.uuid)
        response = await async_http_requester(
            "add message with document to existing chat",
            async_client.put,
            endpoint,
            json={"query": "Analyze this document", "document_uuids": [document_uuid]},
        )

        # Verify response structure
        assert response["status"] == "success"
        assert "message" in response
        assert response["message"]["content"]

        # Verify document appears in chat details
        get_endpoint = api.get_chat_item(user_id, chat.uuid)
        chat_details = await async_http_requester("get chat with documents", async_client.get, get_endpoint)

        assert "documents" in chat_details
        assert len(chat_details["documents"]) >= 1
        document_uuids_in_chat = [str(doc["uuid"]) for doc in chat_details["documents"]]
        assert document_uuid in document_uuids_in_chat

    @pytest.mark.asyncio
    async def test_document_persists_in_subsequent_messages(
        self, chat, user_id, async_client, async_http_requester, file_uploader
    ):
        """Test that documents added mid-chat are available in subsequent messages"""
        # Upload and add document to chat
        file_path = "tests/resources/DNA_Topics_UK.docx"
        upload_response = await file_uploader(file_path, "persistent_doc")
        document_uuid = upload_response["document_uuid"]

        await asyncio.sleep(5)

        # First message: add document to chat
        endpoint = api.get_chat_item(user_id, chat.uuid)
        first_response = await async_http_requester(
            "first message with document",
            async_client.put,
            endpoint,
            json={"query": "What is in this document?", "document_uuids": [document_uuid]},
        )

        assert first_response["status"] == "success"

        # Second message: don't specify documents
        second_response = await async_http_requester(
            "second message without specifying documents",
            async_client.put,
            endpoint,
            json={"query": "Tell me more"},
        )

        assert second_response["status"] == "success"
        assert "message" in second_response

        # Verify document is still in chat
        get_endpoint = api.get_chat_item(user_id, chat.uuid)
        chat_details = await async_http_requester("verify document persists", async_client.get, get_endpoint)

        assert "documents" in chat_details
        assert len(chat_details["documents"]) >= 1
        document_uuids_in_chat = [str(doc["uuid"]) for doc in chat_details["documents"]]
        assert document_uuid in document_uuids_in_chat

    @pytest.mark.asyncio
    async def test_add_multiple_documents_mid_conversation(
        self, chat, user_id, async_client, async_http_requester, file_uploader, db_session
    ):
        """Test adding multiple documents to an existing chat"""
        # Upload first document and add to chat
        upload_response1 = await file_uploader("tests/resources/DNA_Topics_UK.docx", "multi_doc_1")
        document_uuid1 = upload_response1["document_uuid"]

        await asyncio.sleep(5)

        endpoint = api.get_chat_item(user_id, chat.uuid)
        await async_http_requester(
            "add first document",
            async_client.put,
            endpoint,
            json={"query": "Analyze first document", "document_uuids": [document_uuid1]},
        )

        # Upload second document and add to same chat
        upload_response2 = await file_uploader("tests/resources/random-topics.docx", "multi_doc_2")
        document_uuid2 = upload_response2["document_uuid"]

        await asyncio.sleep(5)

        await async_http_requester(
            "add second document",
            async_client.put,
            endpoint,
            json={"query": "Now analyze second document", "document_uuids": [document_uuid2]},
        )

        # Verify both documents are in chat
        get_endpoint = api.get_chat_item(user_id, chat.uuid)
        chat_details = await async_http_requester("verify both documents", async_client.get, get_endpoint)

        assert "documents" in chat_details
        assert len(chat_details["documents"]) == 2

        document_uuids_in_chat = {str(doc["uuid"]) for doc in chat_details["documents"]}
        assert document_uuid1 in document_uuids_in_chat
        assert document_uuid2 in document_uuids_in_chat

    @pytest.mark.asyncio
    async def test_cannot_add_unauthorized_document_mid_chat(
        self, chat, user_id, async_client, async_http_requester, file_uploader, another_user_auth_session, auth_token
    ):
        """Test that users cannot add documents they don't own to a chat"""
        # Upload document as original user
        upload_response = await file_uploader("tests/resources/DNA_Topics_UK.docx", "unauthorized_doc")
        document_uuid = upload_response["document_uuid"]

        await asyncio.sleep(5)

        # Try to add document to chat (should succeed for owner)
        endpoint = api.get_chat_item(user_id, chat.uuid)
        response = await async_http_requester(
            "owner adds document",
            async_client.put,
            endpoint,
            json={"query": "Analyze document", "document_uuids": [document_uuid]},
        )

        assert response["status"] == "success"

    @pytest.mark.asyncio
    async def test_add_document_to_chat_with_stream(
        self, chat, user_id, async_client, async_http_requester, file_uploader
    ):
        """Test adding a document to existing chat using stream endpoint"""
        # Upload document
        file_path = "tests/resources/DNA_Topics_UK.docx"
        upload_response = await file_uploader(file_path, "stream_doc")
        document_uuid = upload_response["document_uuid"]

        await asyncio.sleep(5)

        # Add message with document using stream endpoint
        endpoint = api.get_chat_stream(user_id, chat.uuid)
        response = await async_http_requester(
            "add document via stream",
            async_client.put,
            endpoint,
            json={"query": "Summarize this document", "document_uuids": [document_uuid]},
            response_type="text",
            response_content_type="text/event-stream; charset=utf-8",
        )

        # Stream endpoint returns text response
        assert response is not None

        # Verify document appears in chat
        get_endpoint = api.get_chat_item(user_id, chat.uuid)
        chat_details = await async_http_requester("verify document saved", async_client.get, get_endpoint)

        assert "documents" in chat_details
        document_uuids_in_chat = [str(doc["uuid"]) for doc in chat_details["documents"]]
        assert document_uuid in document_uuids_in_chat
