import logging
from uuid import uuid4

import pytest
from sqlalchemy.future import select

from app.api import ENDPOINTS, ApiConfig
from app.database.models import Document

logger = logging.getLogger()


class TestUserDocuments:
    @pytest.mark.asyncio
    async def test_delete_user_document(self, async_client, user_id, async_http_requester, db_session, file_uploader):
        """
        Tests marking a document mapping, document chunks and document as deleted.

        Asserts:
            The document mapping, document and its associated chunks  marked as deleted.
        """

        api = ENDPOINTS()
        # upload a file
        file_path = "tests/resources/sample_doc_document.doc"
        response = await file_uploader(file_path, "test_delete_document")

        # get document_id from response
        document_uuid = response["document_uuid"]
        api_endpoint = api.user_document(user_id, document_uuid)

        logger.info("Deleting document %s", document_uuid)

        # delete document
        response = await async_http_requester(
            "delete_document",
            async_client.delete,
            api_endpoint,
            response_code=200,
        )

        # get document_id from response
        document_uuid = response["document_uuid"]

        # check document marked as deleted
        document = await db_session.execute(
            select(Document).filter(Document.uuid == document_uuid, Document.deleted_at.is_(None))
        )
        result = document.scalars().all()
        assert result == []

        # check document chunk table is empty
        chunks = await db_session.execute(
            select(Document).filter(Document.uuid == document_uuid, Document.deleted_at.is_(None))
        )
        result = chunks.scalars().all()
        assert result == []

    @pytest.mark.asyncio
    async def test_attempt_to_delete_already_deleted_user_document(
        self, async_client, user_id, async_http_requester, db_session, file_uploader
    ):
        """
        Tests deleting an already deleted document.

        Asserts:
            Http 404 status code returned for already deleted file.
        """

        api = ENDPOINTS()
        # upload a file
        file_path = "tests/resources/sample_doc_document.doc"
        response = await file_uploader(file_path, "test_delete_file")

        # get document_id from response
        document_uuid = response["document_uuid"]
        api_endpoint = api.user_document(user_id, document_uuid)

        logger.info("Deleting document %s", document_uuid)

        # delete document
        await async_http_requester(
            "delete_document",
            async_client.delete,
            api_endpoint,
            response_code=200,
        )

        # delete document second time, where it returns 404
        # for already deleted document.
        await async_http_requester(
            "delete_document_second_time",
            async_client.delete,
            api_endpoint,
            response_code=404,
        )

    @pytest.mark.asyncio
    async def test_try_deleting_non_existent_document(self, async_client, user_id, async_http_requester):
        """
        Tests deleting a non-existent document.

        Asserts:
            Http 404 status code returned for non existing document.
        """

        api = ENDPOINTS()
        api_endpoint = api.user_document(user_id, "bd7f82b7-abd3-403f-8161-02b3a0002fa4")

        # delete document
        await async_http_requester(
            "delete_document",
            async_client.delete,
            api_endpoint,
            response_code=404,
        )

    @pytest.mark.asyncio
    async def test_delete_document_not_owned(
        self,
        user_id,
        async_client,
        async_http_requester,
        file_uploader,
        auth_session,
        auth_token,
        another_user_auth_session,
    ):
        api = ENDPOINTS()
        # upload a file
        file_path = "tests/resources/sample_doc_document.doc"
        response = await file_uploader(file_path, "test_delete_file")

        # get document_id from response
        document_uuid = response["document_uuid"]

        # create endpoint with other user
        non_owning_user = str(uuid4())
        # create session for other user
        other_session = await another_user_auth_session(non_owning_user)
        api_endpoint = api.user_document(non_owning_user, document_uuid)

        logger.info("Deleting document %s", document_uuid)

        # attempt delete document
        non_owning_user_session_params = {
            ApiConfig.USER_KEY_UUID_ALIAS: non_owning_user,
            ApiConfig.SESSION_AUTH_ALIAS: other_session,
            ApiConfig.AUTH_TOKEN_ALIAS: auth_token,
        }
        response = await async_http_requester(
            "delete_document",
            async_client.delete,
            api_endpoint,
            response_code=404,
            headers=non_owning_user_session_params,
        )
        assert response["document_uuid"] == document_uuid
        assert response["message"] == "No Document mapping found"
