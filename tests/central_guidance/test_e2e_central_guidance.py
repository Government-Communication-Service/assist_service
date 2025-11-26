import json
import logging

import pytest

# from pydantic import ValidationError
from sqlalchemy.future import select

from app.api.endpoints import ENDPOINTS
from app.central_guidance.constants import CENTRAL_RAG_INDEX_NAME
from app.database.models import Chat, Document, DocumentChunk, Message, MessageSearchIndexMapping, SearchIndex

logger = logging.getLogger(__name__)
api = ENDPOINTS()


@pytest.mark.asyncio
async def test_list_document_chunks(async_client, async_http_requester, db_session, async_opensearch_client):
    """
    Makes a get request to the endpoint that lists all chunks
    then checks that the number of chunks is the same as recorded in PostgreSQL and OpenSearch.
    """
    # Get list of chunks from API endpoint
    url = ENDPOINTS().document_chunks()
    response = await async_http_requester(
        "GET request to /v1/central-rag/document-chunks endpoint", async_client.get, url=url
    )

    # Get count of chunks from response
    response_chunks = response["document_chunks"]
    response_count = len(response_chunks)

    # Get count of chunks from PostgreSQL
    async with db_session as session:
        stmt = (
            select(DocumentChunk)
            .join(SearchIndex, DocumentChunk.search_index_id == SearchIndex.id)
            .where(SearchIndex.name == CENTRAL_RAG_INDEX_NAME, DocumentChunk.deleted_at.is_(None))
        )
        result = await session.execute(stmt)
        postgres_chunks = result.scalars().all()
        postgres_count = len(postgres_chunks)

        # Get IDs to check in OpenSearch
        opensearch_ids = [chunk.id_opensearch for chunk in postgres_chunks]

    # Get count from OpenSearch
    response = await async_opensearch_client.search(
        index=CENTRAL_RAG_INDEX_NAME,
        body={
            "size": 10000,  # Large enough to get all chunks
            "query": {"match_all": {}},
        },
    )
    opensearch_count = response["hits"]["total"]["value"]

    # Verify counts match
    assert postgres_count == opensearch_count, (
        f"The number of document chunks in PostgreSQL, {postgres_count} "
        f"does not equal the number of chunks in OpenSearch, {opensearch_count}."
    )
    assert response_count == postgres_count, (
        f"The number of document chunks in the api response, {response_count} "
        f"does not equal the number of chunks in PostgreSQL, {postgres_count}."
    )

    # Verify all PostgreSQL IDs exist in OpenSearch
    for opensearch_id in opensearch_ids:
        response = await async_opensearch_client.get(index=CENTRAL_RAG_INDEX_NAME, id=opensearch_id)
        assert response["found"] is True


@pytest.mark.asyncio
async def test_upload_new_document_to_central_rag(
    async_client, async_http_requester, db_session, async_opensearch_client
):
    """
    Uploads a new document with chunks to the API
    then checks that the new document exists in both
    OpenSearch and PostgreSQL.
    """

    doc_name = "Test document upload 1: name"
    doc_description = "Test document upload 1: description"
    doc_url = "https://gcs.civilservice.gov.uk/"

    new_document = [
        {
            "document_name": doc_name,
            "document_url": doc_url,
            "document_description": doc_description,
            "chunk_name": "Test chunk 1: name",
            "chunk_content": "Test chunk 1: description",
        },
        {
            "document_name": doc_name,
            "document_url": doc_url,
            "document_description": doc_description,
            "chunk_name": "Test chunk 2: name",
            "chunk_content": "Test chunk 3: description",
        },
    ]

    response = await async_http_requester(
        "POST to /v1/central-rag/document-chunks endpoint",
        async_client.post,
        url=ENDPOINTS().document_chunks(),
        json=new_document,
    )

    # The response is either True or False
    # Check that the response was not 'False'
    # TODO: Refactor endpoint responses to be consistent with other endpoints.
    assert response

    ### Query PostgreSQL database to check that the entries now exists
    async with db_session as session:
        stmt = (
            select(Document)
            .distinct()
            .join(DocumentChunk, Document.id == DocumentChunk.document_id)
            .join(SearchIndex, DocumentChunk.search_index_id == SearchIndex.id)
            .where(
                SearchIndex.name == CENTRAL_RAG_INDEX_NAME,
                Document.name == doc_name,
                Document.description == doc_description,
                Document.url == doc_url,
                Document.deleted_at.is_(None),
                DocumentChunk.deleted_at.is_(None),
            )
        )
        result = await session.execute(stmt)
        matching_documents = result.scalars().all()

        # There should not be more than one matching documents
        assert len(matching_documents) == 1

        # Check the number of matching chunks is correct
        matching_document = matching_documents[0]
        stmt = (
            select(DocumentChunk)
            .join(
                SearchIndex,
                DocumentChunk.search_index_id == SearchIndex.id,
            )
            .where(
                SearchIndex.name == CENTRAL_RAG_INDEX_NAME,
                DocumentChunk.document_id == matching_document.id,
                DocumentChunk.deleted_at.is_(None),
            )
        )
        result = await session.execute(stmt)
        matching_chunks = result.scalars().all()

        assert len(matching_chunks) == len(new_document)

        opensearch_ids = [chunk.id_opensearch for chunk in matching_chunks]

    ### Query OpenSearch database to check that the entries now exists
    # Check all IDs exist in OpenSearch
    for opensearch_id in opensearch_ids:
        response = await async_opensearch_client.get(index=CENTRAL_RAG_INDEX_NAME, id=opensearch_id)
        assert response["found"] is True

    # Check for duplicates by querying all chunks and comparing name/content
    response = await async_opensearch_client.search(
        index=CENTRAL_RAG_INDEX_NAME,
        body={
            "size": 10000,  # Large enough to get all chunks
            "query": {
                "bool": {
                    "must": [
                        {"match_phrase": {"document_name": doc_name}},
                        {"match_phrase": {"document_url": doc_url}},
                    ]
                }
            },
        },
    )

    hits = response["hits"]["hits"]
    chunk_counts = {}
    for hit in hits:
        chunk_source = hit["_source"]
        chunk_key = (
            chunk_source["document_name"],
            chunk_source["document_url"],
            chunk_source["chunk_name"],
            chunk_source["chunk_content"],
        )
        chunk_counts[chunk_key] = chunk_counts.get(chunk_key, 0) + 1

    duplicates = {k: v for k, v in chunk_counts.items() if v > 1}
    assert not duplicates, (
        f"Found duplicate chunks: {', '.join(f'{k}: {v} occurrences' for k, v in duplicates.items())}"
    )


@pytest.mark.asyncio
async def test_delete_chunk_from_central_rag(async_http_requester, async_client, async_opensearch_client, db_session):
    """
    Makes a get request to central-rag/document-chunks
    Then selects the first document chunks returned
    Then makes a delete request to central-rag/document-chunks
    Then makes another get request and verifies that the chunk is no longer returned
    Then verifies that the chunk has deleted_at populated in the database
    Then verifies that the chunk is no longer in the OpenSearch index.
    """

    async with db_session as session:
        doc_name = "Test document upload 3: testing delete functionality name"
        doc_description = "Test document upload 3: testing delete functionality description"
        doc_url = "https://gcs.civilservice.gov.uk/"
        chunk_name_1 = "Test chunk 8 name: testing delete functionality"
        chunk_description_1 = "Test chunk 8 description: testing delete functionality"

        new_document = [
            {
                "document_name": doc_name,
                "document_url": doc_url,
                "document_description": doc_description,
                "chunk_name": chunk_name_1,
                "chunk_content": chunk_description_1,
            }
        ]

        await async_http_requester(
            "POST to /v1/central-rag/document-chunks endpoint",
            async_client.post,
            url=ENDPOINTS().document_chunks(),
            json=new_document,
        )

        # Get list of chunks from API endpoint
        url = ENDPOINTS().document_chunks()
        response = await async_http_requester(
            "GET request to /v1/central-rag/document-chunks endpoint", async_client.get, url=url
        )

        # Get chunk matching first uploaded chunk
        initial_chunks = response["document_chunks"]
        assert len(initial_chunks) > 0, "No chunks found to delete"
        chunk_to_delete = next(
            chunk
            for chunk in initial_chunks
            if chunk["chunk_name"] == chunk_name_1 and chunk["chunk_content"] == chunk_description_1
        )

        # Check the chunk appears in postgresql
        stmt = (
            select(DocumentChunk)
            .join(SearchIndex, DocumentChunk.search_index_id == SearchIndex.id)
            .where(
                SearchIndex.name == CENTRAL_RAG_INDEX_NAME,
                DocumentChunk.uuid == chunk_to_delete["uuid"],
            )
        )
        result = await session.execute(stmt)
        db_chunk = result.scalar_one()
        assert db_chunk is not None
        assert db_chunk.deleted_at is None

        # Check the chunk appears in opensearch
        response = await async_opensearch_client.get(index=CENTRAL_RAG_INDEX_NAME, id=db_chunk.id_opensearch)
        assert response["found"] is True

        # Delete the chunk
        url_delete = ENDPOINTS().document_chunk(chunk_to_delete["uuid"])
        response = await async_http_requester(
            "DELETE request to /v1/central-rag/document-chunks/uuid endpoint",
            async_client.delete,
            url=url_delete,
        )
        assert response is True

        # Verify chunk no longer appears in GET response
        response = await async_http_requester(
            "GET request to /v1/central-rag/document-chunks endpoint", async_client.get, url=url
        )
        updated_chunks = response["document_chunks"]
        deleted_chunks = [chunk for chunk in updated_chunks if chunk["uuid"] == chunk_to_delete["uuid"]]
        assert len(deleted_chunks) == 0

        # Refresh the session to see the latest database state
        await session.refresh(db_chunk)

        # Verify deleted_at is populated in database
        stmt_after_deletion = (
            select(DocumentChunk)
            .join(SearchIndex, DocumentChunk.search_index_id == SearchIndex.id)
            .where(SearchIndex.name == CENTRAL_RAG_INDEX_NAME, DocumentChunk.uuid == chunk_to_delete["uuid"])
        )
        result_after_deletion = await session.execute(stmt_after_deletion)
        db_chunk_after_deletion = result_after_deletion.scalar_one()
        assert db_chunk_after_deletion.deleted_at is not None

        # Verify chunk no longer exists in OpenSearch
        response = await async_opensearch_client.get(
            index=CENTRAL_RAG_INDEX_NAME,
            id=db_chunk_after_deletion.id_opensearch,
            ignore=[404],  # Ignore 404 error if document not found
        )
        assert response["found"] is False


@pytest.mark.asyncio
async def test_sync_central_index(async_http_requester, async_client, async_opensearch_client, db_session):
    # First check the sync endpoint works
    url = ENDPOINTS().sync_central_index()
    response = await async_http_requester("Send PUT request to central-rag/synchronise", async_client.put, url=url)
    assert response is True

    # Intentionally add chunks to OpenSearch that are not in PostgreSQL
    test_doc = {
        "document_name": "Test sync doc",
        "document_url": "https://test.com",
        "document_description": "Test description",
        "chunk_name": "Test chunk",
        "chunk_content": "Test content",
    }

    response = await async_opensearch_client.index(index=CENTRAL_RAG_INDEX_NAME, body=test_doc, refresh=True)
    test_id = response["_id"]

    # Verify that OpenSearch and PostgreSQL are now out of sync
    # Check OpenSearch has the document
    response = await async_opensearch_client.get(index=CENTRAL_RAG_INDEX_NAME, id=test_id)
    assert response["found"] is True

    # Check PostgreSQL does not have the document
    async with db_session as session:
        stmt = (
            select(Document)
            .join(DocumentChunk, Document.id == DocumentChunk.document_id)
            .join(SearchIndex, DocumentChunk.search_index_id == SearchIndex.id)
            .where(SearchIndex.name == CENTRAL_RAG_INDEX_NAME, Document.name == test_doc["document_name"])
        )
        result = await session.execute(stmt)
        assert len(result.scalars().all()) == 0

    # Run the sync endpoint again
    response = await async_http_requester("Send PUT request to central-rag/synchronise", async_client.put, url=url)
    assert response is True

    # Verify that OpenSearch and PostgreSQL are now in sync
    # The document should be removed from OpenSearch since it's not in PostgreSQL
    response = await async_opensearch_client.get(
        index=CENTRAL_RAG_INDEX_NAME,
        id=test_id,
        ignore=[404],  # Ignore 404 error if document not found
    )
    assert response["found"] is False


@pytest.mark.skip(reason="Flaky test")
async def test_multiple_citations_are_referenced_in_chat_message(
    user_id,
    async_client,
    async_http_requester,
):
    """
    Checks if a message has multiple citations from central document index and these citations
    are returned to the user.
    """
    api = ENDPOINTS()
    logger.debug(f"Creating chat for user ID: {user_id}")
    url = api.chats(user_uuid=user_id)
    response = await async_http_requester(
        "chat_endpoint",
        async_client.post,
        url,
        json={
            "query": "you MUST use the documents MCOM 3.0 and GCS Accessibility Standards to answer this prompt."
            " Give me a brief one paragraph summary of accessible content in GCS",
            "use_rag": True,
        },
    )

    # citations returned not in consistent order, causing odd test failures, sort then check
    expected_citations = (
        '[{"docname": "Inclusive Communications Template",'
        ' "docurl": "https://gcs.civilservice.gov.uk/publications/inclusive-communications-template/'
        '#audience-insight"}, {"docname": "The Modern Communications Operating Model (MCOM) 3.0",'
        ' "docurl": "https://gcs.civilservice.gov.uk/modern-communications-operating-model-3-0/"}]'
    )
    expected_citations_arr = json.loads(expected_citations)

    logger.info("Response received: %s", response)

    actual_citations = response["message"]["citation"]
    actual_citations_arr = json.loads(actual_citations)

    for expected_citation in expected_citations_arr:
        assert expected_citation in actual_citations_arr


@pytest.mark.asyncio
async def test_chat_rag_process_does_not_use_personal_document_index(
    async_client, user_id, chat, async_http_requester, db_session, caplog
):
    """
    Creates a new chat and sends a default message to the chat.
    Then it verifies personal document index is not used in the default RAG process.
    """
    execute = await db_session.execute(select(Chat).filter(Chat.uuid == chat.uuid))
    chat_db_obj = execute.scalar_one()
    execute = await db_session.execute(
        select(Message).filter(Message.chat_id == chat_db_obj.id, Message.role == "user")
    )
    user_messages = list(execute.scalars())
    user_message = user_messages[0]

    execute = await db_session.execute(
        select(MessageSearchIndexMapping).filter(MessageSearchIndexMapping.message_id == user_message.id)
    )

    query_result = list(execute.scalars())

    indices = [r.search_index_id for r in query_result]

    personal_document_index_id = 3
    assert personal_document_index_id not in indices
