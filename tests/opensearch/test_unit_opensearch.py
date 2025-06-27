import logging
import time
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestError

from app.opensearch.service import AsyncOpenSearchOperations, list_indexes

load_dotenv()

logger = logging.getLogger(__name__)
index_name = "py_test_index_1"


@pytest.mark.opensearch
class TestOpensearch:
    async def test_create_index(self, db_session, opensearch_client: OpenSearch):
        logger.info("\nSTARTING test_create_index....\n*******************\n\n")

        exists = False

        for idx in await list_indexes(db_session):
            if idx.name == index_name:
                exists = True
                break
        logger.info(f"Index: {index_name} exists? {exists}")

        if not exists:
            logger.info(f"Creating the Index {index_name}")

            created_index = opensearch_client.indices.create(index_name)

            assert created_index is not None, "Index creation failed"

    def test_add_documents_to_index(self, opensearch_client: OpenSearch):
        logger.info("\nSTARTING test_add_documents_to_index....\n*******************\n\n")
        data = [
            {
                "document_name": "Test Document",
                "document_url": "http://testurl.com",
                "document_description": "A test document",
                "chunk_name": "Introduction",
                "chunk_content": "This is the introduction content.",
            },
            {
                "document_name": "Test Document",
                "document_url": "http://testurl.com",
                "document_description": "A test document",
                "chunk_name": "Core Concept",
                "chunk_content": "This is the core concept content.",
            },
            {
                "document_name": "Test Document",
                "document_url": "http://testurl.com",
                "document_description": "A test document",
                "chunk_name": "Conclusion",
                "chunk_content": "This is the conclusion content.",
            },
        ]

        logger.info(f"Length of Data to be added: {len(data)}")

        for i, doc in enumerate(data):
            response = opensearch_client.index(index=index_name, id=i + 1, body=doc)
            print(f"Document {i + 1} insertion response: {response['result']}")
            logger.info("Sleeping for 5 seconds")
            time.sleep(5)
            assert response["result"] in ["created", "updated"], f"Document insertion failed for id={i}"

    def test_get_index_documents(self, opensearch_client: OpenSearch):
        logger.info("\nSTARTING test_get_index_documents....\n*******************\n\n")

        response = opensearch_client.search(
            index=index_name,
            body={"query": {"match_all": {}}},
            size=1000,  # Adjust size based on the number of documents expected
        )

        assert response is not None, "No response was received"
        assert response["timed_out"] is False, "Request has timed out"
        assert response["_shards"]["successful"] == 1, "Request was not successful"

        logger.info(f"Response of searchall is: {len(response['hits']['hits'])}")

    def test_delete_document_by_document_id(self, opensearch_client: OpenSearch):
        logger.info("\nSTARTING test_delete_document_by_document_id....\n*******************\n\n")

        response = opensearch_client.delete(index=index_name, id="1")

        logger.info(f"Response of deleting a documeht by Document ID and Index: {response}\nSleeping for 3 seconds")

        assert response is not None, "No response was received"
        assert response["result"] == "deleted", "Document was not deleted"
        assert response["_shards"]["failed"] == 0, f"Document deletion for index {index_name} failed"

        time.sleep(3)

    def test_delete_all_documents(self, opensearch_client: OpenSearch):
        logger.info("\nSTARTING test_delete_all_documents....\n*******************\n\n")

        response = opensearch_client.delete_by_query(index=index_name, body={"query": {"match_all": {}}})

        logger.info(f"Response of deleting all documents by Index: {response}")

        assert response is not None, "No response was received"
        assert response["timed_out"] is False, "Request has timed out"
        assert response["failures"] == [], f"Request failed:\n {response['failures']}"

    def test_delete_index(self, opensearch_client: OpenSearch):
        logger.info("\nSTARTING test_delete_index....\n*******************\n\n")

        response = opensearch_client.indices.delete(index=index_name)

        logger.info(f"Response of deleting an Index: {response}")
        assert response is not None, "No response was received"

    @patch("app.opensearch.service.AsyncOpenSearchClient.get")
    @pytest.mark.parametrize(
        "test_scenario, error_type, search_function",
        [
            (
                "request_error",
                RequestError(
                    400,
                    "search_phase_execution_exception",
                    {
                        "error": {
                            "root_cause": [
                                {"reason": "maxClauseCount is set to 1024", "type": "search_phase_execution_exception"}
                            ]
                        }
                    },
                ),
                AsyncOpenSearchOperations.search_for_chunks("test-query", "test-index"),
            ),
            (
                "transport_error",
                RequestError(
                    500,
                    "search_phase_execution_exception",
                    {
                        "error": {
                            "root_cause": [
                                {"reason": "maxClauseCount is set to 1024", "type": "search_phase_execution_exception"}
                            ]
                        }
                    },
                ),
                AsyncOpenSearchOperations.search_for_chunks("test-query", "test-index"),
            ),
            (
                "request_error",
                RequestError(
                    400,
                    "search_phase_execution_exception",
                    {
                        "error": {
                            "root_cause": [
                                {"reason": "maxClauseCount is set to 1024", "type": "search_phase_execution_exception"}
                            ]
                        }
                    },
                ),
                AsyncOpenSearchOperations.search_user_document_chunks("document-id", "test-query", "test-index"),
            ),
            (
                "transport_error",
                RequestError(
                    500,
                    "search_phase_execution_exception",
                    {
                        "error": {
                            "root_cause": [
                                {"reason": "maxClauseCount is set to 1024", "type": "search_phase_execution_exception"}
                            ]
                        }
                    },
                ),
                AsyncOpenSearchOperations.search_user_document_chunks("document-id", "test-query", "test-index"),
            ),
        ],
    )
    async def test_search_handles_managed_request_and_transport_errors(
        self, mock_opensearch_client, test_scenario, error_type, search_function
    ):
        logger.info("Running test scenario %s", test_scenario)
        # Mock the OpenSearch client
        mock_search = AsyncMock()
        mock_opensearch_client.return_value.search = mock_search

        # Simulate error
        mock_search.side_effect = error_type

        chunks = await search_function

        # Assertions
        assert chunks == []
        mock_search.assert_awaited_once()

    @patch("app.opensearch.service.AsyncOpenSearchClient.get")
    @pytest.mark.parametrize(
        "test_scenario, error_type, search_function",
        [
            (
                "request_error",
                RequestError(400, "non-relevant-error"),
                AsyncOpenSearchOperations.search_for_chunks("test-query", "test-index"),
            ),
            (
                "transport_error",
                RequestError(
                    500,
                    "non-relevant-error",
                ),
                AsyncOpenSearchOperations.search_for_chunks("test-query", "test-index"),
            ),
            (
                "request_error",
                RequestError(
                    400,
                    "non-relevant-error",
                ),
                AsyncOpenSearchOperations.search_user_document_chunks("document-id", "test-query", "test-index"),
            ),
            (
                "transport_error",
                RequestError(
                    500,
                    "non-relevant-error",
                ),
                AsyncOpenSearchOperations.search_user_document_chunks("document-id", "test-query", "test-index"),
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_search_raises_unhandled_errors(
        self, mock_opensearch_client, test_scenario, error_type, search_function
    ):
        logger.info("Running test scenario %s", test_scenario)
        # Mock the OpenSearch client
        mock_search = AsyncMock()
        mock_opensearch_client.return_value.search = mock_search

        # Simulate error
        mock_search.side_effect = error_type
        with pytest.raises(Exception) as ex:
            chunks = await search_function

            # Assertions
            assert ex.value == error_type
            assert chunks == []
            mock_search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_multiple_document_chunks_success(self):
        """Test successful search across multiple documents."""
        document_uuids = ["doc-1", "doc-2", "doc-3"]
        query = "test query"
        index_name = "test_index"
        max_size = 30

        # Mock response data
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "chunk-1",
                        "_source": {
                            "document_uuid": "doc-1",
                            "chunk_content": "Content 1",
                            "chunk_name": "Chunk 1",
                            "document_name": "Document 1",
                        },
                        "_score": 1.0,
                    },
                    {
                        "_id": "chunk-2",
                        "_source": {
                            "document_uuid": "doc-2",
                            "chunk_content": "Content 2",
                            "chunk_name": "Chunk 2",
                            "document_name": "Document 2",
                        },
                        "_score": 0.9,
                    },
                ]
            }
        }

        with patch("app.opensearch.service.AsyncOpenSearchClient.get") as mock_client:
            mock_search = AsyncMock()
            mock_client.return_value.search = mock_search
            mock_search.return_value = mock_response

            result = await AsyncOpenSearchOperations.search_multiple_document_chunks(
                document_uuids, query, index_name, max_size
            )

            # Verify the search was called with correct parameters
            expected_search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["document_name", "chunk_name", "chunk_content"],
                                }
                            }
                        ],
                        "filter": [
                            {"terms": {"document_uuid.keyword": document_uuids}},
                        ],
                    }
                },
                "size": max_size,
            }

            mock_search.assert_awaited_once_with(body=expected_search_body, index=index_name)

            # Verify results
            assert len(result) == 2
            assert result[0]["_id"] == "chunk-1"
            assert result[1]["_id"] == "chunk-2"

    @pytest.mark.asyncio
    async def test_get_multiple_document_chunks_success(self):
        """Test successful retrieval of chunks from multiple documents."""
        document_uuids = ["doc-1", "doc-2"]
        index_name = "test_index"
        max_size = 100

        # Mock response data
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "chunk-1",
                        "_source": {
                            "document_uuid": "doc-1",
                            "chunk_content": "Content 1",
                            "chunk_name": "Chunk 1",
                            "document_name": "Document 1",
                        },
                    },
                    {
                        "_id": "chunk-2",
                        "_source": {
                            "document_uuid": "doc-1",
                            "chunk_content": "Content 2",
                            "chunk_name": "Chunk 2",
                            "document_name": "Document 1",
                        },
                    },
                    {
                        "_id": "chunk-3",
                        "_source": {
                            "document_uuid": "doc-2",
                            "chunk_content": "Content 3",
                            "chunk_name": "Chunk 3",
                            "document_name": "Document 2",
                        },
                    },
                ]
            }
        }

        with (
            patch("app.opensearch.service.AsyncOpenSearchClient.get") as mock_client,
            patch("app.logs.LogsHandler.with_logging", new_callable=AsyncMock) as mock_logging,
        ):
            mock_search = AsyncMock()
            mock_client.return_value.search = mock_search
            mock_logging.return_value = mock_response

            result = await AsyncOpenSearchOperations.get_multiple_document_chunks(index_name, document_uuids, max_size)

            # Verify the search parameters
            expected_request_body = {
                "query": {"bool": {"filter": [{"terms": {"document_uuid.keyword": document_uuids}}]}},
                "size": max_size,
            }

            mock_search.assert_called_once_with(expected_request_body, index=index_name)
            mock_logging.assert_awaited_once()

            # Verify results
            assert len(result) == 3
            assert result[0]["_id"] == "chunk-1"
            assert result[1]["_id"] == "chunk-2"
            assert result[2]["_id"] == "chunk-3"

    @pytest.mark.asyncio
    async def test_search_multiple_document_chunks_handles_errors(self):
        """Test error handling in search_multiple_document_chunks."""
        document_uuids = ["doc-1"]
        query = "test query"
        index_name = "test_index"

        # Test with managed error (should return empty list)
        managed_error = RequestError(
            400,
            "search_phase_execution_exception",
            {
                "error": {
                    "root_cause": [
                        {"reason": "maxClauseCount is set to 1024", "type": "search_phase_execution_exception"}
                    ]
                }
            },
        )

        with patch("app.opensearch.service.AsyncOpenSearchClient.get") as mock_client:
            mock_search = AsyncMock()
            mock_client.return_value.search = mock_search
            mock_search.side_effect = managed_error

            result = await AsyncOpenSearchOperations.search_multiple_document_chunks(document_uuids, query, index_name)

            assert result == []

    @pytest.mark.asyncio
    async def test_search_multiple_document_chunks_raises_unhandled_errors(self):
        """Test that unhandled errors are properly raised in search_multiple_document_chunks."""
        document_uuids = ["doc-1"]
        query = "test query"
        index_name = "test_index"

        # Test with unmanaged error (should raise)
        unmanaged_error = RequestError(400, "different_error_type")

        with patch("app.opensearch.service.AsyncOpenSearchClient.get") as mock_client:
            mock_search = AsyncMock()
            mock_client.return_value.search = mock_search
            mock_search.side_effect = unmanaged_error

            with pytest.raises(RequestError):
                await AsyncOpenSearchOperations.search_multiple_document_chunks(document_uuids, query, index_name)

    @pytest.mark.asyncio
    async def test_get_multiple_document_chunks_empty_list(self):
        """Test get_multiple_document_chunks with empty document list."""
        document_uuids = []
        index_name = "test_index"

        mock_response = {"hits": {"hits": []}}

        with (
            patch("app.opensearch.service.AsyncOpenSearchClient.get") as mock_client,
            patch("app.logs.LogsHandler.with_logging", new_callable=AsyncMock) as mock_logging,
        ):
            mock_search = AsyncMock()
            mock_client.return_value.search = mock_search
            mock_logging.return_value = mock_response

            result = await AsyncOpenSearchOperations.get_multiple_document_chunks(index_name, document_uuids)

            assert result == []
