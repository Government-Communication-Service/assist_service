import logging
from unittest.mock import AsyncMock, patch

import pytest
from opensearchpy import RequestError

from app.opensearch.service import AsyncOpenSearchOperations

logger = logging.getLogger(__name__)


@pytest.mark.opensearch
class TestOpensearch:
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
                                    "fields": ["chunk_content"],
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
