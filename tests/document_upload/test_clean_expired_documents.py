"""
Unit tests for clean_expired_documents service function.

Tests cover:
- OpenSearch partial failure (some batches fail)
- OpenSearch total failure (all batches fail)
- DB failure after OpenSearch success
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.document_upload.service import clean_expired_documents

pytestmark = [
    pytest.mark.document_upload,
    pytest.mark.unit,
]


class TestCleanExpiredDocuments:
    """Unit tests for clean_expired_documents function."""

    async def test_no_expired_chunks_returns_zero_counts(self):
        """When no expired chunks exist, returns zero counts."""
        mock_db_session = AsyncMock()

        with patch("app.document_upload.service.DbOperations") as mock_db_ops:
            mock_db_ops.get_expired_chunks_for_cleanup = AsyncMock(return_value=[])

            result = await clean_expired_documents(mock_db_session)

            assert result == {"deleted_count": 0, "failed_count": 0}
            mock_db_ops.mark_chunks_as_deleted.assert_not_called()
            mock_db_ops.mark_documents_as_deleted.assert_not_called()

    async def test_opensearch_partial_failure_only_successful_chunks_deleted_in_db(self):
        """
        When some OpenSearch batches fail, only successful chunks are marked deleted in DB.

        Scenario:
        - 4 expired chunks across 2 batches (batch size = 2)
        - First batch succeeds, second batch fails
        - Only chunks from first batch should be marked deleted in DB
        """
        mock_db_session = AsyncMock()

        # 4 chunks: (chunk_id, doc_id, opensearch_id)
        expired_chunks = [
            (1, 100, "os_id_1"),
            (2, 100, "os_id_2"),
            (3, 101, "os_id_3"),
            (4, 101, "os_id_4"),
        ]

        with (
            patch("app.document_upload.service.DbOperations") as mock_db_ops,
            patch("app.document_upload.service.AsyncOpenSearchOperations") as mock_os_ops,
            patch("app.document_upload.service.config") as mock_config,
        ):
            mock_db_ops.get_expired_chunks_for_cleanup = AsyncMock(return_value=expired_chunks)
            mock_db_ops.mark_chunks_as_deleted = AsyncMock()
            mock_db_ops.mark_documents_as_deleted = AsyncMock()

            mock_config.OPENSEARCH_DELETE_BATCH_SIZE = 2
            mock_config.DOCUMENT_CLEANUP_BATCH_SIZE = 100

            # First batch succeeds, second batch fails
            call_count = 0
            async def mock_delete_chunks(index_name, batch):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("OpenSearch batch delete failed")

            mock_os_ops.delete_document_chunks = AsyncMock(side_effect=mock_delete_chunks)

            result = await clean_expired_documents(mock_db_session)

            # Only first batch (2 chunks) should be deleted
            assert result["deleted_count"] == 2
            assert result["failed_count"] == 2

            # Verify DB operations called with only successful chunk IDs
            mock_db_ops.mark_chunks_as_deleted.assert_called_once()
            deleted_chunk_ids = mock_db_ops.mark_chunks_as_deleted.call_args[0][1]
            assert set(deleted_chunk_ids) == {1, 2}

            mock_db_ops.mark_documents_as_deleted.assert_called_once()
            deleted_doc_ids = mock_db_ops.mark_documents_as_deleted.call_args[0][1]
            assert set(deleted_doc_ids) == {100}  # Both chunks from doc 100

    async def test_opensearch_total_failure_no_db_deletions(self):
        """
        When all OpenSearch batches fail, no DB deletions happen.

        Scenario:
        - 2 expired chunks
        - OpenSearch delete fails for all
        - No DB mark_as_deleted calls should be made
        """
        mock_db_session = AsyncMock()

        expired_chunks = [
            (1, 100, "os_id_1"),
            (2, 101, "os_id_2"),
        ]

        with (
            patch("app.document_upload.service.DbOperations") as mock_db_ops,
            patch("app.document_upload.service.AsyncOpenSearchOperations") as mock_os_ops,
            patch("app.document_upload.service.config") as mock_config,
        ):
            mock_db_ops.get_expired_chunks_for_cleanup = AsyncMock(return_value=expired_chunks)
            mock_db_ops.mark_chunks_as_deleted = AsyncMock()
            mock_db_ops.mark_documents_as_deleted = AsyncMock()

            mock_config.OPENSEARCH_DELETE_BATCH_SIZE = 10
            mock_config.DOCUMENT_CLEANUP_BATCH_SIZE = 100

            # All OpenSearch deletes fail
            mock_os_ops.delete_document_chunks = AsyncMock(
                side_effect=Exception("OpenSearch connection failed")
            )

            result = await clean_expired_documents(mock_db_session)

            # All chunks failed, none deleted
            assert result["deleted_count"] == 0
            assert result["failed_count"] == 2

            # DB operations should NOT be called since no successful OS deletions
            mock_db_ops.mark_chunks_as_deleted.assert_not_called()
            mock_db_ops.mark_documents_as_deleted.assert_not_called()

    async def test_db_failure_after_opensearch_success_raises_exception(self):
        """
        When DB update fails after successful OpenSearch deletion, exception is raised.

        Scenario:
        - OpenSearch delete succeeds
        - DB mark_chunks_as_deleted fails
        - Exception should be raised with appropriate message
        """
        mock_db_session = AsyncMock()

        expired_chunks = [
            (1, 100, "os_id_1"),
            (2, 101, "os_id_2"),
        ]

        with (
            patch("app.document_upload.service.DbOperations") as mock_db_ops,
            patch("app.document_upload.service.AsyncOpenSearchOperations") as mock_os_ops,
            patch("app.document_upload.service.config") as mock_config,
        ):
            mock_db_ops.get_expired_chunks_for_cleanup = AsyncMock(return_value=expired_chunks)
            mock_db_ops.mark_chunks_as_deleted = AsyncMock(
                side_effect=Exception("Database connection error")
            )
            mock_db_ops.mark_documents_as_deleted = AsyncMock()

            mock_config.OPENSEARCH_DELETE_BATCH_SIZE = 10
            mock_config.DOCUMENT_CLEANUP_BATCH_SIZE = 100

            # OpenSearch delete succeeds
            mock_os_ops.delete_document_chunks = AsyncMock()

            with pytest.raises(Exception) as exc_info:
                await clean_expired_documents(mock_db_session)

            assert "Database update failed after OpenSearch deletion" in str(exc_info.value)

            # Verify OpenSearch was called (deletion happened)
            mock_os_ops.delete_document_chunks.assert_called_once()

            # Verify mark_documents_as_deleted was NOT called (failed before reaching it)
            mock_db_ops.mark_documents_as_deleted.assert_not_called()

    async def test_all_batches_succeed_full_deletion(self):
        """
        When all operations succeed, all chunks are marked deleted.

        Scenario:
        - 3 expired chunks
        - OpenSearch delete succeeds
        - DB updates succeed
        - All counts reflect full success
        """
        mock_db_session = AsyncMock()

        expired_chunks = [
            (1, 100, "os_id_1"),
            (2, 100, "os_id_2"),
            (3, 101, "os_id_3"),
        ]

        with (
            patch("app.document_upload.service.DbOperations") as mock_db_ops,
            patch("app.document_upload.service.AsyncOpenSearchOperations") as mock_os_ops,
            patch("app.document_upload.service.config") as mock_config,
        ):
            mock_db_ops.get_expired_chunks_for_cleanup = AsyncMock(return_value=expired_chunks)
            mock_db_ops.mark_chunks_as_deleted = AsyncMock()
            mock_db_ops.mark_documents_as_deleted = AsyncMock()

            mock_config.OPENSEARCH_DELETE_BATCH_SIZE = 10
            mock_config.DOCUMENT_CLEANUP_BATCH_SIZE = 100

            mock_os_ops.delete_document_chunks = AsyncMock()

            result = await clean_expired_documents(mock_db_session)

            assert result["deleted_count"] == 3
            assert result["failed_count"] == 0

            # Verify all chunks marked as deleted
            mock_db_ops.mark_chunks_as_deleted.assert_called_once()
            deleted_chunk_ids = mock_db_ops.mark_chunks_as_deleted.call_args[0][1]
            assert set(deleted_chunk_ids) == {1, 2, 3}

            mock_db_ops.mark_documents_as_deleted.assert_called_once()

    async def test_multiple_batches_with_mixed_results(self):
        """
        Test with multiple batches where middle batch fails.

        Scenario:
        - 6 chunks across 3 batches (batch size = 2)
        - Batch 1 succeeds, Batch 2 fails, Batch 3 succeeds
        - Only chunks from batches 1 and 3 should be deleted
        """
        mock_db_session = AsyncMock()

        expired_chunks = [
            (1, 100, "os_id_1"),
            (2, 100, "os_id_2"),
            (3, 101, "os_id_3"),
            (4, 101, "os_id_4"),
            (5, 102, "os_id_5"),
            (6, 102, "os_id_6"),
        ]

        with (
            patch("app.document_upload.service.DbOperations") as mock_db_ops,
            patch("app.document_upload.service.AsyncOpenSearchOperations") as mock_os_ops,
            patch("app.document_upload.service.config") as mock_config,
        ):
            mock_db_ops.get_expired_chunks_for_cleanup = AsyncMock(return_value=expired_chunks)
            mock_db_ops.mark_chunks_as_deleted = AsyncMock()
            mock_db_ops.mark_documents_as_deleted = AsyncMock()

            mock_config.OPENSEARCH_DELETE_BATCH_SIZE = 2
            mock_config.DOCUMENT_CLEANUP_BATCH_SIZE = 100

            # Batch 1 succeeds, Batch 2 fails, Batch 3 succeeds
            call_count = 0

            async def mock_delete_chunks(index_name, batch):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("OpenSearch batch 2 failed")

            mock_os_ops.delete_document_chunks = AsyncMock(side_effect=mock_delete_chunks)

            result = await clean_expired_documents(mock_db_session)

            # Batches 1 and 3 succeeded (4 chunks), batch 2 failed (2 chunks)
            assert result["deleted_count"] == 4
            assert result["failed_count"] == 2

            # Verify correct chunks marked as deleted (1, 2, 5, 6)
            deleted_chunk_ids = mock_db_ops.mark_chunks_as_deleted.call_args[0][1]
            assert set(deleted_chunk_ids) == {1, 2, 5, 6}
