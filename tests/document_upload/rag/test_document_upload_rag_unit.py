import pytest

from app.document_upload.personal_document_rag import (
    GLOBAL_CHARACTER_LIMIT,
    _apply_fair_document_distribution,
)


class TestPersonalDocumentRAGUnit:
    """Unit tests for personal document RAG constants and fair distribution algorithm."""

    def create_dummy_chunk_scores(self, document_configs: list) -> dict:
        """
        Create dummy chunk scores for testing.

        Args:
            document_configs: List of tuples (doc_uuid, num_chunks, chars_per_chunk, base_score)

        Returns:
            Dictionary mapping chunk_id to chunk info
        """
        chunk_scores = {}
        chunk_counter = 0

        for doc_uuid, num_chunks, chars_per_chunk, base_score in document_configs:
            for i in range(num_chunks):
                chunk_id = f"chunk-{chunk_counter}"
                # Decreasing scores within each document
                score = base_score - (i * 0.01)

                chunk_scores[chunk_id] = {
                    "chunk_data": {
                        "_id": chunk_id,
                        "_source": {
                            "document_uuid": doc_uuid,
                            "chunk_name": f"Chunk {i}",
                            "chunk_content": "A" * chars_per_chunk,
                            "document_name": f"Document {doc_uuid}",
                        },
                        "_score": score,
                    },
                    "total_score": score,
                    "query_count": 1,
                    "final_score": score,
                    "character_count": chars_per_chunk,
                }
                chunk_counter += 1

        return chunk_scores

    @pytest.mark.asyncio
    async def test_single_small_document_all_chunks_included(self):
        """
        Test case: Single small document, 20 chunks, 1000 characters each
        -> Assert all chunks appear AND assert character limit is not breached
        """
        # Create dummy data: 1 document, 20 chunks, 1000 chars each = 20,000 total chars
        document_configs = [("doc-small-1", 20, 1000, 1.0)]
        chunk_scores = self.create_dummy_chunk_scores(document_configs)
        document_uuids = ["doc-small-1"]

        # Apply fair distribution
        result_chunks = await _apply_fair_document_distribution(chunk_scores, document_uuids, GLOBAL_CHARACTER_LIMIT)

        # Calculate total characters
        total_chars = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in result_chunks)

        # Assertions
        assert len(result_chunks) == 20, "All 20 chunks should be included since under character limit"
        assert total_chars == 20000, "Total characters should be exactly 20,000"
        assert total_chars <= GLOBAL_CHARACTER_LIMIT, "Should not exceed character limit"

        # Verify all chunks are from the correct document
        doc_uuids_in_result = set()
        for chunk in result_chunks:
            doc_uuid = chunk.get("_source", {}).get("document_uuid")
            doc_uuids_in_result.add(doc_uuid)

        assert doc_uuids_in_result == {"doc-small-1"}, "All chunks should be from the single document"

    @pytest.mark.asyncio
    async def test_single_large_document_highest_scoring_chunks(self):
        """
        Test case: Single large document, 400 chunks, 1000 characters each
        -> Assert character limit is not breached AND assert highest scoring chunks are included
        """
        # Create dummy data: 1 document, 400 chunks, 1000 chars each = 400,000 total chars (exceeds limit)
        document_configs = [("doc-large-1", 400, 1000, 1.0)]
        chunk_scores = self.create_dummy_chunk_scores(document_configs)
        document_uuids = ["doc-large-1"]

        # Apply fair distribution
        result_chunks = await _apply_fair_document_distribution(chunk_scores, document_uuids, GLOBAL_CHARACTER_LIMIT)

        # Calculate total characters
        total_chars = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in result_chunks)

        # Assertions
        assert total_chars <= GLOBAL_CHARACTER_LIMIT, "Should not exceed character limit"
        assert len(result_chunks) <= 55, "Should not exceed ~55 chunks (55,000 / 1,000)"
        assert len(result_chunks) >= 50, "Should include substantial number of chunks"

        # Verify highest scoring chunks are included
        # The first chunks should have the highest scores (1.0, 0.99, 0.98, etc.)
        result_chunk_ids = [chunk.get("_id") for chunk in result_chunks]

        # Check that early chunks (highest scoring) are included
        assert "chunk-0" in result_chunk_ids, "Highest scoring chunk should be included"
        assert "chunk-1" in result_chunk_ids, "Second highest scoring chunk should be included"
        assert "chunk-2" in result_chunk_ids, "Third highest scoring chunk should be included"

        # Verify all chunks are from the correct document
        doc_uuids_in_result = set()
        for chunk in result_chunks:
            doc_uuid = chunk.get("_source", {}).get("document_uuid")
            doc_uuids_in_result.add(doc_uuid)

        assert doc_uuids_in_result == {"doc-large-1"}, "All chunks should be from the single large document"

    @pytest.mark.asyncio
    async def test_twenty_small_documents_fair_representation(self):
        """
        Test case: 20 small documents (20 chunks each, 1000 characters per chunk)
        -> Assert there is some representation from each document AND assert character limit is respected
        """
        # Create dummy data: 20 documents, 20 chunks each, 1000 chars = 400,000 total chars (exceeds limit)
        document_configs = []
        document_uuids = []

        for i in range(20):
            doc_uuid = f"doc-small-{i}"
            document_configs.append((doc_uuid, 20, 1000, 1.0 - (i * 0.01)))  # Slightly different base scores
            document_uuids.append(doc_uuid)

        chunk_scores = self.create_dummy_chunk_scores(document_configs)

        # Apply fair distribution
        result_chunks = await _apply_fair_document_distribution(chunk_scores, document_uuids, GLOBAL_CHARACTER_LIMIT)

        # Calculate total characters and document representation
        total_chars = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in result_chunks)

        # Analyze representation by document
        doc_representation = {}
        for chunk in result_chunks:
            doc_uuid = chunk.get("_source", {}).get("document_uuid")
            if doc_uuid not in doc_representation:
                doc_representation[doc_uuid] = {"chunks": 0, "chars": 0}
            doc_representation[doc_uuid]["chunks"] += 1
            doc_representation[doc_uuid]["chars"] += len(chunk.get("_source", {}).get("chunk_content", ""))

        # Assertions
        assert total_chars <= GLOBAL_CHARACTER_LIMIT, "Should not exceed character limit"
        assert len(result_chunks) > 0, "Should include some chunks"

        # Fair representation assertions
        represented_docs = len(doc_representation)
        assert represented_docs >= 15, f"Should represent at least 15 out of 20 documents, got {represented_docs}"

        # Each represented document should get some meaningful allocation
        min_expected_chars_per_doc = 1500  # At least 1.5 chunks worth
        for doc_uuid, stats in doc_representation.items():
            assert stats["chars"] >= min_expected_chars_per_doc, (
                f"Document {doc_uuid} should get at least {min_expected_chars_per_doc} chars, got {stats['chars']}"
            )
            assert stats["chunks"] >= 1, f"Document {doc_uuid} should have at least 1 chunk"

    @pytest.mark.asyncio
    async def test_mixed_large_and_small_documents_fair_distribution(self):
        """
        Test case: 10 large documents and 10 small documents
        -> Assert there is some representation from each document AND assert character limit is respected
        """
        # Create dummy data: 10 large docs (50 chunks each) + 10 small docs (10 chunks each)
        document_configs = []
        document_uuids = []

        # 10 large documents (50 chunks each, 1000 chars = 50,000 chars per doc)
        for i in range(10):
            doc_uuid = f"doc-large-{i}"
            document_configs.append((doc_uuid, 50, 1000, 1.0 - (i * 0.01)))
            document_uuids.append(doc_uuid)

        # 10 small documents (10 chunks each, 1000 chars = 10,000 chars per doc)
        for i in range(10):
            doc_uuid = f"doc-small-{i}"
            document_configs.append((doc_uuid, 10, 1000, 0.9 - (i * 0.01)))
            document_uuids.append(doc_uuid)

        chunk_scores = self.create_dummy_chunk_scores(document_configs)

        # Apply fair distribution
        result_chunks = await _apply_fair_document_distribution(chunk_scores, document_uuids, GLOBAL_CHARACTER_LIMIT)

        # Calculate total characters and document representation
        total_chars = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in result_chunks)

        # Analyze representation by document type
        large_doc_representation = {}
        small_doc_representation = {}

        for chunk in result_chunks:
            doc_uuid = chunk.get("_source", {}).get("document_uuid")
            chunk_chars = len(chunk.get("_source", {}).get("chunk_content", ""))

            if doc_uuid.startswith("doc-large-"):
                if doc_uuid not in large_doc_representation:
                    large_doc_representation[doc_uuid] = {"chunks": 0, "chars": 0}
                large_doc_representation[doc_uuid]["chunks"] += 1
                large_doc_representation[doc_uuid]["chars"] += chunk_chars
            elif doc_uuid.startswith("doc-small-"):
                if doc_uuid not in small_doc_representation:
                    small_doc_representation[doc_uuid] = {"chunks": 0, "chars": 0}
                small_doc_representation[doc_uuid]["chunks"] += 1
                small_doc_representation[doc_uuid]["chars"] += chunk_chars

        # Assertions
        assert total_chars <= GLOBAL_CHARACTER_LIMIT, "Should not exceed character limit"
        assert len(result_chunks) > 0, "Should include some chunks"

        # Representation assertions
        total_large_docs_represented = len(large_doc_representation)
        total_small_docs_represented = len(small_doc_representation)

        assert total_large_docs_represented >= 8, (
            f"Should represent at least 8 out of 10 large documents, got {total_large_docs_represented}"
        )
        assert total_small_docs_represented >= 8, (
            f"Should represent at least 8 out of 10 small documents, got {total_small_docs_represented}"
        )

        # Small documents should still get meaningful representation despite large documents
        min_chars_per_small_doc = 1500  # At least 1.5 chunks worth
        for doc_uuid, stats in small_doc_representation.items():
            assert stats["chars"] >= min_chars_per_small_doc, (
                f"Small document {doc_uuid} should get at least {min_chars_per_small_doc} chars, got {stats['chars']}"
            )

        # Large documents should generally get more allocation than small ones
        if large_doc_representation and small_doc_representation:
            avg_large_doc_chars = sum(stats["chars"] for stats in large_doc_representation.values()) / len(
                large_doc_representation
            )
            avg_small_doc_chars = sum(stats["chars"] for stats in small_doc_representation.values()) / len(
                small_doc_representation
            )

            assert avg_large_doc_chars > avg_small_doc_chars, (
                "Large documents should on average get more characters than small documents"
            )

    @pytest.mark.asyncio
    async def test_fair_distribution_edge_cases(self):
        """Test edge cases for the fair distribution algorithm."""

        # Test case 1: Empty inputs
        result_empty = await _apply_fair_document_distribution({}, [], GLOBAL_CHARACTER_LIMIT)
        assert result_empty == [], "Empty inputs should return empty list"

        # Test case 2: Single chunk per document
        document_configs = [
            ("doc-1", 1, 1000, 1.0),
            ("doc-2", 1, 1000, 0.9),
            ("doc-3", 1, 1000, 0.8),
        ]
        chunk_scores = self.create_dummy_chunk_scores(document_configs)
        document_uuids = ["doc-1", "doc-2", "doc-3"]

        result_single = await _apply_fair_document_distribution(chunk_scores, document_uuids, GLOBAL_CHARACTER_LIMIT)

        assert len(result_single) == 3, "Should include all chunks when well under limit"

        # Test case 3: Very small character limit
        small_limit = 5000  # Only enough for ~5 chunks
        result_small_limit = await _apply_fair_document_distribution(chunk_scores, document_uuids, small_limit)

        total_chars_small = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in result_small_limit)
        assert total_chars_small <= small_limit, "Should respect small character limit"
        assert len(result_small_limit) >= 3, "Should still try to represent each document with small limit"
