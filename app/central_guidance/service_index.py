# app.central_guidance.service_index.py
#
# This file holds functions for managing the documents available in the
# 'central guidance' OpenSearch index / PostgreSQL table.
# It includes functions for uploading new chunks, listing existing chunks,
# deleting chunks and synchronising the central guidance OpenSearch index
# with the PostgreSQL database.

import logging
from datetime import datetime
from typing import List
from uuid import UUID

from opensearchpy import NotFoundError, TransportError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.central_guidance.constants import CENTRAL_RAG_INDEX_NAME
from app.central_guidance.schemas import DocumentChunkResponse, ListDocumentChunkResponse
from app.database.db_operations import DbOperations
from app.database.models import (
    Document,
    DocumentChunk,
    SearchIndex,
)
from app.opensearch.schemas import OpenSearchRecord
from app.opensearch.service import create_client

logger = logging.getLogger(__name__)


def get_central_index(db_session: AsyncSession):
    return DbOperations.get_index_by_name(db_session=db_session, name=CENTRAL_RAG_INDEX_NAME)


async def sync_central_index(session: AsyncSession) -> bool:
    """Updates the Central RAG OpenSearch index to be in sync with the PostgreSQL tables."""

    logger.debug("Starting synchronization of OpenSearch with PostgreSQL...")
    opensearch_client = create_client()

    # Delete and recreate central index
    try:
        opensearch_client.indices.delete(index=CENTRAL_RAG_INDEX_NAME)
        logger.info(f"Deleted index: {CENTRAL_RAG_INDEX_NAME}")
    except NotFoundError:
        logger.warning(f"Index {CENTRAL_RAG_INDEX_NAME} not found in OpenSearch. Skipping deletion.")
    except Exception as e:
        logger.error(f"Failed to delete index {CENTRAL_RAG_INDEX_NAME}: {str(e)}")
        raise

    try:
        opensearch_client.indices.create(index=CENTRAL_RAG_INDEX_NAME)
        logger.debug(f"Created index: {CENTRAL_RAG_INDEX_NAME}")
    except Exception as e:
        logger.error(f"Failed to create index {CENTRAL_RAG_INDEX_NAME}: {str(e)}")
        raise

    try:
        # Get the central search index from PostgreSQL
        stmt = select(SearchIndex).where(SearchIndex.name == CENTRAL_RAG_INDEX_NAME)
        result = await session.execute(stmt)
        central_index = result.scalar_one()

        # Get all document chunks for this index
        stmt = select(DocumentChunk).where(
            DocumentChunk.search_index_id == central_index.id, DocumentChunk.deleted_at.is_(None)
        )
        result = await session.execute(stmt)
        chunks = result.scalars().all()

        # Get all documents
        stmt = select(Document).where(Document.deleted_at.is_(None))
        result = await session.execute(stmt)
        documents = result.scalars().all()

    except SQLAlchemyError as e:
        logger.error(f"Failed to query database: {str(e)}")
        raise

    try:
        # Add each chunk to OpenSearch and update PostgreSQL
        updated_at = datetime.now()
        for chunk in chunks:
            # Find associated document
            document = next((d for d in documents if d.id == chunk.document_id), None)
            if not document:
                logger.warning(f"No document found for chunk {chunk.uuid}, skipping")
                continue

            # Create OpenSearch record
            record = OpenSearchRecord(
                document_name=document.name,
                document_url=document.url,
                chunk_name=chunk.name,
                chunk_content=chunk.content,
                document_uuid=document.uuid,
            )

            try:
                # Index in OpenSearch
                response = opensearch_client.index(index=CENTRAL_RAG_INDEX_NAME, body=record.to_opensearch_dict())

                # Update PostgreSQL with new OpenSearch ID
                chunk.id_opensearch = response["_id"]
                chunk.updated_at = updated_at

            except TransportError as e:
                logger.error(f"OpenSearch indexing failed for chunk {chunk.uuid}: {str(e)}")
                raise

        try:
            await session.commit()
            logger.info("Successfully synchronized OpenSearch with PostgreSQL")
            return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to commit changes to database: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Unexpected error during synchronization: {str(e)}")
        raise


async def load_chunks_to_central_rag(
    db_session: AsyncSession,
    data: str,  # A serialised string containing JSON data with the relevant keys
) -> bool:
    """Loads document chunks to the PostgreSQL database and then into OpenSearch."""

    search_index = await get_central_index(db_session)
    opensearch_client = create_client()

    try:
        for row in data:
            document_name = row["document_name"]
            document_url = row["document_url"]
            document_description = row["document_description"]
            chunk_name = row["chunk_name"]
            chunk_content = row["chunk_content"]

            # Check if the Document already exists
            existing_document = await DbOperations.get_existing_document(
                db_session=db_session, name=document_name, url=document_url
            )
            if existing_document is None:
                document = await DbOperations.create_central_document(
                    db_session=db_session,
                    name=document_name,
                    description=document_description,
                    url=document_url,
                )
            else:
                document = existing_document

            # Check that the chunk does not already exist for this document
            existing_chunk = await DbOperations.get_existing_chunk(
                db_session=db_session,
                search_index_id=search_index.id,
                document_id=document.id,
                name=chunk_name,
                content=chunk_content,
            )
            if existing_chunk is None:
                # Prepare data for OpenSearch
                doc_to_index = OpenSearchRecord(
                    document_name, document_url, chunk_name, chunk_content
                ).to_opensearch_dict()

                # Index the document in OpenSearch
                response = opensearch_client.index(index=search_index.name, body=doc_to_index)
                id_opensearch = response["_id"]

                # Insert the document chunk into PostgreSQL
                _ = await DbOperations.add_chunk(
                    db_session=db_session,
                    search_index_id=search_index.id,
                    document_id=document.id,
                    name=chunk_name,
                    content=chunk_content,
                    id_opensearch=id_opensearch,
                )
            else:
                continue

        logger.info("Successfully loaded new document chunks")
        return True

    except Exception as e:
        raise RuntimeError(f"Error loading new document chunks: {e}") from e


async def list_chunks_in_central_rag(
    db_session: AsyncSession, show_deleted_chunks: bool = False
) -> List[DocumentChunk]:
    """Lists all active DocumentChunk instances in the PostgreSQL database."""

    # Get all central documents first
    central_documents = await DbOperations.get_central_documents(db_session)
    if central_documents:
        first_doc = central_documents[0]
        logger.info(f"First central document attributes: {first_doc._mapping.keys()}")
        logger.info(f"First central document values: {first_doc._asdict()}")
    else:
        logger.info("No central documents found")
    # Create lookup dict of id -> name
    document_names = {doc.id: doc.name for doc in central_documents}
    logger.info(f"Document names found in central guidance: {document_names=}")

    # Get all central chunks
    central_index = await get_central_index(db_session)
    chunks = await DbOperations.get_document_chunks_filtered_with_search_index(
        db_session=db_session, search_index_uuid=central_index.uuid, show_deleted_chunks=show_deleted_chunks
    )

    chunks_formatted = [
        DocumentChunkResponse(
            uuid=chunk.uuid,
            created_at=chunk.created_at,
            updated_at=chunk.updated_at,
            deleted_at=chunk.deleted_at,
            document_name=document_names.get(chunk.document_id),
            chunk_name=chunk.name,
            chunk_content=chunk.content,
            id_opensearch=chunk.id_opensearch,
        )
        for chunk in chunks
    ]

    return ListDocumentChunkResponse(document_chunks=chunks_formatted)


# Does not soft delete the Document object as the Document may be used across multiple indexes.
async def delete_chunk_in_central_rag(db_session: AsyncSession, document_chunk_uuid: UUID):
    opensearch_client = create_client()

    deleted_at = datetime.now()

    central_index = await get_central_index(db_session)

    # Start with deleting the document chunk in PostgreSQL
    document_chunk = await DbOperations.get_document_chunk_by_uuid(
        db_session=db_session, document_chunk_uuid=document_chunk_uuid
    )
    if document_chunk:
        document_chunk.deleted_at = deleted_at
    else:
        logger.info(f"No SearchIndex found with uuid: {document_chunk_uuid}")

    response = opensearch_client.delete(index=central_index.name, id=document_chunk.id_opensearch)
    logger.info(f"Raw OpenSearch response to delete request: {response}")
    if response["result"] == "deleted":
        logger.info(f"DocumentChunk '{document_chunk.name}' has been successfully deleted.")
    else:
        raise RuntimeError(f"Failed to delete DocumentChunk '{document_chunk.name}'.")

    return True
