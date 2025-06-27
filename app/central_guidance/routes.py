# ruff: noqa: B008

from uuid import UUID

from fastapi import APIRouter, Body, Depends

from app.api import ENDPOINTS
from app.auth.auth_token import auth_token_validator_no_user
from app.central_guidance.constants import DEFAULT_CHUNKS
from app.central_guidance.schemas import ListDocumentChunkResponse
from app.central_guidance.service_index import (
    delete_chunk_in_central_rag,
    list_chunks_in_central_rag,
    load_chunks_to_central_rag,
    sync_central_index,
)
from app.database.table import async_db_session

router = APIRouter()


@router.get(ENDPOINTS.CENTRAL_RAG_DOCUMENT_CHUNKS, dependencies=[Depends(auth_token_validator_no_user)])
async def get_chunks() -> ListDocumentChunkResponse:
    """Lists all the document chunks stored in the PostgreSQL database."""
    async with async_db_session() as db_session:
        return await list_chunks_in_central_rag(db_session)


@router.post(
    ENDPOINTS.CENTRAL_RAG_DOCUMENT_CHUNKS,
    dependencies=[Depends(auth_token_validator_no_user)],
)
async def create_chunks(chunks: list = Body(DEFAULT_CHUNKS)) -> bool:
    async with async_db_session() as db_session:
        return await load_chunks_to_central_rag(db_session=db_session, data=chunks)


@router.delete(ENDPOINTS.CENTRAL_RAG_DOCUMENT_CHUNK, dependencies=[Depends(auth_token_validator_no_user)])
async def delete_chunk(document_chunk_uuid: UUID) -> bool:
    """Soft deletes document chunks in the PostgreSQL database and hard deletes the document chunk in OpenSearch."""
    async with async_db_session() as db_session:
        return await delete_chunk_in_central_rag(db_session, document_chunk_uuid)


@router.put(ENDPOINTS.CENTRAL_RAG_SYNC, dependencies=[Depends(auth_token_validator_no_user)])
async def sync_indexes() -> bool:
    """Synchronises OpenSearch with the PostgreSQL representation of the search indexes and document chunks."""
    async with async_db_session() as db_session:
        return await sync_central_index(db_session)
