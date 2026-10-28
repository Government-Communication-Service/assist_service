# ruff: noqa: A005
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from pydantic import UUID4, BaseModel, Field

from app.database.models import (
    Document,
    DocumentChunk,
    MessageDocumentChunkMapping,
    SearchIndex,
)


class SearchIndexResponse(BaseModel):
    # id: int
    uuid: UUID4
    created_at: datetime
    updated_at: Optional[datetime] = Field(default=None)
    deleted_at: Optional[datetime] = Field(default=None)
    name: str
    description: str


class ListSearchIndexResponse(BaseModel):
    search_indexes: List[SearchIndexResponse]


class DocumentChunkResponse(BaseModel):
    uuid: UUID4
    created_at: datetime
    updated_at: Optional[datetime] = Field(default=None)
    deleted_at: Optional[datetime] = Field(default=None)
    document_name: str
    chunk_name: str
    chunk_content: str
    id_opensearch: str


class ListDocumentChunkResponse(BaseModel):
    document_chunks: List[DocumentChunkResponse]


# This class collects various SqlAlchemy data objects into a single object to be passed around
@dataclass
class RetrievalResult:
    search_index: SearchIndex
    document_chunk: DocumentChunk
    document: Document
    message_document_chunk_mapping: MessageDocumentChunkMapping = None

    def __str__(self):
        return (
            f"included=({self.message_document_chunk_mapping.use_document_chunk}), "
            f"Document.name=({self.document.name}), "
            f"DocumentChunk.name=({self.document_chunk.name}), "
            f"DocumentChunk.content_truncated=({self.document_chunk.content[0:30]}...{self.document_chunk.content[-30:]})"
        )


@dataclass
class RagRequest:
    use_central_rag: bool
    user_id: int
    query: str
    document_uuids: Optional[List[str]] = None
