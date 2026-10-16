from pydantic import Field, field_validator, model_validator

from app.bedrock.tool_schema import REMOVE_DUPLICATES_DESCRIPTION, LenientList, ToolArgs, ToolDef
from app.config import UPLOADED_DOCUMENTS_MAX_TOOL_QUERIES


class SearchUploadedDocumentsArgs(ToolArgs):
    """Search the documents the user has attached to this chat (listed in the context, with their ids)
    for the passages that best match your queries. Passages are labelled with their document and their
    place in it (for example "chunk 34 of 87") and given back in document order, not relevance order.
    Use it to find where something is mentioned, or to look in particular documents. To read a document
    from the start, use read_uploaded_document instead. Use it whenever the user's request concerns
    their documents."""

    queries: LenientList[str] = Field(
        description=(
            f"1 to {UPLOADED_DOCUMENTS_MAX_TOOL_QUERIES} keyword queries for a BM25 keyword search over the "
            "documents' text. Words are matched exactly, without stemming: 'budget' will not match 'budgets'. "
            "Quote the most relevant words from the user's message in the first query; use synonyms and other "
            "word forms in the others."
        )
    )
    document_ids: LenientList[str] | None = Field(
        default=None,
        description=(
            "Ids of the documents to search, as listed in the context; leave out to search all the attached "
            "documents. Use it to narrow the search when only some documents are relevant."
        ),
    )
    remove_duplicates: bool = Field(default=False, description=REMOVE_DUPLICATES_DESCRIPTION)

    @field_validator("document_ids", mode="before")
    @classmethod
    def _accept_single_id(cls, value: object) -> object:
        """Treat one id sent as a bare string as a list of one; JSON-encoded lists are left for LenientList."""
        if isinstance(value, str) and not value.lstrip().startswith("["):
            return [value]
        return value


SEARCH_UPLOADED_DOCUMENTS = ToolDef("search_uploaded_documents", SearchUploadedDocumentsArgs, strict=False)


class ReadUploadedDocumentArgs(ToolArgs):
    """Read one of the documents the user has attached to this chat, in order, a section at a time.
    Use it to read a document from the start, or to carry on from where a search result or an earlier
    read left off. The result says which chunks it covers and how to continue; long documents take
    several calls. For finding a particular passage in a long document, search_uploaded_documents is
    quicker."""

    document_id: str = Field(description="The document's id, as listed in the context.")
    start_chunk: int = Field(default=1, ge=1, description="The chunk to start from. Default 1, the start.")
    end_chunk: int | None = Field(
        default=None,
        ge=1,
        description=(
            "The last chunk to read, to read a particular stretch, for example the chunks around a search "
            "result. Leave out to read as much as fits. Fewer chunks are returned if the range is too long."
        ),
    )

    @model_validator(mode="after")
    def _check_order(self) -> "ReadUploadedDocumentArgs":
        if self.end_chunk is not None and self.end_chunk < self.start_chunk:
            raise ValueError("'end_chunk' must not be before 'start_chunk'")
        return self


READ_UPLOADED_DOCUMENT = ToolDef("read_uploaded_document", ReadUploadedDocumentArgs, strict=False)
