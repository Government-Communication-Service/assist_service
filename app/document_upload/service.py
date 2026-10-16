from logging import getLogger

from sqlalchemy import func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import config
from app.config import UPLOADED_DOCUMENTS_MAX_READ_CHARS, UPLOADED_DOCUMENTS_MAX_TOOL_QUERIES
from app.database.db_operations import DbOperations
from app.database.models import (
    Document,
    DocumentChunk,
    MessageDocumentChunkMapping,
    MessageSearchIndexMapping,
    RewrittenQuery,
    SearchIndex,
)
from app.database.table import async_db_session
from app.document_upload.constants import PERSONAL_DOCUMENTS_INDEX_NAME
from app.document_upload.personal_document_rag import select_chunks_for_tool
from app.document_upload.utils import (
    PositionedChunk,
    chunk_positions,
    get_document_chunk_mappings,
    get_previously_returned_opensearch_ids,
)
from app.opensearch.service import AsyncOpenSearchOperations

logger = getLogger(__name__)


async def clean_expired_documents(db_session: AsyncSession):
    """
    Delete expired documents from OpenSearch first, then from the database.

    Key improvement:
    - OpenSearch deletion happens BEFORE DB mutation
    - Partial failures are tracked; only successful deletes update DB
    - Failed chunks remain in DB for retry
    """
    try:
        # 1) Getting expired chunks from DB
        expired_chunks = await DbOperations.get_expired_chunks_for_cleanup(db_session)

        if not expired_chunks:
            logger.info("Marked 0 expired document chunk(s) as deleted.")
            logger.info("No opensearch ids found for deletion from opensearch.")
            return {"deleted_count": 0, "failed_count": 0}

        # Build lookup maps
        chunk_id_by_os_id = {os_id: chunk_id for chunk_id, doc_id, os_id in expired_chunks}
        doc_id_by_chunk_id = {chunk_id: doc_id for chunk_id, doc_id, os_id in expired_chunks}
        opensearch_ids = list(chunk_id_by_os_id.keys())
        logger.info("Marked %s expired document chunk(s) as deleted.", len(opensearch_ids))

        deleted_os_ids: list[str] = []
        failed_os_ids: list[str] = []

        # 2) Delete from OpenSearch first (in batches)
        opensearch_delete_batch_size = config.OPENSEARCH_DELETE_BATCH_SIZE
        for i in range(0, len(opensearch_ids), opensearch_delete_batch_size):
            batch = opensearch_ids[i : i + opensearch_delete_batch_size]
            try:
                await AsyncOpenSearchOperations.delete_document_chunks(
                    PERSONAL_DOCUMENTS_INDEX_NAME,
                    batch,
                )
                deleted_os_ids.extend(batch)
                logger.info("Deleted batch of %s chunk(s) from OpenSearch.", len(batch))
            except Exception:
                logger.exception("OpenSearch delete failed for batch starting at index %s", i)
                failed_os_ids.extend(batch)

        # 3) update DB only for successfully deleted OpenSearch docs
        deleted_chunk_ids = [chunk_id_by_os_id[os_id] for os_id in deleted_os_ids]
        deleted_doc_ids = [doc_id_by_chunk_id[chunk_id] for chunk_id in deleted_chunk_ids]
        batch_size = config.DOCUMENT_CLEANUP_BATCH_SIZE
        if deleted_chunk_ids:
            try:
                await DbOperations.mark_chunks_as_deleted(db_session, deleted_chunk_ids, batch_size)
                await DbOperations.mark_documents_as_deleted(db_session, deleted_doc_ids, batch_size)
                logger.info("Successfully deleted %s document chunk(s) from OpenSearch.", len(opensearch_ids))
            except Exception as e:
                logger.exception("Database update failed after OpenSearch deletion.")
                raise Exception(f"Database update failed after OpenSearch deletion: {e}") from e

        # 4) Log failures (chunks remain in DB for future retry)
        if failed_os_ids:
            logger.warning(
                "%s chunk(s) failed OpenSearch deletion and will be retried later.",
                len(failed_os_ids),
            )

        return {
            "deleted_count": len(deleted_chunk_ids),
            "failed_count": len(failed_os_ids),
        }

    except Exception:
        logger.exception("Error during document cleanup")
        raise


UPLOADED_DOCUMENTS_TAG = "uploaded-documents-search-results"
READ_DOCUMENT_TAG = "uploaded-document"
MAX_CHUNKS_PER_READ = 200
"""Cap on chunks loaded for one read, whatever their size; the continuation note covers the rest."""


def _tagged(tag: str, text: str) -> str:
    return f"<{tag}>\n{text}\n</{tag}>"


def _scope_documents(document_ids: list[str] | None, attached: list[str]) -> tuple[list[str], str | None]:
    """Resolve the documents a tool call asks for against the ones attached to the chat.

    Args:
        document_ids: Ids from the model; None or empty means all attached documents.
        attached: Documents attached to the chat, from the request context.

    Returns:
        The attached documents to use, and an error message if any requested id isn't attached.
    """
    requested = [d.strip() for d in document_ids or [] if d.strip()]
    if not requested:
        return attached, None
    by_lower = {u.lower(): u for u in attached}
    unknown = [d for d in requested if d.lower() not in by_lower]
    if unknown:
        return [], (
            f"Unknown document id(s): {', '.join(unknown)}. Use ids from the list of attached documents: "
            f"{', '.join(attached)}."
        )
    return list(dict.fromkeys(by_lower[d.lower()] for d in requested)), None


async def _describe_scope(db_session: AsyncSession, searched: list[str], attached: list[str]) -> str:
    """One line saying which documents a search covered, so a misapplied scope is visible to the model."""
    if len(searched) == len(attached):
        return f"Searched all {len(attached)} attached document(s)."
    names = (
        (await db_session.execute(select(Document.name).where(Document.uuid.in_(searched)).order_by(Document.name)))
        .scalars()
        .all()
    )
    return f"Searched {len(searched)} of {len(attached)} attached documents: {', '.join(names)}."


def format_chunk_for_prompt(result: PositionedChunk) -> str:
    return (
        f"<document-title>{result.document.name}</document-title>\n"
        f"<document-id>{result.document.uuid}</document-id>\n"
        f"<position>chunk {result.position} of {result.total_chunks}</position>\n"
        f"<content>{result.document_chunk.content}</content>"
    )


async def search_uploaded_documents_for_tool(
    queries: list[str],
    document_uuids: list[str],
    message_id: int,
    llm_internal_response_id: int | None,
    db_session: AsyncSession,
    remove_duplicates: bool = False,
    document_ids: list[str] | None = None,
) -> tuple[str, list[dict], bool]:
    """Search the chat's uploaded documents with the chat model's own keyword queries.

    Records the search (message_search_index_mapping), the queries (rewritten_query)
    and the chunks returned (message_document_chunk_mapping).

    Holds no DB connection during the OpenSearch queries: the previously returned chunks are read
    in a short session of their own, and everything else runs on `db_session` once the search is done.

    Args:
        queries: Keyword queries from the model; blanks and duplicates dropped, first
            UPLOADED_DOCUMENTS_MAX_TOOL_QUERIES used.
        document_uuids: Documents attached to the chat, from the request context.
        message_id: This turn's user message id.
        llm_internal_response_id: The logged model call that requested the search.
        db_session: Session to query and record analytics in, used only after the search.
        remove_duplicates: Skip chunks already returned earlier in the chat, this turn included.
        document_ids: Documents to restrict the search to; None or empty for all attached.

    Returns:
        The tool result text, the citations for the documents returned, and whether the call failed.
    """
    if not document_uuids:
        return (_tagged(UPLOADED_DOCUMENTS_TAG, "No documents are attached to this chat."), [], False)

    search_uuids, scope_error = _scope_documents(document_ids, document_uuids)
    if scope_error:
        return _tagged(UPLOADED_DOCUMENTS_TAG, scope_error), [], True

    unique_queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:UPLOADED_DOCUMENTS_MAX_TOOL_QUERIES]
    already_returned = set()
    if remove_duplicates:
        async with async_db_session() as read_session:
            already_returned = await get_previously_returned_opensearch_ids(read_session, message_id)
    chunks, skipped = await select_chunks_for_tool(unique_queries, search_uuids, already_returned)

    search_index = (
        await db_session.execute(select(SearchIndex).where(SearchIndex.name == PERSONAL_DOCUMENTS_INDEX_NAME))
    ).scalar_one()
    db_session.add(
        MessageSearchIndexMapping(
            search_index_id=search_index.id,
            message_id=message_id,
            llm_internal_response_id=llm_internal_response_id,
            use_index=True,
        )
    )
    if unique_queries:
        await db_session.execute(
            insert(RewrittenQuery),
            [
                {
                    "search_index_id": search_index.id,
                    "message_id": message_id,
                    "llm_internal_response_id": llm_internal_response_id,
                    "content": query,
                }
                for query in unique_queries
            ],
        )
    results = await get_document_chunk_mappings(chunks, search_uuids, message_id, db_session, use_chunk=True)
    logger.info(f"Uploaded documents tool search: {len(results)} chunks returned, {skipped} already returned")

    scope = await _describe_scope(db_session, search_uuids, document_uuids)
    note = f"{skipped} matching passage(s) were already provided earlier in the conversation." if skipped else ""
    if not results:
        return (_tagged(UPLOADED_DOCUMENTS_TAG, f"{scope}\nNo new relevant passages found. {note}"), [], False)

    results.sort(key=lambda r: (r.document.name, r.document.id, r.position))
    citations = {}
    parts = [f"<{UPLOADED_DOCUMENTS_TAG}>", scope]
    for i, result in enumerate(results):
        citations[str(result.document.uuid)] = {"docname": result.document.name, "docurl": result.document.url}
        parts.append(f"<result-{i}>\n{format_chunk_for_prompt(result)}\n</result-{i}>")
    if note:
        parts.append(note)
    parts.append(f"</{UPLOADED_DOCUMENTS_TAG}>")
    return ("\n".join(parts), list(citations.values()), False)


async def read_uploaded_document_for_tool(
    document_id: str,
    start_chunk: int,
    document_uuids: list[str],
    message_id: int,
    db_session: AsyncSession,
    end_chunk: int | None = None,
) -> tuple[str, list[dict], bool]:
    """Return a stretch of one uploaded document in reading order, and how to continue.

    Whole chunks are returned, at least one, until UPLOADED_DOCUMENTS_MAX_READ_CHARS is reached. The
    chunks are recorded against the message so a later search with remove_duplicates skips them.

    Args:
        document_id: The document to read; must be one attached to the chat.
        start_chunk: 1-based chunk to start from.
        document_uuids: Documents attached to the chat, from the request context.
        message_id: This turn's user message id.
        db_session: Session to query and record in.
        end_chunk: 1-based last chunk to return; None for as many as fit.

    Returns:
        The tool result text, the citation for the document, and whether the call failed.
    """

    def failure(message: str) -> tuple[str, list[dict], bool]:
        return _tagged(READ_DOCUMENT_TAG, message), [], True

    if not document_uuids:
        return failure("No documents are attached to this chat.")
    read_uuids, scope_error = _scope_documents([document_id], document_uuids)
    if scope_error:
        return failure(scope_error)

    if end_chunk is not None and end_chunk < start_chunk:
        return failure(f"end_chunk {end_chunk} is before start_chunk {start_chunk}.")

    positions = chunk_positions(read_uuids)
    stretch = select(DocumentChunk, Document, positions.c.position, positions.c.total_chunks)
    stretch = stretch.join(positions, positions.c.chunk_id == DocumentChunk.id)
    stretch = stretch.join(Document, Document.id == DocumentChunk.document_id)
    stretch = stretch.where(positions.c.position >= start_chunk)
    if end_chunk is not None:
        stretch = stretch.where(positions.c.position <= end_chunk)
    rows = (await db_session.execute(stretch.order_by(positions.c.position).limit(MAX_CHUNKS_PER_READ))).all()

    if not rows:
        total = (await db_session.execute(select(func.count()).select_from(positions))).scalar_one()
        if total == 0:
            return failure("This document has no readable text yet: it may still be processing, or have none.")
        return failure(f"This document has {total} chunk(s); start_chunk {start_chunk} is past the end.")

    chosen = []
    chars_used = 0
    for row in rows:
        size = len(row.DocumentChunk.content)
        if chosen and chars_used + size > UPLOADED_DOCUMENTS_MAX_READ_CHARS:
            break
        chosen.append(row)
        chars_used += size

    document = chosen[0].Document
    first, last, total = chosen[0].position, chosen[-1].position, chosen[0].total_chunks
    db_session.add_all(
        MessageDocumentChunkMapping(
            message_id=message_id, document_chunk_id=row.DocumentChunk.id, opensearch_score=0.0, use_document_chunk=True
        )
        for row in chosen
    )
    if last < total:
        where = (
            f"Chunks {first}-{last} of {total}. To continue, call read_uploaded_document with "
            f'document_id="{document.uuid}" and start_chunk={last + 1}.'
        )
    else:
        where = f"Chunks {first}-{last} of {total}. End of document."
    name = document.name.replace('"', "'")
    parts = [f'<{READ_DOCUMENT_TAG} id="{document.uuid}" name="{name}">', where]
    parts.extend(f'<chunk n="{row.position}">{row.DocumentChunk.content}</chunk>' for row in chosen)
    parts.append(f"</{READ_DOCUMENT_TAG}>")
    return "\n".join(parts), [{"docname": document.name, "docurl": document.url}], False
