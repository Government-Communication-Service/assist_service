from logging import getLogger

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import config
from app.central_guidance.schemas import RagRequest, RetrievalResult
from app.database.db_operations import DbOperations
from app.database.models import (
    Document,
    DocumentChunk,
    Message,
    MessageSearchIndexMapping,
    SearchIndex,
)
from app.database.table import async_db_session
from app.document_upload.constants import PERSONAL_DOCUMENTS_INDEX_NAME
from app.document_upload.personal_document_rag import personal_document_rag
from app.opensearch.service import AsyncOpenSearchOperations

logger = getLogger(__name__)


def format_doc_chunk_for_prompt(doc_chunk: DocumentChunk, document: Document):
    result_string = (
        f"<document-title>{document.name}</document-title>\n"
        f"<section-title>{doc_chunk.name}</section-title>\n"
        f"<content>{doc_chunk.content}</content>"
    )
    return result_string


async def search_uploaded_documents(rag_request: RagRequest, message: Message, db_session: AsyncSession):
    """
    Runs RAG process to search information related to the user's query and
    inject that information to the user query sent to the LLM model.
    """
    logger.debug("starting to run rag with %s", rag_request)
    try:
        return await rag_pdu(rag_request, message, db_session)
    except Exception:
        logger.exception("Rag process got error")
        raise


async def rag_pdu(rag_request: RagRequest, message: Message, db_session: AsyncSession):
    # rag with personal document index
    pdu_rag_output = (
        await personal_document_rag(
            rag_request,
            message,
            db_session,
        )
        if rag_request.document_uuids
        else {}
    )

    # Unpack the data
    pdu_retrieval_results = pdu_rag_output.get("retrieval_results", [])

    pdu_message_search_index_mappings = pdu_rag_output.get("message_search_index_mappings", [])

    # Use the search results to create an enriched query
    # If the list of retrieval_results is empty, then the returned query will simply be the original user query.
    # The citation_message is appended to the AI message.
    prompt_segment_user_retrieval, citation_message = await compile_document_upload_prompt_segments(
        rag_request,
        pdu_message_search_index_mappings,
        pdu_retrieval_results,
    )
    logger.info("Citation is %s", citation_message)
    return (prompt_segment_user_retrieval, citation_message)


async def compile_document_upload_prompt_segments(
    rag_request: RagRequest,
    user_message_search_index_mappings: list[MessageSearchIndexMapping],
    user_retrieval_results: list[RetrievalResult],
):
    # initial citation and prompt segment values
    citation_message = []
    prompt_segment_user_retrieval = ""

    if user_retrieval_results:
        citation_message = []
        if user_retrieval_results:
            prompt_segment_user_retrieval = "<uploaded-documents-search-results>"
            # user retrievals are stored in single index
            # keep a single doc citation if multiple chunks used from same document.
            user_doc_citations = {}

            for i, retrieval_result in enumerate(user_retrieval_results):
                document = retrieval_result.document
                user_doc_citations[str(document.uuid)] = {"docname": document.name, "docurl": document.url}

                doc_chunk = retrieval_result.document_chunk
                result_string = format_doc_chunk_for_prompt(doc_chunk, document)
                prompt_segment_user_retrieval += f"\n<result-{i}>\n{result_string}\n</result-{i}>"
            prompt_segment_user_retrieval += "\n</uploaded-documents-search-results>"
            citation_message = citation_message + list(user_doc_citations.values())

        return (prompt_segment_user_retrieval, citation_message)

    # central index results
    async with async_db_session() as session:
        # check user indexes
        if (len(user_retrieval_results) == 0) and (len(user_message_search_index_mappings) > 0):
            prompt_segment_user_retrieval = (
                "<uploaded-documents-search-results>\n"
                "The following document(s) were searched but no relevant material was found:\n"
            )
            user_searched_index_ids = [
                message_search_index_mapping.search_index_id
                for message_search_index_mapping in user_message_search_index_mappings
                if message_search_index_mapping.use_index
            ]
            for searched_index_id in user_searched_index_ids:
                execute = await session.execute(
                    select(Document)
                    .distinct(Document.id)
                    .join(DocumentChunk, DocumentChunk.document_id == Document.id)
                    .join(SearchIndex, SearchIndex.id == DocumentChunk.search_index_id)
                    .where(SearchIndex.id == searched_index_id)
                    .where(Document.uuid.in_(rag_request.document_uuids))
                )
                documents = execute.scalars().all()
                for document in documents:
                    prompt_segment_user_retrieval += f"\n<document-title>{document.name}</document-title>"
            prompt_segment_user_retrieval += "\n</uploaded-documents-search-results>"

        return (prompt_segment_user_retrieval, citation_message)


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
