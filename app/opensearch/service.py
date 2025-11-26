import logging
import os
import re
import traceback
from typing import Dict, List, Optional

from dotenv import load_dotenv
from opensearchpy import AsyncOpenSearch, OpenSearch
from opensearchpy.exceptions import RequestError, TransportError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import env_variable
from app.database.db_operations import DbOperations
from app.database.models import SearchIndex
from app.document_upload.constants import PERSONAL_DOCUMENTS_INDEX_NAME
from app.logs import Action, LogsHandler
from app.opensearch.exceptions import DocumentOperationError

load_dotenv()

logging.getLogger("opensearch").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def verify_connection_to_opensearch():
    try:
        client = create_client()
        # Test if the client is created successfully
        assert client is not None

        # Test if the client has the correct attributes
        assert hasattr(client, "search")
        assert hasattr(client, "index")
        assert hasattr(client, "delete")

        # Test if the client can query which indexes are present
        response = client.indices.get_alias("*")
        assert response and len(response) > 0, "No indexes found"
        logger.info("Connection to OpenSearch succesful")
    except Exception as ex:
        traceback_str = traceback.format_exc()
        raise ConnectionError(f"Error connecting with OpenSearch: \n{traceback_str}\n\n") from ex
    logger.info("Succesfully connected to OpenSearch.")
    return


def normalise_string(string):
    """A utility function for making sure index names are allowed by OpenSearch."""

    string = string.strip()
    # Replace all non-letter and non-numerical characters with an underscore
    string = re.sub(r"[^a-zA-Z0-9]", "_", string)
    # Replace all uppercase letters followed by a lowercase letter or a digit with an underscore
    # and the letter in lowercase
    string = re.sub(r"([A-Z]+)([A-Z][a-z]|[0-9])", r"\1_\2", string)
    # Replace all lowercase letters or digits followed by an uppercase letter with the letter, an underscore,
    # and the letter in lowercase
    string = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", string)
    # Convert to lowercase
    string = string.lower()
    # Remove any leading or trailing underscores
    string = string.strip("_")
    # Replace multiple underscores with a single one
    string = re.sub(r"__+", "_", string)

    return string


class OpenSearchClient:
    _instance = None

    @classmethod
    def get_client(cls):
        if cls._instance is None:
            username = os.getenv("OPENSEARCH_USER")
            password = os.getenv("OPENSEARCH_PASSWORD")
            host = os.getenv("OPENSEARCH_HOST")
            port = os.getenv("OPENSEARCH_PORT")
            use_ssl = True
            if env_variable("OPENSEARCH_DISABLE_SSL", False):
                use_ssl = False

            cls._instance = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=(username, password),
                use_ssl=use_ssl,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False,
            )
        return cls._instance


def create_client():
    """Creates or returns existing OpenSearch client."""
    return OpenSearchClient.get_client()


def create_async_client():
    """Creates an AsyncOpenSearch client."""

    username = os.getenv("OPENSEARCH_USER")
    password = os.getenv("OPENSEARCH_PASSWORD")
    host = os.getenv("OPENSEARCH_HOST")
    port = os.getenv("OPENSEARCH_PORT")
    use_ssl = True
    if env_variable("OPENSEARCH_DISABLE_SSL", False):
        use_ssl = False

    client = AsyncOpenSearch(
        hosts=[{"host": host, "port": int(port)}],
        http_auth=(username, password),
        use_ssl=use_ssl,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )

    return client


async def list_indexes(db_session: AsyncSession, include_personal_document_index: bool = False) -> List[SearchIndex]:
    """
    Lists all active indexes in the PostgreSQL database.
    if include_personal_document_index is True, the personal_document_uploads index will be included in the list.
    """
    indexes = await DbOperations.get_active_indexes(
        db_session=db_session,
        personal_documents_index_name=PERSONAL_DOCUMENTS_INDEX_NAME,
        include_personal_document_index=include_personal_document_index,
    )

    return indexes


class AsyncOpenSearchClient:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = create_async_client()
        return cls._instance

    @classmethod
    async def close(cls):
        """Close the async client connection if it exists"""
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None


class AsyncOpenSearchOperations:
    @staticmethod
    def _collect_errors(response: Dict, action_type: str) -> List[str]:
        errors = []
        for item in response.get("items", []):
            action = item.get(action_type, {})
            if "error" in action:
                error_type = action["error"].get("type", "Unknown")
                reason = action["error"].get("reason", "Unknown")
                status = action.get("status", "Unknown")
                doc_id = action.get("_id", "Unknown ID")
                msg = (
                    f"Document {action_type} error for ID {doc_id}: "
                    f"Status {status}, Error Type: {error_type}, Reason: {reason}"
                )
                errors.append(msg)
        return errors

    @staticmethod
    async def index_document_chunks(index: str, document_list: List[Dict]):
        """
        Indexes a list of document chunks into the specified OpenSearch index using Opensearch async bulk operation.
        If the indexing operation encounters errors, it collects the errors and raises a `DocumentOperationError`.

        Args:
            index (str): The name of the OpenSearch index to which documents should be added.
            document_list (List[Dict]): A list of dictionaries representing the documents to be indexed.

        Returns:
            Dict: The response from the OpenSearch bulk indexing API if the operation completes without errors.

        Raises:
            DocumentOperationError: If any errors occur during the indexing operation, this exception is raised
                                    with the details of the errors.

        """

        logger.info("Indexing %d documents to index %s", len(document_list), index)
        docs = []
        doc_header = {"index": {"_index": index}}
        for d in document_list:
            docs.append(doc_header)
            docs.append(d)

        index_action = AsyncOpenSearchClient.get().bulk(body=docs)
        response = await LogsHandler.with_logging(Action.OPENSEARCH_INDEX_DOCUMENT, index_action)

        if not response.get("errors", False):
            return response

        # collect errors and raise exception
        errors = AsyncOpenSearchOperations._collect_errors(response, action_type="index")
        if not errors:
            return response

        exception_msg = "\n".join(errors)
        logger.error(exception_msg)
        raise DocumentOperationError(exception_msg)

    @staticmethod
    async def delete_document_chunks(index: str, ids: List[str]) -> Optional[Dict]:
        """
        Deletes multiple document chunks from the specified OpenSearch index using async bulk.

        If any errors occur during deletion, they are logged, and a `DocumentOperationError`
        is raised with details.

        Args:
            index (str): The name of the OpenSearch index from which the documents will be deleted.
            ids (List[str]): A list of document IDs to be deleted.

        Returns:
            dict: The OpenSearch bulk deletion response if successful and no errors occur.

        Raises:
            DocumentOperationError: If there are errors during the deletion process, an exception
                is raised containing the error details.

        """
        if not ids:
            logger.info("No document IDs provided for deletion. Skipping deletion process.")
            return None

        logger.info("Deleting %s document chunks from index %s", len(ids), index)
        # build bulk request
        request = [{"delete": {"_index": index, "_id": _id}} for _id in ids]
        delete_action = AsyncOpenSearchClient.get().bulk(body=request)
        response = await LogsHandler.with_logging(Action.OPENSEARCH_DELETE_DOCUMENT, delete_action)

        if not response.get("errors", False):
            return response

        errors = AsyncOpenSearchOperations._collect_errors(response, action_type="delete")
        if not errors:
            return response

        exception_msg = "\n".join(errors)
        logger.error(exception_msg)
        raise DocumentOperationError(exception_msg)

    @staticmethod
    async def get_document_chunks(index: str, document_uuid: str, max_size: int = 6) -> List[Dict]:
        """
        Gets document chunks from `from the index and document uuid provided.

        Args:
            index (str): The name of the OpenSearch index from which the documents will be retrieved.
            document_uuid (str): The document uuid to get chunks for.
            max_size (int): The maximum number of document chunks to fetch from index for the document_uuid.
            The default is set to 6 because this method is used to detect if a document contains more than 5 chunks.
        Returns:
            List[Dict]: The document chunks retrieved from the index, from the document_uuid provided

        """

        logger.info(
            "Attempting to retrieve max %s document chunks  from index %s for document %s",
            max_size,
            index,
            document_uuid,
        )
        request_body = {
            "query": {"bool": {"filter": [{"terms": {"document_uuid.keyword": [document_uuid]}}]}},
            "size": max_size,
        }

        action = AsyncOpenSearchClient.get().search(request_body, index=index)
        response = await LogsHandler.with_logging(Action.OPENSEARCH_SEARCH_DOCUMENT, action)
        hit_elements = response["hits"]["hits"]
        logger.info("Found %s document chunks for document %s", len(hit_elements), document_uuid)
        return hit_elements

    @staticmethod
    async def search_user_document_chunks(document_uuid: str, query: str, index: str) -> List[Dict]:
        logger.debug("searching index: %s  with document: %s and user query: %s", index, document_uuid, query)

        search_body = {
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
                        {"terms": {"document_uuid.keyword": [document_uuid]}},
                    ],
                }
            },
            "size": 5,
        }
        logger.debug("built search query %s", search_body)
        # Perform the search
        try:
            response = await AsyncOpenSearchClient.get().search(body=search_body, index=index)
            chunks = response["hits"]["hits"]
            logger.info(
                "%s chunks retrieved from %s using doc %s",
                len(chunks),
                index,
                document_uuid,
            )
            return chunks
        except Exception as e:
            AsyncOpenSearchOperations._handle_search_error(e, query, index, str(search_body))

        # return no match
        return []

    @staticmethod
    async def search_for_chunks(query: str, index: str) -> List[Dict]:
        logger.info("Searching central index for chunks: %s", index)

        # Prepare the search body
        search_body = {
            "query": {"multi_match": {"query": query, "fields": ["document_name", "chunk_name", "chunk_content"]}},
            "size": 3,
        }

        # Perform the search
        try:
            response = await AsyncOpenSearchClient.get().search(body=search_body, index=index)
            chunks = response["hits"]["hits"]
            logger.info(f"{len(chunks)} chunks retrieved from {index}")
            return chunks
        except Exception as e:
            AsyncOpenSearchOperations._handle_search_error(e, query, index, str(search_body))

        # return no match
        return []

    @staticmethod
    def _handle_search_error(ex: Exception, query: str, index: str, search_body: str):
        if not isinstance(ex, (RequestError, TransportError)):
            raise ex

        # skip large text user queries and log as error
        if (
            ex.status_code in [400, 500]
            and ex.error == "search_phase_execution_exception"
            and "maxClauseCount is set to 1024" in str(ex)
        ):
            logger.error("Search error\nIndex:%s\nUser query:%s\nSearch body:%s", index, query, search_body)
        else:
            raise ex

    @staticmethod
    async def search_multiple_document_chunks(
        document_uuids: List[str], query: str, index: str, max_size: int = 50
    ) -> List[Dict]:
        """
        Searches for document chunks across multiple documents using the provided query.

        Args:
            document_uuids (List[str]): List of document UUIDs to search within.
            query (str): The search query to match against document chunks.
            index (str): The name of the OpenSearch index to search in.
            max_size (int): Maximum number of chunks to return. Defaults to 50.

        Returns:
            List[Dict]: List of document chunks matching the query across all specified documents.
        """
        logger.debug("searching index: %s with documents: %s and user query: %s", index, document_uuids, query)

        search_body = {
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
        logger.debug("built search query %s", search_body)

        # Perform the search
        try:
            response = await AsyncOpenSearchClient.get().search(body=search_body, index=index)
            chunks = response["hits"]["hits"]
            logger.info(
                "%s chunks retrieved from %s using docs %s",
                len(chunks),
                index,
                document_uuids,
            )
            return chunks
        except Exception as e:
            AsyncOpenSearchOperations._handle_search_error(e, query, index, str(search_body))

        # return no match
        return []

    @staticmethod
    async def get_multiple_document_chunks(index: str, document_uuids: List[str], max_size: int = 1000) -> List[Dict]:
        """
        Gets all document chunks from multiple documents in the specified index.

        Args:
            index (str): The name of the OpenSearch index from which the documents will be retrieved.
            document_uuids (List[str]): List of document UUIDs to get chunks for.
            max_size (int): The maximum number of document chunks to fetch. Defaults to 1000.

        Returns:
            List[Dict]: All document chunks retrieved from the specified documents.
        """
        logger.info(
            "Attempting to retrieve max %s document chunks from index %s for documents %s",
            max_size,
            index,
            document_uuids,
        )
        request_body = {
            "query": {"bool": {"filter": [{"terms": {"document_uuid.keyword": document_uuids}}]}},
            "size": max_size,
        }

        action = AsyncOpenSearchClient.get().search(request_body, index=index)
        response = await LogsHandler.with_logging(Action.OPENSEARCH_SEARCH_DOCUMENT, action)
        hit_elements = response["hits"]["hits"]
        logger.info("Found %s document chunks for documents %s", len(hit_elements), document_uuids)
        return hit_elements
