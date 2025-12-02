import asyncio
import contextvars
import logging
from enum import Enum, auto
from typing import Awaitable, Callable, TypeVar, Union

T = TypeVar("T")

# configure session-id context
session_id_var = contextvars.ContextVar("session_id", default=None)


class SessionIdFilter(logging.Filter):
    def filter(self, record):
        # Add session_id from contextvar to the log record
        record.session_id = session_id_var.get()
        return True


# === Logging ===
logger = logging.getLogger()

for h in logger.handlers:
    h.addFilter(SessionIdFilter())


def filter_sensitive_headers(notification):
    sensitive_headers = ["auth-token"]
    request = notification.context.get("request")
    if request and "headers" in request:
        for header in sensitive_headers:
            if header in request["headers"]:
                del request["headers"][header]


class Action(str, Enum):
    DB_RETRIEVE_CENTRAL_DOCUMENTS = auto()
    DB_RETRIEVE_USER_DOCUMENTS = auto()
    OPENSEARCH_INDEX_DOCUMENT = auto()
    OPENSEARCH_DELETE_DOCUMENT = auto()
    OPENSEARCH_SEARCH_DOCUMENT = auto()
    DB_RETRIEVE_OPENSEARCH_IDS = auto()
    DB_MARK_USER_DOCUMENT_MAPPING_DELETED = auto()
    DB_MARK_DOCUMENT_DELETED = auto()
    DB_CHECK_USER_DOCUMENT_MAPPINGS = auto()
    DB_RETRIEVE_DOCUMENT = auto()
    DB_RETRIEVE_CHATS = auto()
    DB_RETRIEVE_CHAT = auto()
    UPSERT_USER_BY_UUID = auto()
    INSERT_USER_BY_UUID = auto()
    UPDATE_USER_BY_UUID = auto()
    GET_USER_BY_UUID = auto()
    CREATE_AUTH_SESSSION = auto()
    DB_GET_THEME = auto()
    DB_REVIVE_THEME = auto()
    DB_GET_MESSAGE_BY_UUID = auto()
    DB_GET_MESSAGE_LABELS = auto()
    DB_GET_MESSAGE_BY_ID = auto()
    DB_CREATE_MESSAGE = auto()
    DB_GET_FEEDBACK_SCORE_BY_NAME = auto()
    DB_GET_FEEDBACK_LABEL_BY_LABEL = auto()
    DB_CREATE_FEEDBACK = auto()
    DB_GET_FEEDBACK = auto()
    DB_UPDATE_CHAT_FAVOURITE = auto()
    DB_UPDATE_CHAT_SHARE = auto()
    DB_UPDATE_FEEDBACK = auto()
    DB_DELETE_FEEDBACK = auto()
    DB_GET_EXISTING_INDEX = auto()
    DB_GET_INDEX_BY_NAME = auto()
    DB_GET_INDEX_BY_UUID = auto()
    DB_CREATE_INDEX = auto()
    DB_GET_ACTIVE_INDEXES = auto()
    DB_GET_EXISTING_CHUNK = auto()
    DB_GET_DOCUMENT_CHUNKS = auto()
    DB_GET_EXISTING_DOCUMENT = auto()
    DB_CREATE_DOCUMENT = auto()
    DB_ADD_CHUNK = auto()
    DB_GET_DOCUMENT_CHUNKS_FILTERED_WITH_SEARCH_INDEX = auto()
    DB_GET_DOCUMENT_CHUNKS_BY_UUID = auto()
    DB_LIST_DOCUMENTS = auto()
    DB_PING = auto()
    DB_FETCH_ALL_USE_CASES = auto()
    DB_RETRIEVE_USER_PROMPTS = auto()
    DB_CREATE_USER_PROMPT = auto()
    DB_RETRIEVE_USER_PROMPT = auto()
    DB_UPDATE_USER_PROMPT = auto()
    DB_DELETE_USER_PROMPT = auto()
    DB_UPDATE_CHAT_TITLE = auto()
    DB_ARCHIVE_CHAT = auto()
    DB_UPDATE_MESSAGE = auto()
    DB_UPDATE_THEME = auto()
    DB_UPDATE_USE_CASE = auto()


class LogsHandler:
    logger = logger

    @staticmethod
    def error(error: Exception, task: str = "unamed task"):
        err_message = f"Error occurred during {task}: {error}"
        logger.exception(err_message)

    @staticmethod
    async def with_logging(
        action: Action,
        callback: Union[Callable[..., T], Callable[..., Awaitable[T]]],
    ) -> T:
        """
        Executes a given callable and logs if any error occurs. It supports both
        synchronous and asynchronous callbacks.

        Parameters:
            action (Action): The action being executed, used for logging context.
            callback (Union[Callable[..., T], Callable[..., Awaitable[T]]]):
                A callable representing the action to execute. Can be either
                synchronous or asynchronous.

        Returns:
            T: The result of the callback function.

        Raises:
            Exception: Any exception raised during the callback execution,
            re-raised after logging.
        """
        try:
            logger.debug("Executing %s", action)
            if asyncio.iscoroutine(callback):
                return await callback

            return callback()
        except Exception:
            logger.exception("Got error executing action %s", action)
            raise
