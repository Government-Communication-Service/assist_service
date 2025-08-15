# ruff: noqa: A002
import logging
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from typing import Any, Generic, Optional, TypeVar

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from app.database.database_exception import (
    DatabaseError,
    DatabaseExceptionErrorCode,
)
from app.database.models import (
    LLM,
    AuthSession,
    Chat,
    Feedback,
    FeedbackLabel,
    FeedbackScore,
    LlmInternalResponse,
    Message,
    MessageSearchIndexMapping,
    MessageUserGroupMapping,
    Redaction,
    Theme,
    UseCase,
    User,
    UserGroup,
    UserPrompt,
)

from .database_url import async_database_url, database_url

logger = logging.getLogger(__name__)

engine = create_engine(database_url())

logger.debug(f"table.py using database URL {engine.url}")


class AsyncEngineProvider:
    """
     Singleton provider for an asynchronous SQLAlchemy engine.

    This class manages the lifecycle of a single SQLAlchemy asynchronous engine
    instance, creating it only once on the first request, and reusing it on
    subsequent calls, without re-initializing it.
    """

    __engine = None

    @classmethod
    def get(cls) -> AsyncEngine:
        """
        Retrieve the singleton asynchronous SQLAlchemy engine instance.

        If the engine instance does not exist, this method creates it using
        the configured database URL, otherwise it returns the existing instance.

        Returns:
            AsyncEngine: The singleton instance of the async SQLAlchemy engine.
        """
        if AsyncEngineProvider.__engine is None:
            # pool_size=0 allow app to create connections as required.
            AsyncEngineProvider.__engine = create_async_engine(async_database_url(), pool_size=0)
        return AsyncEngineProvider.__engine


SessionLocal = scoped_session(sessionmaker(bind=engine))
AsyncSessionMaker = sessionmaker(bind=AsyncEngineProvider.get(), class_=AsyncSession, expire_on_commit=False)

T = TypeVar("T")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def async_db_session() -> AsyncSession:
    """
    This function yields a new asynchronous SQLAlchemy session.
    It should typically be called once per request and passed
    to functions requiring database access. Each invocation will yield a new
    session, and any unhandled exceptions will trigger a rollback to maintain
    data integrity.

    Yields:
        AsyncSession: An active SQLAlchemy asynchronous session object.

    Exceptions:
        Any unhandled exception within the context will result in a rollback
        of any uncommitted transactions.

    """
    async with AsyncSessionMaker() as session:
        logger.debug("Starting DB session %s", session.sync_session.hash_key)
        try:
            async with session.begin():
                try:
                    yield session
                except Exception as e:
                    await session.rollback()  # Rollback on error
                    raise e
        finally:
            logger.debug("Closing DB session %s", session.sync_session.hash_key)
            await session.close()


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
    except:
        session.rollback()
        raise
    finally:
        session.close()


class Table(Generic[T]):
    def __init__(self, model: type[T], table_name: Optional[str] = ""):
        self.model = model
        self.table_name = table_name

    def query(self):
        try:
            with get_session() as session:
                return session.query(self.model).all()
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_ERROR,
                message=f"An error occurred when querying the table {self.table_name}: " + f"Original error: {e}",
            ) from e

    def create(self, data: Any) -> T:
        try:
            with get_session() as session:
                obj = self.model(**data)
                session.add(obj)
                session.commit()
                session.flush()
                session.refresh(obj)
                return obj
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.CREATE_ERROR,
                message=f"An error occurred when creating a record in table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def create_or_revive(self, data: Any) -> T:
        """This method creates a new record, but only if there is no suitable record (either deleted or not).
        It determines this by checking if all of the data provided matches a record in the database.
        If all values match, then the old record is revived, rather than creating a new record.

        This is beneficial for themes and use cases where new records with the same information are not necessary.
        """
        try:
            with get_session() as session:
                # Use all entries in the data parameter to build the filter dictionary
                data_to_filter_on = dict(data.items())

                # Query for existing records that match the filter criteria
                obj = session.query(self.model).filter_by(**data_to_filter_on).first()

                if obj:
                    # If an existing record is found, check if it is deleted
                    if obj.deleted_at is not None:
                        # Revive the existing record by setting deleted_at to None
                        obj.deleted_at = None
                        session.add(obj)
                        session.commit()
                        session.refresh(obj)
                else:
                    # If no existing record is found, create a new one
                    obj = self.model(**data)
                    session.add(obj)
                    session.commit()
                    session.refresh(obj)

                return obj
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.CREATE_OR_REVIVE_ERROR,
                message=f"An error occurred when using the `create_or_revive` method on the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def create_model(self, obj: Any) -> T:
        try:
            with get_session() as session:
                session.add(obj)
                session.commit()
                session.flush()
                session.refresh(obj)
                return obj
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.CREATE_ERROR,
                message=f"An error occurred when creating a model in table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def update(self, obj: Any, update_data: dict) -> T:
        try:
            with get_session() as session:
                db_obj = session.merge(obj)
                for key, value in update_data.items():
                    setattr(db_obj, key, value)

                session.commit()
                session.refresh(db_obj)
                return db_obj
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.UPDATE_ERROR,
                message="An error occurred when updating a database record using the `update` method"
                + f" on the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def update_by_uuid(self, uuid: uuid.UUID, data: dict) -> T:
        try:
            with get_session() as session:
                db_obj = (
                    session.query(self.model).filter(self.model.uuid == uuid, self.model.deleted_at.is_(None)).first()
                )
                if db_obj:
                    for key, value in data.items():
                        setattr(db_obj, key, value)
                    session.commit()
                    session.refresh(db_obj)
                    return db_obj
                raise DatabaseError(
                    code=DatabaseExceptionErrorCode.UPDATE_BY_UUID_ERROR,
                    message=f"An error occurred when updating the {self.table_name} table using {uuid=}:"
                    + " no record was found with this UUID.",
                )
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.UPDATE_BY_UUID_ERROR,
                message=f"An error occurred when updating the {self.table_name} table using {uuid=}: "
                + f"Original error: {e}",
            ) from e

    def get(self, record_id: int) -> T | None:
        try:
            with get_session() as session:
                db_obj = (
                    session.query(self.model)
                    .filter(self.model.id == record_id, self.model.deleted_at.is_(None))
                    .first()
                )
                if db_obj:
                    return db_obj
                raise DatabaseError(
                    code=DatabaseExceptionErrorCode.GET_ERROR,
                    message=f"An error occurred when attempting to get a record from the {self.table_name} "
                    + f"table by its internal ID: {record_id=}",
                )
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_ERROR,
                message=f"An error occurred when attempting to get a record from the {self.table_name} table by its "
                + f"internal ID: {record_id=}: "
                + f"Original error: {e}",
            ) from e

    def delete(self, record_id: int) -> None:
        try:
            with get_session() as session:
                obj = session.query(self.model).filter(self.model.deleted_at.is_(None)).get(record_id)
                if obj:
                    obj.deleted_at = datetime.now(UTC)
                    session.commit()
                else:
                    raise DatabaseError(
                        code=DatabaseExceptionErrorCode.DELETE_ERROR,
                        message="An error occurred when attempting to delete a record from the database. "
                        + f"Could not find a record in the table {self.table_name} with the provided "
                        + f"internal ID: {record_id=}",
                    )
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.DELETE_ERROR,
                message=f"An error occurred when attempting to delete a record from the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def soft_delete_by_uuid(self, uuid: uuid.UUID) -> None:
        try:
            with get_session() as session:
                obj = session.query(self.model).filter(self.model.uuid == uuid).first()
                if obj:
                    obj.deleted_at = datetime.now(UTC)
                    session.commit()
                else:
                    raise DatabaseError(
                        code=DatabaseExceptionErrorCode.SOFT_DELETE_BY_UUID_ERROR,
                        message="An error occurred when attempting to soft delete a record from the table "
                        + f"{self.table_name} using {uuid=}: no record was found with this UUID.",
                    )
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.SOFT_DELETE_BY_UUID_ERROR,
                message="An error occurred when soft deleting a database record using the `soft_delete_by_uuid` "
                + f"method on the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def create_batch(self, data_list: list[Any]) -> list[T]:
        try:
            with get_session() as session:
                instances = [self.model(**data) for data in data_list]
                session.add_all(instances)
                session.commit()
                return instances
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.CREATE_BATCH_ERROR,
                message=f"An error occurred when creating a batch of records in table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def get_one_by(self, key: str, value: Any) -> T:
        try:
            with get_session() as session:
                db_obj = (
                    session.query(self.model)
                    .filter(getattr(self.model, key) == value)
                    .filter(self.model.deleted_at.is_(None))
                    .first()
                )
                if db_obj:
                    return db_obj
                raise DatabaseError(
                    code=DatabaseExceptionErrorCode.GET_ONE_BY_ERROR,
                    message=f"An error occurred when using the `get_one_by` method on the table {self.table_name}:"
                    + f" no record found with {key=} {value=}",
                )
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_ONE_BY_ERROR,
                message=f"An error occurred when using the `get_one_by` method on the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def get_by(self, key: str, value: Any) -> list[T]:
        try:
            with get_session() as session:
                db_obj = (
                    session.query(self.model)
                    .filter(getattr(self.model, key) == value)
                    .filter(self.model.deleted_at.is_(None))
                    .all()
                )
                if db_obj:
                    return db_obj
                raise DatabaseError(
                    code=DatabaseExceptionErrorCode.GET_BY_ERROR,
                    message="An error occurred when attempting to retrieve a value from the database using a"
                    + f" key-value pair. The error occurred when querying the {self.table_name} table with the "
                    + f"key-value pair: {key=} {value=}",
                )
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_BY_ERROR,
                message="An error occurred when attempting to retrieve a value from the database using a key-value "
                + f"pair on the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def get_by_uuid(self, uuid: uuid.UUID, include_deleted_records: bool = True) -> T | None:
        try:
            with get_session() as session:
                if include_deleted_records:
                    db_obj = session.query(self.model).filter(self.model.uuid == uuid).first()
                else:
                    db_obj = (
                        session.query(self.model)
                        .filter(self.model.deleted_at.is_(None), self.model.uuid == uuid)
                        .first()
                    )

                if db_obj:
                    return db_obj
                raise DatabaseError(
                    code=DatabaseExceptionErrorCode.GET_BY_UUID_ERROR,
                    message=f"An error occurred when retrieving a record from the {self.table_name} table"
                    + f" using {uuid=}: no record was found with this UUID.",
                )
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_BY_UUID_ERROR,
                message=f"An error occurred when retrieving a record from the {self.table_name} table using {uuid=}: "
                + f"Original error: {e}",
            ) from e

    def most_recent(self) -> T | None:
        try:
            with get_session() as session:
                return session.query(self.model).order_by(self.model.created_at.desc()).first()
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.MOST_RECENT_ERROR,
                message=f"An error occurred when retrieving the most recent record from the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def upsert_by_uuid(self, record_uuid: uuid.UUID, data: Any | None = None) -> T | None:
        if data is None:
            data = {}
        try:
            with get_session() as session:
                obj = session.query(self.model).filter(self.model.uuid == record_uuid).first()
                if not obj:
                    obj = self.create({"uuid": record_uuid, **data})
                return obj
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.UPSERT_BY_UUID_ERROR,
                message=f"An error occurred when using the `upsert_by_uuid` method on the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def fetch_all(self, most_recent=True) -> list[T]:
        try:
            with get_session() as session:
                items = session.query(self.model).filter(self.model.deleted_at.is_(None))
                if most_recent:
                    items = items.order_by(self.model.updated_at.desc())
                return items.all()
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.FETCH_ALL_ERROR,
                message=f"An error occurred when fetching all records from the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    # This method can only be used if the database model has a 'position' field
    # Records with explicit positions come first, in ascending order of position.
    # Records sharing the same position are then ordered by their 'id'.
    # Records with NULL positions come last, ordered by their 'id'.
    # Example: Here is a list of tuples with (id, position):
    # [(1, 1) , (2, 4), (3, null), (4, 2), (5, 9), (6, null), (7, 1)]
    # The returned order should be:
    # [(1, 1) , (7, 1), (4, 2), (2, 4), (5, 9), (3, null), (6, null)]
    def fetch_all_ordered_by_position_or_id(self) -> list[T]:
        try:
            with get_session() as session:
                items = (
                    session.query(self.model)
                    .filter(self.model.deleted_at.is_(None))
                    .order_by(
                        self.model.position.is_(None),
                        self.model.position,
                        self.model.id,
                    )
                    .all()
                )
            return items

        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.FETCH_ALL_ORDERED_BY_POSITION_OR_ID_ERROR,
                message="An error occurred when using the `fetch_all_ordered_by_position_or_id` method on the table"
                + f" {self.table_name}: Original error: {e}",
            ) from e

    def order_by_most_recent(self, items: list[T]) -> list[T]:
        try:
            return items.order_by(self.model.updated_at.desc()).all()
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.ORDER_BY_MOST_RECENT_ERROR,
                message="An error occurred when using the `order_by_most_recent` method on the table"
                + f"{self.table_name}: Original error: {e}",
            ) from e

    def delete_all(self) -> None:
        try:
            with get_session() as session:
                session.query(self.model).filter(self.model.deleted_at.is_(None)).update(
                    {self.model.deleted_at: datetime.now(UTC)}
                )
                session.commit()
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.DELETE_ALL_ERROR,
                message=f"An error occurred when deleting all records from the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def edit_all(self, update_data: dict) -> None:
        try:
            with get_session() as session:
                objects = session.query(self.model).all()
                for obj in objects:
                    for key, value in update_data.items():
                        setattr(obj, key, value)
                session.commit()
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.EDIT_ALL_ERROR,
                message=f"An error occurred when editing all records in the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def upsert(self, data_key_name: str, key_value: Any, data: Optional[dict] = None) -> T:
        if data is None:
            data = {}
        try:
            with get_session() as session:
                obj = session.query(self.model).filter(getattr(self.model, data_key_name) == key_value).first()
                if obj:
                    for key, value in data.items():
                        setattr(obj, key, value)
                    session.commit()
                else:
                    data[data_key_name] = key_value
                    obj = self.model(**data)
                    session.add(obj)
                    session.commit()
                    session.flush()
                    session.refresh(obj)
                return obj
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.UPSERT_ERROR,
                message=f"An error occurred when using the `upsert` method on the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e


class UserTable(Table):
    def __init__(self):
        super().__init__(model=User, table_name="User")


class ChatTable(Table):
    def __init__(self):
        super().__init__(model=Chat, table_name="Chat")

    def get_by_user(self, user_id: str):
        try:
            with get_session() as session:
                chats = session.query(Chat).filter_by(user_id=user_id)
                chats = self.order_by_most_recent(chats)
                return chats
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_ERROR,
                message=f"An error occurred when getting chats by user from the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e


class MessageTable(Table):
    def __init__(self):
        super().__init__(model=Message, table_name="Message")

    def get_by_chat(self, chat_id: str) -> list[Message]:
        try:
            with get_session() as session:
                messages = session.query(Message).filter_by(chat_id=chat_id)
                messages = messages.order_by(self.model.created_at.asc()).all()
                return messages
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_ERROR,
                message=f"An error occurred when getting messages by chat from the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e


class AuthSessionTable(Table):
    def __init__(self):
        super().__init__(model=AuthSession, table_name="AuthSession")


class FeedbackTable(Table):
    def __init__(self):
        super().__init__(model=Feedback, table_name="Feedback")

    def get_by_message_id(self, message_id: str):
        try:
            feedback = self.upsert("message_id", message_id)
            return feedback
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_ERROR,
                message=f"An error occurred when getting feedback by message ID from the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e


class ThemeTable(Table):
    def __init__(self):
        super().__init__(model=Theme, table_name="Theme")


class UseCaseTable(Table):
    def __init__(self):
        super().__init__(model=UseCase, table_name="UseCase")

    # Explanation of ordering:
    # Records with explicit positions come first, in ascending order of position.
    # Records sharing the same position are then ordered by their 'id'.
    # Records with NULL positions come last, ordered by their 'id'.
    # Example: Here is a list of tuples with (id, position):
    # [(1, 1) , (2, 4), (3, null), (4, 2), (5, 9), (6, null), (7, 1)]
    # The returned order should be:
    # [(1, 1) , (7, 1), (4, 2), (2, 4), (5, 9), (3, null), (6, null)]
    def get_by_theme(self, theme_id: str):
        try:
            with get_session() as session:
                messages = (
                    session.query(UseCase)
                    .filter(UseCase.theme_id == theme_id, UseCase.deleted_at.is_(None))
                    .order_by(
                        self.model.position.is_(None),
                        self.model.position,
                        self.model.id,
                    )
                    .all()
                )
                return messages
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_ERROR,
                message=f"An error occurred when getting use cases by theme from the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e


class UserGroupTable(Table):
    def __init__(self):
        super().__init__(model=UserGroup, table_name="UserGroup")

    def get_by_name(self, name: str) -> T | None:
        try:
            with get_session() as session:
                return session.query(self.model).filter(self.model.group == name).first()
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_ONE_BY_ERROR,
                message=f"An error occurred when getting user group by name from the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def upsert_by_name(self, name: str):
        try:
            with get_session() as session:
                obj = session.query(self.model).filter(self.model.group == name).first()
                if not obj:
                    obj = self.create({"group": name})
                return obj
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.UPSERT_ERROR,
                message=f"An error occurred when upserting user group by name in the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e


class MessageUserGroupMappingTable(Table):
    def __init__(self):
        super().__init__(model=MessageUserGroupMapping, table_name="MessageUserGroupMapping")


class LLMTable(Table):
    def __init__(self):
        super().__init__(model=LLM, table_name="LLM")

    def get_by_model(self, model: str) -> T | None:
        try:
            return self.get_one_by("model", model)
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_ONE_BY_ERROR,
                message=f"An error occurred when getting LLM by model from the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e


class RedactionTable(Table):
    def __init__(self):
        super().__init__(model=Redaction, table_name="Redaction")


class FeedbackScoreTable(Table):
    def __init__(self):
        super().__init__(model=FeedbackScore, table_name="FeedbackScore")


class FeedbackLabelTable(Table):
    def __init__(self):
        super().__init__(model=FeedbackLabel, table_name="FeedbackLabel")


class LLMInternalResponseTable(Table):
    def __init__(self):
        super().__init__(model=LlmInternalResponse, table_name="LlmInternalResponse")


class MessageSearchIndexMappingTable(Table):
    def __init__(self):
        super().__init__(model=MessageSearchIndexMapping, table_name="MessageSearchIndexMapping")


class UserPromptTable(Table):
    def __init__(self):
        super().__init__(model=UserPrompt, table_name="UserPrompt")

    def get_by_user(self, user_id: int):
        try:
            with get_session() as session:
                user_prompts = session.query(UserPrompt).filter_by(user_id=user_id, deleted_at=None)
                user_prompts = self.order_by_most_recent(user_prompts)
                return user_prompts
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_ERROR,
                message=f"An error occurred when getting user prompts by user from the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e

    def delete(self, record_id: int) -> None:
        try:
            with get_session() as session:
                obj = session.query(self.model).filter_by(id=record_id, deleted_at=None).first()
                if obj:
                    obj.deleted_at = datetime.now(UTC)
                    session.commit()
                else:
                    raise DatabaseError(
                        code=DatabaseExceptionErrorCode.DELETE_ERROR,
                        message="An error occurred when attempting to delete a record from the database. "
                        + f"Could not find a record in the table {self.table_name} with the provided "
                        + f"internal ID: {record_id=}",
                    )
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.DELETE_ERROR,
                message=f"An error occurred when attempting to delete a record from the table {self.table_name}: "
                + f"Original error: {e}",
            ) from e
