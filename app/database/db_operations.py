import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, List, Mapping, Optional, Tuple, TypeVar

from sqlalchemy import Result, Row, delete, desc, insert, select, text, update
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import contains_eager

from app.database.models import (
    LLM,
    Chat,
    ChatDocumentMapping,
    Document,
    DocumentChunk,
    DocumentUserMapping,
    Feedback,
    FeedbackLabel,
    FeedbackScore,
    GovUkSearchQuery,
    GovUkSearchResult,
    LlmInternalResponse,
    Message,
    SearchIndex,
    Theme,
    UseCase,
    User,
    UserPrompt,
)
from app.database.table import DatabaseError, DatabaseExceptionErrorCode, Table
from app.logs import LogsHandler
from app.logs.logs_handler import Action
from app.themes_use_cases.schemas import (
    ThemeInput,
    UseCaseInputPost,
    UseCaseInputPut,
)
from app.user.schemas import UserCreationInput, UserCreationResponse, UserInput

T = TypeVar("T")

logger = logging.getLogger()


class DbOperations:
    @staticmethod
    async def get_user_documents(db_session: AsyncSession, user: User) -> List[Row]:
        """
        Retrieves  user documents name, uuid, created_at and expired_at fields, ordered alphabetically by document name,
         associated with the user provided

        Args:
            user (User): The user whose documents are to be retrieved.
            db_session(AsyncSession): The database connection session.

        Returns:
            List[Row]: A list of `Row` instances associated with the user.
        """
        user_docs_stmt = (
            select(
                Document.name,
                Document.uuid,
                DocumentUserMapping.expired_at,
                DocumentUserMapping.created_at,
                DocumentUserMapping.last_used,
            )
            .join(DocumentUserMapping, Document.id == DocumentUserMapping.document_id)
            .join(User, User.id == DocumentUserMapping.user_id)
            .filter(User.uuid == user.uuid)
            .filter(DocumentUserMapping.deleted_at.is_(None))
            .filter(Document.deleted_at.is_(None))
            .order_by(desc(DocumentUserMapping.last_used).nulls_last())
        )
        user_result = await LogsHandler.with_logging(
            Action.DB_RETRIEVE_USER_DOCUMENTS, db_session.execute(user_docs_stmt)
        )
        user_documents = user_result.fetchall()
        return user_documents

    @staticmethod
    async def get_central_documents(db_session: AsyncSession) -> List[Row]:
        """
        Retrieves central documents name, uuid, created_at fields,
        ordered alphabetically by document name.

        Returns:
            List[Row]: A list of `Row` instances marked as central.
        """
        central_stmt = (
            select(Document.name, Document.uuid, Document.created_at, Document.id)
            .filter(Document.is_central.is_(True))
            .filter(Document.deleted_at.is_(None))
            .order_by(Document.name)
        )
        central_result = await LogsHandler.with_logging(
            Action.DB_RETRIEVE_CENTRAL_DOCUMENTS, db_session.execute(central_stmt)
        )

        central_documents = central_result.fetchall()
        return central_documents

    @staticmethod
    async def get_document_by_uuid(db_session: AsyncSession, document_uuid: str) -> Optional[int]:
        """
        Retrieves the document by uuid value provided.

        Args:
            db_session(AsyncSession): The database connection session.
            document_uuid (str): The document uuid

        Returns:
            Optional[int]: The  id of the document if present in the database.
        """
        stmt = select(Document.id).filter(Document.uuid == document_uuid).filter(Document.deleted_at.is_(None))
        result = await LogsHandler.with_logging(Action.DB_RETRIEVE_DOCUMENT, db_session.execute(stmt))

        return result.scalar()

    @staticmethod
    async def mark_document_as_deleted(db_session: AsyncSession, document_id: int):
        """
        Marks a document and its associated chunks as deleted by setting the `deleted_at`
        timestamp to the current time.

        Args:
            db_session (AsyncSession): The database connection session.
            document_id (int): The id of the document to mark as deleted.

        Returns:
            None
        """
        update_chunks_stmt = (
            update(DocumentChunk).where(DocumentChunk.document_id == document_id).values(deleted_at=datetime.now())
        )

        update_document_stmt = update(Document).where(Document.id == document_id).values(deleted_at=datetime.now())

        await LogsHandler.with_logging(Action.DB_MARK_DOCUMENT_DELETED, db_session.execute(update_chunks_stmt))
        await LogsHandler.with_logging(Action.DB_MARK_DOCUMENT_DELETED, db_session.execute(update_document_stmt))

    @staticmethod
    async def mark_user_document_mapping_as_deleted(db_session: AsyncSession, document_id: int, user: User) -> Result:
        """
        Updates DocumentUserMapping records as deleted for the document id and user provided.

        Args:
            db_session(AsyncSession): The database connection session.
            document_id (int): The document id
            user (User) The user owning  the document mapping to mark as deleted

        Returns:
            Result: Update result of db operation
        """
        stmt = (
            update(DocumentUserMapping)
            .where(
                DocumentUserMapping.document_id == document_id,
                DocumentUserMapping.user_id == user.id,
                DocumentUserMapping.deleted_at.is_(None),
            )
            .values(deleted_at=datetime.now())
        )
        result = await LogsHandler.with_logging(Action.DB_MARK_USER_DOCUMENT_MAPPING_DELETED, db_session.execute(stmt))
        return result

    @staticmethod
    async def get_opensearch_ids_from_document_chunks(db_session: AsyncSession, document_id: int) -> List[str]:
        """
        Fetches opensearch _id values from document_chunk table where it's associcated with the document_id provided.

        Args:
            db_session(AsyncSession): The database connection session.
            document_id (int): The document id to use for filtering document_chunk table.

        Returns:
            List[str]: List of Opensearch _id values.
        """
        stmt = select(DocumentChunk.id_opensearch).filter(
            DocumentChunk.document_id == document_id, DocumentChunk.deleted_at.is_(None)
        )
        result = await LogsHandler.with_logging(Action.DB_RETRIEVE_OPENSEARCH_IDS, db_session.execute(stmt))
        id_opensearch = result.scalars().all()
        return id_opensearch

    @staticmethod
    async def get_chats_by_user(db_session: AsyncSession, user_id: int) -> List[Chat]:
        """
        Fetches Chats owned by the User with the given  ID, excluding deleted chats.

        Args:
            db_session(AsyncSession): The database connection session.
            user_id (int): The ID of the user owning chats.

        Returns:
            List[Chat]: List of user Chats that are not deleted.
        """
        stmt = (
            select(Chat)
            .where(Chat.user_id == user_id, Chat.deleted_at.is_(None))
            .order_by(desc(Chat.favourite), desc(Chat.updated_at))
        )
        chats_result = await LogsHandler.with_logging(Action.DB_RETRIEVE_CHATS, db_session.execute(stmt))
        chats = chats_result.scalars().all()
        return chats

    @staticmethod
    async def get_chat_with_messages(db_session: AsyncSession, chat_id: int, user_id: int) -> Chat:
        """
        Retrieve chat and messages for a specific chat identified by its UUID and user ID.
        Args:
           db_session (AsyncSession): The asynchronous database session used for querying.
           chat_id (int): The ID of the chat
           user_id (int): The ID of the user

        Returns:
           Chat: The chat with messages and documents referenced in the chat.
        """
        stmt = (
            select(Chat)
            .outerjoin(Message, Chat.id == Message.chat_id)
            .outerjoin(ChatDocumentMapping, Chat.id == ChatDocumentMapping.chat_id)
            .outerjoin(Document, Document.uuid == ChatDocumentMapping.document_uuid)
            .outerjoin(
                DocumentUserMapping,
                (DocumentUserMapping.document_id == Document.id) & (DocumentUserMapping.user_id == user_id),
            )
            .options(
                contains_eager(Chat.chat_document_mapping)
                .contains_eager(ChatDocumentMapping.document)
                .contains_eager(Document.user_mappings)
            )
            .options(contains_eager(Chat.messages))
            .where(Chat.id == chat_id, Chat.user_id == user_id, Chat.deleted_at.is_(None))
            .order_by(Message.created_at, Document.created_at)
        )

        result = await LogsHandler.with_logging(Action.DB_RETRIEVE_CHAT, db_session.execute(stmt))
        chat_result = result.scalars().unique().first()
        if not chat_result:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_ERROR,
                message="Chat not found or has been archived",
            )
        return chat_result

    @staticmethod
    async def fetch_undeleted_chat_documents(
        db_session: AsyncSession, user_id: int, chat_id: int
    ) -> Sequence[Document]:
        """ "
         Fetches undeleted documents that were used in the chat conversation.

        Args:
            db_session (AsyncSession): The SQLAlchemy async session for database interaction.
            chat_id (int): The ID of the chat
            user_id (int): The ID of the user


        Returns:
            Sequence[Document]: A sequence of Documents not marked as deleted.

        """
        stmt = (
            select(Document)
            .join(DocumentUserMapping, (DocumentUserMapping.document_id == Document.id))
            .join(ChatDocumentMapping, ChatDocumentMapping.document_uuid == Document.uuid)
            .where(
                DocumentUserMapping.deleted_at.is_(None),
                DocumentUserMapping.user_id == user_id,
                ChatDocumentMapping.chat_id == chat_id,
            )
        )
        _exec = await db_session.execute(stmt)
        return _exec.scalars().all()

    @staticmethod
    async def save_records(db_session: AsyncSession, model: T, values: List[Mapping[str, Any]]) -> List[T]:
        """
        Inserts multiple records into the database for a given model and returns the inserted records.

        Args:
            db_session (AsyncSession): The database session used for executing the insert statement.
            model (T): The SQLAlchemy ORM model class representing the table.
            values (List[Mapping[str, Any]]): A list of dictionaries containing
            the column-value mappings for the rows to insert.

        Returns:
            List[T]: A list of model instances corresponding to the inserted rows.
        """
        stmt = insert(model).values(values).returning(model)
        exc = await db_session.execute(stmt)
        return exc.scalars().all()

    @staticmethod
    async def update_document_expiry_and_last_used_time(
        db_session: AsyncSession, document_uuids: List[str], user_id: int
    ):
        """
        Updates the expired_at field of DocumentUserMapping record with the new expiry date provided, for the
        user and document uuids provided.

        Args:
            db_session(AsyncSession): The database connection session.
            document_uuids (List[str]): The document uuids to use for update condition on DocumentUserMapping
            user_id (int): The user id to use for update condition on DocumentUserMapping

        Returns:
            None
        """
        now = datetime.now()
        new_expiry_date = datetime.today() + timedelta(days=90)

        stmt = (
            update(DocumentUserMapping)
            .where(DocumentUserMapping.document_id == Document.id)
            .where(Document.uuid.in_(document_uuids))
            .where(DocumentUserMapping.user_id == user_id)
            .where(DocumentUserMapping.deleted_at.is_(None))
            .values(expired_at=new_expiry_date, last_used=now)
        )

        await db_session.execute(stmt)

    @staticmethod
    async def delete_expired_documents(db_session: AsyncSession) -> List[str]:
        """
        Updates the `DocumentUserMapping` table records that are expired, as deleted if they are not already marked
        as deleted. It checks expired_at column against current timestamp for determining if a row needs deletion mark.

        Updates the `Document` table to mark documents with the affected `document_id` values as deleted.

        Updates the `DocumentChunk` table to mark chunks with the affected `document_id` values as deleted.
        and returns `id_opensearch`  from DocumentChunk row affected

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.

        Returns:
            List[str]: A list of id_opensearch from DocumentChunk rows affected,
            otherwise empty list, if no expired records found.

        """
        current_date = datetime.now()
        doc_user_map_stmt = (
            update(DocumentUserMapping)
            .where(DocumentUserMapping.expired_at < current_date, DocumentUserMapping.deleted_at.is_(None))
            .values(deleted_at=current_date)
            .returning(DocumentUserMapping.document_id)
        )
        result = await LogsHandler.with_logging(
            Action.DB_MARK_USER_DOCUMENT_MAPPING_DELETED, db_session.execute(doc_user_map_stmt)
        )
        document_ids = result.scalars().all()

        if document_ids:
            doc_stmt = update(Document).where(Document.id.in_(document_ids)).values(deleted_at=current_date)
            await LogsHandler.with_logging(Action.DB_MARK_DOCUMENT_DELETED, db_session.execute(doc_stmt))

            doc_chunk_stmt = (
                update(DocumentChunk)
                .where(DocumentChunk.document_id.in_(document_ids))
                .values(deleted_at=current_date)
                .returning(DocumentChunk.id_opensearch)
            )
            result = await LogsHandler.with_logging(Action.DB_MARK_DOCUMENT_DELETED, db_session.execute(doc_chunk_stmt))
            id_opensearch_list = result.scalars().all()
            return id_opensearch_list

        return []

    @staticmethod
    async def upsert_user_by_uuid(db_session: AsyncSession, user_key_uuid: str) -> User | None:
        user_stmt = select(User).where(User.uuid == user_key_uuid)
        user_result = await LogsHandler.with_logging(Action.GET_USER_BY_UUID, db_session.execute(user_stmt))
        user = user_result.scalars().first()
        if user is not None:
            return user
        try:
            new_user_stmt = insert(User).values(uuid=user_key_uuid).returning(User)
            user = await LogsHandler.with_logging(Action.UPSERT_USER_BY_UUID, db_session.execute(new_user_stmt))
            return user.scalars().first()
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.UPSERT_BY_UUID_ERROR,
                message="An error occurred when using the `upsert_user_by_uuid` "
                + "method on the table {User.__tablename__}: "
                + f"Original error: {e}",
            ) from e

    async def theme_create_or_revive(db_session: AsyncSession, theme_input: ThemeInput) -> Theme:
        # Query for existing records that match the filter criteria
        stmt = select(Theme).where(
            Theme.title == theme_input.title,
            Theme.subtitle == theme_input.subtitle,
            Theme.position == theme_input.position,
        )

        result = await LogsHandler.with_logging(Action.DB_GET_THEME, db_session.execute(stmt))
        theme = result.scalars().first()

        if theme:
            # If an existing record is found, check if it is deleted
            if theme.deleted_at is not None:
                # Revive the existing record by setting deleted_at to None
                stmt = update(Theme).where(Theme.id == theme.id).values(deleted_at=None).returning(Theme)
                result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))
                theme = result.scalars().first()

        else:
            # If no existing record is found, create a new one
            stmt = (
                insert(Theme)
                .values(title=theme_input.title, subtitle=theme_input.subtitle, position=theme_input.position)
                .returning(Theme)
            )
            result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))
            theme = result.scalars().first()

        return theme

    @staticmethod
    async def get_theme(
        db_session: AsyncSession, theme_uuid: UUID, include_deleted_records: bool = False
    ) -> Optional[Theme]:
        if include_deleted_records:
            stmt = select(Theme).where(Theme.uuid == theme_uuid)
        else:
            stmt = select(Theme).where(Theme.uuid == theme_uuid, Theme.deleted_at.is_(None))

        result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))
        return result.scalars().first()

    @staticmethod
    async def get_message_by_uuid(db_session: AsyncSession, message_uuid: str) -> Optional[Message]:
        stmt = select(Message).where(Message.uuid == message_uuid)
        result = await LogsHandler.with_logging(Action.DB_GET_MESSAGE_BY_UUID, db_session.execute(stmt))
        message = result.scalars().first()
        return message

    @staticmethod
    async def get_message_feedback_labels_list(db_session: AsyncSession) -> List[Optional[FeedbackLabel]]:
        stmt = select(FeedbackLabel)
        result = await LogsHandler.with_logging(Action.DB_GET_MESSAGE_LABELS, db_session.execute(stmt))
        labels = result.scalars().all()
        return labels

    @staticmethod
    async def get_feedback(db_session: AsyncSession, message_id: int) -> Optional[Feedback]:
        stmt = select(Feedback).where(Feedback.message_id == message_id)
        result = await LogsHandler.with_logging(Action.DB_GET_FEEDBACK, db_session.execute(stmt))
        feedback = result.scalars().first()
        return feedback

    @staticmethod
    async def upsert_feedback(
        db_session: AsyncSession,
        message_id: int,
        feedback_score_id: int,
        freetext: str,
        feedback_label_id: Optional[int] = None,
    ) -> Feedback:
        stmt = select(Feedback).where(Feedback.message_id == message_id)
        result = await LogsHandler.with_logging(Action.DB_GET_FEEDBACK, db_session.execute(stmt))
        feedback = result.scalars().first()
        if feedback is None:
            stmt = (
                insert(Feedback)
                .values(
                    message_id=message_id,
                    feedback_score_id=feedback_score_id,
                    freetext=freetext,
                    feedback_label_id=feedback_label_id,
                )
                .returning(Feedback)
            )
            result = await LogsHandler.with_logging(Action.DB_CREATE_FEEDBACK, db_session.execute(stmt))
            feedback = result.scalars().first()
            return feedback
        stmt = (
            update(Feedback)
            .where(Feedback.id == feedback.id)
            .values(
                deleted_at=None,
                message_id=message_id,
                feedback_score_id=feedback_score_id,
                freetext=freetext,
                feedback_label_id=feedback_label_id,
            )
            .returning(Feedback)
        )
        result = await LogsHandler.with_logging(Action.DB_CREATE_FEEDBACK, db_session.execute(stmt))
        feedback = result.scalars().first()
        return feedback

    @staticmethod
    async def delete_feedback(db_session: AsyncSession, feedback: Feedback) -> Optional[Feedback]:
        stmt = update(Feedback).where(Feedback.id == feedback.id).values(deleted_at=datetime.now())
        _ = await LogsHandler.with_logging(Action.DB_UPDATE_FEEDBACK, db_session.execute(stmt))
        stmt = select(Feedback).where(Feedback.id == feedback.id)
        result = await LogsHandler.with_logging(Action.DB_GET_FEEDBACK, db_session.execute(stmt))
        return result.scalars().first()

    @staticmethod
    async def get_feedback_score_by_name(db_session: AsyncSession, score: str) -> Optional[FeedbackScore]:
        stmt = select(FeedbackScore).where(FeedbackScore.score == score)
        result = await LogsHandler.with_logging(Action.DB_GET_FEEDBACK_SCORE_BY_NAME, db_session.execute(stmt))

        return result.scalars().first()

    @staticmethod
    async def get_themes(db_session: AsyncSession) -> List[Theme]:
        stmt = (
            select(Theme)
            .filter(Theme.deleted_at.is_(None))
            .order_by(
                Theme.position.is_(None),
                Theme.position,
                Theme.id,
            )
        )

        result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))

        return result.scalars().all()

    @staticmethod
    async def get_feedback_label_by_name(db_session: AsyncSession, feedback_label: str) -> Optional[FeedbackLabel]:
        stmt = select(FeedbackLabel).where(FeedbackLabel.uuid == feedback_label)
        result = await LogsHandler.with_logging(Action.DB_GET_FEEDBACK_LABEL_BY_LABEL, db_session.execute(stmt))

        return result.scalars().first()

    @staticmethod
    async def get_existing_index(db_session: AsyncSession, name: str) -> SearchIndex:
        """
        Returns an existing SearchIndex with the given name.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            name (str): The name of the SearchIndex.

        Returns:
            SearchIndex: A SearchIndex with the given name.
        """
        stmt = select(SearchIndex).where(SearchIndex.name == name, SearchIndex.deleted_at.is_(None))
        result = await LogsHandler.with_logging(Action.DB_GET_EXISTING_INDEX, db_session.execute(stmt))
        existing_index = result.scalars().first()
        return existing_index

    @staticmethod
    async def get_index_by_name(db_session: AsyncSession, name: str) -> SearchIndex:
        """
        Returns an existing SearchIndex with the given name.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            name (str): The name attribute of the SearchIndex .

        Returns:
            SearchIndex: A SearchIndex with the given UUID.
        """
        stmt = select(SearchIndex).where(SearchIndex.name == name, SearchIndex.deleted_at.is_(None))
        result = await LogsHandler.with_logging(Action.DB_GET_INDEX_BY_NAME, db_session.execute(stmt))
        existing_index = result.scalars().first()
        return existing_index

    @staticmethod
    async def get_index_by_uuid(db_session: AsyncSession, index_uuid: UUID) -> SearchIndex:
        """
        Returns an existing SearchIndex with the given UUID.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            index_uuid (UUID): The UUID of the SearchIndex.

        Returns:
            SearchIndex: A SearchIndex with the given UUID.
        """
        stmt = select(SearchIndex).where(SearchIndex.uuid == index_uuid, SearchIndex.deleted_at.is_(None))
        result = await LogsHandler.with_logging(Action.DB_GET_INDEX_BY_UUID, db_session.execute(stmt))
        existing_index = result.scalars().first()
        return existing_index

    @staticmethod
    async def create_index(db_session: AsyncSession, name: str, description: str) -> SearchIndex:
        """
        Creates and returns a new SearchIndex with the given name and description.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            name (str): The name of new SearchIndex.
            description (str): The description of new SearchIndex.

        Returns:
            SearchIndex: A new SearchIndex.
        """
        stmt = insert(SearchIndex).values(name=name, description=description).returning(SearchIndex)
        existing_index = await LogsHandler.with_logging(Action.DB_CREATE_INDEX, db_session.execute(stmt))
        return existing_index

    @staticmethod
    async def get_active_indexes(
        db_session: AsyncSession,
        personal_documents_index_name: str,
        include_personal_document_index: bool = False,
    ) -> List[SearchIndex]:
        """
        Returns a list of acive SearchIndexes with the given name.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            personal_documents_index_name (str): The name of the SearchIndex.
            include_personal_document_index (bool): The UUID of the SearchIndex.

        Returns:
            List[SearchIndex]: A list of SearchIndex with the given UUID.
        """
        if include_personal_document_index:
            stmt = select(SearchIndex).filter(
                SearchIndex.deleted_at.is_(None), SearchIndex.name == personal_documents_index_name
            )
        else:
            stmt = select(SearchIndex).filter(
                SearchIndex.deleted_at.is_(None), SearchIndex.name != personal_documents_index_name
            )

        result = await LogsHandler.with_logging(Action.DB_GET_ACTIVE_INDEXES, db_session.execute(stmt))

        return result.scalars().all()

    @staticmethod
    async def theme_update(db_session: AsyncSession, theme_uuid: UUID, theme_input: ThemeInput) -> Optional[Theme]:
        stmt = (
            update(Theme)
            .where(Theme.uuid == theme_uuid)
            .values(title=theme_input.title, subtitle=theme_input.subtitle, position=theme_input.position)
            .returning(Theme)
        )
        result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))
        return result.scalars().first()

    @staticmethod
    async def theme_soft_delete_by_uuid(db_session: AsyncSession, theme_uuid: UUID) -> Optional[Theme]:
        stmt = update(Theme).where(Theme.uuid == theme_uuid).values(deleted_at=datetime.now())
        _ = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))

    @staticmethod
    async def use_case_create_or_revive(
        db_session: AsyncSession, theme_uuid: UUID, use_case_input: UseCaseInputPost
    ) -> UseCase:
        theme = await DbOperations.get_theme(db_session=db_session, theme_uuid=theme_uuid)
        # Query for existing records that match the filter criteria
        stmt = select(UseCase).where(
            UseCase.title == use_case_input.title,
            UseCase.instruction == use_case_input.instruction,
            UseCase.user_input_form == use_case_input.user_input_form,
            UseCase.theme_id == theme.id,
            UseCase.position == use_case_input.position,
        )

        result = await LogsHandler.with_logging(Action.DB_GET_THEME, db_session.execute(stmt))
        use_case = result.scalars().first()

        if use_case:
            # If an existing record is found, check if it is deleted
            if use_case.deleted_at is not None:
                # Revive the existing record by setting deleted_at to None
                # use_case.deleted_at = None
                # stmt = update(UseCase).where(UseCase.id == use_case.id).values(deleted_at=use_case.deleted_at)
                stmt = update(UseCase).where(UseCase.id == use_case.id).values(deleted_at=None).returning(UseCase)
                use_case = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))
                return use_case.scalars().first()
        else:
            # If no existing record is found, create a new one
            stmt = (
                insert(UseCase)
                .values(
                    title=use_case_input.title,
                    instruction=use_case_input.instruction,
                    user_input_form=use_case_input.user_input_form,
                    theme_id=theme.id,
                    position=use_case_input.position,
                )
                .returning(UseCase)
            )
            result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))
            use_case = result.scalars().first()

        return use_case

    @staticmethod
    async def get_use_case(
        db_session: AsyncSession, theme_uuid: UUID, use_case_uuid: UUID
    ) -> Optional[Tuple[UseCase, Theme]]:
        # Verify that the use case belongs to this theme UUID
        use_case_stmt = select(UseCase).where(UseCase.uuid == use_case_uuid, UseCase.deleted_at.is_(None))
        use_case_result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(use_case_stmt))
        use_case = use_case_result.scalars().first()

        theme_stmt = select(Theme).where(Theme.uuid == theme_uuid)
        theme_result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(theme_stmt))
        theme = theme_result.scalars().first()

        if theme and use_case:
            if use_case.theme_id != theme.id:
                raise DatabaseError(
                    code=DatabaseExceptionErrorCode.USE_CASE_NOT_UNDER_THIS_THEME_ERROR,
                    message=f"The use case UUID '{use_case_uuid}'"
                    + f"is not associated with the theme UUID '{theme_uuid}'",
                )

            # Proceed if condition is met
            theme_stmt = select(Theme).where(Theme.id == use_case.theme_id)
            theme_result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(theme_stmt))
            theme = theme_result.scalars().first()

        return theme, use_case

    @staticmethod
    async def use_case_get_by_uuid_no_theme(
        db_session: AsyncSession, use_case_uuid: UUID, include_deleted_records: bool = False
    ) -> Optional[UseCase]:
        # Verify that the use case belongs to this theme UUID
        if include_deleted_records:
            use_case_stmt = select(UseCase).where(UseCase.uuid == use_case_uuid)
        else:
            use_case_stmt = select(UseCase).where(UseCase.uuid == use_case_uuid, UseCase.deleted_at.is_(None))

        use_case_result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(use_case_stmt))
        use_case = use_case_result.scalars().first()
        return use_case

    @staticmethod
    async def theme_get_by_id(db_session: AsyncSession, theme_id: id) -> Optional[Theme]:
        stmt = select(Theme).where(Theme.id == theme_id)
        result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))
        theme = result.scalars().first()
        return theme

    @staticmethod
    async def use_case_update_by_uuid(
        db_session: AsyncSession, theme: Theme, use_case_uuid: UUID, use_case_input: UseCaseInputPut
    ) -> Optional[UseCase]:
        stmt = (
            update(UseCase)
            .where(UseCase.uuid == use_case_uuid)
            .values(
                theme_id=theme.id,
                title=use_case_input.title,
                instruction=use_case_input.instruction,
                user_input_form=use_case_input.user_input_form,
                position=use_case_input.position,
            )
        )
        _ = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))
        stmt = select(UseCase).where(UseCase.uuid == use_case_uuid)
        result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))
        return result.scalars().first()

    @staticmethod
    async def use_case_soft_delete_by_uuid(db_session: AsyncSession, use_case_uuid: UUID) -> Optional[bool]:
        stmt = update(UseCase).where(UseCase.uuid == use_case_uuid).values(deleted_at=datetime.now())
        _ = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))
        return

    @staticmethod
    async def use_case_get_by_theme(db_session: AsyncSession, theme_uuid: UUID) -> List[UseCase]:
        stmt = (
            select(UseCase)
            .filter(UseCase.theme_id == theme_uuid, UseCase.deleted_at.is_(None))
            .order_by(
                UseCase.position.is_(None),
                UseCase.position,
                UseCase.id,
            )
        )
        result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))
        return result.scalars().all()

    @staticmethod
    async def theme_fetch_all_ordered_by_position_or_id(db_session: AsyncSession) -> List[Theme]:
        stmt = (
            select(Theme)
            .filter(Theme.deleted_at.is_(None))
            .order_by(
                Theme.position.is_(None),
                Theme.position,
                Theme.id,
            )
        )
        result = await LogsHandler.with_logging(Action.DB_REVIVE_THEME, db_session.execute(stmt))
        return result.scalars().all()

    @staticmethod
    async def theme_delete_all(db_session: AsyncSession) -> List[bool]:
        stmt = update(Theme).where(Theme.deleted_at is not None).values(deleted_at=datetime.now())
        _ = await LogsHandler.with_logging(Action.DB_UPDATE_THEME, db_session.execute(stmt))
        # await db_session.commit()

    @staticmethod
    async def use_case_delete_all(db_session: AsyncSession) -> List[bool]:
        stmt = update(UseCase).where(UseCase.deleted_at is not None).values(deleted_at=datetime.now())
        _ = await LogsHandler.with_logging(Action.DB_UPDATE_USE_CASE, db_session.execute(stmt))
        # await db_session.commit()

    async def get_document_chunks(db_session: AsyncSession, search_index: SearchIndex) -> List[DocumentChunk]:
        """
        Returns a list of DocumentChunks belonging to the given search_index.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            search_index (SearchIndex): Parent SearchIndex.

        Returns:
            List[DocumentChunk]: A list of DocumentChunks belonging to the given search_index.
        """
        stmt = select(SearchIndex).where(SearchIndex.id == search_index.id, DocumentChunk.deleted_at.is_(None))
        result = await LogsHandler.with_logging(Action.DB_GET_DOCUMENT_CHUNKS, db_session.execute(stmt))
        document_chunks = result.scalars().all()
        return document_chunks

    @staticmethod
    async def get_existing_chunk(
        db_session: AsyncSession, search_index_id: int, document_id: int, name: str, content: str
    ) -> DocumentChunk:
        """
        Returns a DocumentChunk with the given name and content.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            search_index (SearchIndex): Parent SearchIndex.

        Returns:
            List[DocumentChunk]: A list of DocumentChunks belonging to the given search_index.
        """
        stmt = (
            select(DocumentChunk)
            .join(SearchIndex, DocumentChunk.search_index_id == SearchIndex.id)
            .where(
                DocumentChunk.document_id == document_id,
                SearchIndex.id == search_index_id,
                DocumentChunk.name == name,
                DocumentChunk.content == content,
                DocumentChunk.deleted_at.is_(None),
            )
        )
        result = await LogsHandler.with_logging(Action.DB_GET_EXISTING_CHUNK, db_session.execute(stmt))
        existing_chunk = result.scalars().first()
        return existing_chunk

    @staticmethod
    async def get_existing_document(db_session: AsyncSession, name: str, url: str) -> Document:
        """
        Returns a Document with the given name and url.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            name (str): Document name.
            url (str): Document url.

        Returns:
            Document: A Document with the given name and url.
        """
        stmt = select(Document).where(Document.name == name, Document.url == url)
        result = await LogsHandler.with_logging(Action.DB_GET_EXISTING_DOCUMENT, db_session.execute(stmt))
        document = result.scalars().first()
        return document

    @staticmethod
    async def create_central_document(db_session: AsyncSession, name: str, description: str, url: str) -> Document:
        """
        Creates a Document in the central RAG with the given name and url.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            name (str): Document name.
            description (str): Document description.
            url (str): Document url.

        Returns:
            Document: A Document with the given name, description and url.
        """
        stmt = insert(Document).values(name=name, description=description, url=url, is_central=True).returning(Document)
        document = await LogsHandler.with_logging(Action.DB_CREATE_DOCUMENT, db_session.execute(stmt))
        return document.scalars().first()

    @staticmethod
    async def add_chunk(
        db_session: AsyncSession,
        search_index_id: int,
        document_id: int,
        name: str,
        content: str,
        id_opensearch: str,
    ) -> DocumentChunk:
        """
        Adds a DocumentChunk for the given Document to the given SearchIndex.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            search_index_id (int): SearchIndex ID.
            document_id (int): Parent Document ID.
            name (str): Document name.
            content (str): DocumentChunk content.
            id_opensearch (int): OpenSearch Index ID.

        Returns:
            DocumentChunk: New DocumentChunk.
        """
        stmt = (
            insert(DocumentChunk)
            .values(
                search_index_id=search_index_id,
                document_id=document_id,
                name=name,
                content=content,
                id_opensearch=id_opensearch,
            )
            .returning(DocumentChunk)
        )
        document = await LogsHandler.with_logging(Action.DB_ADD_CHUNK, db_session.execute(stmt))
        return document

    @staticmethod
    async def get_document_chunks_filtered_with_search_index(
        db_session: AsyncSession,
        search_index_uuid: str,
        show_deleted_chunks: bool = False,
    ) -> List[DocumentChunk]:
        """
        Returns a list of DocumentChunks for the given SearchIndex ID.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            search_index_uuid (str): SearchIndex UUID.
            show_deleted_chunks (bool): Parent Document ID.

        Returns:
            List[DocumentChunk]: A list of DocumentChunks.
        """
        if show_deleted_chunks:
            stmt = select(DocumentChunk).join(SearchIndex).filter(SearchIndex.uuid == search_index_uuid)
        else:
            stmt = (
                select(DocumentChunk)
                .join(SearchIndex)
                .filter(SearchIndex.uuid == search_index_uuid, DocumentChunk.deleted_at.is_(None))
            )

        result = await LogsHandler.with_logging(
            Action.DB_GET_DOCUMENT_CHUNKS_FILTERED_WITH_SEARCH_INDEX, db_session.execute(stmt)
        )

        return result.scalars().all()

    @staticmethod
    async def get_document_chunk_by_uuid(db_session: AsyncSession, document_chunk_uuid: UUID) -> DocumentChunk:
        """
        Returns a DocumentChunk with the given UUID.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            document_chunk_uuid (UUID): DocumentChunk UUID.

        Returns:
            DocumentChunk: A DocumentChunks with the given UUID.
        """
        stmt = select(DocumentChunk).filter(DocumentChunk.uuid == document_chunk_uuid)
        result = await LogsHandler.with_logging(Action.DB_GET_DOCUMENT_CHUNKS_BY_UUID, db_session.execute(stmt))
        return result.scalars().first()

    @staticmethod
    async def list_documents(db_session: AsyncSession, show_deleted_documents: bool = False) -> List[Document]:
        """
        Returns a list of all Documents, optionally including deleted documents.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            show_deleted_documents (bool): Optionally include deleted documents.

        Returns:
            List[Document]: A list of all Documents.
        """
        if show_deleted_documents:
            stmt = select(Document)
        else:
            stmt = select(Document).filter(Document.deleted_at.is_(None))

        result = await LogsHandler.with_logging(Action.DB_LIST_DOCUMENTS, db_session.execute(stmt))
        return result.scalars().all()

    async def get_ping_database(db_session: AsyncSession):
        try:
            # Attempt to execute a simple SELECT statement
            stmt = text("SELECT 1")
            ping_result = await LogsHandler.with_logging(Action.DB_PING, db_session.execute(stmt))
            _ = ping_result.scalars().first()
            return {"message": "Successfully connected to the database."}
        except SQLAlchemyError as e:
            # If an error occurs, return an HTTP error response
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.GET_ERROR,
                message="An error occurred when attempting to connect to the database. " + f"Original error: {e}",
            ) from e

    @staticmethod
    async def get_all_use_cases(db_session: AsyncSession):
        stmt = select(UseCase)
        use_case_result = await LogsHandler.with_logging(Action.DB_FETCH_ALL_USE_CASES, db_session.execute(stmt))
        use_cases = use_case_result.scalars().fetch_all()

        return use_cases

    @staticmethod
    async def reset_themes_use_cases(db_session: AsyncSession, data: dict):
        for theme_name, use_cases in data.items():
            theme_stmt = insert(Theme).values(value=theme_name).returning(Theme)
            theme = await LogsHandler.with_logging(Action.DB_FETCH_ALL_USE_CASES, db_session.execute(theme_stmt))
            for use_case_data in use_cases:
                use_case_stmt = (
                    insert(UseCase).values(**use_case_data, system_prompt="", theme_id=theme.id).returning(UseCase)
                )
                use_case = await LogsHandler.with_logging(
                    Action.DB_FETCH_ALL_USE_CASES, db_session.execute(use_case_stmt)
                )

        return use_case.fetch_all()

    # lib/user.py db operations
    #
    @staticmethod
    async def create_user(db_session: AsyncSession, user_input: UserCreationInput) -> UserCreationResponse:
        """
        Creates a new user in the database if one doesn't already exist with the given UUID.

        Args:
            db_session (AsyncSession): The database session for executing queries
            user_input (UserCreationInput): The user data containing UUID and profile information

        Returns:
            UserCreationResponse: A response object containing:
                - success (bool): True if user was created, False if user already exists
                - message (str): Description of the operation result

        Raises:
            SQLAlchemyError: If there's an error during database operations
        """

        async def _create_user():
            # Check if user already exists
            existing_user = await db_session.execute(select(User).where(User.uuid == user_input.uuid))

            if existing_user.scalar_one_or_none():
                msg = f"User already exists with UUID: {user_input.uuid}"
                logger.info(msg)
                return UserCreationResponse(success=False, message=msg)

            new_user = User(
                uuid=user_input.uuid,
                job_title=user_input.job_title,
                region=user_input.region,
                sector=user_input.sector,
                organisation=user_input.organisation,
                grade=user_input.grade,
                communicator_role=user_input.communicator_role,
            )

            db_session.add(new_user)

            msg = f"Successfully created user with UUID: {user_input.uuid}"
            logger.info(msg)
            return UserCreationResponse(success=True, message=msg)

        return await LogsHandler.with_logging(Action.INSERT_USER_BY_UUID, _create_user())

    @staticmethod
    async def update_user(db_session: AsyncSession, user_uuid: UUID, user_input: UserInput) -> UserCreationResponse:
        """
        Updates an existing user in the database with the provided profile information.

        Args:
            db_session (AsyncSession): The database session for executing queries
            user_uuid (str): The UUID of the user to update
            user_input (UserInput): The updated user data containing profile information

        Returns:
            UserCreationResponse: A response object containing:
                - success (bool): True if user was updated, False if user doesn't exist
                - message (str): Description of the operation result

        Raises:
            SQLAlchemyError: If there's an error during database operations
        """

        async def _update_user():
            # Check if user exists
            existing_user = await db_session.execute(select(User).where(User.uuid == user_uuid))
            user = existing_user.scalar_one_or_none()

            if not user:
                msg = f"User not found with UUID: {user_uuid}"
                logger.info(msg)
                return UserCreationResponse(success=False, message=msg)

            # Update user fields
            user.job_title = user_input.job_title
            user.region = user_input.region
            user.sector = user_input.sector
            user.organisation = user_input.organisation
            user.grade = user_input.grade
            user.communicator_role = user_input.communicator_role

            msg = f"Successfully updated user with UUID: {user_uuid}"
            logger.info(msg)
            return UserCreationResponse(success=True, message=msg)

        return await LogsHandler.with_logging(Action.UPDATE_USER_BY_UUID, _update_user())

    # lib/user_prompt.py db operations
    #
    @staticmethod
    async def get_user_prompts(db_session: AsyncSession, user: User) -> List[Row]:
        """
        Retrieves UserPrompt(s) ordered alphabetically by date/time of creation,
        associated with the user provided

        Args:
            db_session(AsyncSession): The database connection session.
            user (User): The user whose documents are to be retrieved.

        Returns:
            List[Row]: A list of `Row` UserPrompt instances associated with the user.
        """
        stmt = (
            select(UserPrompt)
            .where(UserPrompt.user_id == user.id)
            .filter(UserPrompt.deleted_at.is_(None))
            .order_by(UserPrompt.created_at)
        )
        user_prompt_result = await LogsHandler.with_logging(Action.DB_RETRIEVE_USER_PROMPTS, db_session.execute(stmt))
        user_prompts = user_prompt_result.scalars().all()

        return user_prompts

    @staticmethod
    async def insert_single_user_prompt(db_session: AsyncSession, user_id: int, title: str, content: str) -> UserPrompt:
        """
        Creates a user prompt using title, user_id and content provided in the database and
        returns newly created user prompt
        Args:
            db_session (AsyncSession): The asynchronous database session used for querying.
            user_id (int): The ID of the user
            title (str): The title of the user prompt
            content (str): The content of the user prompt

        Returns:
            UserPrompt: The user prompt with body, title, and date of creation.
        """
        stmt = insert(UserPrompt).values(user_id=user_id, title=title, content=content).returning(UserPrompt)
        user_prompt_result = await LogsHandler.with_logging(Action.DB_CREATE_USER_PROMPT, db_session.execute(stmt))
        user_document = user_prompt_result.scalars().unique().one()

        return user_document

    @staticmethod
    async def get_single_user_prompt(db_session: AsyncSession, user_id: str, user_prompt_uuid: UUID) -> UserPrompt:
        """
        Retrieve specific user prompt identified by its UUID and user ID.
        Args
            db_session (AsyncSession): The asynchronous database session used for querying.
            user_id (str): The ID of the user
            user_prompt_uuid (UUID): The UUID of the user prompt
        Returns:
            UserPrompt: The user prompt with body, title, and date of creation.
        """
        stmt = (
            select(UserPrompt)
            .where(UserPrompt.uuid == user_prompt_uuid)
            .filter(UserPrompt.deleted_at.is_(None))
            .filter(UserPrompt.user_id == int(user_id))
            .order_by(UserPrompt.created_at)
        )
        user_prompt_result = await LogsHandler.with_logging(Action.DB_RETRIEVE_USER_PROMPT, db_session.execute(stmt))
        user_document = user_prompt_result.scalars().unique().first()
        return user_document

    @staticmethod
    async def update_single_user_prompt(
        db_session: AsyncSession, user_id: int, user_prompt_uuid: UUID, title: str, content: str
    ) -> UserPrompt:
        """
        Update specific user prompt identified by its id and user ID.

        Args:
            db_session (AsyncSession): The asynchronous database session used for querying.
            user_id (int): The ID of the user
            user_prompt_uuid (UUID): The UUID of the user prompt
            title (str): The title of the user prompt
            content (str): The content of the user prompt
        Returns:
            UserPrompt: The user prompt with body, title, and date of creation.
        """
        stmt = (
            update(UserPrompt)
            .values(user_id=user_id, uuid=user_prompt_uuid, title=title, content=content)
            .where(UserPrompt.user_id == user_id, UserPrompt.uuid == user_prompt_uuid)
            .returning(UserPrompt)
        )
        user_prompt_result = await LogsHandler.with_logging(Action.DB_UPDATE_USER_PROMPT, db_session.execute(stmt))
        user_document = user_prompt_result.scalars().unique().one()
        return user_document

    @staticmethod
    async def delete_single_user_prompt(db_session: AsyncSession, user_id: int, user_prompt_uuid: UUID) -> Result:
        """
        Delete specific user prompt identified by its id and user ID.
        Args:
            db_session (AsyncSession): The asynchronous database session used for querying.
            user_id (int): The ID of the user
            user_prompt_uuid (UUID): The ID of the user prompt

        Returns:
            Response
        """
        stmt = delete(UserPrompt).where(UserPrompt.user_id == user_id, UserPrompt.uuid == user_prompt_uuid)
        user_prompt_result = await LogsHandler.with_logging(Action.DB_DELETE_USER_PROMPT, db_session.execute(stmt))
        return user_prompt_result

    @staticmethod
    async def retrieve_analytics(
        db_session: AsyncSession, table: Table, start_date: Optional[datetime.date], end_date: Optional[datetime.date]
    ) -> Any:
        data = []
        if start_date and end_date:
            if start_date == end_date:
                stmt = select(table).where(table.created_at == start_date).order_by(desc(table.created_at))
            else:
                stmt = (
                    select(table).where(table.created_at.between(start_date, end_date)).order_by(desc(table.created_at))
                )
            data = await LogsHandler.with_logging(Action.DB_MARK_DOCUMENT_DELETED, db_session.execute(stmt))

        if data:
            data_dict = [msg.__dict__ for msg in data]
            return data_dict
        return []  # Return an empty list if 'data' is falsy

    @staticmethod
    async def chat_update_title(db_session: AsyncSession, chat: Chat, title: str) -> Chat:
        """
        Updates the chat title.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            chat (Chat): Chat object.

        Returns:
            Chat: A Chat object with an updated title,
        """

        try:
            chat_stmt = update(Chat).values(title=title).where(Chat.id == chat.id).returning(Chat)
            result = await LogsHandler.with_logging(Action.DB_UPDATE_CHAT_TITLE, db_session.execute(chat_stmt))
            return result.scalars().first()
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.UPDATE_ERROR,
                message=f"An error occurred when using the `update_chat` method on the table {Chat.__tablename__}: "
                + f"Original error: {e}",
            ) from e

    @staticmethod
    async def chat_update_favourite(db_session: AsyncSession, chat: Chat, favourite: bool) -> Chat:
        """
        Updates the chat favourite status.
        Toggling the favourite status does not update the chat's updated_at timestamp,
            hence we pass in the existing value.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            chat (Chat): Chat object.
            favourite (bool): New favourite status.

        Returns:
            Chat: A Chat object with updated favourite status.
        """
        try:
            chat_stmt = (
                update(Chat)
                .values(favourite=favourite, updated_at=chat.updated_at)
                .where(Chat.id == chat.id)
                .returning(Chat)
            )
            result = await LogsHandler.with_logging(Action.DB_UPDATE_CHAT_FAVOURITE, db_session.execute(chat_stmt))
            return result.scalars().first()
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.UPDATE_ERROR,
                message=f"An error occurred in the `chat_update_favourite` method on the table {Chat.__tablename__}:"
                + f"Original error: {e}",
            ) from e

    @staticmethod
    async def chat_archive(db_session: AsyncSession, chat: Chat) -> Chat:
        """
        Archives a chat by setting the deleted_at timestamp.

        Args:
            db_session (AsyncSession): The asynchronous SQLAlchemy session for executing database queries.
            chat (Chat): Chat object to be archived.

        Returns:
            Chat: A Chat object with updated deleted_at timestamp.
        """
        try:
            chat_stmt = update(Chat).values(deleted_at=datetime.now()).where(Chat.id == chat.id).returning(Chat)
            result = await LogsHandler.with_logging(Action.DB_ARCHIVE_CHAT, db_session.execute(chat_stmt))
            return result.scalars().first()
        except Exception as e:
            raise DatabaseError(
                code=DatabaseExceptionErrorCode.UPDATE_ERROR,
                message=f"An error occurred in the `chat_archive` method on the table {Chat.__tablename__}: "
                + f"Original error: {e}",
            ) from e

    @staticmethod
    async def insert_llm_internal_response_id_query(
        db_session: AsyncSession,
        web_browsing_llm: LLM,
        content: str,
        tokens_in: int,
        tokens_out: int,
        completion_cost: int,
    ) -> LlmInternalResponse:
        # Log the raw LLM response to the database to allow:
        # - tracking the cost of the query and the raw response
        # - debugging issues with the query
        stmt = (
            insert(LlmInternalResponse)
            .values(
                llm_id=web_browsing_llm.id,
                content=content,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                completion_cost=completion_cost,
            )
            .returning(LlmInternalResponse)
        )
        response = await db_session.execute(stmt)
        llm_internal_response = response.scalars().unique().one()

        return llm_internal_response

    @staticmethod
    async def insert_gov_uk_search_query(
        db_session: AsyncSession,
        llm_internal_response_id: int,
        message_id: int,
        query: str,
    ):
        stmt = (
            insert(GovUkSearchQuery)
            .values(llm_internal_response_id=llm_internal_response_id, message_id=message_id, content=query)
            .returning(GovUkSearchQuery)
        )
        response = await db_session.execute(stmt)
        gov_uk_search_query_result = response.scalars().unique().one()

        return gov_uk_search_query_result

    @staticmethod
    async def insert_gov_uk_search_result(
        db_session: AsyncSession,
        llm_internal_response_id: int,
        message_id: int,
        gov_uk_search_query_id: int,
        url: str,
        content: str,
        is_used: bool,
        position: int,
    ):
        stmt = (
            insert(GovUkSearchResult)
            .values(
                llm_internal_response_id=llm_internal_response_id,
                message_id=message_id,
                gov_uk_search_query_id=gov_uk_search_query_id,
                url=url,
                content=content,
                is_used=is_used,
                position=position,
            )
            .returning(GovUkSearchResult)
        )
        response = await db_session.execute(stmt)
        gov_uk_search_result = response.scalars().unique().one()

        return gov_uk_search_result
