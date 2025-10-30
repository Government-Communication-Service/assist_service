from enum import Enum
from typing import Type

from sqlalchemy import (
    DECIMAL,
    JSON,
    VARCHAR,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import text

from .database_url import database_url


class CommonMixin:
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(UUID(as_uuid=True), server_default=text("uuid_generate_v4()"))
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(
        DateTime,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("CURRENT_TIMESTAMP"),
    )
    deleted_at = Column(DateTime, nullable=True)

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    def dict(self):
        return self.__dict__

    def client_response(self, extra=None):
        if extra is None:
            extra = {}
        data = {
            "uuid": str(self.uuid),
            "title": self.title if hasattr(self, "title") else None,
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
            **extra,
        }

        data = {k: v for k, v in data.items() if v is not None}

        return data


Base = declarative_base(cls=CommonMixin)


class User(Base):
    __tablename__ = "user"
    job_title = Column(String(255), nullable=True)
    region = Column(String(255), nullable=True)
    sector = Column(String(255), nullable=True)
    organisation = Column(String(255), nullable=True)
    grade = Column(String(255), nullable=True)
    communicator_role = Column(Boolean, nullable=True)

    def client_response(self):
        return {
            **super().client_response(),
            "job_title": self.job_title,
            "region": self.region,
            "sector": self.sector,
            "organisation": self.organisation,
            "grade": self.grade,
            "communicator_role": self.communicator_role,
        }


class UserGroup(Base):
    __tablename__ = "user_group"
    group = Column(String(255), nullable=False)

    def client_response(self):
        return {
            **super().client_response(),
            "group": self.group,
        }


class LLM(Base):
    __tablename__ = "llm"
    model = Column(String(255), nullable=False)
    provider = Column(String(255), nullable=False)
    input_cost_per_token = Column(Float, nullable=True)
    output_cost_per_token = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)

    def client_response(self):
        return {
            **super().client_response(),
            "model": self.model,
            "provider": self.provider,
        }


class AuthSession(Base):
    __tablename__ = "auth_session"
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    job_title = Column(String(255), nullable=True)
    region = Column(String(255), nullable=True)
    sector = Column(String(255), nullable=True)
    organisation = Column(String(255), nullable=True)
    grade = Column(String(255), nullable=True)
    communicator_role = Column(Boolean, nullable=True)

    def client_response(self):
        return {
            **super().client_response(),
            "user_id": self.user_id,
            "job_title": self.job_title,
            "region": self.region,
            "sector": self.sector,
            "organisation": self.organisation,
            "grade": self.grade,
            "communicator_role": self.communicator_role,
        }


class Chat(Base):
    __tablename__ = "chat"
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    use_case_id = Column(Integer, ForeignKey("use_case.id"))
    title = Column(String(255), nullable=False, default="Default title")
    from_open_chat = Column(Boolean, nullable=False)
    use_rag = Column(Boolean, nullable=True, server_default="true")
    use_gov_uk_search_api = Column(Boolean, nullable=True, server_default="false")
    use_smart_targets = Column(Boolean, nullable=True, server_default="false")
    favourite = Column(Boolean, nullable=False, server_default="false")

    def client_response(self):
        return {
            **super().client_response(
                {
                    "from_open_chat": self.from_open_chat,
                    "title": self.title,
                    "use_rag": self.use_rag,
                    "use_gov_uk_search_api": self.use_gov_uk_search_api,
                    "use_smart_targets": self.use_smart_targets,
                    "favourite": self.favourite,
                }
            ),
        }

    chat_document_mapping = relationship("ChatDocumentMapping", back_populates="chat")
    messages = relationship("Message", back_populates="chat")


class Message(Base):
    __tablename__ = "message"
    chat_id = Column(Integer, ForeignKey("chat.id"), nullable=False)
    content = Column(Text, nullable=False)
    content_enhanced_with_rag = Column(Text, nullable=True)
    citation = Column(Text, nullable=True)
    sources = Column(Text, nullable=True)
    role = Column(Text, nullable=False)
    tokens = Column(Integer, nullable=False)
    auth_session_id = Column(Integer, ForeignKey("auth_session.id"), nullable=False)
    parent_message_id = Column(Integer, ForeignKey("message.id"), nullable=True)
    redaction_id = Column(Integer, ForeignKey("redaction.id"))
    interrupted = Column(Boolean, nullable=False)
    completion_cost = Column(Numeric, nullable=True)
    llm_id = Column(Integer, ForeignKey("llm.id"), nullable=True)
    summary = Column(Text, nullable=True)
    summary_llm_response_id = Column(Integer, ForeignKey("llm_internal_response.id"), nullable=True)
    chat = relationship("Chat", back_populates="messages")

    def client_response(self):
        return super().client_response(
            {
                "content": self.content,
                "role": self.role,
                "redaction_id": self.redaction_id,
                "interrupted": self.interrupted,
                "citation": self.citation,
            },
        )


class Redaction(Base):
    __tablename__ = "redaction"
    redacted = Column(Boolean, nullable=False)
    alert_level = Column(Integer, nullable=False)
    alert_message = Column(Text, nullable=False)
    redaction_reason = Column(String(255), nullable=False)

    def client_response(self):
        return {
            **super().client_response(),
            "redacted": self.redacted,
            "alert_level": self.alert_level,
            "alert_message": self.alert_message,
            "redaction_message": None,
            "redaction_alert_level": None,
        }


class MessageUserGroupMapping(Base):
    __tablename__ = "message_user_group_mapping"
    message_id = Column(Integer, ForeignKey("message.id"), nullable=False)
    user_group_id = Column(Integer, ForeignKey("user_group.id"), nullable=False)

    def client_response(self):
        return {
            **super().client_response(),
            "message_id": self.message_id,
            "user_group_id": self.user_group_id,
        }


class Feedback(Base):
    __tablename__ = "feedback"
    message_id = Column(Integer, ForeignKey("message.id"), nullable=False)
    feedback_score_id = Column(Integer, ForeignKey("feedback_score.id"))
    feedback_label_id = Column(Integer, ForeignKey("feedback_label.id"), nullable=False)
    freetext = Column(Text)

    def client_response(self):
        return {
            **super().client_response(),
            "message_id": self.message_id,
            "feedback_score_id": self.feedback_score_id,
            "feedback_label_id": self.feedback_label_id,
            "freetext": self.freetext,
        }


class FeedbackScore(Base):
    __tablename__ = "feedback_score"
    score = Column(String(255), nullable=False)

    def client_response(self):
        return {
            **super().client_response(),
            "score": self.score,
        }


class FeedbackLabel(Base):
    __tablename__ = "feedback_label"
    label = Column(String(255), nullable=False)

    def client_response(self):
        return {
            **super().client_response(),
            "label": self.label,
        }


class Theme(Base):
    __tablename__ = "theme"
    title = Column(Text, nullable=False)
    subtitle = Column(Text, nullable=False)
    position = Column(
        Integer, nullable=True
    )  # The 'position' column is to ensure that the order of records is explicit.
    # This is to avoid a situation where themes / use cases are shown in a random order on the frontend.

    def client_response(self):
        return super().client_response({"title": self.title, "subtitle": self.subtitle, "position": self.position})

    def __str__(self):
        return f"title: {self.title}; subtitle: {self.subtitle}; position: {self.position}"


class UseCase(Base):
    __tablename__ = "use_case"
    theme_id = Column(Integer, ForeignKey("theme.id"), nullable=False)
    title = Column(Text, nullable=False)
    system_prompt = Column(Text, nullable=True)
    instruction = Column(Text, nullable=False)
    user_input_form = Column(Text, nullable=False)
    position = Column(
        Integer, nullable=True
    )  # The 'position' column is to ensure that the order of records is explicit.
    # This is to avoid a situation where themes / use cases are shown in a random order on the frontend.

    def client_response(self, theme_uuid=None):
        return super().client_response(
            {
                "theme_uuid": theme_uuid,
                "title": self.title,
                "instruction": self.instruction,
                "user_input_form": self.user_input_form,
                "position": self.position,
            }
        )


class UserAction(Base):
    __tablename__ = "user_action"
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    auth_session_id = Column(Integer, ForeignKey("auth_session.id"), nullable=False)
    action_type_id = Column(Integer, ForeignKey("action_type.id"), nullable=False)
    action_properties = Column(JSON)


class ActionType(Base):
    __tablename__ = "action_type"
    action_name = Column(String(255), nullable=True)


### RAG models
# The first three classes (SearchIndex, RewrittenQuery and DocumentChunk) represent real data objects.


# The search index represents the OpenSearch index.
# The OpenSearch index contains a set of document chunks. Searches within an index are limited
# to the document chunks in that index.
class SearchIndex(Base):
    __tablename__ = "search_index"

    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)


# A document chunk is a section of a document.
# At runtime, the content is searched and given an opensearch score.
class DocumentChunk(Base):
    __tablename__ = "document_chunk"

    search_index_id = Column(Integer, ForeignKey("search_index.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("document.id"), nullable=False)
    name = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    id_opensearch = Column(
        String(255),
        nullable=False,
    )  # This is not an ID anywhere in the PostgreSQL database, but it is the id in the opensearch index.

    def __str__(self):
        return f"name=('{self.name}'), content_truncated=('{self.content[0:20]}...{self.content[-20]}')"


# A document is the original document that was chunked
class Document(Base):
    __tablename__ = "document"

    name = Column(String(255), nullable=False)
    description = Column(Text)
    url = Column(Text)
    is_central = Column(Boolean, nullable=False)
    user_mappings = relationship("DocumentUserMapping", back_populates="document")
    chat_document_mapping = relationship("ChatDocumentMapping", back_populates="document")


class DocumentUserMapping(Base):
    __tablename__ = "document_user_mapping"

    document_id = Column(Integer, ForeignKey("document.id"), nullable=False, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False, primary_key=True)
    auth_session_id = Column(Integer, ForeignKey("auth_session.id"), nullable=False)
    expired_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    document = relationship("Document", back_populates="user_mappings")
    last_used = Column(DateTime)


class ChatDocumentMapping(Base):
    __tablename__ = "chat_document_mapping"

    chat_id = Column(Integer, ForeignKey("chat.id"), primary_key=True)
    document_uuid = Column(UUID(as_uuid=True), ForeignKey("document.uuid"), primary_key=True)

    chat = relationship("Chat", back_populates="chat_document_mapping")
    document = relationship("Document", back_populates="chat_document_mapping")


# A rewritten query is when an LLM takes the user query and re-writes it so that it performs better in OpenSearch.
class RewrittenQuery(Base):
    __tablename__ = "rewritten_query"

    search_index_id = Column(Integer, ForeignKey("search_index.id"), nullable=False)
    message_id = Column(Integer, ForeignKey("message.id"), nullable=False)
    llm_internal_response_id = Column(Integer, ForeignKey("llm_internal_response.id"), nullable=False)
    content = Column(Text, nullable=False)


# The following mapping tables are used to record the journey through the RAG pipeline.


# This mapping table links the users message to the document chunks that were retrieved.
# The table includes information about the final relevance of the document chunk as determined by an LLM.
class MessageDocumentChunkMapping(Base):
    __tablename__ = "message_document_chunk_mapping"

    message_id = Column(Integer, ForeignKey("message.id"), nullable=False)
    document_chunk_id = Column(Integer, ForeignKey("document_chunk.id"), nullable=False)
    llm_internal_response_id = Column(Integer, ForeignKey("llm_internal_response.id"))
    opensearch_score = Column(Float, nullable=False)
    use_document_chunk = Column(Boolean)


# This mapping table links the user message to the search indexes considered by the RAG chain.
# The table contains information about whether the LLM thought the index was relevant.
class MessageSearchIndexMapping(Base):
    __tablename__ = "message_search_index_mapping"

    search_index_id = Column(Integer, ForeignKey("search_index.id"), nullable=False)
    message_id = Column(Integer, ForeignKey("message.id"), nullable=False)
    llm_internal_response_id = Column(Integer, ForeignKey("llm_internal_response.id"))
    use_index = Column(Boolean, nullable=False)


# The following 2 tables are to help record and manage the internal LLM calls.

# The llm internal response is a way to record the exact tokens from an internal llm call alongside the completion cost
# and the number of tokens in the response.


# The system prompt is an instruction given to an LLM.
class SystemPrompt(Base):
    __tablename__ = "system_prompt"

    name = Column(VARCHAR(255), nullable=False)
    content = Column(Text, nullable=False)


class LlmInternalResponse(Base):
    __tablename__ = "llm_internal_response"

    llm_id = Column(Integer, ForeignKey("llm.id"), nullable=False)
    system_prompt_id = Column(Integer, ForeignKey("system_prompt.id"), nullable=True)
    content = Column(Text, nullable=False)
    tokens_in = Column(Integer, nullable=False)
    tokens_out = Column(Integer, nullable=False)
    completion_cost = Column(DECIMAL(precision=10, scale=8), nullable=False)


# The user prompt is an LLM prompt stored in the user's Personalised Prompt Library.
class UserPrompt(Base):
    __tablename__ = "user_prompt"

    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    title = Column(VARCHAR(255), nullable=False)
    content = Column(Text, nullable=False)
    description = Column(Text, nullable=True)

    def client_response(self):
        payload = {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "description": self.description,
        }
        return super().client_response(payload)


# The following tables are used to record the journey through the Gov UK Search API pipeline.


class GovUkSearchResult(Base):
    __tablename__ = "gov_uk_search_result"

    message_id = Column(Integer, ForeignKey("message.id"), nullable=False)
    gov_uk_search_query_id = Column(Integer, ForeignKey("gov_uk_search_query.id"), nullable=False)
    llm_internal_response_id = Column(Integer, ForeignKey("llm_internal_response.id"), nullable=False)
    content = Column(Text, nullable=False)
    url = Column(Text, nullable=False)
    is_used = Column(Boolean, nullable=False)
    position = Column(Integer, nullable=False)


class GovUkSearchQuery(Base):
    __tablename__ = "gov_uk_search_query"

    llm_internal_response_id = Column(Integer, ForeignKey("llm_internal_response.id"), nullable=False)
    message_id = Column(Integer, ForeignKey("message.id"))
    content = Column(Text, nullable=False)


class UseGovUkSearchDecision(Base):
    __tablename__ = "use_gov_uk_search_decision"

    llm_internal_response_id = Column(Integer, ForeignKey("llm_internal_response.id"), nullable=False)
    message_id = Column(Integer, ForeignKey("message.id"))
    decision = Column(Boolean, nullable=False)


engine = create_engine(database_url())
Session = sessionmaker(bind=engine)


class ModelEnum(str, Enum):
    user = "user"
    chat = "chat"
    message = "message"
    auth_session = "auth_session"
    user_group = "user_group"
    message_user_group_mapping = "message_user_group_mapping"
    redaction = "redaction"
    theme = "theme"
    use_case = "use_case"
    user_action = "user_action"
    action_type = "action_type"
    feedback = "feedback"
    feedback_score = "feedback_score"
    feedback_label = "feedback_label"
    llm = "llm"
    search_index = "search_index"
    document_chunk = "document_chunk"
    document = "document"
    rewritten_query = "rewritten_query"
    message_document_chunk_mapping = "message_document_chunk_mapping"
    message_search_index_mapping = "message_search_index_mapping"
    system_prompt = "system_prompt"
    llm_internal_response = "llm_internal_response"
    user_prompt = "user_prompt"


def get_model_class(model: ModelEnum) -> Type["Base"]:
    model_mapping = {
        ModelEnum.user: User,
        ModelEnum.chat: Chat,
        ModelEnum.message: Message,
        ModelEnum.auth_session: AuthSession,
        ModelEnum.user_group: UserGroup,
        ModelEnum.message_user_group_mapping: MessageUserGroupMapping,
        ModelEnum.redaction: Redaction,
        ModelEnum.feedback: Feedback,
        ModelEnum.theme: Theme,
        ModelEnum.use_case: UseCase,
        ModelEnum.user_action: UserAction,
        ModelEnum.action_type: ActionType,
        ModelEnum.feedback_score: FeedbackScore,
        ModelEnum.feedback_label: FeedbackLabel,
        ModelEnum.llm: LLM,
        ModelEnum.search_index: SearchIndex,
        ModelEnum.document_chunk: DocumentChunk,
        ModelEnum.document: Document,
        ModelEnum.rewritten_query: RewrittenQuery,
        ModelEnum.message_document_chunk_mapping: MessageDocumentChunkMapping,
        ModelEnum.message_search_index_mapping: MessageSearchIndexMapping,
        ModelEnum.system_prompt: SystemPrompt,
        ModelEnum.llm_internal_response: LlmInternalResponse,
        ModelEnum.user_prompt: UserPrompt,
    }
    return model_mapping[model]


Index("ix_user_uuid", User.uuid)
Index("ix_chat_uuid", Chat.uuid)
Index("ix_message_uuid", Message.uuid)
Index("ix_auth_session_uuid", AuthSession.uuid)
Index("ix_feedback_message_id", Feedback.message_id)
Index("ix_user_prompt_created_at", UserPrompt.created_at)
