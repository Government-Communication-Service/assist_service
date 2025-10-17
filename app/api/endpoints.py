from uuid import UUID


class ENDPOINTS:
    def __init__(self, api_version: str = "v1"):
        self.api_version = f"/{api_version}"

    SESSIONS = "/auth-sessions"
    #
    # Endpoint patterns as class attributes
    CHATS = "/chats/users/{user_uuid}"
    CHAT_CREATE_STREAM = "/chats/users/{user_uuid}/stream"
    CHATS_BY_USER = "/chats/users/{user_uuid}/chats"
    CHAT_ITEM = "/chats/users/{user_uuid}/chats/{chat_uuid}"
    CHAT_UPDATE_STREAM = "/chats/users/{user_uuid}/chats/{chat_uuid}/stream"
    CHAT_MESSAGES = "/chats/users/{user_uuid}/chats/{chat_uuid}/messages"
    CHAT_TITLE = "/chats/users/{user_uuid}/chats/{chat_uuid}/title"
    CHAT_FAVOURITE = "/chats/users/{user_uuid}/chats/{chat_uuid}/favourite"
    CHAT_ARCHIVE = "/chats/users/{user_uuid}/chats/{chat_uuid}/archive"
    #
    #
    PROMPTS_BULK = "/prompts/bulk"
    PROMPTS_THEMES = "/prompts/themes"
    PROMPTS_THEME = "/prompts/themes/{theme_uuid}"
    PROMPTS_THEMES_USE_CASES = "/prompts/themes/{theme_uuid}/use-cases"
    PROMPTS_THEMES_USE_CASE = "/prompts/themes/{theme_uuid}/use-cases/{use_case_uuid}"
    PROMPTS_USE_CASE = "/prompts/use-cases/{use_case_uuid}"

    #
    USER_GET_CHATS = "/chats/users/{user_uuid}/chats"
    USER_DOCUMENTS = "/users/{user_uuid}/documents"
    USER_DOCUMENT = "/users/{user_uuid}/documents/{document_uuid}"

    #
    USER_PROMPTS = "/users/{user_uuid}/prompts"
    USER_PROMPT = "/users/{user_uuid}/prompts/{user_prompt_uuid}"

    #
    USER = "/user/{user_uuid}"
    USERS = "/users"

    #
    MESSAGE_FEEDBACK = "/chats/users/{user_uuid}/chats/{message_uuid}/feedback"
    FEEDBACK_LABELS = "/chat/messages/feedback/labels"

    # Endpoints for managing central RAG
    CENTRAL_RAG_SYNC = "/central-rag/synchronise"
    CENTRAL_RAG_DOCUMENT_CHUNKS = "/central-rag/document-chunks"
    CENTRAL_RAG_DOCUMENT_CHUNK = "/central-rag/document-chunks/{document_chunk_uuid}"

    # Add new endpoint for cleaning up expired message content
    CHAT_CLEANUP_EXPIRED_CONTENT = "/chats/messages/content/expired"

    def build_url(self, pattern: str, **kwargs) -> str:
        """Builds a URL from a pattern and keyword arguments to replace placeholders."""
        return f"{self.api_version}{pattern.format(**kwargs)}"

    # Methods for building specific endpoints
    def chats(self, user_uuid: UUID) -> str:
        return self.build_url(self.CHATS, user_uuid=user_uuid)

    def user_documents(self, user_uuid: UUID) -> str:
        return self.build_url(self.USER_DOCUMENTS, user_uuid=user_uuid)

    def user_document(self, user_uuid: UUID, document_uuid: UUID) -> str:
        return self.build_url(self.USER_DOCUMENT, user_uuid=user_uuid, document_uuid=document_uuid)

    def get_user_prompts(self, user_uuid: UUID) -> str:
        return self.build_url(self.USER_PROMPTS, user_uuid=user_uuid)

    def create_user_prompt(self, user_uuid: UUID) -> str:
        return self.build_url(self.USER_PROMPTS, user_uuid=user_uuid)

    def get_user_prompt(self, user_uuid: UUID, user_prompt_uuid: UUID) -> str:
        return self.build_url(self.USER_PROMPT, user_uuid=user_uuid, user_prompt_uuid=user_prompt_uuid)

    def create_user(self) -> str:
        return self.build_url(self.USERS)

    def update_user(self, user_uuid: UUID) -> str:
        return self.build_url(self.USER, user_uuid=user_uuid)

    def patch_user_prompt(self, user_uuid: UUID, user_prompt_uuid: UUID) -> str:
        return self.build_url(self.USER_PROMPT, user_uuid=user_uuid, user_prompt_uuid=user_prompt_uuid)

    def delete_user_prompt(self, user_uuid: UUID, user_prompt_uuid: UUID) -> str:
        return self.build_url(self.USER_PROMPT, user_uuid=user_uuid, user_prompt_uuid=user_prompt_uuid)

    def create_chat_stream(self, user_uuid: UUID) -> str:
        return self.build_url(self.CHAT_CREATE_STREAM, user_uuid=user_uuid)

    def get_chat_stream(self, user_uuid: UUID, chat_uuid: UUID) -> str:
        return self.build_url(self.CHAT_UPDATE_STREAM, user_uuid=user_uuid, chat_uuid=chat_uuid)

    def get_chats_by_user(self, user_uuid: UUID) -> str:
        return self.build_url(self.CHATS_BY_USER, user_uuid=user_uuid)

    def get_chat_item(self, user_uuid: UUID, chat_uuid: UUID) -> str:
        return self.build_url(self.CHAT_ITEM, user_uuid=user_uuid, chat_uuid=chat_uuid)

    def add_message_feedback(self, user_uuid: UUID, message_uuid: UUID) -> str:
        return self.build_url(self.MESSAGE_FEEDBACK, user_uuid=user_uuid, message_uuid=message_uuid)

    def get_feedback_labels(self) -> str:
        return self.build_url(self.FEEDBACK_LABELS)

    def get_chat_messages(self, user_uuid: UUID, chat_uuid: UUID) -> str:
        return self.build_url(self.CHAT_MESSAGES, user_uuid=user_uuid, chat_uuid=chat_uuid)

    def create_chat_title(self, user_uuid: UUID, chat_uuid: UUID) -> str:
        return self.build_url(self.CHAT_TITLE, user_uuid=user_uuid, chat_uuid=chat_uuid)

    def archive_chat(self, user_uuid: UUID, chat_uuid: UUID) -> str:
        return self.build_url(self.CHAT_ARCHIVE, user_uuid=user_uuid, chat_uuid=chat_uuid)

    def themes_use_cases(self, theme_uuid: UUID) -> str:
        return self.build_url(self.PROMPTS_THEMES_USE_CASES, theme_uuid=theme_uuid)

    def themes_use_case(self, theme_uuid: UUID, use_case_uuid: UUID) -> str:
        return self.build_url(
            self.PROMPTS_THEMES_USE_CASE,
            theme_uuid=theme_uuid,
            use_case_uuid=use_case_uuid,
        )

    def get_use_case(self, use_case_uuid: UUID) -> str:
        return self.build_url(self.PROMPTS_USE_CASE, use_case_uuid=use_case_uuid)

    def get_sessions(self) -> str:
        return self.build_url(self.SESSIONS)

    def prompts_bulk(self) -> str:
        return self.build_url(self.PROMPTS_BULK)

    def get_themes(self) -> str:
        return self.build_url(self.PROMPTS_THEMES)

    def post_theme(self) -> str:
        return self.build_url(self.PROMPTS_THEMES)

    def theme(self, theme_uuid: UUID) -> str:
        return self.build_url(self.PROMPTS_THEME, theme_uuid=theme_uuid)

    def document_chunks(self) -> str:
        return self.build_url(self.CENTRAL_RAG_DOCUMENT_CHUNKS)

    def document_chunk(self, document_chunk_uuid: UUID) -> str:
        return self.build_url(self.CENTRAL_RAG_DOCUMENT_CHUNK, document_chunk_uuid=document_chunk_uuid)

    def sync_central_index(self) -> str:
        return self.build_url(self.CENTRAL_RAG_SYNC)
