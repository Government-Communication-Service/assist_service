import os

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import (
    AWSSecretsManagerSettingsSource,
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,  # treat empty string env vars as unset (use field default)
    )

    # --- secrets ---
    auth_secret_key: SecretStr
    auth_secret_key_2: SecretStr | None = None
    postgres_password: SecretStr
    opensearch_password: SecretStr
    bugsnag_api_key: SecretStr | None = None

    # --- database ---
    postgres_db: str = "copilot"
    postgres_user: str = "postgres"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # --- opensearch ---
    opensearch_user: str = "admin"
    opensearch_host: str = "opensearch-node1"
    opensearch_port: int = 9200
    opensearch_disable_ssl: bool = False
    sync_central_indexes_on_startup: bool = False
    # When True, OpenSearch is stubbed out: the boot-time connection check is
    # skipped and clients return empty results. Lets the app run
    # without an OpenSearch cluster. Pair with USE_RAG=false.
    opensearch_mocked: bool = False

    # --- server ---
    port: int = 5312
    url_hostname: str | None = None

    @model_validator(mode="after")
    def set_url_hostname_default(self) -> "AppSettings":
        if self.url_hostname is None:
            self.url_hostname = f"http://localhost:{self.port}"
        return self

    # --- AWS / infra ---
    aws_default_region: str = "eu-west-2"
    aws_bedrock_region1: str = "eu-west-2"
    aws_bedrock_region2: str = "eu-west-1"
    aws_bedrock_regions_max_retries: int = 3
    s3_errordocs_bucket: str = "assist-error-docs"
    cloudwatch_log_group: str | None = None
    cloudwatch_log_stream: str | None = None

    # --- GCS ---
    gcs_data_api_url: str | None = None

    # --- feature flags ---
    is_dev: bool = False
    use_jwt_token: bool = False
    show_auth_token_generator: bool = False
    show_detailed_error_messages: bool = False
    show_header_params_in_docs: bool = False
    show_developer_endpoints_in_docs: bool = False
    use_rag: bool = True
    use_default_llm_response: bool = False
    smart_targets_service_disabled: bool = False
    debug_mode: bool = False
    debug_logging: bool = False
    litellm_logging: bool = False

    # --- monitoring ---
    bugsnag_release_stage: str | None = None
    disable_bugsnag_logging: bool = False
    disable_cloudwatch_logging: bool = False

    # --- LLM / Bedrock ---
    llm_default_provider: str = "bedrock"
    llm_default_model: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    llm_chat_response_model: str = "anthropic.claude-sonnet-4-6"
    llm_chat_title_model: str = "anthropic.claude-haiku-4-5-20251001-v1:0"
    llm_index_router: str = "anthropic.claude-haiku-4-5-20251001-v1:0"
    llm_opensearch_query_generator: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    llm_chunk_reviewer: str = "anthropic.claude-haiku-4-5-20251001-v1:0"
    llm_govuk_query_generator: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    llm_document_relevancy_model: str = "anthropic.claude-haiku-4-5-20251001-v1:0"
    llm_gov_uk_search_followup_assessment: str = "anthropic.claude-haiku-4-5-20251001-v1:0"
    llm_smart_targets_model: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    llm_compaction_summarisation_model: str = "anthropic.claude-haiku-4-5-20251001-v1:0"
    llm_style_guide_model: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"

    # --- style guide ---
    style_guide_llm_batch_size: int = 10
    style_guide_max_document_chars: int = 100000
    style_guide_max_chunk_chars: int = 50000

    # --- timeouts / batch sizes ---
    stream_first_chunk_timeout: float = 20.0
    opensearch_delete_batch_size: int = 100
    document_cleanup_batch_size: int = 1000
    document_processing_timeout_seconds: int = 118
    compaction_token_threshold: int = 160000

    # --- gov.uk ---
    whitelisted_urls: list[str] = ["https://www.gov.uk"]
    blacklisted_urls: list[str] = ["https://www.gov.uk/publications"]
    web_browsing_timeout: int = 300
    gov_uk_base_url: str = "https://www.gov.uk"
    gov_uk_search_max_count: int = 10

    # --- test helpers ---
    test_user_groups: str = ""
    test_session_uuid: str | None = None
    test_user_uuid: str | None = None
    test_chat_uuid: str | None = None
    default_user_key_uuid: str | None = None

    @field_validator("postgres_port", "opensearch_port", "port")
    @classmethod
    def valid_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"port must be between 1 and 65535, got {v}")
        return v

    @field_validator(
        "stream_first_chunk_timeout",
        "web_browsing_timeout",
        "document_processing_timeout_seconds",
        "opensearch_delete_batch_size",
        "document_cleanup_batch_size",
        "compaction_token_threshold",
        "style_guide_max_document_chars",
        "style_guide_max_chunk_chars",
        "style_guide_llm_batch_size",
        "gov_uk_search_max_count",
        "aws_bedrock_regions_max_retries",
        mode="after",
    )
    @classmethod
    def positive(cls, v: int | float) -> int | float:
        if v <= 0:
            raise ValueError(f"value must be positive, got {v}")
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        **kwargs: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources = list(super().settings_customise_sources(settings_cls, **kwargs))
        secret_name = os.environ.get("APP_SECRET_NAME")
        if secret_name:
            region = os.environ.get("AWS_DEFAULT_REGION", "eu-west-2")
            sources.append(
                AWSSecretsManagerSettingsSource(
                    settings_cls,
                    secret_id=secret_name,
                    region_name=region,
                )
            )
        return tuple(sources)


settings = AppSettings()

# ── Module-level exports — existing importers unchanged ──────────────────────

DATA_DIR = "data"
IS_DEV = settings.is_dev
URL_HOSTNAME = settings.url_hostname

# AWS
AWS_DEFAULT_REGION = settings.aws_default_region
S3_ERRORDOCS_BUCKET = settings.s3_errordocs_bucket

# GCS
GCS_DATA_API_URL = settings.gcs_data_api_url

# Feature flags
USE_RAG = settings.use_rag
SMART_TARGETS_SERVICE_DISABLED = settings.smart_targets_service_disabled

# OpenSearch (non-secret)
OPENSEARCH_USER = settings.opensearch_user
OPENSEARCH_HOST = settings.opensearch_host
OPENSEARCH_PORT = settings.opensearch_port
OPENSEARCH_DISABLE_SSL = settings.opensearch_disable_ssl
SYNC_CENTRAL_INDEXES_ON_STARTUP = settings.sync_central_indexes_on_startup
OPENSEARCH_MOCKED = settings.opensearch_mocked

# LLM / Bedrock
LLM_DEFAULT_PROVIDER = settings.llm_default_provider
AWS_BEDROCK_REGION1 = settings.aws_bedrock_region1
AWS_BEDROCK_REGION2 = settings.aws_bedrock_region2
AWS_BEDROCK_REGIONS_MAX_RETRIES = settings.aws_bedrock_regions_max_retries
STREAM_FIRST_CHUNK_TIMEOUT = settings.stream_first_chunk_timeout
LLM_DEFAULT_MODEL = settings.llm_default_model
LLM_CHAT_RESPONSE_MODEL = settings.llm_chat_response_model
LLM_CHAT_TITLE_MODEL = settings.llm_chat_title_model
LLM_INDEX_ROUTER = settings.llm_index_router
LLM_OPENSEARCH_QUERY_GENERATOR = settings.llm_opensearch_query_generator
LLM_CHUNK_REVIEWER = settings.llm_chunk_reviewer
LLM_GOVUK_QUERY_GENERATOR = settings.llm_govuk_query_generator
LLM_DOCUMENT_RELEVANCY_MODEL = settings.llm_document_relevancy_model
LLM_GOV_UK_SEARCH_FOLLOWUP_ASSESSMENT = settings.llm_gov_uk_search_followup_assessment
LLM_SMART_TARGETS_MODEL = settings.llm_smart_targets_model
LLM_COMPACTION_SUMMARISATION_MODEL = settings.llm_compaction_summarisation_model
COMPACTION_TOKEN_THRESHOLD = settings.compaction_token_threshold

# Style guide
STYLE_GUIDE_LLM_BATCH_SIZE = settings.style_guide_llm_batch_size
STYLE_GUIDE_LLM_MODEL = settings.llm_style_guide_model
STYLE_GUIDE_MAX_DOCUMENT_CHARS = settings.style_guide_max_document_chars
STYLE_GUIDE_MAX_CHUNK_CHARS = settings.style_guide_max_chunk_chars

# Document cleanup
OPENSEARCH_DELETE_BATCH_SIZE = settings.opensearch_delete_batch_size
DOCUMENT_CLEANUP_BATCH_SIZE = settings.document_cleanup_batch_size
DOCUMENT_PROCESSING_TIMEOUT_SECONDS = settings.document_processing_timeout_seconds

# Gov.uk
WHITELISTED_URLS = settings.whitelisted_urls
BLACKLISTED_URLS = settings.blacklisted_urls
WEB_BROWSING_TIMEOUT = settings.web_browsing_timeout
GOV_UK_BASE_URL = settings.gov_uk_base_url
GOV_UK_SEARCH_MAX_COUNT = settings.gov_uk_search_max_count

# Test helpers
TEST_USER_GROUPS = settings.test_user_groups
