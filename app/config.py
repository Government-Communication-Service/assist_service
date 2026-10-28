import os
from enum import Enum

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    AWSSecretsManagerSettingsSource,
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class ThinkingLevel(str, Enum):
    """Effort level for Sonnet 5.5 adaptive thinking on the main chat endpoint.

    disabled — lowest thinking setting (cheapest). Sonnet 5.5 rejects `thinking: disabled`, so this
    sends `between_tools`: no extended thinking, only short progress notes between tool calls.
    low/medium/high/xhigh/max — adaptive thinking at increasing depth/cost.
    Set via CHAT_THINKING_LEVEL env var (default disabled); override per-request via the
    thinking_level body parameter.
    """

    disabled = "disabled"
    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"
    max = "max"


class CacheTtl(str, Enum):
    """Time-to-live for the conversation prompt-cache breakpoint.

    five_minutes — 1.25x write cost, breaks even after 2 requests.
    one_hour — 2x write cost, worth it when users pause mid-conversation.
    """

    five_minutes = "5m"
    one_hour = "1h"


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
    # Async engine pool, per worker: up to db_pool_size + db_max_overflow connections.
    # Sync engines keep SQLAlchemy's defaults (5 + 10).
    db_pool_size: int = 10
    # 0 means no overflow beyond db_pool_size.
    db_max_overflow: int = 20
    # Seconds to wait for a free connection before raising.
    db_pool_timeout_seconds: int = 30
    # Connections older than this are replaced on checkout.
    db_pool_recycle_seconds: int = 1800
    # Test each connection on checkout, so a dropped one is replaced instead of failing the request.
    db_pool_pre_ping: bool = True

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

    @model_validator(mode="after")
    def bedrock_retry_budget_fits_request_timeout(self) -> "AppSettings":
        if self.bedrock_retry_budget_seconds >= self.request_timeout_seconds:
            raise ValueError(
                f"bedrock_retry_budget_seconds ({self.bedrock_retry_budget_seconds}) must be less than "
                f"request_timeout_seconds ({self.request_timeout_seconds})"
            )
        return self

    # --- AWS / infra ---
    aws_default_region: str = "eu-west-2"
    aws_bedrock_region1: str = "eu-west-2"
    aws_bedrock_region2: str = "eu-west-1"
    aws_bedrock_regions_max_retries: int = 3
    # Non-stream Bedrock calls: jittered exponential backoff between retries, and a wall-clock
    # budget for the whole call including retries. The budget must fit inside request_timeout_seconds.
    bedrock_retry_backoff_base_seconds: float = 0.5
    bedrock_retry_backoff_max_seconds: float = 4.0
    bedrock_retry_budget_seconds: float = 110.0
    s3_errordocs_bucket: str = "assist-error-docs"
    cloudwatch_log_group: str | None = None
    cloudwatch_log_stream: str | None = None

    # --- GCS ---
    gcs_data_api_url: str | None = None

    # --- feature flags ---
    is_dev: bool = False
    use_rag: bool = True
    # Old SMART_TARGETS_* env var names are still read so existing deployments keep working.
    campaign_benchmarks_service_disabled: bool = Field(
        False, validation_alias=AliasChoices("campaign_benchmarks_service_disabled", "smart_targets_service_disabled")
    )
    debug_mode: bool = False
    debug_logging: bool = False

    # --- Probably unused ---
    use_jwt_token: bool = False
    show_auth_token_generator: bool = False
    show_header_params_in_docs: bool = False
    show_developer_endpoints_in_docs: bool = False
    use_default_llm_response: bool = False
    litellm_logging: bool = False

    # --- auth ---
    # When True, PUT /v1/user/{user_uuid} needs a matching User-Key-UUID header and that user's Session-Auth.
    # Off by default until the frontend sends both headers on that call.
    user_update_requires_session_auth: bool = False

    # --- monitoring ---
    bugsnag_release_stage: str | None = None
    disable_bugsnag_logging: bool = False
    disable_cloudwatch_logging: bool = False

    # --- LLM / Bedrock ---
    llm_default_provider: str = "bedrock"
    llm_default_model: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    llm_chat_response_model: str = "anthropic.claude-sonnet-5-5"
    llm_chat_title_model: str = "anthropic.claude-haiku-5-5"
    llm_index_router: str = "anthropic.claude-haiku-5-5"
    llm_opensearch_query_generator: str = "anthropic.claude-haiku-5-5"
    llm_chunk_reviewer: str = "anthropic.claude-haiku-5-5"
    llm_govuk_query_generator: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    llm_document_relevancy_model: str = "anthropic.claude-haiku-5-5"
    llm_gov_uk_search_followup_assessment: str = "anthropic.claude-haiku-5-5"
    llm_campaign_benchmarks_model: str = Field(
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        validation_alias=AliasChoices("llm_campaign_benchmarks_model", "llm_smart_targets_model"),
    )
    llm_style_guide_model: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    chat_thinking_level: ThinkingLevel = ThinkingLevel.disabled
    # Total attempts (not retries) for the combined title+classification Haiku call; 1 = no retry.
    classification_title_max_attempts: int = 2

    # --- System prompt constants ---
    system_prompt_caching_enabled: bool = True
    system_prompt_model_name: str = "Claude Sonnet 5.5"
    system_prompt_model_cutoff: str = "end of June 2026"  # no capital at start
    system_prompt_assist_about: str = "https://connect.communications.gov.uk/assist/guidance-and-support"
    system_prompt_assist_how_to_use: str = "https://www.communications.gov.uk/publication/about-assist"
    system_prompt_assist_my_docs: str = "https://connect.communications.gov.uk/assist/my-documents"
    system_prompt_assist_home: str = "https://connect.communications.gov.uk/assist"
    system_prompt_audience_seg: str = "https://www.communications.gov.uk/publication/government-audience-segmentation"

    # --- style guide ---
    style_guide_llm_batch_size: int = 10
    style_guide_max_document_chars: int = 100_000
    style_guide_max_chunk_chars: int = 50_000
    style_guide_max_output_tokens: int = 4096

    # --- timeouts / batch sizes ---
    stream_first_chunk_timeout: float = 20.0
    # Whole-request timeout applied by the API middleware.
    request_timeout_seconds: int = 120
    bedrock_api_read_timeout_seconds: int = 115
    bedrock_api_connect_timeout_seconds: int = 3
    opensearch_index_request_timeout_seconds: int = 30
    opensearch_delete_batch_size: int = 100
    document_cleanup_batch_size: int = 1000
    document_chunk_insert_batch_size: int = 1000
    document_processing_timeout_seconds: int = 118
    max_table_chunk_chars: int = 1500

    # --- compaction ---
    compaction_enabled: bool = True
    compaction_token_threshold: int = 120_000
    compaction_max_summary_tokens: int = 12_000
    # Never compact until at least this many messages since the last compaction
    compaction_min_messages: int = 4
    # Feature flag — when False, summaries are written but not used.
    compaction_use_conversation_summary: bool = True
    # A compaction lock older than this is treated as stale and can be reacquired
    compaction_lock_stale_after_minutes: int = 5
    compaction_cache_control_enabled: bool = True

    # --- prompt caching (conversation) ---
    message_cache_control_enabled: bool = True
    message_cache_ttl: CacheTtl = CacheTtl.five_minutes
    cache_read_cost_multiplier: float = 0.1
    cache_write_cost_multiplier: float = 1.25

    # --- agent loop ---
    # Model calls per chat turn; past this, the loop nudges the model to answer instead of calling tools.
    agent_loop_max_iterations: int = 5
    # Tool calls executed per model call; extra calls get an is_error result.
    agent_loop_max_calls_per_iteration: int = 5
    # Fraction of chat_model_context_window_tokens past which the loop nudges the model to answer.
    agent_loop_context_budget_fraction: float = 0.8
    # Cumulative tool_result chars per turn past which no further tool call runs and the model is
    # nudged to answer. With the per-call cap below, a turn's tool results are at most this plus one
    # capped result, regardless of per-tool caps.
    agent_loop_max_result_chars_per_turn: int = 200_000
    # A single tool_result longer than this is truncated.
    agent_loop_max_result_chars_per_call: int = 200_000
    # Wall-clock limit for one tool call; past it the call is cancelled and the model gets an is_error result.
    agent_loop_tool_timeout_seconds: float = 90.0
    # Model-written text in tool status labels is shortened to this length.
    tool_status_label_max_query_chars: int = 80

    # --- helper LLM calls (relevance, filter selection, title) ---
    # User turns in the recent conversation shown to helper calls are cut to this many chars
    # (head and tail kept), and the whole recent conversation to the second limit (oldest turns dropped).
    helper_call_max_user_turn_chars: int = 4_000
    helper_call_max_context_chars: int = 16_000
    # This turn's user message, as sent to the relevance checks and the title call, is cut to this many chars.
    helper_call_max_query_chars: int = 8_000

    # --- central guidance RAG ---
    max_central_guidance_results: int = 12
    max_central_guidance_chunk_chars: int = 10_000
    central_guidance_max_tool_queries: int = 3
    # Recent conversation turns shown to the chunk relevance evaluator.
    central_guidance_evaluator_recent_turns: int = 6

    # --- uploaded documents ---
    uploaded_documents_max_tool_queries: int = 3
    # Characters of uploaded-document passages returned per search.
    uploaded_documents_max_result_chars: int = 55_000
    # Characters of an uploaded document returned per read_uploaded_document call.
    uploaded_documents_max_read_chars: int = 30_000

    # --- campaign benchmarks ---
    campaign_benchmarks_max_tool_metrics: int = Field(
        4, validation_alias=AliasChoices("campaign_benchmarks_max_tool_metrics", "smart_targets_max_tool_metrics")
    )
    # Recent conversation turns shown to the filter selector.
    campaign_benchmarks_recent_turns: int = Field(
        6, validation_alias=AliasChoices("campaign_benchmarks_recent_turns", "smart_targets_recent_turns")
    )

    # --- audience segments ---
    audience_segments_max_tool_segments: int = 4
    audience_segments_library_cache_seconds: int = 600
    # After a failed library refresh, how long to wait before calling the API again (stale data is served meanwhile).
    audience_segments_library_retry_backoff_seconds: int = 60
    audience_segments_api_timeout_seconds: float = 10.0

    # --- context window ---
    # The main chat model's total context window. The agent loop withdraws tools once a request
    # nears it (see agent_loop_context_budget_fraction).
    chat_model_context_window_tokens: int = 1_000_000

    # --- gov.uk ---
    whitelisted_urls: list[str] = ["https://www.gov.uk"]
    blacklisted_urls: list[str] = ["https://www.gov.uk/publications"]
    web_browsing_timeout: int = 300
    gov_uk_base_url: str = "https://www.gov.uk"
    # Results fetched per search term: default and maximum.
    gov_uk_search_default_count: int = 5
    gov_uk_search_max_count: int = 10
    gov_uk_search_max_document_chars: int = 9000
    # Characters of a page returned by get_gov_uk_page.
    gov_uk_page_max_chars: int = 100_000
    # get_gov_uk_pdf: characters returned per call, and the largest PDF downloaded.
    gov_uk_pdf_max_chars: int = 40_000
    gov_uk_pdf_max_bytes: int = 5_000_000
    gov_uk_search_max_tool_search_terms: int = 3
    # Pages returned per search.
    gov_uk_search_max_tool_documents: int = 5
    # Characters of each page shown to the relevancy check.
    gov_uk_search_relevancy_excerpt_chars: int = 6000
    # Pages assessed per relevancy call.
    gov_uk_search_relevancy_batch_size: int = 10
    gov_uk_http_timeout_seconds: float = 20.0

    # --- dev tooling ---
    # Do not set this variable in production - it will cause every request to be logged to file
    log_full_invocation_request_to_file_path: str = ""

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

    @field_validator("db_max_overflow")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"value must not be negative, got {v}")
        return v

    @field_validator(
        "db_pool_size",
        "db_pool_timeout_seconds",
        "db_pool_recycle_seconds",
        "stream_first_chunk_timeout",
        "web_browsing_timeout",
        "document_processing_timeout_seconds",
        "opensearch_delete_batch_size",
        "document_cleanup_batch_size",
        "document_chunk_insert_batch_size",
        "max_table_chunk_chars",
        "compaction_token_threshold",
        "compaction_max_summary_tokens",
        "compaction_min_messages",
        "compaction_lock_stale_after_minutes",
        "cache_read_cost_multiplier",
        "cache_write_cost_multiplier",
        "style_guide_max_document_chars",
        "style_guide_max_chunk_chars",
        "style_guide_llm_batch_size",
        "gov_uk_search_max_count",
        "gov_uk_search_max_document_chars",
        "gov_uk_page_max_chars",
        "gov_uk_pdf_max_chars",
        "gov_uk_pdf_max_bytes",
        "max_central_guidance_results",
        "max_central_guidance_chunk_chars",
        "agent_loop_max_iterations",
        "agent_loop_max_calls_per_iteration",
        "chat_model_context_window_tokens",
        "agent_loop_context_budget_fraction",
        "agent_loop_max_result_chars_per_turn",
        "agent_loop_max_result_chars_per_call",
        "agent_loop_tool_timeout_seconds",
        "tool_status_label_max_query_chars",
        "central_guidance_max_tool_queries",
        "helper_call_max_user_turn_chars",
        "helper_call_max_context_chars",
        "helper_call_max_query_chars",
        "central_guidance_evaluator_recent_turns",
        "uploaded_documents_max_tool_queries",
        "uploaded_documents_max_result_chars",
        "uploaded_documents_max_read_chars",
        "campaign_benchmarks_max_tool_metrics",
        "campaign_benchmarks_recent_turns",
        "audience_segments_max_tool_segments",
        "audience_segments_library_cache_seconds",
        "audience_segments_library_retry_backoff_seconds",
        "audience_segments_api_timeout_seconds",
        "gov_uk_search_default_count",
        "gov_uk_search_max_tool_search_terms",
        "gov_uk_search_max_tool_documents",
        "gov_uk_search_relevancy_excerpt_chars",
        "gov_uk_search_relevancy_batch_size",
        "gov_uk_http_timeout_seconds",
        "style_guide_max_output_tokens",
        "request_timeout_seconds",
        "bedrock_api_read_timeout_seconds",
        "bedrock_api_connect_timeout_seconds",
        "opensearch_index_request_timeout_seconds",
        "aws_bedrock_regions_max_retries",
        "bedrock_retry_backoff_base_seconds",
        "bedrock_retry_backoff_max_seconds",
        "bedrock_retry_budget_seconds",
        "classification_title_max_attempts",
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
        """Use AWS Secrets manager

        The priority order is:
            kwargs
            environment variables
            .env file
            AWS secrets manager
            defaults
        """
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
USER_UPDATE_REQUIRES_SESSION_AUTH = settings.user_update_requires_session_auth
URL_HOSTNAME = settings.url_hostname

# AWS
AWS_DEFAULT_REGION = settings.aws_default_region
S3_ERRORDOCS_BUCKET = settings.s3_errordocs_bucket

# GCS
GCS_DATA_API_URL = settings.gcs_data_api_url

# Feature flags
USE_RAG = settings.use_rag
CAMPAIGN_BENCHMARKS_SERVICE_DISABLED = settings.campaign_benchmarks_service_disabled

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
BEDROCK_RETRY_BACKOFF_BASE_SECONDS = settings.bedrock_retry_backoff_base_seconds
BEDROCK_RETRY_BACKOFF_MAX_SECONDS = settings.bedrock_retry_backoff_max_seconds
BEDROCK_RETRY_BUDGET_SECONDS = settings.bedrock_retry_budget_seconds
STREAM_FIRST_CHUNK_TIMEOUT = settings.stream_first_chunk_timeout
BEDROCK_API_READ_TIMEOUT_SECONDS = settings.bedrock_api_read_timeout_seconds
BEDROCK_API_CONNECT_TIMEOUT_SECONDS = settings.bedrock_api_connect_timeout_seconds
LLM_DEFAULT_MODEL = settings.llm_default_model
LLM_CHAT_RESPONSE_MODEL = settings.llm_chat_response_model
LLM_CHAT_TITLE_MODEL = settings.llm_chat_title_model
LLM_INDEX_ROUTER = settings.llm_index_router
LLM_OPENSEARCH_QUERY_GENERATOR = settings.llm_opensearch_query_generator
LLM_CHUNK_REVIEWER = settings.llm_chunk_reviewer
LLM_GOVUK_QUERY_GENERATOR = settings.llm_govuk_query_generator
LLM_DOCUMENT_RELEVANCY_MODEL = settings.llm_document_relevancy_model
LLM_GOV_UK_SEARCH_FOLLOWUP_ASSESSMENT = settings.llm_gov_uk_search_followup_assessment
LLM_CAMPAIGN_BENCHMARKS_MODEL = settings.llm_campaign_benchmarks_model
CHAT_THINKING_LEVEL = settings.chat_thinking_level

# Central guidance RAG
MAX_CENTRAL_GUIDANCE_RESULTS = settings.max_central_guidance_results
MAX_CENTRAL_GUIDANCE_CHUNK_CHARS = settings.max_central_guidance_chunk_chars
CENTRAL_GUIDANCE_MAX_TOOL_QUERIES = settings.central_guidance_max_tool_queries
CENTRAL_GUIDANCE_EVALUATOR_RECENT_TURNS = settings.central_guidance_evaluator_recent_turns

# Uploaded documents
UPLOADED_DOCUMENTS_MAX_TOOL_QUERIES = settings.uploaded_documents_max_tool_queries
UPLOADED_DOCUMENTS_MAX_RESULT_CHARS = settings.uploaded_documents_max_result_chars
UPLOADED_DOCUMENTS_MAX_READ_CHARS = settings.uploaded_documents_max_read_chars

# Campaign benchmarks
CAMPAIGN_BENCHMARKS_MAX_TOOL_METRICS = settings.campaign_benchmarks_max_tool_metrics
CAMPAIGN_BENCHMARKS_RECENT_TURNS = settings.campaign_benchmarks_recent_turns

# Audience segments
AUDIENCE_SEGMENTS_MAX_TOOL_SEGMENTS = settings.audience_segments_max_tool_segments
AUDIENCE_SEGMENTS_LIBRARY_CACHE_SECONDS = settings.audience_segments_library_cache_seconds
AUDIENCE_SEGMENTS_LIBRARY_RETRY_BACKOFF_SECONDS = settings.audience_segments_library_retry_backoff_seconds
AUDIENCE_SEGMENTS_API_TIMEOUT_SECONDS = settings.audience_segments_api_timeout_seconds

# Agent loop
TOOL_STATUS_LABEL_MAX_QUERY_CHARS = settings.tool_status_label_max_query_chars

# Style guide
STYLE_GUIDE_LLM_BATCH_SIZE = settings.style_guide_llm_batch_size
STYLE_GUIDE_LLM_MODEL = settings.llm_style_guide_model
STYLE_GUIDE_MAX_DOCUMENT_CHARS = settings.style_guide_max_document_chars
STYLE_GUIDE_MAX_CHUNK_CHARS = settings.style_guide_max_chunk_chars
STYLE_GUIDE_MAX_OUTPUT_TOKENS = settings.style_guide_max_output_tokens

# Document cleanup
OPENSEARCH_DELETE_BATCH_SIZE = settings.opensearch_delete_batch_size
DOCUMENT_CLEANUP_BATCH_SIZE = settings.document_cleanup_batch_size
DOCUMENT_CHUNK_INSERT_BATCH_SIZE = settings.document_chunk_insert_batch_size
DOCUMENT_PROCESSING_TIMEOUT_SECONDS = settings.document_processing_timeout_seconds
MAX_TABLE_CHUNK_CHARS = settings.max_table_chunk_chars
OPENSEARCH_INDEX_REQUEST_TIMEOUT_SECONDS = settings.opensearch_index_request_timeout_seconds
REQUEST_TIMEOUT_SECONDS = settings.request_timeout_seconds

# Gov.uk
WHITELISTED_URLS = settings.whitelisted_urls
BLACKLISTED_URLS = settings.blacklisted_urls
WEB_BROWSING_TIMEOUT = settings.web_browsing_timeout
GOV_UK_BASE_URL = settings.gov_uk_base_url
GOV_UK_SEARCH_MAX_COUNT = settings.gov_uk_search_max_count
GOV_UK_SEARCH_MAX_DOCUMENT_CHARS = settings.gov_uk_search_max_document_chars
GOV_UK_PAGE_MAX_CHARS = settings.gov_uk_page_max_chars
GOV_UK_PDF_MAX_CHARS = settings.gov_uk_pdf_max_chars
GOV_UK_PDF_MAX_BYTES = settings.gov_uk_pdf_max_bytes
GOV_UK_SEARCH_DEFAULT_COUNT = settings.gov_uk_search_default_count
GOV_UK_SEARCH_MAX_TOOL_SEARCH_TERMS = settings.gov_uk_search_max_tool_search_terms
GOV_UK_SEARCH_MAX_TOOL_DOCUMENTS = settings.gov_uk_search_max_tool_documents
GOV_UK_SEARCH_RELEVANCY_EXCERPT_CHARS = settings.gov_uk_search_relevancy_excerpt_chars
GOV_UK_SEARCH_RELEVANCY_BATCH_SIZE = settings.gov_uk_search_relevancy_batch_size
GOV_UK_HTTP_TIMEOUT_SECONDS = settings.gov_uk_http_timeout_seconds

# Dev tooling
LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH = settings.log_full_invocation_request_to_file_path

# Test helpers
TEST_USER_GROUPS = settings.test_user_groups
