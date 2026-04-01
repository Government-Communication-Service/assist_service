import os
from typing import Union

from dotenv import load_dotenv


def load_environment_variables():
    if os.path.exists("../.env"):
        load_dotenv("../.env")
        # print("Loaded environment variables from .env file.")


def env_variable(name: str, default=None) -> Union[str, bool]:
    value = os.getenv(name, default)
    if value and str(value).lower() == "false":
        return False
    if value and str(value).lower() == "true":
        return True
    return value


### --- Environment Configuration --- ###

IS_DEV = env_variable("IS_DEV")
URL_HOSTNAME = os.getenv("URL_HOSTNAME", "http://localhost:" + os.getenv("PORT", "5312"))
DATA_DIR = "data"

### --- AWS Configuration --- ###

AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-2")

### --- S3 Configuration --- ###

S3_ERRORDOCS_BUCKET = os.getenv("S3_ERRORDOCS_BUCKET", "assist-error-docs")

### --- LLM / Bedrock Configuration --- ###

LLM_DEFAULT_PROVIDER = "bedrock"
AWS_BEDROCK_REGION1 = "us-west-2"
AWS_BEDROCK_REGION2 = "us-east-1"
AWS_BEDROCK_REGIONS_MAX_RETRIES: int = 3
# Timeout in seconds to wait for the first chunk when streaming
# If no data arrives within this time, trigger failover
STREAM_FIRST_CHUNK_TIMEOUT: float = float(os.getenv("STREAM_FIRST_CHUNK_TIMEOUT", "10.0"))

# The default LLM used by BedrockHandler
# Note that, when building an instance of BedrockHandler, a region prefix is added
# E.g. if the region is 'us', a 'us.' prefix is added to the model name.
# This is necessary to handle cross-region inference, which we use as a failover mechanism
# anthropic.claude-sonnet-4-5-20250929-v1:0
LLM_DEFAULT_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"

### --- Chat Configuration --- ###

# This LLM generates the final response to the user's query.
# This model should ideally be of thie highest quality.
LLM_CHAT_RESPONSE_MODEL = "anthropic.claude-sonnet-4-6"

# This LLM generates the title of the user's chat.
LLM_CHAT_TITLE_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"

### --- Central Guidance Configuration --- ###

# This LLM determines if the user query should be enriched
# by the central guidance, or not.
LLM_INDEX_ROUTER = "anthropic.claude-haiku-4-5-20251001-v1:0"

# This LLM takes a user's message and returns a set of OpenSearch queries
LLM_OPENSEARCH_QUERY_GENERATOR = "anthropic.claude-sonnet-4-5-20250929-v1:0"

# This LLM determines if the retrieved document chunk
# should be included in the main LLM context.
LLM_CHUNK_REVIEWER = "anthropic.claude-haiku-4-5-20251001-v1:0"


### --- GOV.UK Configuration --- ###

LLM_GOVUK_QUERY_GENERATOR = "anthropic.claude-sonnet-4-5-20250929-v1:0"
LLM_DOCUMENT_RELEVANCY_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"
LLM_GOV_UK_SEARCH_FOLLOWUP_ASSESMENT = "anthropic.claude-haiku-4-5-20251001-v1:0"


### --- GCS Data API Configuration --- ###
GCS_DATA_API_URL = os.getenv("GCS_DATA_API_URL")

### --- Smart Targets Configuration --- ###
SMART_TARGETS_SERVICE_DISABLED = env_variable("SMART_TARGETS_SERVICE_DISABLED", False)
LLM_SMART_TARGETS_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"


### --- Compaction Configuration --- ###

# This LLM generates summaries of messages for compaction
LLM_COMPACTION_SUMMARISATION_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"

# Token threshold for triggering compaction (160k tokens)
COMPACTION_TOKEN_THRESHOLD = 160000

if env_variable("LLM_DEFAULT_MODEL"):
    LLM_DEFAULT_MODEL = env_variable("LLM_DEFAULT_MODEL")

WHITELISTED_URLS = ["https://www.gov.uk"]
BLACKLISTED_URLS = ["https://www.gov.uk/publications"]
WEB_BROWSING_TIMEOUT = 300
GOV_UK_BASE_URL = "https://www.gov.uk"
GOV_UK_SEARCH_MAX_COUNT = 10

### --- style guide configuration --- ###

# Number of rules to batch together for LLM validation
STYLE_GUIDE_LLM_BATCH_SIZE = 10

# Default LLM model for style guide checking
STYLE_GUIDE_LLM_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"

# Maximum number of characters in a document before rejecting with an error message
STYLE_GUIDE_MAX_DOCUMENT_CHARS = int(os.getenv("STYLE_GUIDE_MAX_DOCUMENT_CHARS", "100000"))

# Maximum number of characters per chunk when splitting large documents for LLM calls
STYLE_GUIDE_MAX_CHUNK_CHARS = int(os.getenv("STYLE_GUIDE_MAX_CHUNK_CHARS", "50000"))

### --- Document Cleanup Configuration --- ###

# Number of documents to delete in each batch when cleaning up expired documents from OpenSearch
OPENSEARCH_DELETE_BATCH_SIZE = int(os.getenv("OPENSEARCH_DELETE_BATCH_SIZE", "100"))
# Number of documents to delete in each batch when cleaning up expired documents from the database
DOCUMENT_CLEANUP_BATCH_SIZE = int(os.getenv("DOCUMENT_CLEANUP_BATCH_SIZE", "1000"))

# Document parsing timeout (seconds) for personal uploads
DOCUMENT_PROCESSING_TIMEOUT_SECONDS = int(os.getenv("DOCUMENT_PROCESSING_TIMEOUT_SECONDS", "118"))
