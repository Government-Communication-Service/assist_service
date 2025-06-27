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

BYPASS_SESSION_VALIDATOR = env_variable("BYPASS_SESSION_VALIDATOR")
BYPASS_AUTH_VALIDATOR = env_variable("BYPASS_AUTH_VALIDATOR")


### --- LLM / Bedrock Configuration --- ###

LLM_DEFAULT_PROVIDER = "bedrock"
AWS_BEDROCK_REGION1 = "us-west-2"
AWS_BEDROCK_REGION2 = "us-east-1"

# The default LLM used by BedrockHandler
# Note that, when building an instance of BedrockHandler, a region prefix is added
# E.g. if the region is 'us', a 'us.' prefix is added to the model name.
# This is necessary to handle cross-region inference, which we use as a failover mechanism
# anthropic.claude-sonnet-4-20250514-v1:0
LLM_DEFAULT_MODEL = "anthropic.claude-3-7-sonnet-20250219-v1:0"

### --- Central Guidance Configuration --- ###

# This LLM determines if the user query should be enriched
# by the central guidance, or not.
LLM_INDEX_ROUTER = "anthropic.claude-3-5-haiku-20241022-v1:0"

# This LLM takes a user's message and returns a set of OpenSearch queries
LLM_OPENSEARCH_QUERY_GENERATOR = "anthropic.claude-3-7-sonnet-20250219-v1:0"

# This LLM determines if the retrieved document chunk
# should be included in the main LLM context.
LLM_CHUNK_REVIEWER = "anthropic.claude-3-5-haiku-20241022-v1:0"


### --- GOV.UK Configuration --- ###

LLM_DOCUMENT_RELEVANCY_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"
LLM_GOV_UK_SEARCH_FOLLOWUP_ASSESMENT = "anthropic.claude-3-5-haiku-20241022-v1:0"

if env_variable("LLM_DEFAULT_MODEL"):
    LLM_DEFAULT_MODEL = env_variable("LLM_DEFAULT_MODEL")

WHITELISTED_URLS = ["https://www.gov.uk"]
BLACKLISTED_URLS = ["https://www.gov.uk/publications"]
WEB_BROWSING_TIMEOUT = 300
GOV_UK_BASE_URL = "https://www.gov.uk"
GOV_UK_SEARCH_MAX_COUNT = 10
