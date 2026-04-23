import os
from pathlib import Path

from dotenv import load_dotenv

# Load from .env.dev in the repo root if it exists
_repo_root = Path(__file__).parent.parent.parent
load_dotenv(_repo_root / ".env.dev")
load_dotenv(_repo_root / ".env", override=False)

BASE_URL = os.getenv("BASE_URL", "https://dev.api.copilot.gcs.civilservice.gov.uk")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://connect.communications.gov.uk")
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "")

# UUID of the "GOV.UK style guide checker" use case on dev.
# Override via STYLE_GUIDE_USE_CASE_UUID in .env/.env.dev if this changes.
if not AUTH_SECRET_KEY:
    raise RuntimeError("AUTH_SECRET_KEY is not set — populate .env.dev or .env before running")

STYLE_GUIDE_USE_CASE_UUID = os.getenv("STYLE_GUIDE_USE_CASE_UUID")

# Browser-like User-Agent required to pass AWS WAF bot control
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
