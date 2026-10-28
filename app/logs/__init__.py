from app.config import settings

from .bugsnag_logger import BugsnagLogger
from .logs_handler import *

__all__ = ["BugsnagLogger"]

BUGSNAG_API_KEY: str | None = settings.bugsnag_api_key.get_secret_value() if settings.bugsnag_api_key else None
BUGSNAG_RELEASE_STAGE = settings.bugsnag_release_stage
DISABLE_BUGSNAG_LOGGING = settings.disable_bugsnag_logging

BUGSNAG_ENABLED = bool(not DISABLE_BUGSNAG_LOGGING and BUGSNAG_API_KEY and BUGSNAG_RELEASE_STAGE)
