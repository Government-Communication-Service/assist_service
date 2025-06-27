import os

from .bugsnag_logger import BugsnagLogger
from .logs_handler import *

__all__ = ["BugsnagLogger"]

BUGSNAG_API_KEY = os.getenv("BUGSNAG_API_KEY")
BUGSNAG_RELEASE_STAGE = os.getenv("BUGSNAG_RELEASE_STAGE")
DISABLE_BUGSNAG_LOGGING = os.getenv("DISABLE_BUGSNAG_LOGGING", False)

BUGSNAG_ENABLED = bool(not DISABLE_BUGSNAG_LOGGING and BUGSNAG_API_KEY and BUGSNAG_RELEASE_STAGE)
