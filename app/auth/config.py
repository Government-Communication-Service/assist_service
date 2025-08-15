import os

AUTH_TOKEN = os.getenv("AUTH_SECRET_KEY")
AUTH_TOKEN_2 = os.getenv("AUTH_SECRET_KEY_2")

DEFAULT_AUTH_TOKEN = None  # AUTH_TOKEN if IS_DEV else None
DEFAULT_USER_KEY_UUID = None  # os.getenv("DEFAULT_USER_KEY_UUID") if IS_DEV else None
