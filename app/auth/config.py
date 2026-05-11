from app.config import settings

AUTH_TOKEN: str = settings.auth_secret_key.get_secret_value()
AUTH_TOKEN_2: str | None = settings.auth_secret_key_2.get_secret_value() if settings.auth_secret_key_2 else None

DEFAULT_AUTH_TOKEN = None
DEFAULT_USER_KEY_UUID = None
