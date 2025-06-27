import os

from fastapi import Header

from app.config import IS_DEV, env_variable

HEADER_DEFAULTS = {
    # 'include_in_schema': os.getenv("SHOW_HEADER_PARAMS_IN_DOCS", True) == True,
    "include_in_schema": True,
}


def param_checker(env_value, other_value=None):
    if env_value:
        if other_value:
            return other_value
        return env_value

    return ...


def header_param(alias, default, description):
    return Header(default=default, alias=alias, description=description, **HEADER_DEFAULTS)


class ApiParams:
    USER_UUID = param_checker(os.getenv("TEST_USER_UUID"))
    USER_PROMPT_UUID = param_checker(os.getenv("TEST_USER_PROMPT_UUID"))
    SESSION_UUID = param_checker(os.getenv("TEST_SESSION_UUID"))
    AUTH_TOKEN = ...

    if IS_DEV and env_variable("BYPASS_AUTH_VALIDATOR"):
        AUTH_TOKEN = "BYPASSED"


def header_documentation(main: str, extra: str):
    return " ".join([main, extra if IS_DEV else ""])


class ApiConfig:
    SESSION_AUTH_ALIAS = "Session-Auth"
    AUTH_TOKEN_ALIAS = "Auth-Token"
    USER_KEY_UUID_ALIAS = "User-Key-UUID"
    USER_GROUPS_ALIAS = "User-Groups"

    # Header objects referencing class attributes for aliasing
    SESSION_AUTH = header_param(
        default=ApiParams.SESSION_UUID,
        alias=SESSION_AUTH_ALIAS,
        description=header_documentation(
            "The session token generated through the backend `/session` endpoint, to be used across the other "
            "Copilot endpoints.",
            "This can be disabled in the dev environment by setting the `BYPASS_SESSION_VALIDATOR` environment "
            "variable to `True` (`BYPASS_SESSION_VALIDATOR=True`).",
        ),
    )

    AUTH_TOKEN = header_param(
        default=ApiParams.AUTH_TOKEN,
        alias=AUTH_TOKEN_ALIAS,
        description=header_documentation(
            "The authentication token generated on the client side.",
            "This can be disabled in the dev environment by setting the `BYPASS_AUTH_VALIDATOR` environment "
            "variable to `True` (`BYPASS_AUTH_VALIDATOR=True`).",
        ),
    )

    USER_KEY_UUID = header_param(
        default=ApiParams.USER_UUID,
        alias=USER_KEY_UUID_ALIAS,
        description="""The user UUID as taken from the official GCS database, fetched from the GCS client.
        This is not verified against the GCS database by the backend due to security reasons, but is stored across
        the calls in order for better segmentation of the data by the analytics team.""",
    )

    USER_GROUPS = header_param(
        default=os.getenv("TEST_USER_GROUPS", ""),
        alias=USER_GROUPS_ALIAS,
        description="A comma-separated list of groups assigned to the user from GCS Connect which can be used to "
        "filter queries.",
    )

    PARAMS = ApiParams
