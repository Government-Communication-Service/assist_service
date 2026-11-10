from enum import Enum, auto

from app.exceptions.app_error import AppError


class DatabaseExceptionErrorCode(Enum):
    CREATE_ERROR = auto()
    CREATE_OR_REVIVE_ERROR = auto()
    UPDATE_ERROR = auto()
    UPDATE_BY_UUID_ERROR = auto()
    GET_ERROR = auto()
    DELETE_ERROR = auto()
    SOFT_DELETE_BY_UUID_ERROR = auto()
    CREATE_BATCH_ERROR = auto()
    GET_ONE_BY_ERROR = auto()
    GET_BY_ERROR = auto()
    GET_BY_UUID_ERROR = auto()
    MOST_RECENT_ERROR = auto()
    UPSERT_BY_UUID_ERROR = auto()
    FETCH_ALL_ERROR = auto()
    FETCH_ALL_ORDERED_BY_POSITION_OR_ID_ERROR = auto()
    ORDER_BY_MOST_RECENT_ERROR = auto()
    DELETE_ALL_ERROR = auto()
    EDIT_ALL_ERROR = auto()
    UPSERT_ERROR = auto()
    USE_CASE_NOT_UNDER_THIS_THEME_ERROR = auto()
    DEFAULT = auto()


class DatabaseError(AppError):
    """Database operation failure. Not-found codes map to 404; every other code is a 500."""

    def __init__(self, code: DatabaseExceptionErrorCode, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def expected(self) -> bool:
        return self.code in (
            DatabaseExceptionErrorCode.GET_BY_UUID_ERROR,
            DatabaseExceptionErrorCode.USE_CASE_NOT_UNDER_THIS_THEME_ERROR,
        )

    @property
    def status_code(self) -> int:
        return 404 if self.expected else 500

    @property
    def detail(self) -> str:
        if self.code == DatabaseExceptionErrorCode.GET_BY_UUID_ERROR:
            return "Record not found"
        if self.code == DatabaseExceptionErrorCode.USE_CASE_NOT_UNDER_THIS_THEME_ERROR:
            return self.message
        return AppError.detail
