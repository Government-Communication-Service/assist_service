from enum import Enum, auto


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


class DatabaseError(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
