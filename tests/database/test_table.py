import logging

import pytest

from app.database.database_exception import DatabaseError
from app.database.table import MessageTable

logger = logging.getLogger(__name__)


class TestDatabaseException:
    def test_bad_get_by_id(self):
        """Tests if a bad ID submitted to the Table.get method will raise a DatabaseException"""

        # Choose a very large integer ID that is unlikely to return a record
        # in the local database
        bad_id = 999999999999
        message_table = MessageTable()

        with pytest.raises(DatabaseError) as exc_info:
            message_table.get(bad_id)

        assert exc_info.value.code, "No error code was given with the DataBase exception"
        assert exc_info.value.message, "No message was given with the DataBase exception"
        logger.info(f"DatabaseException was raised successfully: {exc_info.value.code=} {exc_info.value.message=}")

    def test_bad_get_by_uuid(self):
        """Tests if a bad UUID submitted to get_by_uuid will raise a DatabaseException"""
        logger.info("Starting test that a bad UUID submitted to get_by_uuid will raise a DatabaseException")

        # This is a randomly generated UUID.
        bad_uuid = "e7d260f0-c4ba-4c04-9a42-72a75e2e01c6"

        logger.info(f"{bad_uuid=}")
        message_table = MessageTable()

        with pytest.raises(DatabaseError) as exc_info:
            message_table.get_by_uuid(bad_uuid)

        assert exc_info.value.code, "No error code was given with the DataBase exception"
        assert exc_info.value.message, "No message was given with the DataBase exception"
        logger.info(f"DatabaseException was raised successfully: {exc_info.value.code=} {exc_info.value.message=}")
