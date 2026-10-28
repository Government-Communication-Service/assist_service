"""Tests for themes_use_cases/lib.py functions."""

import logging
from datetime import datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.responses import Response
from starlette.status import HTTP_404_NOT_FOUND

from app.chat.schemas import SuccessResponse, ThemeResponse, UseCaseResponse
from app.database.db_operations import DbOperations
from app.database.table import DatabaseError
from app.themes_use_cases.lib import (
    create_theme,
    soft_delete_theme,
    soft_delete_use_case,
    update_theme,
    update_use_case,
)
from app.themes_use_cases.schemas import ThemeInput, UseCaseInputPut

logger = logging.getLogger(__name__)


@pytest.fixture
def theme_input():
    """Sample ThemeInput for testing."""
    return ThemeInput(
        title="Test Theme",
        subtitle="Test Subtitle",
        position=1,
    )


@pytest.fixture
def mock_theme():
    """Mock theme object for testing."""
    now = datetime.now()
    mock = Mock()
    mock.uuid = uuid4()
    mock.id = 1
    mock.title = "Test Theme"
    mock.subtitle = "Test Subtitle"
    mock.position = 1
    mock.client_response = Mock(return_value={
        "uuid": str(mock.uuid),
        "title": "Test Theme",
        "subtitle": "Test Subtitle",
        "position": 1,
        "created_at": now,
        "updated_at": now,
    })
    return mock


@pytest.fixture
def mock_use_case():
    """Mock use case object for testing."""
    now = datetime.now()
    mock = Mock()
    mock.uuid = uuid4()
    mock.id = 1
    mock.title = "Test Use Case"
    mock.instruction = "Test Instruction"
    mock.user_input_form = "{}"
    mock.position = 1
    mock.theme_id = 1
    mock.client_response = Mock(return_value={
        "uuid": str(mock.uuid),
        "title": "Test Use Case",
        "instruction": "Test Instruction",
        "user_input_form": "{}",
        "position": 1,
        "created_at": now,
        "updated_at": now,
    })
    return mock


class TestCreateTheme:
    """Tests for create_theme function."""

    async def test_create_theme_success(self, db_session, theme_input, mock_theme, mocker):
        """Should return ThemeResponse on success."""
        mocker.patch.object(
            DbOperations,
            "theme_create_or_revive",
            new_callable=AsyncMock,
            return_value=mock_theme
        )
        mocker.patch.object(
            DbOperations,
            "get_theme",
            new_callable=AsyncMock,
            return_value=mock_theme
        )

        result = await create_theme(db_session, theme_input)

        assert isinstance(result, ThemeResponse)
        assert result.title == "Test Theme"

    async def test_create_theme_not_found_after_creation(self, db_session, theme_input, mock_theme, mocker):
        """Should return 404 if theme cannot be retrieved after creation."""
        mocker.patch.object(
            DbOperations,
            "theme_create_or_revive",
            new_callable=AsyncMock,
            return_value=mock_theme
        )
        mocker.patch.object(
            DbOperations,
            "get_theme",
            new_callable=AsyncMock,
            return_value=None
        )

        result = await create_theme(db_session, theme_input)

        assert isinstance(result, Response)
        assert result.status_code == HTTP_404_NOT_FOUND


class TestUpdateTheme:
    """Tests for update_theme function."""

    async def test_update_theme_success(self, db_session, theme_input, mock_theme, mocker):
        """Should return ThemeResponse on success."""
        theme_uuid = uuid4()
        mocker.patch.object(
            DbOperations,
            "theme_update",
            new_callable=AsyncMock,
            return_value=mock_theme
        )

        result = await update_theme(db_session, theme_uuid, theme_input)

        assert isinstance(result, ThemeResponse)
        assert result.title == "Test Theme"

    async def test_update_theme_not_found(self, db_session, theme_input, mocker):
        """Should return 404 if theme not found."""
        theme_uuid = uuid4()
        mocker.patch.object(
            DbOperations,
            "theme_update",
            new_callable=AsyncMock,
            # Simulate theme not found - theme_update returns None when no matching theme exists
            return_value=None
        )

        result = await update_theme(db_session, theme_uuid, theme_input)

        assert isinstance(result, Response)
        assert result.status_code == HTTP_404_NOT_FOUND


class TestSoftDeleteTheme:
    """Tests for soft_delete_theme function."""

    async def test_soft_delete_theme_success(self, db_session, mock_theme, mocker):
        """Should return SuccessResponse on success."""
        theme_uuid = uuid4()
        mocker.patch.object(
            DbOperations,
            "get_theme",
            new_callable=AsyncMock,
            return_value=mock_theme
        )
        mocker.patch.object(
            DbOperations,
            "theme_soft_delete_by_uuid",
            new_callable=AsyncMock,
            return_value=None
        )

        result = await soft_delete_theme(db_session, theme_uuid)

        assert isinstance(result, SuccessResponse)

    async def test_soft_delete_theme_not_found(self, db_session, mocker):
        """Should return 404 if theme not found."""
        theme_uuid = uuid4()
        mocker.patch.object(
            DbOperations,
            "get_theme",
            new_callable=AsyncMock,
            return_value=None
        )

        result = await soft_delete_theme(db_session, theme_uuid)

        assert isinstance(result, Response)
        assert result.status_code == HTTP_404_NOT_FOUND


class TestUpdateUseCase:
    """Tests for update_use_case function."""

    async def test_update_use_case_success(self, db_session, mock_theme, mock_use_case, mocker):
        """Should return UseCaseResponse on success."""
        theme_uuid = uuid4()
        use_case_uuid = uuid4()

        use_case_input = UseCaseInputPut(
            theme_uuid=mock_theme.uuid,
            title="Updated Use Case",
            instruction="Updated Instruction",
            user_input_form="{}",
            position=1,
        )

        mocker.patch.object(
            DbOperations,
            "use_case_get_by_uuid_no_theme",
            new_callable=AsyncMock,
            return_value=mock_use_case
        )
        mocker.patch.object(
            DbOperations,
            "get_theme",
            new_callable=AsyncMock,
            return_value=mock_theme
        )
        mocker.patch.object(
            DbOperations,
            "use_case_update_by_uuid",
            new_callable=AsyncMock,
            return_value=mock_use_case
        )

        result = await update_use_case(db_session, theme_uuid, use_case_uuid, use_case_input)

        assert isinstance(result, UseCaseResponse)

    async def test_update_use_case_not_found(self, db_session, mocker):
        """Should return 404 if use case not found."""
        theme_uuid = uuid4()
        use_case_uuid = uuid4()

        use_case_input = UseCaseInputPut(
            theme_uuid=uuid4(),
            title="Updated Use Case",
            instruction="Updated Instruction",
            user_input_form="{}",
            position=1,
        )

        mocker.patch.object(
            DbOperations,
            "use_case_get_by_uuid_no_theme",
            new_callable=AsyncMock,
            return_value=None
        )

        result = await update_use_case(db_session, theme_uuid, use_case_uuid, use_case_input)

        assert isinstance(result, Response)
        assert result.status_code == HTTP_404_NOT_FOUND

    async def test_update_use_case_wrong_theme(self, db_session, mock_theme, mock_use_case, mocker):
        """Should raise DatabaseError if use case belongs to different theme."""
        theme_uuid = uuid4()
        use_case_uuid = uuid4()
        mock_use_case.theme_id = 999

        use_case_input = UseCaseInputPut(
            theme_uuid=mock_theme.uuid,
            title="Updated Use Case",
            instruction="Updated Instruction",
            user_input_form="{}",
            position=1,
        )

        mocker.patch.object(
            DbOperations,
            "use_case_get_by_uuid_no_theme",
            new_callable=AsyncMock,
            return_value=mock_use_case
        )
        mocker.patch.object(
            DbOperations,
            "get_theme",
            new_callable=AsyncMock,
            return_value=mock_theme
        )

        with pytest.raises(DatabaseError):
            await update_use_case(db_session, theme_uuid, use_case_uuid, use_case_input)


class TestSoftDeleteUseCase:
    """Tests for soft_delete_use_case function."""

    async def test_soft_delete_use_case_success(self, db_session, mock_theme, mock_use_case, mocker):
        """Should return SuccessResponse on success."""
        theme_uuid = uuid4()
        use_case_uuid = uuid4()

        mocker.patch.object(
            DbOperations,
            "use_case_get_by_uuid_no_theme",
            new_callable=AsyncMock,
            return_value=mock_use_case
        )
        mocker.patch.object(
            DbOperations,
            "get_theme",
            new_callable=AsyncMock,
            return_value=mock_theme
        )
        mocker.patch.object(
            DbOperations,
            "use_case_soft_delete_by_uuid",
            new_callable=AsyncMock,
            return_value=None
        )

        result = await soft_delete_use_case(db_session, theme_uuid, use_case_uuid)

        assert isinstance(result, SuccessResponse)

    async def test_soft_delete_use_case_not_found(self, db_session, mocker):
        """Should return 404 if use case not found."""
        theme_uuid = uuid4()
        use_case_uuid = uuid4()

        mocker.patch.object(
            DbOperations,
            "use_case_get_by_uuid_no_theme",
            new_callable=AsyncMock,
            return_value=None
        )

        result = await soft_delete_use_case(db_session, theme_uuid, use_case_uuid)

        assert isinstance(result, Response)
        assert result.status_code == HTTP_404_NOT_FOUND

    async def test_soft_delete_use_case_wrong_theme(self, db_session, mock_theme, mock_use_case, mocker):
        """Should raise DatabaseError if use case belongs to different theme."""
        theme_uuid = uuid4()
        use_case_uuid = uuid4()
        mock_use_case.theme_id = 999

        mocker.patch.object(
            DbOperations,
            "use_case_get_by_uuid_no_theme",
            new_callable=AsyncMock,
            return_value=mock_use_case
        )
        mocker.patch.object(
            DbOperations,
            "get_theme",
            new_callable=AsyncMock,
            return_value=mock_theme
        )

        with pytest.raises(DatabaseError):
            await soft_delete_use_case(db_session, theme_uuid, use_case_uuid)

