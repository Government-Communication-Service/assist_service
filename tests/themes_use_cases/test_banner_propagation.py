"""Tests for banner propagation from use cases to parent themes.

Tests for banner propagation from use cases to parent themes:
- Theme with no banner receives propagated banner from use case.
- Theme banner expiry is extended when use case has a later expiry.
- Theme banner expiry is NOT shortened when use case has an earlier expiry.
"""

import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.chat.schemas import UseCaseResponse
from app.database.db_operations import DbOperations
from app.themes_use_cases.lib import (
    _propagate_banner_to_theme,
    create_use_case,
    update_use_case,
)
from app.themes_use_cases.schemas import UseCaseInputPost, UseCaseInputPut

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def theme_uuid():
    return uuid4()


@pytest.fixture
def use_case_uuid():
    return uuid4()


@pytest.fixture
def make_mock_theme():
    """Factory to create a mock theme with configurable banner settings."""

    def _make(
        show_update_banner=False,
        banner_type=None,
        banner_until=None,
        theme_id=1,
    ):
        now = datetime.now()
        mock = Mock()
        mock.uuid = uuid4()
        mock.id = theme_id
        mock.title = "Banner Test Theme"
        mock.subtitle = "banner theme"
        mock.position = 0
        mock.show_update_banner = show_update_banner
        mock.banner_type = banner_type
        mock.banner_until = banner_until
        mock.client_response = Mock(
            return_value={
                "uuid": str(mock.uuid),
                "title": mock.title,
                "subtitle": mock.subtitle,
                "position": mock.position,
                "show_update_banner": show_update_banner,
                "banner_type": banner_type,
                "banner_until": banner_until,
                "created_at": now,
                "updated_at": now,
            }
        )
        return mock

    return _make


@pytest.fixture
def make_mock_use_case():
    """Factory to create a mock use case."""

    def _make(theme_id=1):
        now = datetime.now()
        mock = Mock()
        mock.uuid = uuid4()
        mock.id = 1
        mock.title = "Banner test use-case"
        mock.instruction = "This is banner test use case"
        mock.user_input_form = "N/A"
        mock.position = 0
        mock.theme_id = theme_id
        mock.client_response = Mock(
            return_value={
                "uuid": str(mock.uuid),
                "title": mock.title,
                "instruction": mock.instruction,
                "user_input_form": mock.user_input_form,
                "position": mock.position,
                "show_update_banner": True,
                "banner_type": "updated",
                "banner_until": None,
                "created_at": now,
                "updated_at": now,
            }
        )
        return mock

    return _make


# ---------------------------------------------------------------------------
# Theme has no banner, Should propagate
# ---------------------------------------------------------------------------


class TestThemeNoBannerPropagation:
    """Theme has no banner,  banner should propagate from use case."""

    async def test_propagate_banner_to_theme_with_no_banner(self, db_session, make_mock_theme, mocker):
        """_propagate_banner_to_theme should set theme banner when theme has none."""
        theme = make_mock_theme(show_update_banner=False, banner_type=None, banner_until=None)
        use_case_banner_type = "updated"
        use_case_banner_until = datetime.now() + timedelta(days=30)

        mock_theme_update = mocker.patch.object(
            DbOperations,
            "theme_update",
            new_callable=AsyncMock,
            return_value=theme,
        )

        await _propagate_banner_to_theme(
            db_session=db_session,
            theme=theme,
            use_case_banner_type=use_case_banner_type,
            use_case_banner_until=use_case_banner_until,
        )

        # Verify theme_update was called
        mock_theme_update.assert_called_once()
        call_kwargs = mock_theme_update.call_args
        theme_input = call_kwargs.kwargs["theme_input"]
        assert theme_input.show_update_banner is True
        assert theme_input.banner_type == "updated"
        assert theme_input.banner_until == use_case_banner_until

    async def test_update_use_case_propagates_banner_to_theme_with_no_banner(
        self, db_session, theme_uuid, use_case_uuid, make_mock_theme, make_mock_use_case, mocker
    ):
        """Full update_use_case flow: theme has no banner, use case enables banner, theme gets banner."""
        theme = make_mock_theme(show_update_banner=False, banner_type=None, banner_until=None)
        use_case = make_mock_use_case(theme_id=theme.id)
        use_case_banner_until = datetime.now() + timedelta(days=30)

        use_case_input = UseCaseInputPut(
            theme_uuid=theme.uuid,
            title="Banner test use-case - Test A",
            instruction="This is banner test use case",
            user_input_form="N/A",
            position=0,
            show_update_banner=True,
            banner_type="updated",
            banner_until=use_case_banner_until,
        )

        mocker.patch.object(
            DbOperations,
            "use_case_get_by_uuid_no_theme",
            new_callable=AsyncMock,
            return_value=use_case,
        )
        mocker.patch.object(
            DbOperations,
            "get_theme",
            new_callable=AsyncMock,
            return_value=theme,
        )
        mocker.patch.object(
            DbOperations,
            "use_case_update_by_uuid",
            new_callable=AsyncMock,
            return_value=use_case,
        )
        mock_theme_update = mocker.patch.object(
            DbOperations,
            "theme_update",
            new_callable=AsyncMock,
            return_value=theme,
        )

        result = await update_use_case(db_session, theme_uuid, use_case_uuid, use_case_input)

        assert isinstance(result, UseCaseResponse)
        # Verify propagation happened
        mock_theme_update.assert_called_once()
        theme_input = mock_theme_update.call_args.kwargs["theme_input"]
        assert theme_input.show_update_banner is True
        assert theme_input.banner_type == "updated"
        assert theme_input.banner_until == use_case_banner_until


# ---------------------------------------------------------------------------
# Extend expiry when use case has later date
# ---------------------------------------------------------------------------


class TestExtendBannerExpiry:
    """Theme has 'updated' banner. Use case sends a later expiry, theme should extend."""

    async def test_propagate_extends_expiry(self, db_session, make_mock_theme, mocker):
        """Theme expiry Apr 15, use case expiry Sep 2, theme should extend to Sep 2."""
        theme_expiry = datetime(2026, 4, 15, 12, 0, 0)
        use_case_expiry = datetime(2026, 9, 2, 12, 0, 0)

        theme = make_mock_theme(
            show_update_banner=True,
            banner_type="updated",
            banner_until=theme_expiry,
        )

        mock_theme_update = mocker.patch.object(
            DbOperations,
            "theme_update",
            new_callable=AsyncMock,
            return_value=theme,
        )

        await _propagate_banner_to_theme(
            db_session=db_session,
            theme=theme,
            use_case_banner_type="updated",
            use_case_banner_until=use_case_expiry,
        )

        mock_theme_update.assert_called_once()
        theme_input = mock_theme_update.call_args.kwargs["theme_input"]
        assert theme_input.show_update_banner is True
        assert theme_input.banner_type == "updated"
        assert theme_input.banner_until == use_case_expiry

    async def test_update_use_case_extends_theme_expiry(
        self, db_session, theme_uuid, use_case_uuid, make_mock_theme, make_mock_use_case, mocker
    ):
        """Full flow: theme 'updated' expiry Apr 15, use case sends Sep 2, theme extends."""
        theme_expiry = datetime(2026, 4, 15, 12, 0, 0)
        use_case_expiry = datetime(2026, 9, 2, 12, 0, 0)

        theme = make_mock_theme(
            show_update_banner=True,
            banner_type="updated",
            banner_until=theme_expiry,
        )
        use_case = make_mock_use_case(theme_id=theme.id)

        use_case_input = UseCaseInputPut(
            theme_uuid=theme.uuid,
            title="Banner test use-case - Test C1",
            instruction="This is banner test use case",
            user_input_form="N/A",
            position=0,
            show_update_banner=True,
            banner_type="updated",
            banner_until=use_case_expiry,
        )

        mocker.patch.object(
            DbOperations,
            "use_case_get_by_uuid_no_theme",
            new_callable=AsyncMock,
            return_value=use_case,
        )
        mocker.patch.object(
            DbOperations,
            "get_theme",
            new_callable=AsyncMock,
            return_value=theme,
        )
        mocker.patch.object(
            DbOperations,
            "use_case_update_by_uuid",
            new_callable=AsyncMock,
            return_value=use_case,
        )
        mock_theme_update = mocker.patch.object(
            DbOperations,
            "theme_update",
            new_callable=AsyncMock,
            return_value=theme,
        )

        result = await update_use_case(db_session, theme_uuid, use_case_uuid, use_case_input)

        assert isinstance(result, UseCaseResponse)
        mock_theme_update.assert_called_once()
        theme_input = mock_theme_update.call_args.kwargs["theme_input"]
        assert theme_input.show_update_banner is True
        assert theme_input.banner_type == "updated"
        assert theme_input.banner_until == use_case_expiry


# ---------------------------------------------------------------------------
# Don't shorten expiry when use case has earlier date
# ---------------------------------------------------------------------------


class TestDontShortenBannerExpiry:
    """Theme has 'updated' banner. Use case sends earlier expiry, theme should NOT shorten."""

    async def test_propagate_does_not_shorten_expiry(self, db_session, make_mock_theme, mocker):
        """Theme expiry Jun 1, use case expiry Mar 1, theme should stay Jun 1 (no update)."""
        theme_expiry = datetime(2026, 6, 1, 12, 0, 0)
        use_case_expiry = datetime(2026, 3, 1, 12, 0, 0)

        theme = make_mock_theme(
            show_update_banner=True,
            banner_type="updated",
            banner_until=theme_expiry,
        )

        mock_theme_update = mocker.patch.object(
            DbOperations,
            "theme_update",
            new_callable=AsyncMock,
            return_value=theme,
        )

        await _propagate_banner_to_theme(
            db_session=db_session,
            theme=theme,
            use_case_banner_type="updated",
            use_case_banner_until=use_case_expiry,
        )

        # theme_update should NOT be called — expiry is not extended
        mock_theme_update.assert_not_called()

    async def test_update_use_case_does_not_shorten_theme_expiry(
        self, db_session, theme_uuid, use_case_uuid, make_mock_theme, make_mock_use_case, mocker
    ):
        """Full flow: theme 'updated' expiry Jun 1, use case sends Mar 1, theme stays Jun 1."""
        theme_expiry = datetime.now() + timedelta(days=60)
        use_case_expiry = datetime.now() + timedelta(days=10)

        theme = make_mock_theme(
            show_update_banner=True,
            banner_type="updated",
            banner_until=theme_expiry,
        )
        use_case = make_mock_use_case(theme_id=theme.id)

        use_case_input = UseCaseInputPut(
            theme_uuid=theme.uuid,
            title="Banner test use-case - Test C2",
            instruction="This is banner test use case",
            user_input_form="N/A",
            position=0,
            show_update_banner=True,
            banner_type="updated",
            banner_until=use_case_expiry,
        )

        mocker.patch.object(
            DbOperations,
            "use_case_get_by_uuid_no_theme",
            new_callable=AsyncMock,
            return_value=use_case,
        )
        mocker.patch.object(
            DbOperations,
            "get_theme",
            new_callable=AsyncMock,
            return_value=theme,
        )
        mocker.patch.object(
            DbOperations,
            "use_case_update_by_uuid",
            new_callable=AsyncMock,
            return_value=use_case,
        )
        mock_theme_update = mocker.patch.object(
            DbOperations,
            "theme_update",
            new_callable=AsyncMock,
            return_value=theme,
        )

        result = await update_use_case(db_session, theme_uuid, use_case_uuid, use_case_input)

        assert isinstance(result, UseCaseResponse)
        # Propagation should NOT call theme_update since expiry is earlier
        mock_theme_update.assert_not_called()


# ---------------------------------------------------------------------------
# Additional edge cases for _propagate_banner_to_theme
# ---------------------------------------------------------------------------


class TestPropagationEdgeCases:
    """Additional edge cases for banner propagation logic."""

    async def test_theme_new_banner_not_overridden_by_updated(self, db_session, make_mock_theme, mocker):
        """Theme has 'new' banner. Use case has 'updated', theme should NOT be overridden."""
        theme = make_mock_theme(
            show_update_banner=True,
            banner_type="new",
            banner_until=datetime(2026, 5, 1, 12, 0, 0),
        )

        mock_theme_update = mocker.patch.object(
            DbOperations,
            "theme_update",
            new_callable=AsyncMock,
            return_value=theme,
        )

        await _propagate_banner_to_theme(
            db_session=db_session,
            theme=theme,
            use_case_banner_type="updated",
            use_case_banner_until=datetime(2026, 6, 1, 12, 0, 0),
        )

        # Should NOT update — "new" is higher priority than "updated"
        mock_theme_update.assert_not_called()

    async def test_theme_new_banner_expiry_extended_by_new_use_case(self, db_session, make_mock_theme, mocker):
        """Theme has 'new' banner expiry Apr 1, use case 'new' with expiry Jun 1, theme should extend."""
        theme_expiry = datetime(2026, 4, 1, 12, 0, 0)
        use_case_expiry = datetime(2026, 6, 1, 12, 0, 0)

        theme = make_mock_theme(
            show_update_banner=True,
            banner_type="new",
            banner_until=theme_expiry,
        )

        mock_theme_update = mocker.patch.object(
            DbOperations,
            "theme_update",
            new_callable=AsyncMock,
            return_value=theme,
        )

        await _propagate_banner_to_theme(
            db_session=db_session,
            theme=theme,
            use_case_banner_type="new",
            use_case_banner_until=use_case_expiry,
        )

        mock_theme_update.assert_called_once()
        theme_input = mock_theme_update.call_args.kwargs["theme_input"]
        assert theme_input.banner_type == "new"
        assert theme_input.banner_until == use_case_expiry

    async def test_theme_new_banner_expiry_not_shortened_by_new_use_case(self, db_session, make_mock_theme, mocker):
        """Theme has 'new' banner expiry Jun 1, use case 'new' with expiry Apr 1, theme should NOT shorten."""
        theme_expiry = datetime(2026, 6, 1, 12, 0, 0)
        use_case_expiry = datetime(2026, 4, 1, 12, 0, 0)

        theme = make_mock_theme(
            show_update_banner=True,
            banner_type="new",
            banner_until=theme_expiry,
        )

        mock_theme_update = mocker.patch.object(
            DbOperations,
            "theme_update",
            new_callable=AsyncMock,
            return_value=theme,
        )

        await _propagate_banner_to_theme(
            db_session=db_session,
            theme=theme,
            use_case_banner_type="new",
            use_case_banner_until=use_case_expiry,
        )

        mock_theme_update.assert_not_called()

    async def test_no_propagation_when_use_case_banner_disabled(
        self, db_session, theme_uuid, use_case_uuid, make_mock_theme, make_mock_use_case, mocker
    ):
        """Use case banner is disabled, should NOT propagate to theme."""
        theme = make_mock_theme(show_update_banner=False)
        use_case = make_mock_use_case(theme_id=theme.id)

        use_case_input = UseCaseInputPut(
            theme_uuid=theme.uuid,
            title="No banner use case",
            instruction="No banner",
            user_input_form="N/A",
            position=0,
            show_update_banner=False,
        )

        mocker.patch.object(
            DbOperations,
            "use_case_get_by_uuid_no_theme",
            new_callable=AsyncMock,
            return_value=use_case,
        )
        mocker.patch.object(
            DbOperations,
            "get_theme",
            new_callable=AsyncMock,
            return_value=theme,
        )
        mocker.patch.object(
            DbOperations,
            "use_case_update_by_uuid",
            new_callable=AsyncMock,
            return_value=use_case,
        )
        mock_theme_update = mocker.patch.object(
            DbOperations,
            "theme_update",
            new_callable=AsyncMock,
            return_value=theme,
        )

        result = await update_use_case(db_session, theme_uuid, use_case_uuid, use_case_input)

        assert isinstance(result, UseCaseResponse)
        # No propagation since use case banner is disabled
        mock_theme_update.assert_not_called()


# ---------------------------------------------------------------------------
# Create use case banner propagation (POST)
# ---------------------------------------------------------------------------


class TestCreateUseCaseBannerPropagation:
    """Banner propagation when creating a new use case (POST, not PUT)."""

    def _mock_create_flow(self, mocker, theme, use_case):
        """Set up common mocks for create_use_case flow."""
        mocker.patch.object(
            DbOperations,
            "get_theme",
            new_callable=AsyncMock,
            return_value=theme,
        )
        mocker.patch.object(
            DbOperations,
            "use_case_create_or_revive",
            new_callable=AsyncMock,
            return_value=use_case,
        )
        mocker.patch.object(
            DbOperations,
            "get_use_case",
            new_callable=AsyncMock,
            return_value=(theme, use_case),
        )
        mock_theme_update = mocker.patch.object(
            DbOperations,
            "theme_update",
            new_callable=AsyncMock,
            return_value=theme,
        )
        return mock_theme_update

    async def test_create_use_case_propagates_new_banner_to_theme_with_no_banner(
        self, db_session, make_mock_theme, make_mock_use_case, mocker
    ):
        """Theme has no banner. Creating use case with 'new' banner propagates to theme."""
        theme = make_mock_theme(show_update_banner=False, banner_type=None, banner_until=None)
        use_case = make_mock_use_case(theme_id=theme.id)
        use_case_expiry = datetime.now() + timedelta(days=30)

        use_case_input = UseCaseInputPost(
            title="New predefined prompt - Test A",
            instruction="This is a new predefined prompt",
            user_input_form="N/A",
            position=99,
            show_update_banner=True,
            banner_type="new",
            banner_until=use_case_expiry,
        )

        mock_theme_update = self._mock_create_flow(mocker, theme, use_case)

        result = await create_use_case(db_session, theme.uuid, use_case_input)

        assert isinstance(result, UseCaseResponse)
        mock_theme_update.assert_called_once()
        theme_input = mock_theme_update.call_args.kwargs["theme_input"]
        assert theme_input.show_update_banner is True
        assert theme_input.banner_type == "new"
        assert theme_input.banner_until == use_case_expiry

    async def test_create_use_case_extends_theme_new_banner_expiry(
        self, db_session, make_mock_theme, make_mock_use_case, mocker
    ):
        """Theme has 'new' banner expiry Apr 15, new use case 'new' expiry Jun 1, theme should extend."""
        theme_expiry = datetime(2026, 4, 15, 12, 0, 0)
        use_case_expiry = datetime(2026, 6, 1, 12, 0, 0)

        theme = make_mock_theme(
            show_update_banner=True,
            banner_type="new",
            banner_until=theme_expiry,
        )
        use_case = make_mock_use_case(theme_id=theme.id)

        use_case_input = UseCaseInputPost(
            title="New predefined prompt - Test B",
            instruction="This is a new predefined prompt",
            user_input_form="N/A",
            position=99,
            show_update_banner=True,
            banner_type="new",
            banner_until=use_case_expiry,
        )

        mock_theme_update = self._mock_create_flow(mocker, theme, use_case)

        result = await create_use_case(db_session, theme.uuid, use_case_input)

        assert isinstance(result, UseCaseResponse)
        mock_theme_update.assert_called_once()
        theme_input = mock_theme_update.call_args.kwargs["theme_input"]
        assert theme_input.banner_type == "new"
        assert theme_input.banner_until == use_case_expiry

    async def test_create_use_case_does_not_shorten_theme_new_banner_expiry(
        self, db_session, make_mock_theme, make_mock_use_case, mocker
    ):
        """Theme has 'new' banner expiry Jun 1, new use case 'new' expiry Mar 1, theme should NOT shorten."""
        theme_expiry = datetime.now() + timedelta(days=60)
        use_case_expiry = datetime.now() + timedelta(days=10)

        theme = make_mock_theme(
            show_update_banner=True,
            banner_type="new",
            banner_until=theme_expiry,
        )
        use_case = make_mock_use_case(theme_id=theme.id)

        use_case_input = UseCaseInputPost(
            title="New predefined prompt - Test C",
            instruction="This is a new predefined prompt",
            user_input_form="N/A",
            position=99,
            show_update_banner=True,
            banner_type="new",
            banner_until=use_case_expiry,
        )

        mock_theme_update = self._mock_create_flow(mocker, theme, use_case)

        result = await create_use_case(db_session, theme.uuid, use_case_input)

        assert isinstance(result, UseCaseResponse)
        # Should NOT shorten expiry
        mock_theme_update.assert_not_called()

    async def test_create_use_case_no_propagation_when_banner_disabled(
        self, db_session, make_mock_theme, make_mock_use_case, mocker
    ):
        """Creating use case with banner disabled, should NOT propagate to theme."""
        theme = make_mock_theme(show_update_banner=False)
        use_case = make_mock_use_case(theme_id=theme.id)

        use_case_input = UseCaseInputPost(
            title="No banner use case",
            instruction="No banner",
            user_input_form="N/A",
            position=99,
            show_update_banner=False,
        )

        mock_theme_update = self._mock_create_flow(mocker, theme, use_case)

        result = await create_use_case(db_session, theme.uuid, use_case_input)

        assert isinstance(result, UseCaseResponse)
        mock_theme_update.assert_not_called()
