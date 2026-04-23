from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.themes_use_cases.schemas import ThemeInput, UseCaseInputPost, UseCaseInputPut


class TestBannerValidation:
    def test_theme_rejects_invalid_banner_type(self):
        with pytest.raises(ValidationError):
            ThemeInput(
                title="Theme",
                subtitle="Subtitle",
                show_update_banner=True,
                banner_type="invalid",
                banner_until=datetime.now(UTC).replace(tzinfo=None) + timedelta(days=1),
            )

    def test_use_case_post_rejects_invalid_banner_type(self):
        with pytest.raises(ValidationError):
            UseCaseInputPost(
                title="Use case",
                instruction="Instruction",
                user_input_form="Form",
                show_update_banner=True,
                banner_type="invalid",
                banner_until=datetime.now(UTC).replace(tzinfo=None) + timedelta(days=1),
            )

    @pytest.mark.parametrize(
        "factory, kwargs",
        [
            (
                ThemeInput,
                {
                    "title": "Theme",
                    "subtitle": "Subtitle",
                    "show_update_banner": True,
                    "banner_type": "new",
                },
            ),
            (
                UseCaseInputPost,
                {
                    "title": "Use case",
                    "instruction": "Instruction",
                    "user_input_form": "Form",
                    "show_update_banner": True,
                    "banner_type": "updated",
                },
            ),
            (
                UseCaseInputPut,
                {
                    "title": "Use case",
                    "instruction": "Instruction",
                    "user_input_form": "Form",
                    "theme_uuid": uuid4(),
                    "show_update_banner": True,
                    "banner_type": "updated",
                },
            ),
        ],
    )
    def test_banner_until_rejects_past_datetime(self, factory, kwargs):
        with pytest.raises(ValidationError, match="Banner until cannot be in the past"):
            factory(**kwargs, banner_until=datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1))

    def test_use_case_post_accepts_null_banner_when_disabled(self):
        """Removing banner from a use-case should accept null banner_type and banner_until."""
        model = UseCaseInputPost(
            title="Use case",
            instruction="Instruction",
            user_input_form="Form",
            show_update_banner=False,
            banner_type=None,
            banner_until=None,
        )
        assert model.banner_type is None
        assert model.banner_until is None

    def test_use_case_put_accepts_null_banner_when_disabled(self):
        """Removing banner from a use-case should accept null banner_type and banner_until."""
        model = UseCaseInputPut(
            title="Use case",
            instruction="Instruction",
            user_input_form="Form",
            theme_uuid=uuid4(),
            show_update_banner=False,
            banner_type=None,
            banner_until=None,
        )
        assert model.banner_type is None
        assert model.banner_until is None

    def test_theme_accepts_null_banner_when_disabled(self):
        """Removing banner from a theme should accept null banner_type and banner_until."""
        model = ThemeInput(
            title="Theme",
            subtitle="Subtitle",
            show_update_banner=False,
            banner_type=None,
            banner_until=None,
        )
        assert model.banner_type is None
        assert model.banner_until is None
