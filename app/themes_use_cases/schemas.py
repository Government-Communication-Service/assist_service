from datetime import UTC, datetime
from typing import Literal, Optional, get_args
from uuid import UUID

from pydantic import BaseModel, model_validator

BannerType = Literal["new", "updated"]
ALLOWED_BANNER_TYPES = set(get_args(BannerType))


def validate_banner_fields(values):
    """Validate the banner_type and banner_until are provided when show_update_banner is True."""
    if not isinstance(values, dict):
        return values

    if not values.get("show_update_banner", False):
        return values

    banner_type = values.get("banner_type")
    banner_until = values.get("banner_until")

    # Required presence checks
    if not banner_type:
        raise ValueError("Banner type is required when showing an update banner")
    if not banner_until:
        raise ValueError("Banner until is required when showing an update banner")

    # Validate banner_type is an allowed value
    if banner_type not in ALLOWED_BANNER_TYPES:
        raise ValueError("Banner type must be 'new' or 'updated'")

    # Parse and validate banner_until
    if isinstance(banner_until, str):
        try:
            banner_until = datetime.fromisoformat(banner_until)
        except ValueError as exc:
            raise ValueError("Banner until must be a valid ISO datetime") from exc
    elif not isinstance(banner_until, datetime):
        raise ValueError("Banner until must be a valid datetime")

    # Normalize to naive UTC so the comparison below works for both
    # naive and tz-aware inputs (DB column is naive UTC).
    if banner_until.tzinfo is not None:
        banner_until = banner_until.astimezone(UTC).replace(tzinfo=None)

    if banner_until < datetime.now(UTC).replace(tzinfo=None):
        raise ValueError("Banner until cannot be in the past")

    values["banner_until"] = banner_until
    return values


# These input models are used to configure FastAPI to collect data in the body
class ThemeInput(BaseModel):
    """Model for creating a new theme with a title and subtitle."""

    title: str
    subtitle: str
    position: Optional[int] = None
    show_update_banner: bool = False
    banner_type: Optional[BannerType] = None
    banner_until: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def validate_banner(cls, values):
        return validate_banner_fields(values)


class UseCaseInputPost(BaseModel):
    """Model for creating a new use case associated with a theme, including title, instruction, and user input form."""

    title: str
    instruction: str
    user_input_form: str
    position: Optional[int] = None
    show_update_banner: bool = False
    banner_type: Optional[BannerType] = None
    banner_until: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def validate_banner(cls, values):
        return validate_banner_fields(values)


class UseCaseInputPut(BaseModel):
    """Model for updating a new use case associated with a theme, including title, instruction, and user input form."""

    title: str
    instruction: str
    user_input_form: str
    position: Optional[int] = None
    theme_uuid: UUID
    show_update_banner: bool = False
    banner_type: Optional[BannerType] = None
    banner_until: Optional[datetime] = None

    @model_validator(mode="before")
    @classmethod
    def validate_banner(cls, values):
        return validate_banner_fields(values)


# This model is used in both the input and outputs of endpoints for bulk operations.
# It is used in bulk operations, where the use_case
# and theme for each prompt are collapsed into a single 'prebuilt prompt' data structure.
class PrebuiltPrompt(BaseModel):
    """Model representing a prebuilt prompt combining theme and use case details."""

    theme_title: str
    theme_subtitle: str
    theme_position: Optional[int] = None
    theme_show_update_banner: bool = False
    theme_banner_type: Optional[BannerType] = None
    theme_banner_until: Optional[datetime] = None
    use_case_title: str
    use_case_instruction: str
    use_case_user_input_form: str
    use_case_position: Optional[int] = None
    use_case_show_update_banner: bool = False
    use_case_banner_type: Optional[BannerType] = None
    use_case_banner_until: Optional[datetime] = None
