from typing import Optional
from uuid import UUID

from pydantic import BaseModel


# These input models are used to configure FastAPI to collect data in the body
class ThemeInput(BaseModel):
    """Model for creating a new theme with a title and subtitle."""

    title: str
    subtitle: str
    position: Optional[int] = None


class UseCaseInputPost(BaseModel):
    """Model for creating a new use case associated with a theme, including title, instruction, and user input form."""

    title: str
    instruction: str
    user_input_form: str
    position: Optional[int] = None


class UseCaseInputPut(BaseModel):
    """Model for updating a new use case associated with a theme, including title, instruction, and user input form."""

    title: str
    instruction: str
    user_input_form: str
    position: Optional[int] = None
    theme_uuid: UUID


# This model is used in both the input and outputs of endpoints for bulk operations.
# It is used in bulk operations, where the use_case
# and theme for each prompt are collapsed into a single 'prebuilt prompt' data structure.
class PrebuiltPrompt(BaseModel):
    """Model representing a prebuilt prompt combining theme and use case details."""

    theme_title: str
    theme_subtitle: str
    theme_position: Optional[int] = None
    use_case_title: str
    use_case_instruction: str
    use_case_user_input_form: str
    use_case_position: Optional[int] = None
