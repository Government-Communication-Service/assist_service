"""
Fixtures for style guide e2e tests.

Provides a `style_guide_use_case_id` fixture that creates the required
"GOV.UK style guide checker" theme and use case in the database so that
the chat endpoint activates the style guide checker for the test.
"""
import pytest

from app.database.table import ThemeTable, UseCaseTable


@pytest.fixture
def style_guide_use_case_id():
    """
    Creates a 'GOV.UK style guide checker' theme and use case in the DB.

    Returns the use case UUID (string) for use as `use_case_id` in chat
    request payloads.  Tears down the created records after each test.
    """
    theme_table = ThemeTable()
    theme = theme_table.create(
        {
            "title": "GOV.UK style guide checker",
            "subtitle": "Check content against the GOV.UK Style Guide",
            "position": 99,
        }
    )

    use_case_table = UseCaseTable()
    use_case = use_case_table.create(
        {
            "theme_id": theme.id,
            "title": "Check my content against the style guide",
            "instruction": "Check the provided content against the GOV.UK Style Guide.",
            "user_input_form": "Content: [USER PROMPT]",
            "position": 1,
        }
    )

    yield str(use_case.uuid)

    # Teardown – use soft_delete_by_uuid to avoid the deprecated .get() call in delete()
    use_case_table.soft_delete_by_uuid(use_case.uuid)
    theme_table.soft_delete_by_uuid(theme.uuid)
