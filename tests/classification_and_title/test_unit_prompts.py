from unittest.mock import MagicMock

import pytest

from app.classification_and_title.prompts import (
    build_title_and_classification_system_prompt,
    build_title_and_classification_tool,
)

pytestmark = [pytest.mark.unit]


def make_classification(title):
    classification = MagicMock()
    classification.title = title
    return classification


class TestBuildSystemPrompt:
    def test_static_block_lists_all_categories_plus_other(self):
        classifications = [make_classification("Media handling and press releases"), make_classification("Research")]

        blocks = build_title_and_classification_system_prompt(classifications, [], None)

        static_text = blocks[0]["text"]
        assert "Media handling and press releases" in static_text
        assert "Research" in static_text
        assert "Other: Does not fit any of the above categories" in static_text

    def test_no_job_title_or_documents_produces_single_block(self):
        blocks = build_title_and_classification_system_prompt([], [], None)

        assert len(blocks) == 1

    def test_job_title_is_appended_as_dynamic_block(self):
        blocks = build_title_and_classification_system_prompt([], [], "Senior Press Officer")

        assert len(blocks) == 2
        assert "Senior Press Officer" in blocks[1]["text"]

    def test_document_names_are_appended_as_dynamic_block(self):
        blocks = build_title_and_classification_system_prompt([], ["Budget report.pdf"], None)

        assert len(blocks) == 2
        assert "Budget report.pdf" in blocks[1]["text"]

    def test_job_title_and_documents_share_the_same_dynamic_block(self):
        blocks = build_title_and_classification_system_prompt([], ["Budget report.pdf"], "Senior Press Officer")

        assert len(blocks) == 2
        assert "Senior Press Officer" in blocks[1]["text"]
        assert "Budget report.pdf" in blocks[1]["text"]


class TestBuildTool:
    def test_category_enum_includes_all_classifications_plus_other(self):
        classifications = [make_classification("Media handling and press releases"), make_classification("Research")]

        tool = build_title_and_classification_tool(classifications)

        category_enum = tool["input_schema"]["properties"]["category"]["enum"]
        assert category_enum == ["Media handling and press releases", "Research", "Other"]

    def test_task_type_enum_includes_research(self):
        tool = build_title_and_classification_tool([])

        task_type_enum = tool["input_schema"]["properties"]["task_type"]["enum"]
        assert "Research" in task_type_enum

    def test_task_type_enum_includes_unknown(self):
        tool = build_title_and_classification_tool([])

        task_type_enum = tool["input_schema"]["properties"]["task_type"]["enum"]
        assert "Unknown" in task_type_enum

    def test_required_fields(self):
        tool = build_title_and_classification_tool([])

        assert tool["input_schema"]["required"] == ["title", "category", "task_type", "discipline"]
