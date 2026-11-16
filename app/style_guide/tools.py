from pydantic import Field, model_validator

from app.bedrock.tool_schema import ToolArgs, ToolDef
from app.config import StyleGuideThoroughness


class CheckStyleGuideArgs(ToolArgs):
    """Check content against the GOV.UK style guide (A to Z and related rules). Returns the issues found,
    one entry per rule with its guidance, how many times it was found and one example. It does not return
    a corrected version. Use it when the user asks for a style guide check, or for content to be checked
    or made compliant with GOV.UK style. Give exactly one of text or document_id. To check text from the
    user's message or the conversation, copy just that text into text, leaving out any request around it.

    By default, say briefly what was checked, then go through the issues grouped by rule, quoting the
    original wording and the correction. If the user asked for something narrower, such as only the
    issues or one section, answer that instead. Issues are listed most important first: clarity,
    accessibility and inclusive language, then formatting, then specific terms. The checker errs towards
    reporting, so check each issue against the text, find its other occurrences there, and leave out any
    that are fine in context. If the result lists issues it left out or rules it didn't run, mention them
    briefly."""

    text: str | None = Field(default=None, description="The text to check, copied exactly.")
    document_id: str | None = Field(
        default=None, description="The id of one attached document to check, as listed in the context."
    )
    thoroughness: StyleGuideThoroughness | None = Field(
        default=None,
        description=(
            "Which rules to run. lite: clarity, accessibility, inclusive language and core formatting such "
            "as dates and numbers. balanced (the default): also specific terms, spellings and abbreviations. "
            "thorough: also capitalisation of individual terms; use it when the user wants every detail "
            "checked, or asks about capitalisation."
        ),
    )
    context: str | None = Field(
        default=None,
        description=(
            "Optional guidance for the checker: what the content is, who it is for, or what the user wants "
            "checked. Not itself checked."
        ),
    )

    @model_validator(mode="after")
    def _check_one_source(self) -> "CheckStyleGuideArgs":
        if (self.text is None) == (self.document_id is None):
            raise ValueError("Give exactly one of 'text' or 'document_id'")
        return self


CHECK_STYLE_GUIDE = ToolDef("check_style_guide", CheckStyleGuideArgs, strict=False)
