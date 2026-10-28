class StyleGuideCheckError(Exception):
    """A style guide check couldn't be completed."""


class PartialStyleGuideCheckError(StyleGuideCheckError):
    """Some model validation calls failed; the others' violations are kept.

    The first underlying failure is the exception's __cause__.

    Attributes:
        violations: Violations found by the calls that succeeded.
        not_checked: Titles of the rules in the failed calls, without duplicates.
    """

    def __init__(self, message: str, violations: list[dict], not_checked: list[str] | None = None):
        super().__init__(message)
        self.violations = violations
        self.not_checked = list(dict.fromkeys(not_checked or []))
