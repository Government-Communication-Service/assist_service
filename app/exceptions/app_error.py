class AppError(Exception):
    """Base class for exceptions that map to an HTTP response.

    A single handler (`app/exceptions/handlers.py`) turns any subclass into a response, so modules declare their own
    mapping here rather than registering a handler. Subclasses override the class attributes, or define them as
    properties when they depend on the instance.

    Attributes:
        status_code: HTTP status of the response.
        detail: Message returned in the response body. Must not contain user-supplied text.
        expected: True for normal API behaviour (not found, conflicts, bad headers): logged at info level and not
            reported to Bugsnag. False: logged at error level, which reports to Bugsnag.
    """

    status_code: int = 500
    detail: str = "An internal error occurred."
    expected: bool = False

    def content(self) -> dict:
        """Returns the JSON response body."""
        return {"detail": self.detail}
