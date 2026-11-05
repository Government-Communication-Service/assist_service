# Custom exceptions for Gov UK Search feature


class GovUKPageNotFoundError(Exception):
    """GOV.UK returned 404 for the page."""
