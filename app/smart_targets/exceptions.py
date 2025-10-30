class SmartTargetsConnectionError(Exception):
    """Triggered when a connection to the Smart Targets tool does not succeed"""

    ...


class GetSmartTargetsMetricsError(Exception):
    """Triggered when metrics ere not retrieved from the Smart Targets tool"""
