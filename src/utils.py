"""
Shared utility functions and constants for the forecasting pipeline.
"""

CHANNELS = ["google", "meta", "bing"]
FORECAST_PERIODS = [30, 60, 90]
SEED = 42


def get_project_root():
    """Get absolute path to project root"""
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def safe_divide(numerator, denominator, default=0.0):
    """Divide safely, returning default if denominator is 0"""
    return numerator / denominator if denominator != 0 else default


def format_currency(value):
    """Format a number as currency string"""
    return f"${value:,.2f}"