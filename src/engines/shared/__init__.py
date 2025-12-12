"""Shared base components for inference and validation engines."""

from src.engines.shared.handlers import (
    BaseMetricsHandler,
    BaseProgressHandler,
    BaseResultsHandler,
)
from src.engines.shared.utils import safe_getattr

__all__ = [
    "BaseMetricsHandler",
    "BaseProgressHandler",
    "BaseResultsHandler",
    "safe_getattr",
]
