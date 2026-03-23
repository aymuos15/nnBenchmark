"""
Inference module for model evaluation.

Uses Ignite-based EvaluationEngine for event-driven inference and validation.
"""

from src.engines.inference.engine import EvaluationEngine
from src.engines.inference.handlers import (
    InferenceMetricsHandler,
    InferenceProgressHandler,
    InferenceResultsHandler,
)
from src.engines.inference.run import run_inference
from src.engines.inference.strategy import (
    FullVolumeInferer,
    InferenceStrategy,
    SlidingWindowInferer,
    create_inferer,
)

__all__ = [
    "run_inference",
    "EvaluationEngine",
    "InferenceMetricsHandler",
    "InferenceProgressHandler",
    "InferenceResultsHandler",
    "InferenceStrategy",
    "FullVolumeInferer",
    "SlidingWindowInferer",
    "create_inferer",
]
