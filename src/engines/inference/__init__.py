"""
Inference module for model evaluation.

Uses Ignite-based EvaluationEngine for event-driven inference and validation.
"""

from src.engines.inference.engine import EvaluationEngine, InferenceEngine
from src.engines.inference.handlers import (
    InferenceMetricsHandler,
    InferenceProgressHandler,
    InferenceResultsHandler,
)
from src.engines.inference.restoration import (
    convert_predictions_to_original_space,
    get_padding_for_divisibility,
    pad_nd_image,
    revert_padding,
    uncrop_predictions,
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
    "InferenceEngine",  # Backward compatibility alias
    "InferenceMetricsHandler",
    "InferenceProgressHandler",
    "InferenceResultsHandler",
    "InferenceStrategy",
    "FullVolumeInferer",
    "SlidingWindowInferer",
    "create_inferer",
    "pad_nd_image",
    "get_padding_for_divisibility",
    "uncrop_predictions",
    "revert_padding",
    "convert_predictions_to_original_space",
]
