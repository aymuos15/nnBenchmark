"""Metric registry for creating evaluation metrics from configuration.

This module provides a registry-based factory for creating MONAI metrics.
Unlike other factories, the metric registry builds multiple metrics at once
and returns them as a dictionary mapping full type names to instances.
"""

from typing import Any

from monai import metrics

from src.factory.base_registry import BaseRegistry
from src.factory.metrics.cc import CCMetric


class MetricRegistry(BaseRegistry):
    """Registry for creating metrics from configuration.

    The registry maintains a mapping of metric names to their MONAI classes
    and provides methods for building multiple metrics with native parameters.

    Unlike other registries that return single instances, this registry
    returns a dictionary of metrics since evaluation typically uses multiple
    metrics simultaneously.
    """

    def __init__(self) -> None:
        """Initialize the metric registry with default MONAI metrics."""
        super().__init__()
        self._register_default_metrics()

    def _register_default_metrics(self) -> None:
        """Register default MONAI evaluation metrics."""
        # Register common MONAI metrics for segmentation
        self.register("DiceMetric", getattr(metrics, "DiceMetric"))
        self.register("SurfaceDiceMetric", getattr(metrics, "SurfaceDiceMetric"))
        self.register(
            "HausdorffDistanceMetric", getattr(metrics, "HausdorffDistanceMetric")
        )
        self.register(
            "SurfaceDistanceMetric", getattr(metrics, "SurfaceDistanceMetric")
        )
        self.register("MeanIoU", getattr(metrics, "MeanIoU"))
        self.register(
            "ConfusionMatrixMetric", getattr(metrics, "ConfusionMatrixMetric")
        )
        # Register custom Connected Components metric
        self.register("CCMetric", CCMetric)

    def build(self, config: dict[str, Any]) -> dict[str, Any]:
        """Build all metrics from configuration.

        This method builds multiple metrics at once from a configuration
        dictionary and returns them as a dictionary mapping full type names
        to metric instances.

        Args:
            config: Configuration dictionary with 'metrics' section containing
                   a list of metric configurations. Each metric config must have:
                   - type: Metric class name (full name, e.g., 'DiceMetric')
                   - Other fields are native MONAI parameters

        Returns:
            Dictionary mapping metric type names to metric instances.

        Raises:
            KeyError: If a metric type is not registered
            TypeError: If config parameters don't match metric signature
        """
        metric_dict: dict[str, Any] = {}

        for metric_cfg in config["metrics"]:
            metric_type = metric_cfg["type"]
            metric_class = self._validate_type(metric_type)

            # Extract parameters (everything except 'type')
            metric_params = self._extract_params(metric_cfg)

            # Instantiate metric and store with full type name as key
            metric_dict[metric_type] = metric_class(**metric_params)

        return metric_dict


# Create a global registry instance
metric_registry = MetricRegistry()
