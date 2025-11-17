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

        For metrics with duplicate types (e.g., multiple CCMetric instances),
        unique names are auto-generated using the metric_type parameter if available.
        For example, "CCMetric" with metric_type="dice" becomes "CCMetric_dice".

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
        metric_type_counts: dict[str, int] = {}  # Track metric type occurrences

        for metric_cfg in config["metrics"]:
            metric_type = metric_cfg["type"]
            metric_class = self._validate_type(metric_type)

            # Extract parameters (everything except 'type')
            metric_params = self._extract_params(metric_cfg)

            # Generate unique key for this metric instance
            metric_key = self._generate_metric_key(
                metric_type, metric_cfg, metric_type_counts
            )

            # Instantiate metric and store with generated key
            metric_dict[metric_key] = metric_class(**metric_params)

        return metric_dict

    @staticmethod
    def _generate_metric_key(
        metric_type: str,
        metric_cfg: dict[str, Any],
        type_counts: dict[str, int],
    ) -> str:
        """Generate a unique key for a metric instance.

        For metrics with a 'metric_type' parameter (e.g., CCMetric), uses that
        to create a descriptive name like "CCMetric_dice" or "CCMetric_surface_dice".
        For duplicate metric types without metric_type parameter, appends a counter.

        Args:
            metric_type: The metric class name (e.g., 'CCMetric')
            metric_cfg: The full metric configuration dict
            type_counts: Dictionary tracking occurrences of each metric type

        Returns:
            Unique key for this metric instance (e.g., "CCMetric_dice")
        """
        # Track how many times we've seen this metric type
        type_counts[metric_type] = type_counts.get(metric_type, 0) + 1
        count = type_counts[metric_type]

        # If this metric has a 'metric_type' parameter, use it for naming
        # (commonly used by CCMetric for dice/surface_dice variants)
        if "metric_type" in metric_cfg:
            subtype = metric_cfg["metric_type"]
            return f"{metric_type}_{subtype}"

        # If this is the first occurrence of this metric type, use it as-is
        if count == 1:
            return metric_type

        # For duplicate metric types without metric_type param, append counter
        return f"{metric_type}_{count}"


# Create a global registry instance
metric_registry = MetricRegistry()
