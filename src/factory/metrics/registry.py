"""Metric registry for creating evaluation metrics from configuration.

This module provides a registry-based factory for creating MONAI metrics.
Unlike other factories, the metric registry builds multiple metrics at once
and returns them as a dictionary mapping full type names to instances.
"""

from typing import Any

from monai import metrics


class MetricRegistry:
    """Registry for creating metrics from configuration.

    The registry maintains a mapping of metric names to their MONAI classes
    and provides methods for building multiple metrics with native parameters.

    Unlike other registries that return single instances, this registry
    returns a dictionary of metrics since evaluation typically uses multiple
    metrics simultaneously.

    Example:
        >>> registry = MetricRegistry()
        >>> config = {
        ...     "metrics": [
        ...         {"type": "DiceMetric", "include_background": False},
        ...         {"type": "SurfaceDiceMetric", "include_background": False}
        ...     ]
        ... }
        >>> metric_dict = registry.build(config)
        >>> # Returns: {"DiceMetric": DiceMetric(...), "SurfaceDiceMetric": SurfaceDiceMetric(...)}
    """

    def __init__(self) -> None:
        """Initialize the metric registry with default MONAI metrics."""
        self._registry: dict[str, type] = {}
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

    def register(self, name: str, metric_class: type) -> None:
        """Register a metric class with the given name.

        Args:
            name: Name to register the metric under (typically the class name)
            metric_class: The metric class to register

        Raises:
            ValueError: If the name is already registered
        """
        if name in self._registry:
            raise ValueError(
                f"Metric '{name}' is already registered. "
                f"Use a different name or unregister first."
            )
        self._registry[name] = metric_class

    def unregister(
        self, name: str
    ) -> (
        None
    ):  # Part of public API for registry management  # noqa: D401  # pragma: no cover
        """Remove a metric from the registry.

        Part of the public API for registry management.

        Args:
            name: Name of the metric to unregister

        Raises:
            KeyError: If the metric name is not registered
        """
        if name not in self._registry:
            raise KeyError(f"Metric '{name}' is not registered")
        del self._registry[name]

    def list_available(self) -> list[str]:
        """Get a list of all registered metric names.

        Returns:
            Sorted list of registered metric names
        """
        return sorted(self._registry.keys())

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
            Example: {"DiceMetric": DiceMetric(...), "SurfaceDiceMetric": SurfaceDiceMetric(...)}

        Raises:
            KeyError: If a metric type is not registered
            TypeError: If config parameters don't match metric signature

        Example:
            >>> config = {
            ...     "metrics": [
            ...         {
            ...             "type": "DiceMetric",
            ...             "include_background": False,
            ...             "reduction": "mean_batch",
            ...             "num_classes": 3
            ...         },
            ...         {
            ...             "type": "SurfaceDiceMetric",
            ...             "include_background": False,
            ...             "reduction": "mean_batch",
            ...             "class_thresholds": [2.0, 2.0]
            ...         }
            ...     ]
            ... }
            >>> metric_dict = registry.build(config)
        """
        metric_dict: dict[str, Any] = {}

        for metric_cfg in config["metrics"]:
            metric_type = metric_cfg["type"]

            if metric_type not in self._registry:
                available = ", ".join(self.list_available())
                raise KeyError(
                    f"Metric type '{metric_type}' is not registered. "
                    f"Available metrics: {available}"
                )

            metric_class = self._registry[metric_type]

            # Extract parameters (everything except 'type')
            metric_params = {k: v for k, v in metric_cfg.items() if k != "type"}

            # Instantiate metric and store with full type name as key
            metric_dict[metric_type] = metric_class(**metric_params)

        return metric_dict


# Create a global registry instance
metric_registry = MetricRegistry()
