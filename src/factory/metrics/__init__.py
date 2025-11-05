"""Metric factory for creating evaluation metrics from configuration."""

from src.factory.metrics.cc import CCMetric
from src.factory.metrics.registry import MetricRegistry, metric_registry

__all__ = ["CCMetric", "MetricRegistry", "metric_registry"]
