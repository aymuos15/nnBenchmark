"""Base registry class for factory pattern component registration.

This module provides a base class for all component registries (losses, metrics,
models, optimizers, transforms) to reduce code duplication and ensure consistent
registry behavior across the framework.
"""

from abc import ABC
from typing import Any, Callable


class BaseRegistry(ABC):
    """Base class for component registries.

    Provides common functionality for:
    - Registry management (register, unregister, list_available)
    - Parameter extraction from config
    - Type validation and error handling

    Subclasses should:
    1. Call super().__init__() in their __init__
    2. Implement _get_component_name() to return the component type name
    3. Call _register_default_components() to populate the registry
    4. Implement build() with appropriate signature for their component type
    """

    def __init__(self) -> None:
        """Initialize the base registry with an empty registry dictionary."""
        self._registry: dict[str, type | Callable] = {}

    def register(self, name: str, component_class: type | Callable) -> None:
        """Register a component class or factory function with the given name.

        Args:
            name: Name to register the component under
            component_class: The component class or factory function to register

        Raises:
            ValueError: If the name is already registered
        """
        if name in self._registry:
            raise ValueError(
                f"{self._get_component_name()} '{name}' is already registered. "
                f"Use a different name or unregister first."
            )
        self._registry[name] = component_class

    def unregister(
        self, name: str
    ) -> None:  # Part of public API for registry management  # noqa: D401
        """Remove a component from the registry.

        Part of the public API for registry management.

        Args:
            name: Name of the component to unregister

        Raises:
            KeyError: If the component name is not registered
        """
        if name not in self._registry:
            raise KeyError(f"{self._get_component_name()} '{name}' is not registered")
        del self._registry[name]

    def list_available(self) -> list[str]:
        """Get a list of all registered component names.

        Returns:
            Sorted list of registered component names
        """
        return sorted(self._registry.keys())

    def _get_component_name(self) -> str:
        """Get the name of the component type for error messages.

        Subclasses should override this to provide component-specific names.

        Returns:
            Component name (e.g., "Loss", "Metric", "Model")
        """
        return self.__class__.__name__.replace("Registry", "")

    @staticmethod
    def _extract_params(
        config: dict[str, Any], exclude_keys: set[str] | None = None
    ) -> dict[str, Any]:
        """Extract parameters from config, excluding specific keys.

        Args:
            config: Configuration dictionary
            exclude_keys: Keys to exclude from extraction. Defaults to {"type"}

        Returns:
            Dictionary of extracted parameters
        """
        if exclude_keys is None:
            exclude_keys = {"type"}
        return {k: v for k, v in config.items() if k not in exclude_keys}

    def _validate_type(self, type_name: str) -> type | Callable:  # type: ignore[return-value]
        """Validate and return component type from registry.

        Args:
            type_name: Name of the component type to validate

        Returns:
            The component class or factory function

        Raises:
            KeyError: If type_name is not registered
        """
        if type_name not in self._registry:
            available = ", ".join(self.list_available())
            component_lower = self._get_component_name().lower()
            raise KeyError(
                f"{self._get_component_name()} type '{type_name}' is not registered. "
                f"Available {component_lower}s: {available}"
            )
        return self._registry[type_name]
