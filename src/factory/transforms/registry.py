"""Transform registry for creating data transformation pipelines from configuration.

This module provides a registry-based factory for creating MONAI transform pipelines.
Transforms are composed from a 'common' section plus mode-specific transforms ('train', 'val', 'test').
"""

from typing import Any

from monai import transforms


class TransformRegistry:
    """Registry for creating transform pipelines from configuration.

    The registry handles building MONAI transform pipelines by combining
    common transforms with mode-specific transforms. The composition logic
    is simplified: common transforms followed by mode-specific transforms.
    """

    def __init__(self) -> None:
        """Initialize the transform registry.

        Note: We don't pre-register transforms since MONAI provides
        all transforms via getattr(transforms, type) dynamically.
        """
        self._registry: dict[str, type] = {}

    def register(self, name: str, transform_class: type) -> None:
        """Register a transform class with the given name.

        Args:
            name: Name to register the transform under
            transform_class: The transform class to register

        Raises:
            ValueError: If the name is already registered
        """
        if name in self._registry:
            raise ValueError(
                f"Transform '{name}' is already registered. "
                f"Use a different name or unregister first."
            )
        self._registry[name] = transform_class

    def unregister(
        self, name: str
    ) -> (
        None
    ):  # Part of public API for registry management  # noqa: D401  # pragma: no cover
        """Remove a transform from the registry.

        Part of the public API for registry management.

        Args:
            name: Name of the transform to unregister

        Raises:
            KeyError: If the transform name is not registered
        """
        if name not in self._registry:
            raise KeyError(f"Transform '{name}' is not registered")
        del self._registry[name]

    def list_available(self) -> list[str]:
        """Get a list of all registered transform names.

        Returns:
            Sorted list of registered transform names
        """
        return sorted(self._registry.keys())

    def build(self, config: dict[str, Any], mode: str = "train") -> transforms.Compose:  # type: ignore[name-defined]
        """Build transform pipeline from configuration.

        This method builds a MONAI Compose pipeline by:
        1. Validating required sections exist ('common' and specified mode)
        2. Building all transforms from 'common' section
        3. Appending all transforms from mode-specific section
        4. Wrapping result in Compose

        **Simplified Logic**: Common transforms + mode-specific transforms (appended).
        No special ToTensord insertion logic - transforms are composed in order.

        Args:
            config: Configuration dictionary with 'transforms' section containing:
                   - common: List of transform configs for all modes
                   - train/val/test: List of mode-specific transform configs
            mode: Transform mode ('train', 'val', or 'test')

        Returns:
            MONAI Compose object containing the transform pipeline

        Raises:
            KeyError: If required sections ('common' or mode) are missing
        """
        # Validate required sections exist
        if "common" not in config["transforms"]:
            raise KeyError(
                "Missing 'common' section in transforms config. "
                "Transforms must use the format with 'common', 'train', 'val', and 'test' sections."
            )

        if mode not in config["transforms"]:
            raise KeyError(
                f"Missing '{mode}' section in transforms config. "
                f"Please define transforms for mode '{mode}'."
            )

        transform_list: list[Any] = []

        # Build common transforms
        for t_cfg in config["transforms"]["common"]:
            transform_cls = self._get_transform_class(t_cfg["type"])
            transform_params = {k: v for k, v in t_cfg.items() if k != "type"}
            transform_list.append(transform_cls(**transform_params))

        # Append mode-specific transforms
        for t_cfg in config["transforms"][mode]:
            transform_cls = self._get_transform_class(t_cfg["type"])
            transform_params = {k: v for k, v in t_cfg.items() if k != "type"}
            transform_list.append(transform_cls(**transform_params))

        return transforms.Compose(transform_list)  # type: ignore[attr-defined]

    def _get_transform_class(self, transform_type: str) -> type:
        """Get transform class by name.

        First checks the registry, then falls back to MONAI transforms module.

        Args:
            transform_type: Name of the transform class

        Returns:
            Transform class

        Raises:
            AttributeError: If transform type not found in registry or MONAI
        """
        # Check registry first
        if transform_type in self._registry:
            return self._registry[transform_type]

        # Fall back to MONAI transforms
        if hasattr(transforms, transform_type):
            return getattr(transforms, transform_type)

        raise AttributeError(
            f"Transform '{transform_type}' not found in registry or MONAI transforms. "
            f"Available in registry: {self.list_available()}"
        )


# Create a global registry instance
transform_registry = TransformRegistry()
