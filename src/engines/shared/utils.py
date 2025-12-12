"""Shared utility functions for engine modules."""

from typing import Any


def safe_getattr(module: Any, name: str, module_name: str) -> type:
    """Safely get an attribute from a module with helpful error messages.

    Args:
        module: The module to get the attribute from
        name: The attribute name to retrieve
        module_name: Human-readable module name for error messages

    Returns:
        The requested attribute (class)

    Raises:
        ValueError: If attribute not found, with list of available options
    """
    try:
        return getattr(module, name)
    except AttributeError as e:
        # Get available public attributes
        available = sorted([n for n in dir(module) if not n.startswith('_')])
        # Show first 20 options
        options_str = ", ".join(available[:20])
        if len(available) > 20:
            options_str += f", ... and {len(available) - 20} more"
        raise ValueError(
            f"'{name}' not found in {module_name}. "
            f"Available options: {options_str}"
        ) from e
