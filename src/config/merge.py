"""Configuration merging utilities for handling config inheritance and overrides."""

from typing import Any


class ConfigValidationError(Exception):
    """Raised when config override validation fails."""

    pass


def deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge overrides into base dictionary.

    Args:
        base: Base dictionary to merge into
        overrides: Dictionary with override values

    Returns:
        New merged dictionary (does not modify input dicts)
    """
    result = base.copy()

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_dicts(result[key], value)
        else:
            # Override the value
            result[key] = value

    return result


def validate_override_keys(
    base: dict[str, Any], overrides: dict[str, Any], path: str = ""
) -> None:
    """
    Validate that all override keys exist in the base config.

    Special case: If overrides contains a "type" key that differs from base,
    the entire section is being replaced (e.g., changing loss function),
    so we skip validation for that section.

    Args:
        base: Base configuration dictionary
        overrides: Override configuration dictionary
        path: Current path in the config hierarchy (for error messages)

    Raises:
        ConfigValidationError: If an override key doesn't exist in base config
    """
    # If changing "type", we're replacing the entire component, so skip validation
    if "type" in overrides and "type" in base and overrides["type"] != base["type"]:
        return

    for key, value in overrides.items():
        current_path = f"{path}.{key}" if path else key

        if key not in base:
            raise ConfigValidationError(
                f"Override key '{current_path}' does not exist in base config. "
                f"Available keys at this level: {list(base.keys())}"
            )

        # If both are dicts, recursively validate nested keys
        if isinstance(value, dict):
            if not isinstance(base[key], dict):
                raise ConfigValidationError(
                    f"Override key '{current_path}' is a dict, but base config has type {type(base[key]).__name__}"
                )
            validate_override_keys(base[key], value, current_path)


def load_config_with_inheritance(
    config: dict[str, Any], config_loader_func: callable
) -> dict[str, Any]:
    """
    Load config with base_config inheritance support.

    Args:
        config: Loaded configuration dictionary (may contain base_config key)
        config_loader_func: Function to load base config (passed to avoid circular imports)

    Returns:
        Merged configuration dictionary

    Raises:
        ConfigValidationError: If override validation fails
    """
    if "base_config" not in config:
        # No inheritance, return as-is
        return config

    base_config_path = config["base_config"]
    overrides = config.get("overrides", {})

    # Load base config
    base_config = config_loader_func(base_config_path)

    # Validate that all override keys exist in base config
    validate_override_keys(base_config, overrides)

    # Merge overrides into base config
    merged_config = deep_merge_dicts(base_config, overrides)

    return merged_config
