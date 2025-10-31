"""
Configuration validation utilities for nnBenchmark.
Provides helpers for validating configuration files and ensuring required fields exist.
"""

from typing import Any


def validate_required_field(
    config: dict[str, Any], path: list[str], field_name: str, example: str | None = None
) -> None:
    """
    Validate that a required field exists in nested config.

    Args:
        config: Configuration dictionary
        path: Path to nested field (e.g., ["dataset", "fold"])
        field_name: Human-readable field name for error message
        example: Optional example value to show in error message

    Raises:
        ValueError: If field doesn't exist
    """
    current = config
    for i, key in enumerate(path[:-1]):
        if key not in current:
            section = " -> ".join(path[: i + 1])
            raise ValueError(f"Missing section '{section}' in configuration")
        current = current[key]

    field_key = path[-1]
    if field_key not in current:
        section = " -> ".join(path[:-1])
        example_text = f" (e.g., {example})" if example else ""
        raise ValueError(
            f"'{field_name}' parameter is required in {section} config. "
            f"Please specify {field_name}{example_text}"
        )


def validate_model_config(config: dict[str, Any]) -> None:
    """
    Validate model configuration structure.

    Supports both flat and nested config formats:
    - Flat: All params at model level (backward compatibility)
    - Nested: Shared params + model-specific nested sections

    Ensures that:
    1. model.type exists and is valid ("DynUNet" or "UNet")
    2. Required shared parameters exist
    3. For nested configs, appropriate model-specific section exists
    4. Model-specific required parameters are present

    Args:
        config: Configuration dictionary with 'model' section

    Raises:
        ValueError: If validation fails
    """
    if "model" not in config:
        raise ValueError("Missing 'model' section in configuration")

    model_cfg = config["model"]

    # Validate model type
    if "type" not in model_cfg:
        raise ValueError(
            "Missing 'type' field in model config. "
            "Please specify model type (e.g., type: DynUNet or type: UNet)"
        )

    model_type = model_cfg["type"]
    valid_types = ["DynUNet", "UNet"]
    if model_type not in valid_types:
        raise ValueError(
            f"Invalid model type '{model_type}'. "
            f"Must be one of: {valid_types}"
        )

    # Determine if config is nested
    nested_keys = {"DynUNet", "UNet"}
    is_nested = any(key in model_cfg for key in nested_keys)

    if is_nested:
        # Nested config: validate shared params + model-specific section exists
        if model_type not in model_cfg:
            raise ValueError(
                f"Nested config detected but '{model_type}' section not found. "
                f"Please provide '{model_type}' section with model-specific parameters."
            )

        # Validate shared parameters exist
        shared_required = ["spatial_dims", "in_channels", "out_channels"]
        for param in shared_required:
            if param not in model_cfg:
                raise ValueError(
                    f"Missing required shared parameter '{param}' in model config"
                )

        # Get model-specific params
        model_specific = model_cfg[model_type]

        # Validate model-specific required parameters
        if model_type == "DynUNet":
            _validate_dynunet_params(model_specific, is_nested=True)
        elif model_type == "UNet":
            _validate_unet_params(model_specific, is_nested=True)
    else:
        # Flat config: validate all required params at top level
        required = ["spatial_dims", "in_channels", "out_channels"]
        for param in required:
            if param not in model_cfg:
                raise ValueError(
                    f"Missing required parameter '{param}' in model config"
                )

        # Validate model-specific params at top level
        if model_type == "DynUNet":
            _validate_dynunet_params(model_cfg, is_nested=False)
        elif model_type == "UNet":
            _validate_unet_params(model_cfg, is_nested=False)


def _validate_dynunet_params(params: dict[str, Any], is_nested: bool) -> None:
    """Validate DynUNet-specific parameters.

    Args:
        params: Parameter dictionary to validate
        is_nested: Whether this is a nested model-specific section

    Raises:
        ValueError: If required parameters are missing
    """
    required = ["filters", "kernel_size", "strides", "upsample_kernel_size"]
    section = "DynUNet section" if is_nested else "model config"

    for param in required:
        if param not in params:
            raise ValueError(
                f"Missing required DynUNet parameter '{param}' in {section}"
            )


def _validate_unet_params(params: dict[str, Any], is_nested: bool) -> None:
    """Validate UNet-specific parameters.

    Args:
        params: Parameter dictionary to validate
        is_nested: Whether this is a nested model-specific section

    Raises:
        ValueError: If required parameters are missing
    """
    required = ["channels", "strides"]
    section = "UNet section" if is_nested else "model config"

    for param in required:
        if param not in params:
            raise ValueError(
                f"Missing required UNet parameter '{param}' in {section}"
            )


def validate_deep_supervision_config(config: dict[str, Any]) -> None:
    """
    Validate deep supervision configuration in model section.

    Ensures that:
    1. If deep_supervision is enabled, ds_weights must be provided
    2. ds_weights must be a non-empty list of floats
    3. Optionally warn if weights don't follow decreasing pattern

    Args:
        config: Configuration dictionary with 'model' section

    Raises:
        ValueError: If validation fails
    """
    model_cfg = config.get("model", {})
    deep_supervision = model_cfg.get("deep_supervision", False)

    if not deep_supervision:
        # Deep supervision disabled is valid
        return

    # If enabled, ds_weights must exist
    if "ds_weights" not in model_cfg:
        raise ValueError(
            "deep_supervision is enabled but 'ds_weights' is not specified. "
            "Please provide ds_weights as a list of floats in model config. "
            "Example: ds_weights: [1.0, 0.5, 0.25]"
        )

    ds_weights = model_cfg["ds_weights"]

    # Validate ds_weights is a list
    if not isinstance(ds_weights, list):
        raise ValueError(f"ds_weights must be a list, got {type(ds_weights).__name__}")

    # Validate ds_weights is non-empty
    if not ds_weights:
        raise ValueError(
            "ds_weights must be non-empty. "
            "Provide at least one weight for the final output."
        )

    # Validate all elements are numbers
    for i, weight in enumerate(ds_weights):
        if not isinstance(weight, (int, float)):
            raise ValueError(
                f"ds_weights[{i}] must be a number, got {type(weight).__name__}"
            )
        if weight <= 0:
            raise ValueError(f"ds_weights[{i}] must be positive, got {weight}")


def validate_sliding_window_config(config: dict[str, Any]) -> None:
    """
    Validate sliding window inference configuration.

    Ensures that if sliding_window is enabled in testing config:
    1. All required parameters are present
    2. overlap is a float between 0.0 and 0.99
    3. sw_batch_size is a positive integer
    4. mode is one of: gaussian, constant
    5. padding_mode is one of: constant, edge, reflect, wrap

    Args:
        config: Configuration dictionary with optional 'testing.sliding_window' section

    Raises:
        ValueError: If sliding_window is enabled but invalid
    """
    testing_cfg = config.get("testing", {})
    sliding_window_cfg = testing_cfg.get("sliding_window", {})

    # If sliding_window section doesn't exist or is empty, validation passes
    if not sliding_window_cfg:
        return

    enabled = sliding_window_cfg.get("enabled", False)

    # If not enabled, validation passes (all other parameters are optional)
    if not enabled:
        return

    # Validate overlap parameter
    if "overlap" in sliding_window_cfg:
        overlap = sliding_window_cfg["overlap"]
        if not isinstance(overlap, (int, float)):
            raise ValueError(
                f"sliding_window 'overlap' must be a number, got {type(overlap).__name__}"
            )
        if not (0.0 <= overlap < 1.0):
            raise ValueError(
                f"sliding_window 'overlap' must be between 0.0 and 0.99, got {overlap}"
            )

    # Validate sw_batch_size parameter
    if "sw_batch_size" in sliding_window_cfg:
        sw_batch_size = sliding_window_cfg["sw_batch_size"]
        if not isinstance(sw_batch_size, int):
            raise ValueError(
                f"sliding_window 'sw_batch_size' must be an integer, "
                f"got {type(sw_batch_size).__name__}"
            )
        if sw_batch_size <= 0:
            raise ValueError(
                f"sliding_window 'sw_batch_size' must be positive, got {sw_batch_size}"
            )

    # Validate mode parameter
    valid_modes = ["gaussian", "constant"]
    if "mode" in sliding_window_cfg:
        mode = sliding_window_cfg["mode"]
        if mode not in valid_modes:
            raise ValueError(
                f"sliding_window 'mode' must be one of {valid_modes}, got '{mode}'"
            )

    # Validate padding_mode parameter
    valid_padding_modes = ["constant", "edge", "reflect", "wrap"]
    if "padding_mode" in sliding_window_cfg:
        padding_mode = sliding_window_cfg["padding_mode"]
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"sliding_window 'padding_mode' must be one of {valid_padding_modes}, "
                f"got '{padding_mode}'"
            )


def validate_metrics_config(
    config: dict[str, Any], metric_dict: dict[str, Any]
) -> tuple[str, list[str]]:
    """
    Validate checkpoint_metric and plot_metrics configuration.

    Ensures that:
    1. checkpoint_metric is specified in training config
    2. plot_metrics is specified in training config
    3. Both checkpoint_metric and all plot_metrics exist in defined metrics

    Args:
        config: Configuration dictionary with 'training' and 'metrics' sections
        metric_dict: Dictionary of built metrics (from build_metrics)

    Returns:
        Tuple of (checkpoint_metric, plot_metrics)

    Raises:
        ValueError: If validation fails
    """
    available_metrics = list(metric_dict.keys())

    # Validate checkpoint_metric exists
    if "checkpoint_metric" not in config["training"]:
        raise ValueError(
            "Missing required field 'checkpoint_metric' in training config. "
            f"Please specify which metric to use for best model selection. "
            f"Available metrics: {available_metrics}"
        )

    # Validate plot_metrics exists
    if "plot_metrics" not in config["training"]:
        raise ValueError(
            "Missing required field 'plot_metrics' in training config. "
            f"Please specify which metrics to plot (as a list). "
            f"Available metrics: {available_metrics}"
        )

    checkpoint_metric: str = config["training"]["checkpoint_metric"]
    plot_metrics: list[str] = config["training"]["plot_metrics"]

    # Validate checkpoint_metric is in defined metrics
    if checkpoint_metric not in metric_dict:
        raise ValueError(
            f"checkpoint_metric '{checkpoint_metric}' not found in defined metrics. "
            f"Available metrics: {available_metrics}"
        )

    # Validate all plot_metrics are in defined metrics
    for pm in plot_metrics:
        if pm not in metric_dict:
            raise ValueError(
                f"plot_metric '{pm}' not found in defined metrics. "
                f"Available metrics: {available_metrics}"
            )

    return checkpoint_metric, plot_metrics
