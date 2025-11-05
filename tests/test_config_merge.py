"""
Tests for config merge functionality and inheritance.
Tests the merge utilities and load_config with base_config support.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest
import yaml

from src.config.load import load_config
from src.config.merge import (
    ConfigValidationError,
    deep_merge_dicts,
    validate_override_keys,
)


class TestDeepMergeDicts:
    """Tests for deep_merge_dicts function."""

    def test_merge_flat_dicts(self) -> None:
        """Test merging flat dictionaries."""
        base = {"a": 1, "b": 2, "c": 3}
        overrides = {"b": 20, "d": 4}

        result = deep_merge_dicts(base, overrides)

        assert result == {"a": 1, "b": 20, "c": 3, "d": 4}
        # Ensure original dicts are not modified
        assert base == {"a": 1, "b": 2, "c": 3}
        assert overrides == {"b": 20, "d": 4}

    def test_merge_nested_dicts(self) -> None:
        """Test merging nested dictionaries."""
        base = {
            "dataset": {"name": "Dataset001", "fold": 0},
            "training": {"epochs": 100, "batch_size": 2},
        }
        overrides = {
            "training": {"epochs": 200, "learning_rate": 0.001},
        }

        result = deep_merge_dicts(base, overrides)

        assert result == {
            "dataset": {"name": "Dataset001", "fold": 0},
            "training": {"epochs": 200, "batch_size": 2, "learning_rate": 0.001},
        }

    def test_merge_deep_nested_dicts(self) -> None:
        """Test merging deeply nested dictionaries."""
        base = {
            "level1": {
                "level2": {
                    "level3": {"a": 1, "b": 2},
                    "x": 10,
                },
                "y": 20,
            }
        }
        overrides = {
            "level1": {
                "level2": {
                    "level3": {"b": 200},
                    "z": 30,
                }
            }
        }

        result = deep_merge_dicts(base, overrides)

        assert result == {
            "level1": {
                "level2": {
                    "level3": {"a": 1, "b": 200},
                    "x": 10,
                    "z": 30,
                },
                "y": 20,
            }
        }

    def test_merge_override_with_different_type(self) -> None:
        """Test that overriding with different type replaces the value."""
        base = {"config": {"nested": {"value": 100}}}
        overrides = {"config": {"nested": "string_value"}}

        result = deep_merge_dicts(base, overrides)

        # When override is not a dict, it replaces the entire value
        assert result == {"config": {"nested": "string_value"}}


class TestValidateOverrideKeys:
    """Tests for validate_override_keys function."""

    def test_valid_override_keys(self) -> None:
        """Test validation passes for valid override keys."""
        base = {
            "dataset": {"name": "Dataset001", "fold": 0},
            "training": {"epochs": 100, "batch_size": 2},
        }
        overrides = {
            "training": {"epochs": 200},
        }

        # Should not raise any exception
        validate_override_keys(base, overrides)

    def test_invalid_top_level_key(self) -> None:
        """Test validation fails for non-existent top-level key."""
        base = {
            "dataset": {"name": "Dataset001"},
            "training": {"epochs": 100},
        }
        overrides = {
            "invalid_key": {"value": 123},
        }

        with pytest.raises(ConfigValidationError) as excinfo:
            validate_override_keys(base, overrides)

        assert "invalid_key" in str(excinfo.value)
        assert "does not exist in base config" in str(excinfo.value)

    def test_invalid_nested_key(self) -> None:
        """Test validation fails for non-existent nested key."""
        base = {
            "training": {"epochs": 100, "batch_size": 2},
        }
        overrides = {
            "training": {"non_existent_param": 999},
        }

        with pytest.raises(ConfigValidationError) as excinfo:
            validate_override_keys(base, overrides)

        assert "training.non_existent_param" in str(excinfo.value)
        assert "does not exist in base config" in str(excinfo.value)

    def test_invalid_deeply_nested_key(self) -> None:
        """Test validation fails for non-existent deeply nested key."""
        base = {
            "level1": {
                "level2": {
                    "valid_key": 123,
                }
            }
        }
        overrides = {
            "level1": {
                "level2": {
                    "invalid_key": 456,
                }
            }
        }

        with pytest.raises(ConfigValidationError) as excinfo:
            validate_override_keys(base, overrides)

        assert "level1.level2.invalid_key" in str(excinfo.value)

    def test_override_dict_where_base_is_not_dict(self) -> None:
        """Test validation fails when trying to override non-dict with dict."""
        base = {
            "training": {"epochs": 100},
        }
        overrides = {
            "training": {
                "epochs": {"nested": "value"},  # epochs is int in base, not dict
            }
        }

        with pytest.raises(ConfigValidationError) as excinfo:
            validate_override_keys(base, overrides)

        assert "training.epochs" in str(excinfo.value)
        assert "is a dict" in str(excinfo.value)

    def test_changing_type_allows_new_keys(self) -> None:
        """Test that changing 'type' field allows adding new keys (component replacement)."""
        base = {
            "loss": {
                "type": "DiceCELoss",
                "to_onehot_y": True,
                "softmax": True,
                "batch": True,
            }
        }
        overrides = {
            "loss": {
                "type": "BlobLoss",
                "base_loss": "DiceLoss",  # New key not in base
                "main_weight": 3.0,  # New key not in base
                "blob_weight": 1.0,  # New key not in base
            }
        }

        # Should not raise - type change means full replacement
        validate_override_keys(base, overrides)

    def test_type_change_in_model(self) -> None:
        """Test that changing model type allows different model-specific params."""
        base = {
            "model": {
                "type": "DynUNet",
                "spatial_dims": 3,
                "filters": [32, 64, 128],
            }
        }
        overrides = {
            "model": {
                "type": "UNet",
                "channels": [32, 64, 128],  # Different param name for UNet
                "strides": [2, 2],  # UNet-specific param
            }
        }

        # Should not raise - type change means full replacement
        validate_override_keys(base, overrides)


class TestLoadConfigWithInheritance:
    """Tests for load_config with base_config support."""

    def test_load_config_without_base(self, temp_dir: str) -> None:
        """Test loading regular config without base_config works as before."""
        config_path = os.path.join(temp_dir, "simple_config.yaml")
        config_data = {
            "dataset": {"name": "Dataset001"},
            "training": {"epochs": 100},
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        result = load_config(config_path)

        assert result == config_data

    def test_load_config_with_base_and_overrides(self, temp_dir: str) -> None:
        """Test loading config with base_config and overrides."""
        # Create base config
        base_config_path = os.path.join(temp_dir, "base_fold_0.yaml")
        base_config = {
            "dataset": {"name": "Dataset001", "fold": 0, "batch_size": 2},
            "training": {"epochs": 100, "learning_rate": 0.001},
        }
        with open(base_config_path, "w") as f:
            yaml.dump(base_config, f)

        # Create override config
        override_config_path = os.path.join(temp_dir, "quick_test.yaml")
        override_config = {
            "base_config": "base_fold_0.yaml",
            "overrides": {
                "training": {"epochs": 10},
            },
        }
        with open(override_config_path, "w") as f:
            yaml.dump(override_config, f)

        # Load override config
        result = load_config(override_config_path)

        # Should have merged config with overridden epochs
        assert result["dataset"]["name"] == "Dataset001"
        assert result["dataset"]["fold"] == 0
        assert result["training"]["epochs"] == 10  # Overridden
        assert result["training"]["learning_rate"] == 0.001  # From base

    def test_load_config_with_absolute_base_path(self, temp_dir: str) -> None:
        """Test loading config with absolute base_config path."""
        # Create base config
        base_config_path = os.path.join(temp_dir, "base_config.yaml")
        base_config = {
            "training": {"epochs": 100},
        }
        with open(base_config_path, "w") as f:
            yaml.dump(base_config, f)

        # Create override config in different directory
        override_dir = os.path.join(temp_dir, "experiments")
        os.makedirs(override_dir, exist_ok=True)
        override_config_path = os.path.join(override_dir, "override.yaml")
        override_config = {
            "base_config": base_config_path,  # Absolute path
            "overrides": {
                "training": {"epochs": 50},
            },
        }
        with open(override_config_path, "w") as f:
            yaml.dump(override_config, f)

        result = load_config(override_config_path)

        assert result["training"]["epochs"] == 50

    def test_load_config_with_relative_base_path(self, temp_dir: str) -> None:
        """Test loading config with relative base_config path."""
        # Create directory structure:
        # temp_dir/
        #   docs/datasets/Dataset001/fold_0.yaml
        #   experiments/test.yaml
        docs_dir = os.path.join(temp_dir, "docs", "datasets", "Dataset001")
        os.makedirs(docs_dir, exist_ok=True)

        base_config_path = os.path.join(docs_dir, "fold_0.yaml")
        base_config = {
            "training": {"epochs": 200},
        }
        with open(base_config_path, "w") as f:
            yaml.dump(base_config, f)

        experiments_dir = os.path.join(temp_dir, "experiments")
        os.makedirs(experiments_dir, exist_ok=True)
        override_config_path = os.path.join(experiments_dir, "test.yaml")
        override_config = {
            "base_config": "../docs/datasets/Dataset001/fold_0.yaml",
            "overrides": {
                "training": {"epochs": 400},
            },
        }
        with open(override_config_path, "w") as f:
            yaml.dump(override_config, f)

        result = load_config(override_config_path)

        assert result["training"]["epochs"] == 400

    def test_load_config_with_invalid_override_key(self, temp_dir: str) -> None:
        """Test that invalid override keys raise ConfigValidationError."""
        # Create base config
        base_config_path = os.path.join(temp_dir, "base.yaml")
        base_config = {
            "training": {"epochs": 100},
        }
        with open(base_config_path, "w") as f:
            yaml.dump(base_config, f)

        # Create override config with invalid key
        override_config_path = os.path.join(temp_dir, "override.yaml")
        override_config = {
            "base_config": "base.yaml",
            "overrides": {
                "training": {"invalid_param": 999},
            },
        }
        with open(override_config_path, "w") as f:
            yaml.dump(override_config, f)

        with pytest.raises(ConfigValidationError) as excinfo:
            load_config(override_config_path)

        assert "invalid_param" in str(excinfo.value)

    def test_load_config_multiple_nested_overrides(self, temp_dir: str) -> None:
        """Test loading config with multiple nested overrides."""
        # Create base config
        base_config_path = os.path.join(temp_dir, "base.yaml")
        base_config = {
            "dataset": {
                "name": "Dataset001",
                "spatial_size": [64, 64, 64],
                "cache": {"enabled": True, "rate": 1.0},
            },
            "training": {
                "epochs": 100,
                "batch_size": 2,
                "learning_rate": 0.001,
            },
            "optimizer": {
                "type": "Adam",
                "weight_decay": 0.0001,
            },
        }
        with open(base_config_path, "w") as f:
            yaml.dump(base_config, f)

        # Create override config
        override_config_path = os.path.join(temp_dir, "override.yaml")
        override_config = {
            "base_config": "base.yaml",
            "overrides": {
                "dataset": {
                    "cache": {"rate": 0.5},  # Only override cache rate
                },
                "training": {
                    "epochs": 400,
                    "batch_size": 4,
                },
                "optimizer": {
                    "weight_decay": 0.00001,
                },
            },
        }
        with open(override_config_path, "w") as f:
            yaml.dump(override_config, f)

        result = load_config(override_config_path)

        # Check that overrides were applied
        assert result["training"]["epochs"] == 400
        assert result["training"]["batch_size"] == 4
        assert result["training"]["learning_rate"] == 0.001  # Not overridden

        assert result["dataset"]["cache"]["rate"] == 0.5  # Overridden
        assert result["dataset"]["cache"]["enabled"] is True  # Not overridden
        assert result["dataset"]["name"] == "Dataset001"  # Not overridden

        assert result["optimizer"]["weight_decay"] == 0.00001  # Overridden
        assert result["optimizer"]["type"] == "Adam"  # Not overridden


@pytest.fixture
def temp_dir() -> str:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
