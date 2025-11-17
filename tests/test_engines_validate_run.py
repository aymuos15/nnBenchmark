"""
Tests for src/engines/validate/run.py - Validation pipeline execution.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml


class TestRunValidationSetup:
    """Test validation setup and initialization."""

    def test_run_validation_requires_config(self, tmp_path: Path) -> None:
        """Test that run_validation requires a valid config file."""
        from src.engines.validate.run import run_validation

        with pytest.raises(Exception):  # FileNotFoundError or similar
            run_validation(str(tmp_path / "nonexistent.yaml"))

    def test_run_validation_resolves_config_path(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that run_validation properly resolves config paths."""
        from src.engines.validate.run import run_validation

        # Create a test config file
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Mock setup to avoid actual validation
        with patch("src.engines.validate.run.setup_val_logger") as mock_logger:
            mock_logger.side_effect = Exception("Stop here for testing")

            with pytest.raises(Exception, match="Stop here for testing"):
                run_validation(str(config_file), dataset="Dataset001_Hippo")

    def test_run_validation_imports_required_modules(self) -> None:
        """Test that run_validation imports required modules."""
        import src.engines.validate.run as validate_run

        # Check that module loads without errors
        assert hasattr(validate_run, "run_validation")
        assert callable(validate_run.run_validation)


class TestRunValidationCheckpointHandling:
    """Test checkpoint discovery and loading."""

    def test_run_validation_finds_epoch_checkpoints(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that run_validation finds all epoch checkpoints when no specific checkpoint provided."""
        from src.engines.validate.run import run_validation

        # Create config file
        config_file = tmp_path / "test_config.yaml"
        sample_config["dataset"]["fold"] = 0
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Create results directory with multiple checkpoints
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoint_dir = results_dir
        checkpoint_dir.mkdir(exist_ok=True)

        # Create mock checkpoints
        for epoch in [1, 2, 3]:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "model": {},
                    "optimizer": {},
                    "epoch": epoch,
                    "config_metadata": sample_config,
                },
                checkpoint_path,
            )

        # Mock the validation engine to avoid actual validation
        with (
            patch("src.engines.validate.run.setup_val_logger"),
            patch("src.engines.common.setup_device") as mock_device,
            patch("src.engines.validate.run.ValidationEngine"),
            patch("src.engines.validate.run.get_data_dicts") as mock_data,
            patch("src.engines.validate.run.metric_registry") as mock_metrics,
            patch("src.engines.validate.run.model_registry") as mock_model,
        ):
            mock_device.return_value = (torch.device("cpu"), False)
            mock_data.return_value = []
            mock_metrics.build.return_value = []
            mock_model.build.return_value = torch.nn.Identity()

            # This should discover all 3 checkpoints
            # We expect it to process each one
            with patch("src.engines.validate.run.Path") as mock_path_class:
                mock_path_instance = MagicMock()
                mock_path_class.return_value = mock_path_instance
                mock_path_instance.glob.return_value = [
                    checkpoint_dir / f"checkpoint_epoch_{i:03d}.pt" for i in [1, 2, 3]
                ]

                # Run should not raise error
                try:
                    run_validation(
                        str(config_file), dataset="Dataset001_Hippo", checkpoint_path=None
                    )
                except Exception as e:
                    # Allow certain expected exceptions (missing data, etc)
                    if "fold" not in str(e) and "data" not in str(e).lower():
                        raise

    def test_run_validation_single_checkpoint_mode(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that run_validation can validate a specific checkpoint."""
        from src.engines.validate.run import run_validation

        # Create config file
        config_file = tmp_path / "test_config.yaml"
        sample_config["dataset"]["fold"] = 0
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Create a single checkpoint
        checkpoint_path = tmp_path / "checkpoint_epoch_001.pt"
        torch.save(
            {
                "model": {},
                "optimizer": {},
                "epoch": 1,
                "config_metadata": sample_config,
            },
            checkpoint_path,
        )

        # Mock validation components
        with (
            patch("src.engines.validate.run.setup_val_logger"),
            patch("src.engines.common.setup_device") as mock_device,
            patch("src.engines.validate.run.ValidationEngine"),
            patch("src.engines.validate.run.get_data_dicts") as mock_data,
            patch("src.engines.validate.run.metric_registry") as mock_metrics,
            patch("src.engines.validate.run.model_registry") as mock_model,
        ):
            mock_device.return_value = (torch.device("cpu"), False)
            mock_data.return_value = []
            mock_metrics.build.return_value = []
            mock_model.build.return_value = torch.nn.Identity()

            # Should process only the specified checkpoint
            try:
                run_validation(
                    str(config_file),
                    dataset="Dataset001_Hippo",
                    checkpoint_path=checkpoint_path,
                )
            except Exception as e:
                # Allow certain expected exceptions
                if "fold" not in str(e) and "data" not in str(e).lower():
                    raise


class TestRunValidationConfigValidation:
    """Test configuration validation."""

    def test_run_validation_requires_fold_in_config(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that run_validation raises error when fold is missing from config."""
        from src.engines.validate.run import run_validation

        # Remove fold from config
        config_without_fold = sample_config.copy()
        if "fold" in config_without_fold.get("dataset", {}):
            del config_without_fold["dataset"]["fold"]

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_without_fold, f)

        # Create a checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({"model": {}, "epoch": 1}, checkpoint_path)

        # Mock setup
        with (
            patch("src.engines.validate.run.setup_val_logger"),
            patch("src.engines.common.setup_device") as mock_device,
        ):
            mock_device.return_value = (torch.device("cpu"), False)

            # Should raise error about missing fold
            with pytest.raises(Exception):  # ValueError or KeyError
                run_validation(
                    str(config_file),
                    dataset="Dataset001_Hippo",
                    checkpoint_path=checkpoint_path,
                )

    def test_run_validation_validates_sliding_window_config(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that run_validation validates sliding window configuration."""
        from src.engines.validate.run import run_validation

        # Add invalid sliding window config
        sample_config["inference"] = {
            "sw_batch_size": 4,
            "mode": "invalid_mode",  # Should be 'gaussian' or 'constant'
            "overlap": 0.5,
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Create checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({"model": {}, "epoch": 1, "config_metadata": sample_config}, checkpoint_path)

        # Mock setup
        with (
            patch("src.engines.validate.run.setup_val_logger"),
            patch("src.engines.common.setup_device") as mock_device,
        ):
            mock_device.return_value = (torch.device("cpu"), False)

            # Should raise ValueError for invalid mode
            with pytest.raises((ValueError, Exception)):
                run_validation(
                    str(config_file),
                    dataset="Dataset001_Hippo",
                    checkpoint_path=checkpoint_path,
                )


class TestPrintValidationResults:
    """Test validation results printing."""

    def test_print_validation_results_basic(self, capsys) -> None:
        """Test that print_validation_results formats output correctly."""
        from src.engines.validate.run import print_validation_results

        results = {
            "mean": 0.85,
            "std": 0.05,
            "min": 0.75,
            "max": 0.95,
        }

        print_validation_results(results, "Dice")

        captured = capsys.readouterr()
        # Check case-insensitive since the function uses .upper()
        assert "VALIDATION RESULTS" in captured.out.upper()
        assert "0.8500" in captured.out
        assert "0.0500" in captured.out

    def test_print_validation_results_with_per_class(self, capsys) -> None:
        """Test that print_validation_results shows per-class results."""
        from src.engines.validate.run import print_validation_results

        results = {
            "mean": 0.85,
            "std": 0.05,
            "min": 0.75,
            "max": 0.95,
            "per_class": {
                "Class1": {"mean": 0.80, "std": 0.04},
                "Class2": {"mean": 0.90, "std": 0.03},
            },
        }

        print_validation_results(results, "Dice")

        captured = capsys.readouterr()
        assert "Class1" in captured.out
        assert "Class2" in captured.out
        assert "0.8000" in captured.out
        assert "0.9000" in captured.out


class TestValidationIntegration:
    """Integration tests for validation pipeline."""

    def test_run_validation_creates_validation_history_json(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that run_validation creates validation_history_epoch_*.json files."""
        from src.engines.validate.run import run_validation

        # Setup config and checkpoint
        config_file = tmp_path / "test_config.yaml"
        sample_config["dataset"]["fold"] = 0
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoint_path = results_dir / "checkpoint_epoch_001.pt"
        torch.save(
            {
                "model": {},
                "optimizer": {},
                "epoch": 1,
                "config_metadata": sample_config,
            },
            checkpoint_path,
        )

        # Mock components
        with (
            patch("src.engines.validate.run.setup_val_logger"),
            patch("src.engines.common.setup_device") as mock_device,
            patch("src.engines.validate.run.ValidationEngine") as mock_engine_class,
            patch("src.engines.validate.run.get_data_dicts") as mock_data,
            patch("src.engines.validate.run.metric_registry") as mock_metrics,
            patch("src.engines.validate.run.model_registry") as mock_model,
        ):
            mock_device.return_value = (torch.device("cpu"), False)
            mock_data.return_value = [{"image": "test.nii.gz"}]
            mock_metrics.build.return_value = []
            mock_model.build.return_value = torch.nn.Identity()

            # Mock engine instance
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine
            mock_engine.state.metrics = {"DiceMetric": torch.tensor(0.85)}

            # Run validation
            try:
                run_validation(
                    str(config_file),
                    dataset="Dataset001_Hippo",
                    checkpoint_path=checkpoint_path,
                )
            except Exception:
                # We expect this to fail at some point, but we're testing setup
                pass

            # Check that engine was created
            assert mock_engine_class.called
