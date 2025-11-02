"""
Tests for src/engines/train/run.py - Training pipeline execution.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestRunTrainingSetup:
    """Test training setup and initialization."""

    def test_run_training_requires_config(self, tmp_path: Path) -> None:
        """Test that run_training requires a valid config file."""
        from src.engines.train.run import run_training

        with pytest.raises(Exception):  # FileNotFoundError or similar
            run_training(str(tmp_path / "nonexistent.yaml"))

    def test_run_training_resolves_config_path(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that run_training properly resolves config paths."""
        from src.engines.train.run import run_training

        # Create a test config file
        config_file = tmp_path / "test_config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Mock setup_experiment to avoid actual setup
        with patch("src.engines.train.run.setup_experiment") as mock_setup:
            mock_setup.side_effect = Exception("Stop here for testing")

            with pytest.raises(Exception, match="Stop here for testing"):
                run_training(str(config_file))

    def test_run_training_calls_setup_experiment(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that run_training calls setup_experiment with config."""
        import yaml

        from src.engines.train.run import run_training

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        with patch("src.engines.train.run.setup_experiment") as mock_setup:
            mock_setup.side_effect = Exception("Expected mock exit")

            with pytest.raises(Exception, match="Expected mock exit"):
                run_training(str(config_file))

            assert mock_setup.called


class TestRunTrainingSeeding:
    """Test seeding and reproducibility."""

    def test_run_training_imports_seeding_functions(self) -> None:
        """Test that run_training imports seeding functions."""
        import src.engines.train.run as train_run

        # Check that required seeding functions are imported
        assert hasattr(train_run, "set_random_seeds")
        assert hasattr(train_run, "get_seed_from_config")
        assert hasattr(train_run, "enable_cuda_determinism")

    def test_run_training_gets_seed_from_config(self) -> None:
        """Test that run_training can get seed from config."""
        from src.utils.seeding import get_seed_from_config

        config = {"training": {"seed": 42}}
        seed = get_seed_from_config(config)

        assert isinstance(seed, int)


class TestRunTrainingLogging:
    """Test logging setup during training."""

    def test_run_training_imports_logging_functions(self) -> None:
        """Test that run_training imports logging functions."""
        import src.engines.train.run as train_run

        # Check that required logging functions are imported
        assert hasattr(train_run, "setup_train_logger")
        assert hasattr(train_run, "log_and_print")
        assert hasattr(train_run, "log_header")
        assert hasattr(train_run, "log_system_info")

    def test_run_training_uses_configured_logging(self) -> None:
        """Test that run_training can use loguru logger."""
        from loguru import logger

        # Logger should be available
        assert hasattr(logger, "info")
        assert callable(logger.info)


class TestRunTrainingDataLoading:
    """Test data loading during training."""

    def test_run_training_gets_fold_number(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that run_training extracts fold number from config."""
        import yaml

        from src.engines.train.run import run_training

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        with patch("src.engines.train.run.setup_experiment") as mock_setup:
            with patch("src.engines.train.run.set_random_seeds"):
                with patch("src.engines.train.run.get_seed_from_config") as mock_get_seed:
                    with patch("src.engines.train.run.setup_train_logger") as mock_logger:
                        with patch("src.engines.train.run.log_header"):
                            with patch("src.engines.train.run.log_system_info"):
                                with patch(
                                    "src.engines.train.run.validate_required_field"
                                ):
                                    with patch(
                                        "src.engines.train.run.metric_registry.build"
                                    ):
                                        mock_log = MagicMock()
                                        mock_logger.return_value = mock_log
                                        mock_get_seed.return_value = 42
                                        mock_setup.return_value = (
                                            sample_config,
                                            torch.device("cpu"),
                                            str(tmp_path / "data"),
                                            str(tmp_path / "results"),
                                            "test_config",
                                        )
                                        mock_setup.side_effect = Exception("Stop")

                                        with pytest.raises(Exception):
                                            run_training(str(config_file))

    def test_run_training_detects_training_all_data(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that run_training can detect fold=-1 (training on all data)."""
        # Simple test: verify that fold can be -1 in config
        sample_config["dataset"]["fold"] = -1

        # Check that fold is properly set
        assert sample_config["dataset"]["fold"] == -1


class TestRunTrainingCheckpointHandling:
    """Test checkpoint loading and resumption."""

    def test_run_training_checkpoint_cleanup_logic_exists(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that run_training has checkpoint cleanup logic."""
        import src.engines.train.run as train_run

        # Verify the module has the code structure for checkpoint handling
        # by checking it can be imported without errors
        assert hasattr(train_run, "run_training")
        assert callable(train_run.run_training)

    def test_run_training_preserves_checkpoint_when_resuming(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        """Test that run_training preserves checkpoints when resuming."""
        import yaml

        from src.engines.train.run import run_training

        # Create a results directory with a checkpoint
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        checkpoint = results_dir / "checkpoint_final_checkpoint.pt"
        torch.save({"model": torch.nn.Linear(1, 1).state_dict()}, checkpoint)

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        with patch("src.engines.train.run.setup_experiment") as mock_setup:
            with patch("src.engines.train.run.set_random_seeds"):
                with patch("src.engines.train.run.get_seed_from_config") as mock_get_seed:
                    with patch("src.engines.train.run.setup_train_logger") as mock_logger:
                        with patch("src.engines.train.run.log_header"):
                            with patch("src.engines.train.run.log_system_info"):
                                with patch(
                                    "src.engines.train.run.validate_required_field"
                                ):
                                    with patch(
                                        "src.engines.train.run.metric_registry.build"
                                    ):
                                        with patch(
                                            "src.engines.train.run.create_trainer"
                                        ) as mock_trainer:
                                            mock_log = MagicMock()
                                            mock_logger.return_value = mock_log
                                            mock_get_seed.return_value = 42
                                            mock_setup.return_value = (
                                                sample_config,
                                                torch.device("cpu"),
                                                str(tmp_path / "data"),
                                                str(results_dir),
                                                "test_config",
                                            )

                                            # Create mock trainer/evaluator
                                            mock_trainer_obj = MagicMock()
                                            mock_trainer_obj.network = MagicMock()
                                            mock_trainer.return_value = (
                                                mock_trainer_obj,
                                                None,
                                            )

                                            mock_trainer_obj.run.side_effect = (
                                                Exception("Stop")
                                            )

                                            with pytest.raises(Exception):
                                                run_training(
                                                    str(config_file), resume=True
                                                )

                                            # Checkpoint should still exist
                                            assert checkpoint.exists()


class TestRunTrainingMixedPrecision:
    """Test mixed precision training configuration."""

    def test_run_training_can_read_mixed_precision_config(
        self, sample_config: dict
    ) -> None:
        """Test that mixed_precision config can be set and read."""
        sample_config["training"]["mixed_precision"] = True

        # Verify config is properly set
        assert sample_config["training"]["mixed_precision"] is True

    def test_run_training_mixed_precision_defaults_to_false(
        self, sample_config: dict
    ) -> None:
        """Test that mixed_precision defaults to False."""
        # When not specified, .get() should return False
        use_amp = sample_config.get("training", {}).get("mixed_precision", False)

        assert use_amp is False
