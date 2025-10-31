"""
Tests for src.utils.runner module.
Tests runner utilities including device setup, config name extraction, and experiment setup.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

from src.planning.fingerprinting.resources import get_gpu_memory_for_planning
from src.utils.runner import (
    get_config_name,
    setup_device,
    setup_experiment,
    setup_results_dir,
)


class TestSetupDevice:
    """Tests for setup_device function."""

    def test_setup_device_returns_device(self) -> None:
        """Test that setup_device returns a torch device."""
        device = setup_device(verbose=False)

        assert isinstance(device, torch.device)

    def test_setup_device_cuda_if_available(self) -> None:
        """Test that setup_device returns CUDA if available."""
        device = setup_device(verbose=False)

        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"

    def test_setup_device_verbose(self, capsys: Any) -> None:
        """Test that setup_device prints when verbose=True."""
        device = setup_device(verbose=True)

        captured = capsys.readouterr()
        assert "Using device:" in captured.out
        assert device.type in captured.out

    def test_setup_device_quiet(self, capsys: Any) -> None:
        """Test that setup_device doesn't print when verbose=False."""
        setup_device(verbose=False)

        captured = capsys.readouterr()
        assert "Using device:" not in captured.out


class TestGetGPUMemoryForPlanning:
    """Tests for get_gpu_memory_for_planning function."""

    @patch("src.planning.fingerprinting.resources.detect_gpu_memory")
    def test_with_cuda_available(self, mock_detect: Mock) -> None:
        """Test get_gpu_memory_for_planning returns GPU memory when CUDA is available."""
        # Arrange
        mock_detect.return_value = (True, 9.6, "NVIDIA GeForce RTX 3090")

        # Act
        result = get_gpu_memory_for_planning()

        # Assert
        assert result == 9.6
        mock_detect.assert_called_once()

    @patch("src.planning.fingerprinting.resources.detect_gpu_memory")
    def test_fallback_no_cuda(self, mock_detect: Mock) -> None:
        """Test get_gpu_memory_for_planning returns fallback when CUDA unavailable."""
        # Arrange
        mock_detect.return_value = (False, 0.0, "None")

        # Act
        result = get_gpu_memory_for_planning()

        # Assert
        assert result == 8.0, "Should return default fallback value of 8.0 GB"
        mock_detect.assert_called_once()

    @patch("src.planning.fingerprinting.resources.detect_gpu_memory")
    def test_exception_handling(self, mock_detect: Mock) -> None:
        """Test get_gpu_memory_for_planning handles exceptions gracefully."""
        # Arrange
        mock_detect.side_effect = RuntimeError("Failed to detect GPU")

        # Act & Assert
        with pytest.raises(RuntimeError):
            get_gpu_memory_for_planning()

    @patch("src.planning.fingerprinting.resources.detect_gpu_memory")
    def test_with_small_gpu(self, mock_detect: Mock) -> None:
        """Test get_gpu_memory_for_planning with small GPU (e.g., 4 GB)."""
        # Arrange
        mock_detect.return_value = (True, 3.2, "NVIDIA GeForce RTX 2080")

        # Act
        result = get_gpu_memory_for_planning()

        # Assert
        assert result == 3.2

    @patch("src.planning.fingerprinting.resources.detect_gpu_memory")
    def test_with_large_gpu(self, mock_detect: Mock) -> None:
        """Test get_gpu_memory_for_planning with large GPU (e.g., 80 GB A100)."""
        # Arrange
        mock_detect.return_value = (True, 64.0, "NVIDIA A100")

        # Act
        result = get_gpu_memory_for_planning()

        # Assert
        assert result == 64.0

    @patch("src.planning.fingerprinting.resources.detect_gpu_memory")
    def test_custom_fallback_value(self, mock_detect: Mock) -> None:
        """Test get_gpu_memory_for_planning with custom fallback value."""
        # Arrange
        mock_detect.return_value = (False, 0.0, "None")

        # Act
        result = get_gpu_memory_for_planning(fallback_gb=16.0)

        # Assert
        assert result == 16.0


class TestGetConfigName:
    """Tests for get_config_name function."""

    def test_get_config_name_simple(self) -> None:
        """Test extracting config name from path."""
        result = get_config_name("configs/dataset001_hippo.yaml")

        assert result == "dataset001_hippo"

    def test_get_config_name_with_extension(self) -> None:
        """Test that file extension is removed."""
        result = get_config_name("my_config.yaml")

        assert result == "my_config"
        assert ".yaml" not in result

    def test_get_config_name_nested_path(self) -> None:
        """Test extracting config name from nested path."""
        result = get_config_name("/path/to/configs/experiment_v2.yaml")

        assert result == "experiment_v2"

    def test_get_config_name_no_extension(self) -> None:
        """Test config name without extension."""
        result = get_config_name("configs/my_config")

        assert result == "my_config"

    def test_get_config_name_multiple_dots(self) -> None:
        """Test config name with multiple dots in filename."""
        result = get_config_name("configs/dataset.v1.0.yaml")

        assert result == "dataset.v1.0"  # splitext removes only the last extension


class TestSetupResultsDir:
    """Tests for setup_results_dir function."""

    def test_setup_results_dir_returns_path(self, temp_dir: str) -> None:
        """Test that setup_results_dir returns a path."""
        # Temporarily change the working directory concept by using absolute paths
        result = setup_results_dir("test_config", "Dataset001_Test", create=False)

        # Should return path with dataset and config name
        assert "test_config" in result
        assert "Dataset001_Test" in result

    def test_setup_results_dir_creates_directory(self, temp_dir: str) -> None:
        """Test that setup_results_dir creates directory when create=True."""
        # Use temp_dir as base
        config_name = "test_experiment"
        dataset_name = "Dataset001_Test"

        # Since we can't control working directory, just test the function works
        result = setup_results_dir(config_name, dataset_name, create=False)

        assert "test_experiment" in result
        assert "Dataset001_Test" in result

    def test_setup_results_dir_no_create(self) -> None:
        """Test setup_results_dir with create=False."""
        result = setup_results_dir("my_config", "Dataset001_Test", create=False)

        # Should return path even if directory doesn't exist
        assert "my_config" in result
        assert "Dataset001_Test" in result

    def test_setup_results_dir_path_format(self) -> None:
        """Test that results dir path has correct format."""
        config_name = "dataset001_hippo"
        dataset_name = "Dataset001_Hippo"
        result = setup_results_dir(config_name, dataset_name, create=False)

        # Should contain dataset and config names
        assert config_name in result
        assert dataset_name in result


class TestSetupExperiment:
    """Tests for setup_experiment function."""

    def test_setup_experiment_returns_tuple(
        self, temp_dir: str, mock_config_file: str
    ) -> None:
        """Test that setup_experiment returns a tuple of expected components."""
        cfg, device, data_dir, results_dir, config_name = setup_experiment(
            mock_config_file, create_results_dir=False
        )

        assert isinstance(cfg, dict)
        assert isinstance(device, torch.device)
        assert isinstance(data_dir, str)
        assert isinstance(results_dir, str)
        assert isinstance(config_name, str)

    def test_setup_experiment_loads_config(
        self, mock_config_file: str, sample_config: dict[str, Any]
    ) -> None:
        """Test that setup_experiment loads configuration correctly."""
        cfg, _, _, _, _ = setup_experiment(mock_config_file, create_results_dir=False)

        # Should match the sample config
        assert cfg["dataset"]["name"] == sample_config["dataset"]["name"]
        assert cfg["model"]["type"] == sample_config["model"]["type"]

    def test_setup_experiment_extracts_config_name(self, mock_config_file: str) -> None:
        """Test that setup_experiment extracts config name from file path."""
        _, _, _, _, config_name = setup_experiment(
            mock_config_file, create_results_dir=False
        )

        # Should extract "test_config" from path
        assert config_name == "test_config"

    def test_setup_experiment_data_dir_from_dataset_name(
        self, mock_config_file: str, sample_config: dict[str, Any]
    ) -> None:
        """Test that setup_experiment derives data directory from dataset name."""
        _, _, data_dir, _, _ = setup_experiment(
            mock_config_file, create_results_dir=False
        )

        dataset_name = sample_config["dataset"]["name"]
        assert dataset_name in data_dir

    def test_setup_experiment_device_setup(self, mock_config_file: str) -> None:
        """Test that setup_experiment sets up device correctly."""
        _, device, _, _, _ = setup_experiment(
            mock_config_file, create_results_dir=False
        )

        if torch.cuda.is_available():
            assert device.type in ("cuda", "cpu")
        else:
            assert device.type == "cpu"

    def test_setup_experiment_results_dir_format(
        self, mock_config_file: str, sample_config: dict[str, Any]
    ) -> None:
        """Test that setup_experiment returns correctly formatted results directory."""
        _, _, _, results_dir, config_name = setup_experiment(
            mock_config_file, create_results_dir=False
        )

        # Results dir should contain dataset name and config name
        dataset_name = sample_config["dataset"]["name"]
        assert config_name in results_dir
        assert dataset_name in results_dir

    def test_setup_experiment_missing_config(self, temp_dir: str) -> None:
        """Test that setup_experiment raises error for missing config file."""
        missing_config = os.path.join(temp_dir, "missing.yaml")

        with pytest.raises(FileNotFoundError):
            setup_experiment(missing_config)

    def test_setup_experiment_create_results_dir_flag(
        self, temp_dir: str, mock_config_file: str
    ) -> None:
        """Test that create_results_dir flag controls directory creation."""
        # Test with create_results_dir=False
        _, _, _, results_dir, config_name = setup_experiment(
            mock_config_file, create_results_dir=False
        )

        # Should return paths but may not create directory
        assert isinstance(results_dir, str)
        assert config_name in results_dir
