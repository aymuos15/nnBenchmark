"""
Unit tests for GPU detection functionality.
Tests get_gpu_memory_for_planning function from planning module.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from src.planning.fingerprinting.resources import get_gpu_memory_for_planning


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
