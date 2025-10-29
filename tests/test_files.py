"""
Tests for src.utils.files module.
Tests file system utilities including case ID extraction, JSON I/O, and directory creation.
"""

from __future__ import annotations

import json
import os

import pytest

from src.utils.files import (
    ensure_directory,
    extract_base_name_for_label,
    extract_case_id,
    load_json,
    save_json,
)


class TestExtractCaseId:
    """Tests for extract_case_id function."""

    def test_nifti_with_channel_suffix_removed(self) -> None:
        """Test NIfTI filename with channel suffix removed."""
        result = extract_case_id("Hippo_001_0000.nii.gz", remove_channel_suffix=True)
        assert result == "Hippo_001"

    def test_nifti_with_channel_suffix_kept(self) -> None:
        """Test NIfTI filename with channel suffix kept."""
        result = extract_case_id("Hippo_001_0000.nii.gz", remove_channel_suffix=False)
        assert result == "Hippo_001_0000"

    def test_jpg_with_channel_suffix_removed(self) -> None:
        """Test JPG filename with channel suffix removed."""
        result = extract_case_id("ISIC_0000000_0000.jpg", remove_channel_suffix=True)
        assert result == "ISIC_0000000"

    def test_jpg_with_channel_suffix_kept(self) -> None:
        """Test JPG filename with channel suffix kept."""
        result = extract_case_id("ISIC_0000000_0000.jpg", remove_channel_suffix=False)
        assert result == "ISIC_0000000_0000"

    def test_full_path(self) -> None:
        """Test with full path instead of just filename."""
        result = extract_case_id("/path/to/Hippo_001_0000.nii.gz")
        assert result == "Hippo_001"

    def test_no_channel_suffix(self) -> None:
        """Test filename without channel suffix pattern."""
        result = extract_case_id("case_001.nii.gz")
        assert result == "case_001"

    def test_non_standard_suffix(self) -> None:
        """Test filename with non-standard suffix (not 4 digits)."""
        result = extract_case_id("case_001_00.nii.gz")
        assert result == "case_001_00"


class TestExtractBaseNameForLabel:
    """Tests for extract_base_name_for_label function."""

    def test_nifti_image(self) -> None:
        """Test NIfTI image filename."""
        base_name, label_ext = extract_base_name_for_label("Hippo_001_0000.nii.gz")
        assert base_name == "Hippo_001"
        assert label_ext == ".nii.gz"

    def test_jpg_image(self) -> None:
        """Test JPG image filename (should return PNG for label)."""
        base_name, label_ext = extract_base_name_for_label("ISIC_0000000_0000.jpg")
        assert base_name == "ISIC_0000000"
        assert label_ext == ".png"

    def test_jpeg_image(self) -> None:
        """Test JPEG image filename (should return PNG for label)."""
        base_name, label_ext = extract_base_name_for_label("image_001_0000.jpeg")
        assert base_name == "image_001"
        assert label_ext == ".png"

    def test_png_image(self) -> None:
        """Test PNG image filename."""
        base_name, label_ext = extract_base_name_for_label("image_001_0000.png")
        assert base_name == "image_001"
        assert label_ext == ".png"


class TestEnsureDirectory:
    """Tests for ensure_directory function."""

    def test_create_new_directory(self, temp_dir: str) -> None:
        """Test creating a new directory."""
        new_dir = os.path.join(temp_dir, "new_directory")
        result = ensure_directory(new_dir)

        assert result == new_dir
        assert os.path.exists(new_dir)
        assert os.path.isdir(new_dir)

    def test_create_nested_directories(self, temp_dir: str) -> None:
        """Test creating nested directories."""
        nested_dir = os.path.join(temp_dir, "level1", "level2", "level3")
        result = ensure_directory(nested_dir)

        assert result == nested_dir
        assert os.path.exists(nested_dir)
        assert os.path.isdir(nested_dir)

    def test_existing_directory(self, temp_dir: str) -> None:
        """Test with existing directory (should not raise error)."""
        result = ensure_directory(temp_dir)
        assert result == temp_dir
        assert os.path.exists(temp_dir)


class TestLoadJson:
    """Tests for load_json function."""

    def test_load_valid_json(self, temp_dir: str) -> None:
        """Test loading valid JSON file."""
        test_data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        json_path = os.path.join(temp_dir, "test.json")

        with open(json_path, "w") as f:
            json.dump(test_data, f)

        result = load_json(json_path, "test JSON")
        assert result == test_data

    def test_load_missing_file(self, temp_dir: str) -> None:
        """Test loading non-existent JSON file raises FileNotFoundError."""
        missing_path = os.path.join(temp_dir, "missing.json")

        with pytest.raises(FileNotFoundError) as excinfo:
            load_json(missing_path, "test JSON")

        assert "test JSON not found" in str(excinfo.value)
        assert missing_path in str(excinfo.value)


class TestSaveJson:
    """Tests for save_json function."""

    def test_save_json(self, temp_dir: str) -> None:
        """Test saving data to JSON file."""
        test_data = {"epoch": 1, "loss": 0.5, "metrics": [0.7, 0.8]}
        json_path = os.path.join(temp_dir, "output.json")

        save_json(test_data, json_path)

        assert os.path.exists(json_path)

        # Verify contents
        with open(json_path, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data

    def test_save_json_custom_indent(self, temp_dir: str) -> None:
        """Test saving JSON with custom indentation."""
        test_data = {"key": "value"}
        json_path = os.path.join(temp_dir, "output.json")

        save_json(test_data, json_path, indent=4)

        # Check that file was created
        assert os.path.exists(json_path)

    def test_save_json_overwrites_existing(self, temp_dir: str) -> None:
        """Test that saving JSON overwrites existing file."""
        json_path = os.path.join(temp_dir, "output.json")

        # Save first data
        save_json({"old": "data"}, json_path)

        # Save new data (should overwrite)
        new_data = {"new": "data"}
        save_json(new_data, json_path)

        # Verify only new data exists
        loaded_data = load_json(json_path, "test")
        assert loaded_data == new_data
        assert "old" not in loaded_data
