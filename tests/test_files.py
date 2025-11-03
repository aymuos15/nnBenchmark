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

    @pytest.mark.parametrize(
        "filename,remove_suffix,expected",
        [
            pytest.param(
                "Hippo_001_0000.nii.gz", True, "Hippo_001", id="nifti_remove_suffix"
            ),
            pytest.param(
                "Hippo_001_0000.nii.gz", False, "Hippo_001_0000", id="nifti_keep_suffix"
            ),
            pytest.param(
                "ISIC_0000000_0000.jpg", True, "ISIC_0000000", id="jpg_remove_suffix"
            ),
            pytest.param(
                "ISIC_0000000_0000.jpg",
                False,
                "ISIC_0000000_0000",
                id="jpg_keep_suffix",
            ),
            pytest.param(
                "/path/to/Hippo_001_0000.nii.gz", True, "Hippo_001", id="full_path"
            ),
            pytest.param("case_001.nii.gz", True, "case_001", id="no_channel_suffix"),
            pytest.param(
                "case_001_00.nii.gz", True, "case_001_00", id="non_standard_suffix"
            ),
        ],
    )
    def test_extract_case_id(
        self, filename: str, remove_suffix: bool, expected: str
    ) -> None:
        """Test extract_case_id with various filename patterns.

        Parameters:
        - filename: Input filename to extract case ID from
        - remove_suffix: Whether to remove channel suffix (default True)
        - expected: Expected case ID result
        """
        result = extract_case_id(filename, remove_channel_suffix=remove_suffix)
        assert result == expected


class TestExtractBaseNameForLabel:
    """Tests for extract_base_name_for_label function."""

    @pytest.mark.parametrize(
        "filename,expected_base,expected_ext",
        [
            pytest.param("Hippo_001_0000.nii.gz", "Hippo_001", ".nii.gz", id="nifti"),
            pytest.param(
                "ISIC_0000000_0000.jpg", "ISIC_0000000", ".png", id="jpg_returns_png"
            ),
            pytest.param(
                "image_001_0000.jpeg", "image_001", ".png", id="jpeg_returns_png"
            ),
            pytest.param("image_001_0000.png", "image_001", ".png", id="png"),
        ],
    )
    def test_extract_base_name_for_label(
        self, filename: str, expected_base: str, expected_ext: str
    ) -> None:
        """Test extract_base_name_for_label with various image formats.

        Parameters:
        - filename: Input filename
        - expected_base: Expected base name without suffix and extension
        - expected_ext: Expected label extension (images converted to PNG)
        """
        base_name, label_ext = extract_base_name_for_label(filename)
        assert base_name == expected_base
        assert label_ext == expected_ext


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
