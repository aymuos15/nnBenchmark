"""
File system utilities for nnBenchmark.
Provides helpers for directory creation, case ID extraction, file path handling, JSON I/O, and image loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from monai.transforms.io.dictionary import LoadImaged
from numpy.typing import NDArray


def detect_file_type(filepath: str) -> str:
    """
    Detect file type from filepath extension.

    Args:
        filepath: Path or filename to check

    Returns:
        File type: 'nifti', 'png', 'jpeg', or 'unknown'
    """
    filepath_lower = filepath.lower()

    if filepath_lower.endswith(".nii.gz") or filepath_lower.endswith(".nii"):
        return "nifti"
    elif filepath_lower.endswith(".png"):
        return "png"
    elif filepath_lower.endswith(".jpg") or filepath_lower.endswith(".jpeg"):
        return "jpeg"
    else:
        return "unknown"


def extract_case_id(filepath: str, remove_channel_suffix: bool = True) -> str:
    """
    Extract case ID from a file path consistently.

    Args:
        filepath: Full path or filename (e.g., "Hippo_001_0000.nii.gz", "ISIC_0000000_0000.jpg")
        remove_channel_suffix: If True, remove channel suffix like _0000

    Returns:
        Case ID (e.g., "Hippo_001", "ISIC_0000000")
    """
    # Get basename if full path provided
    filename = Path(filepath).name

    # Remove extension(s)
    if filename.endswith(".nii.gz"):
        base_name = filename.replace(".nii.gz", "")
    else:
        base_name = Path(filename).stem

    # Remove channel suffix if requested
    if remove_channel_suffix and "_" in base_name:
        # Check if last part looks like a channel suffix (e.g., _0000)
        parts = base_name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
            return parts[0]

    return base_name


def extract_base_name_for_label(img_name: str) -> tuple[str, str]:
    """
    Extract base name and appropriate label extension from image filename.
    Used for finding corresponding label files.

    Args:
        img_name: Image filename (e.g., "Hippo_001_0000.nii.gz", "ISIC_0000000_0000.jpg")

    Returns:
        Tuple of (base_name, label_extension)
    """
    file_type = detect_file_type(img_name)

    if file_type == "nifti":
        base_name = img_name.replace(".nii.gz", "").rsplit("_", 1)[0]
        label_ext = ".nii.gz"
    elif file_type == "jpeg":
        # JPG/JPEG images typically have PNG labels
        name_parts = img_name.rsplit(".", 1)
        name_no_ext = name_parts[0]
        base_name = name_no_ext.rsplit("_", 1)[0] if "_" in name_no_ext else name_no_ext
        label_ext = ".png"
    else:
        # PNG and other formats
        name_parts = img_name.rsplit(".", 1)
        name_no_ext = name_parts[0]
        base_name = name_no_ext.rsplit("_", 1)[0] if "_" in name_no_ext else name_no_ext
        label_ext = f".{name_parts[1]}" if len(name_parts) > 1 else ".png"

    return base_name, label_ext


def ensure_directory(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to create

    Returns:
        The path (for convenience in chaining)
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str, description: str = "JSON file") -> dict[str, Any]:
    """
    Load JSON file with error handling.

    Args:
        path: Path to JSON file
        description: Description of the file for error messages (e.g., "dataset.json", "splits")

    Returns:
        Dictionary containing JSON data

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"{description} not found at {path}")

    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: str, indent: int = 2) -> None:
    """
    Save data to JSON file with consistent formatting.

    Args:
        data: Dictionary to save
        path: Path where JSON file will be saved
        indent: Indentation level for JSON formatting (default: 2)
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_nifti_data(image_path: str) -> NDArray:
    """
    Load NIfTI image data using MONAI's LoadImaged transform.

    Uses MONAI's robust image loading with proper metadata handling and error handling.

    Args:
        image_path: Path to NIfTI file (.nii or .nii.gz)

    Returns:
        NumPy array containing image data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid NIfTI image
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Use MONAI's LoadImaged for consistent, robust image loading
    # ensure_channel_first=False to preserve NIfTI's native channel-first format
    # (NIfTI files in preprocessed format already have channel dimension)
    loader = LoadImaged(keys=["image"], ensure_channel_first=False)
    data_dict = loader({"image": image_path})
    image_data = data_dict["image"].numpy()

    return image_data


def load_nifti_with_metadata(image_path: str) -> tuple[NDArray, tuple[float, ...]]:
    """
    Load image data and spacing using MONAI's LoadImaged transform.

    Supports NIfTI (.nii, .nii.gz) and image formats (PNG, JPEG).
    Uses MONAI's robust image loading with proper metadata handling.
    Returns data in channel-first format.

    Args:
        image_path: Path to image file (NIfTI, PNG, or JPEG)

    Returns:
        Tuple of (image_data, spacing)
        - image_data: NumPy array in channel-first format (C, H, W) for 2D or (C, D, H, W) for 3D
        - spacing: Tuple of voxel spacings for each spatial dimension

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid image or metadata is missing
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Use MONAI's LoadImaged for consistent, robust image loading
    # ensure_channel_first=False to preserve NIfTI's native channel-first format
    # PNG/JPEG files need manual channel dimension handling (done in fingerprint.py)
    loader = LoadImaged(keys=["image"], ensure_channel_first=False)
    data_dict = loader({"image": image_path})
    image_tensor = data_dict["image"]

    # Convert to numpy after extracting metadata
    image_data = image_tensor.numpy()

    # Extract spacing from MONAI MetaTensor's affine matrix
    # MetaTensor stores affine as an attribute
    if hasattr(image_tensor, "affine"):
        affine = (
            image_tensor.affine.numpy()
            if hasattr(image_tensor.affine, "numpy")
            else np.array(image_tensor.affine)
        )
        # Compute spacing from affine matrix (norm of each column vector)
        spacing = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
    else:
        # Fallback for images without affine (PNG/JPEG): standard isotropic spacing
        # Use spatial dimensions (excluding channel dimension at index 0)
        ndim_spatial = image_data.ndim - 1  # Subtract 1 for channel dimension
        if ndim_spatial == 2:
            spacing = (1.0, 1.0)
        else:
            spacing = (1.0, 1.0, 1.0)

    return image_data, spacing
