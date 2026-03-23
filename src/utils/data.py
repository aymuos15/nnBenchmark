"""
Data loading and dataset preparation utilities for nnBenchmark.
Provides helpers for creating MONAI data dictionaries from dataset.json and splits.
"""

from pathlib import Path
from typing import Any

from loguru import logger

from src.config.load import load_splits
from src.utils.files import extract_base_name_for_label, extract_case_id, load_json


def _build_case_to_paths_mapping(data_dir: str) -> dict[str, dict[str, str]]:
    """
    Build a mapping from case IDs to preprocessed image/label file paths.

    This helper eliminates code duplication between get_data_dicts and get_test_data_dicts.
    Uses preprocessed data from nnBench_preprocessed directory (imagesTr/labelsTr).

    The preprocessing step (crop to nonzero regions) must be completed via 'nnBench.plan'
    before training or inference. This ensures all data follows nnU-Net v2.4.1 preprocessing.

    Args:
        data_dir: Dataset directory in nnBench_raw (contains dataset.json and splits.json)

    Returns:
        Dictionary mapping case_id (filename) to {"image": path, "label": path}

    Raises:
        FileNotFoundError: If preprocessed directories don't exist
    """
    from src.config.paths import get_preprocessed_root

    data_dir_path = Path(data_dir)
    dataset_name = data_dir_path.name

    # Get preprocessed directory from environment
    preprocessed_root = get_preprocessed_root()
    preprocessed_dir = preprocessed_root / dataset_name
    images_dir = preprocessed_dir / "imagesTr"
    labels_dir = preprocessed_dir / "labelsTr"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed images directory not found: {images_dir}\n"
            "Please run 'nnBench.plan --dataset {dataset_name}' first to preprocess the dataset.\n"
            "Preprocessing crops images to nonzero regions (nnU-Net v2.4.1 style)."
        )
    if not labels_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed labels directory not found: {labels_dir}\n"
            "Please run 'nnBench.plan --dataset {dataset_name}' first to preprocess the dataset.\n"
            "Preprocessing crops labels to nonzero regions (nnU-Net v2.4.1 style)."
        )

    # Create a mapping from case_id (filename) to file paths
    case_to_paths: dict[str, dict[str, str]] = {}
    skipped_cases: list[str] = []

    # Scan all training images
    image_files = sorted(images_dir.glob("*_0000.*"))
    for img_file in image_files:
        if img_file.name.startswith("._"):  # Skip macOS metadata files
            continue

        # Extract base case name (e.g., "Hippo_001_0000.nii.gz" -> "Hippo_001")
        base_name = extract_case_id(img_file.name, remove_channel_suffix=True)

        # Find matching label file
        label_file = None
        for label_ext in [".nii.gz", ".nii", ".png", ".jpg", ".jpeg"]:
            potential_label = labels_dir / f"{base_name}{label_ext}"
            if potential_label.exists():
                label_file = potential_label
                break

        if label_file:
            case_to_paths[img_file.name] = {
                "image": str(img_file),
                "label": str(label_file),
            }
        else:
            # Log cases that are skipped due to missing labels
            skipped_cases.append(base_name)

    if skipped_cases:
        logger.warning(
            f"Skipped {len(skipped_cases)} cases due to missing label files: {skipped_cases[:10]}"
            f"{'...' if len(skipped_cases) > 10 else ''}"
        )

    return case_to_paths


def get_data_dicts(
    data_dir: str, fold: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Get train/val data dictionaries using fold-based splits.

    Args:
        data_dir: Dataset directory containing dataset.json and splits.json
        fold: Fold number to use

    Returns:
        Tuple of (train_data_dicts, val_data_dicts)
    """
    train_cases, val_cases = load_splits(data_dir, fold)
    case_to_paths = _build_case_to_paths_mapping(data_dir)

    train_data = [case_to_paths[c] for c in train_cases if c in case_to_paths]
    val_data = [case_to_paths[c] for c in val_cases if c in case_to_paths]
    return train_data, val_data


def get_test_data_dicts(
    data_dir: str, fold: int | None, use_test_set: bool = False
) -> list[dict[str, str]]:
    """
    Get test data dictionaries.

    Args:
        data_dir: Dataset directory
        fold: Fold number (required if use_test_set=False)
        use_test_set: If True, use dedicated test set (imagesTs/labelsTs).
                      If False, use validation split from fold.

    Returns:
        List of data dictionaries with image/label paths

    Raises:
        ValueError: If use_test_set=False and fold is None
    """
    if not use_test_set:
        if fold is None:
            raise ValueError("fold parameter is required when use_test_set=False")
        _, val_data = get_data_dicts(data_dir, fold)
        return val_data

    # Use dedicated test set (imagesTs/labelsTs)
    images_dir = Path(data_dir) / "imagesTs"
    labels_dir = Path(data_dir) / "labelsTs"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Test set directory not found: {images_dir}\n"
            f"This dataset does not have a dedicated test set (imagesTs/labelsTs).\n"
            f"Use '--use-val-split' flag to test on the validation split instead:\n"
            f"  nnBench.inference --config <config.yaml> --use-val-split"
        )

    images = sorted(images_dir.glob("*"))
    if not images:
        raise ValueError(
            f"No images found in test set directory: {images_dir}\n"
            f"The directory exists but contains no files.\n"
            f"Use '--use-val-split' flag to test on the validation split instead."
        )

    data_dicts: list[dict[str, str]] = []
    for img_path in images:
        base_name, label_ext = extract_base_name_for_label(img_path.name)
        label_path = labels_dir / f"{base_name}{label_ext}"
        if label_path.exists():
            data_dicts.append({"image": str(img_path), "label": str(label_path)})

    if not data_dicts:
        raise ValueError(
            f"No valid image-label pairs found in test set.\n"
            f"Images directory: {images_dir} (found {len(images)} images)\n"
            f"Labels directory: {labels_dir}\n"
            f"This dataset may not have a dedicated test set (imagesTs/labelsTs).\n"
            f"Use '--use-val-split' flag to test on the validation split instead:\n"
            f"  nnBench.inference --config <config.yaml> --use-val-split"
        )

    return data_dicts


def get_class_labels(data_dir: str, include_background: bool = False) -> dict[int, str]:
    """
    Load class labels from dataset.json.

    Args:
        data_dir: Dataset directory containing dataset.json
        include_background: If False, skip background class (index 0)

    Returns:
        Dictionary mapping class index to class name
    """
    dataset_json_path = str(Path(data_dir) / "dataset.json")
    dataset_info: dict[str, Any] = load_json(dataset_json_path, "dataset.json")

    # nnU-Net format: {"background": 0, "class1": 1, ...} (name -> index)
    labels_dict = dataset_info.get("labels", {})
    class_labels = {idx: name for name, idx in labels_dict.items()}

    if not include_background:
        class_labels.pop(0, None)

    return class_labels
