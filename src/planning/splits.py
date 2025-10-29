"""
Generate k-fold cross-validation splits for nnUNet-style datasets.

Note: This module uses scikit-learn's random_state parameter for seeding.
For end-to-end reproducibility, ensure set_random_seeds() from src.utils.seeding
is called before using this module in training/evaluation workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold, StratifiedKFold

from src.utils.files import extract_case_id, load_json, load_nifti_data, save_json
from src.utils.seeding import set_random_seeds


def load_dataset_json(dataset_path: str) -> dict[str, Any]:
    """Load dataset.json from the dataset folder."""
    dataset_json_path = str(Path(dataset_path) / "dataset.json")
    return load_json(dataset_json_path, "dataset.json")


def extract_case_identifiers(
    dataset_json: dict[str, Any], dataset_path: str | Path
) -> list[str]:
    """
    Extract case identifiers by scanning the imagesTr directory.
    Uses the full image filename as the case identifier for consistency.

    Example: "./imagesTr/Hippo_001_0000.nii.gz" -> "Hippo_001_0000.nii.gz"

    Args:
        dataset_json: Dataset metadata dict (for compatibility, not used for extraction)
        dataset_path: Path to dataset directory

    Returns:
        List of case identifiers, or empty list if imagesTr directory doesn't exist
        or contains no matching images.
    """
    case_identifiers: list[str] = []

    # Always scan imagesTr directory
    images_dir = Path(dataset_path) / "imagesTr"
    if not images_dir.exists():
        return case_identifiers

    image_files = sorted(images_dir.glob("*_0000.*"))
    for img_file in image_files:
        if not img_file.name.startswith("._"):  # Skip macOS metadata files
            case_identifiers.append(img_file.name)

    return case_identifiers


def get_labels_for_stratification(
    dataset_path: str, case_identifiers: list[str]
) -> list[int]:
    """
    Read labels for stratified splitting.
    Returns a label class for each case based on the presence of foreground classes.
    Uses MONAI's LoadImaged for robust label loading.
    """
    labels: list[int] = []
    for case_id in case_identifiers:
        # Extract base case name from full filename (e.g., "Hippo_001_0000.nii.gz" -> "Hippo_001")
        base_name = extract_case_id(case_id, remove_channel_suffix=True)
        label_path = str(Path(dataset_path) / "labelsTr" / f"{base_name}.nii.gz")

        if not Path(label_path).exists():
            raise FileNotFoundError(f"Label not found: {label_path}")

        # Load label using MONAI for robust, consistent handling
        label_data: NDArray = load_nifti_data(label_path)

        # Use the most frequent non-background class as the label
        unique: NDArray
        counts: NDArray
        unique, counts = np.unique(label_data, return_counts=True)
        # Filter out background (0)
        non_bg_mask = unique != 0
        if np.any(non_bg_mask):
            non_bg_unique = unique[non_bg_mask]
            non_bg_counts = counts[non_bg_mask]
            dominant_class = non_bg_unique[np.argmax(non_bg_counts)]
        else:
            dominant_class = 0

        labels.append(int(dominant_class))

    return labels


def create_splits(
    case_identifiers: list[str],
    n_folds: int = 5,
    stratified: bool = False,
    dataset_path: str | None = None,
    seed: int = 12345,
) -> dict[str, dict[str, list[str]]]:
    """
    Create k-fold splits with reproducible seeding.

    Args:
        case_identifiers: List of case IDs
        n_folds: Number of folds
        stratified: Whether to use stratified splitting
        dataset_path: Path to dataset (required if stratified=True)
        seed: Random seed for reproducibility (also sets global random state)

    Returns:
        Dictionary with fold splits
    """
    # Ensure reproducibility by setting all random seeds
    set_random_seeds(seed)

    case_identifiers_array: NDArray = np.array(case_identifiers)

    if stratified:
        if dataset_path is None:
            raise ValueError("dataset_path is required for stratified splitting")

        labels = get_labels_for_stratification(dataset_path, case_identifiers)
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits_iterator = kfold.split(case_identifiers_array, labels)
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits_iterator = kfold.split(case_identifiers_array)

    splits: dict[str, dict[str, list[str]]] = {}
    for fold_idx, (train_idx, val_idx) in enumerate(splits_iterator):
        train_cases: list[str] = case_identifiers_array[train_idx].tolist()
        val_cases: list[str] = case_identifiers_array[val_idx].tolist()

        splits[f"fold_{fold_idx}"] = {"train": train_cases, "val": val_cases}

    return splits


def create_all_split(
    case_identifiers: list[str],
) -> dict[str, dict[str, list[str]]]:
    """
    Create a special split that uses all cases for training without validation.

    This is useful for:
    - Final production model after cross-validation
    - Small datasets where validation reduces training data too much

    Args:
        case_identifiers: List of all case IDs

    Returns:
        Dictionary with a single "fold_-1" entry containing all cases in train
        and empty validation set
    """
    return {
        "fold_-1": {
            "train": case_identifiers,
            "val": [],
        }
    }


def save_splits(splits: dict[str, dict[str, list[str]]], output_path: str) -> None:
    """Save splits to JSON file."""
    save_json(splits, output_path)
