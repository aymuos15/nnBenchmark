#!/usr/bin/env python3
"""
Automatic dataset preparation and initialization.
Generates dataset.json and splits.json from actual data in imagesTr/labelsTr directories.

This script should be run after setting up a dataset with proper directory structure:
  dataset_dir/
    ├── imagesTr/          (training images)
    ├── labelsTr/          (training labels)
    ├── imagesTs/          (optional test images)
    └── labelsTs/          (optional test labels)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from loguru import logger

from src.planning.splits import create_splits, save_splits
from src.preprocessing.cropping import crop_to_nonzero
from src.utils.files import extract_case_id, load_nifti_with_metadata


def extract_training_cases(dataset_path: str | Path) -> list[dict[str, str]]:
    """
    Extract training cases from imagesTr and labelsTr directories.

    Returns list of dicts with 'image' and 'label' keys.
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "imagesTr"
    labels_dir = dataset_path / "labelsTr"

    if not images_dir.exists():
        raise FileNotFoundError(f"imagesTr directory not found: {images_dir}")

    # Find all training images
    training_cases = []
    image_files = sorted(images_dir.glob("*_0000.*"))

    if not image_files:
        logger.warning(f"No training images found in {images_dir}")
        return []

    for img_file in image_files:
        # Extract case ID (remove _0000 suffix and extension)
        base_name = extract_case_id(img_file.name, remove_channel_suffix=True)

        # Find corresponding label file
        label_found = False
        for label_ext in [".nii.gz", ".nii", ".png", ".jpg", ".jpeg"]:
            label_path = labels_dir / f"{base_name}{label_ext}"
            if label_path.exists():
                training_cases.append(
                    {
                        "image": f"./imagesTr/{img_file.name}",
                        "label": f"./labelsTr/{label_path.name}",
                    }
                )
                label_found = True
                break

        if not label_found:
            logger.warning(f"No matching label found for {img_file.name}")

    logger.info(f"Found {len(training_cases)} training cases")
    return training_cases


def extract_test_cases(dataset_path: str | Path) -> list[str]:
    """
    Extract test cases from imagesTs directory.

    Returns list of image paths.
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "imagesTs"

    if not images_dir.exists():
        logger.info("No imagesTs directory found, skipping test cases")
        return []

    test_cases = []
    image_files = sorted(images_dir.glob("*_0000.*"))

    for img_file in image_files:
        test_cases.append(f"./imagesTs/{img_file.name}")

    if test_cases:
        logger.info(f"Found {len(test_cases)} test cases")

    return test_cases


def preprocess_and_crop_dataset(
    dataset_path: str | Path,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Preprocess dataset by cropping to nonzero regions (matching nnU-Net).

    This function:
    1. Loads each training image and segmentation
    2. Crops to nonzero bounding box
    3. Saves cropped versions to output directories
    4. Stores bounding box and original shape in properties

    Args:
        dataset_path: Path to dataset directory with imagesTr/labelsTr
        output_dir: Directory to save cropped images (default: same directory with _cropped suffix)
        force: Overwrite existing cropped files

    Returns:
        Dictionary mapping case_id to properties dict containing:
        - crop_bbox: Bounding box coordinates
        - original_shape: Original image shape
        - cropped_shape: Cropped image shape
        - spacing: Voxel spacing
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "imagesTr"
    labels_dir = dataset_path / "labelsTr"

    if not images_dir.exists():
        raise FileNotFoundError(f"imagesTr directory not found: {images_dir}")

    # Set output directories
    if output_dir is None:
        images_output_dir = dataset_path / "imagesTr_cropped"
        labels_output_dir = dataset_path / "labelsTr_cropped"
    else:
        output_dir = Path(output_dir)
        images_output_dir = output_dir / "imagesTr"
        labels_output_dir = output_dir / "labelsTr"

    # Create output directories
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Properties for all cases
    properties_dict: dict[str, Any] = {}

    # Find all training images
    image_files = sorted(images_dir.glob("*_0000.*"))
    logger.info(f"Found {len(image_files)} training images to preprocess")

    for idx, img_file in enumerate(image_files, 1):
        base_name = extract_case_id(img_file.name, remove_channel_suffix=True)
        logger.info(f"[{idx}/{len(image_files)}] Processing {base_name}...")

        # Find corresponding label file
        label_path = None
        for label_ext in [".nii.gz", ".nii", ".png", ".jpg", ".jpeg"]:
            candidate = labels_dir / f"{base_name}{label_ext}"
            if candidate.exists():
                label_path = candidate
                break

        if label_path is None:
            logger.warning(f"  ⚠ No matching label found for {base_name}, skipping")
            continue

        # Load image and get metadata
        try:
            img_data, spacing = load_nifti_with_metadata(str(img_file))

            # Ensure channel-first format: (C, H, W) for 2D or (C, H, W, D) for 3D
            if img_data.ndim == 2:
                # 2D image (H, W) from PNG/JPG - add channel dimension only
                img_data = np.expand_dims(img_data, axis=0)  # (1, H, W)
            elif img_data.ndim == 3:
                # 3D image (H, W, D) - add channel dimension
                img_data = np.expand_dims(img_data, axis=0)  # (1, H, W, D)
            elif img_data.ndim != 4:
                logger.warning(f"  ⚠ Unexpected image shape {img_data.shape}, skipping")
                continue

            # Load segmentation
            seg_data, _ = load_nifti_with_metadata(str(label_path))
            # Ensure segmentation has channel dimension and matches image dimensions
            if seg_data.ndim == 2:
                # 2D segmentation (H, W) - add channel dimension only
                seg_data = np.expand_dims(seg_data, axis=0)  # (1, H, W)
            elif seg_data.ndim == 3:
                # 3D segmentation (H, W, D) - add channel dimension
                seg_data = np.expand_dims(seg_data, axis=0)  # (1, H, W, D)

        except Exception as e:
            logger.warning(f"  ⚠ Failed to load {base_name}: {e}")
            continue

        # Crop to nonzero
        cropped_img, cropped_seg, bbox = crop_to_nonzero(img_data, seg_data)

        # Ensure cropped_seg is not None
        if cropped_seg is None:
            logger.warning(f"  ⚠ Failed to crop segmentation for {base_name}, skipping")
            continue

        logger.info(
            f"  Original shape: {img_data.shape} → Cropped shape: {cropped_img.shape} "
            f"(bbox: {bbox})"
        )

        # Output paths for cropped data
        output_img_path = images_output_dir / img_file.name
        output_seg_path = labels_output_dir / label_path.name

        # Check if already exists
        if output_img_path.exists() and not force:
            logger.warning(
                f"  Cropped file already exists: {output_img_path.name}, skipping"
            )
            # Still add to properties
            properties_dict[base_name] = {
                "crop_bbox": bbox,
                "original_shape": list(img_data.shape),
                "cropped_shape": list(cropped_img.shape),
                "spacing": list(spacing) if spacing else [1.0, 1.0, 1.0],
            }
            continue

        # Save cropped image in original format (NIfTI or PNG/JPG)
        try:
            if str(img_file).endswith((".nii", ".nii.gz")):
                # For NIfTI: save as NIfTI with affine metadata
                affine = None
                try:
                    orig_nib = nib.load(str(img_file))  # type: ignore[attr-defined]
                    affine = (
                        orig_nib.affine if hasattr(orig_nib, "affine") else np.eye(4)  # type: ignore[attr-defined]
                    )
                except Exception:
                    affine = None

                if affine is None:
                    affine = np.eye(4)

                # Create new NIfTI with cropped data and affine
                img_nib = nib.Nifti1Image(cropped_img, affine=affine)  # type: ignore[attr-defined]
                seg_nib = nib.Nifti1Image(cropped_seg, affine=affine)  # type: ignore[attr-defined]

                nib.save(img_nib, str(output_img_path))  # type: ignore[attr-defined]
                nib.save(seg_nib, str(output_seg_path))  # type: ignore[attr-defined]
            else:
                # For PNG/JPG: save as PNG maintaining original format
                from PIL import Image

                # Remove channel dimension for image saving
                # Format expects (H, W) or (H, W, C), we have (C, H, W) or (C, H, W, D)
                if cropped_img.ndim == 3 and cropped_img.shape[0] == 1:
                    # (1, H, W) → (H, W)
                    img_to_save = cropped_img[0]
                elif cropped_img.ndim == 4 and cropped_img.shape[0] == 1:
                    # (1, H, W, D) → take first slice (H, W)
                    img_to_save = cropped_img[0, :, :, 0]
                else:
                    img_to_save = cropped_img

                if cropped_seg.ndim == 3 and cropped_seg.shape[0] == 1:
                    seg_to_save = cropped_seg[0]
                elif cropped_seg.ndim == 4 and cropped_seg.shape[0] == 1:
                    seg_to_save = cropped_seg[0, :, :, 0]
                else:
                    seg_to_save = cropped_seg

                # Convert image to uint8 (0-255 range for display)
                if img_to_save.max() <= 1.0:
                    img_to_save = (img_to_save * 255).astype(np.uint8)
                else:
                    img_to_save = img_to_save.astype(np.uint8)

                # Convert segmentation to uint8 (preserve class indices, do NOT scale)
                # Labels should remain as class indices [0, 1, 2, ...], not [0, 255]
                seg_to_save = seg_to_save.astype(np.uint8)

                # Save as PNG keeping original filename
                Image.fromarray(img_to_save).save(str(output_img_path))
                Image.fromarray(seg_to_save).save(str(output_seg_path))

            logger.info("  ✓ Saved cropped image and segmentation")

        except Exception as e:
            logger.error(f"  ✗ Failed to save cropped {base_name}: {e}")
            continue

        # Store properties for this case
        properties_dict[base_name] = {
            "crop_bbox": bbox,
            "original_shape": list(img_data.shape),
            "cropped_shape": list(cropped_img.shape),
            "spacing": list(spacing) if spacing else [1.0, 1.0, 1.0],
        }

    logger.info(f"✓ Preprocessed {len(properties_dict)} cases")
    return properties_dict


def prepare_dataset(
    dataset_path: str | Path,
    dataset_name: str | None = None,
    channel: str = "Unknown",
    num_classes: int = 2,
    description: str = "",
    force: bool = False,
    preprocess: bool = True,
    preprocessed_dir: str | Path | None = None,
) -> None:
    """
    Prepare a dataset by generating dataset.json and splits.json.

    By default, preprocesses dataset by cropping to nonzero regions (matching nnU-Net v2.4.1).
    This is the first preprocessing step that removes background zero regions.

    PREPROCESSING (DEFAULT: ENABLED):
    - Creates binary mask of non-zero voxels (any channel)
    - Applies morphological hole-filling for robust foreground detection
    - Crops both images and labels to bounding box
    - Saves cropped versions to imagesTr_cropped/ and labelsTr_cropped/
    - Stores bounding box coordinates and metadata for inference restoration

    IMPACT BY DATASET TYPE:
    - Brain MRI (skull-stripped): 25-50% size reduction → MAJOR SPEEDUP
    - Organ CT (KiTS, Liver): <5% reduction → minimal impact
    - Pre-cropped data: ~0% change → no effect

    TO DISABLE PREPROCESSING:
    - Option 1: Pass preprocess=False to this function
    - Option 2: Use CLI flag: python prepare_dataset.py <path> --no-preprocess
    - Option 3: Remove cropped images if created: rm -rf imagesTr_cropped/ labelsTr_cropped/

    Args:
        dataset_path: Path to dataset directory
        dataset_name: Name of the dataset (defaults to directory name)
        channel: Channel type (e.g., 'MRI', 'CT')
        num_classes: Number of output classes (including background)
        description: Dataset description
        force: Overwrite existing files
        preprocess: If True (default), crop images to nonzero regions and save cropped versions.
                   Set to False to skip preprocessing and use original images.
        preprocessed_dir: Directory to save splits.json to (defaults to raw dataset directory)

    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    if dataset_name is None:
        dataset_name = dataset_path.name

    # Extract training cases
    training_cases = extract_training_cases(str(dataset_path))

    if not training_cases:
        raise ValueError(f"No training cases found in {dataset_path}/imagesTr")

    # Extract test cases
    test_cases = extract_test_cases(str(dataset_path))

    # Create dataset.json (metadata only - no training/test case lists)
    dataset_json: dict[str, Any] = {
        "name": dataset_name,
        "description": description,
        "modality": {"0": channel},
        "labels": {str(i): f"class_{i}" for i in range(num_classes)},
    }

    # Save dataset.json
    dataset_json_path = dataset_path / "dataset.json"

    if dataset_json_path.exists() and not force:
        logger.warning(f"dataset.json already exists: {dataset_json_path}")
        logger.info("Use force=True to overwrite")
        return

    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    logger.info(f"✓ Created {dataset_json_path}")

    # Create splits.json
    if preprocessed_dir is None:
        splits_json_path = dataset_path / "splits.json"
    else:
        splits_json_path = Path(preprocessed_dir) / "splits.json"
        Path(preprocessed_dir).mkdir(parents=True, exist_ok=True)

    if splits_json_path.exists() and not force:
        logger.warning(f"splits.json already exists: {splits_json_path}")
        logger.info("Use force=True to overwrite")
        return

    # Extract case identifiers (just the filenames)
    case_identifiers = [case["image"].split("/")[-1] for case in training_cases]

    # Create 5-fold cross-validation splits
    splits = create_splits(
        case_identifiers=case_identifiers,
        n_folds=5,
        stratified=False,
        seed=12345,
    )

    # Save splits
    save_splits(splits, str(splits_json_path))
    logger.info(f"✓ Created {splits_json_path}")

    # Preprocess dataset if requested
    properties = None
    if preprocess:
        logger.info("\n" + "=" * 60)
        logger.info("Starting dataset preprocessing (crop to nonzero)...")
        logger.info("=" * 60)
        properties = preprocess_and_crop_dataset(dataset_path, force=force)
        logger.info("=" * 60)

    logger.info("\nDataset prepared successfully!")
    logger.info(f"  Training cases: {len(training_cases)}")
    logger.info(f"  Test cases: {len(test_cases)}")
    logger.info("  Folds: 5")
    if preprocess and properties:
        logger.info(f"  Preprocessed cases: {len(properties)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare dataset by generating dataset.json and splits.json"
    )
    parser.add_argument("dataset_path", help="Path to dataset directory")
    parser.add_argument("--name", help="Dataset name (default: directory name)")
    parser.add_argument(
        "--channel", default="Unknown", help="Channel type (e.g., MRI, CT)"
    )
    parser.add_argument(
        "--num-classes", type=int, default=2, help="Number of classes (default: 2)"
    )
    parser.add_argument("--description", default="", help="Dataset description")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--no-preprocess",
        action="store_false",
        dest="preprocess",
        help="Skip preprocessing (crop to nonzero)",
    )

    args = parser.parse_args()

    prepare_dataset(
        dataset_path=args.dataset_path,
        dataset_name=args.name,
        channel=args.channel,
        num_classes=args.num_classes,
        description=args.description,
        force=args.force,
        preprocess=args.preprocess,
    )
