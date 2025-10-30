#!/usr/bin/env python3
"""
nnUNet Dataset Format Converter for Cellpose

Converts Cellpose instance segmentation data to nnUNet semantic segmentation format.

This script:
1. Unzips train.zip and test.zip
2. Converts instance masks to semantic masks (background=0, cell=1)
3. Reorganizes files into nnUNet format
4. Creates dataset.json metadata file

Instance Segmentation → Semantic Segmentation:
- Original: Each cell has unique ID (1, 2, 3, ..., 1305)
- Converted: All cells → 1, background → 0
"""

import json
import zipfile
from pathlib import Path
from typing import Optional, Set

import numpy as np
from PIL import Image


def unzip_data(dataset_dir: Path, zip_file: str) -> Optional[Path]:
    """
    Unzip a dataset file.

    Args:
        dataset_dir: Path to dataset directory
        zip_file: Name of zip file (e.g., 'train.zip' or 'test.zip')

    Returns:
        Path to extracted directory or None if failed
    """
    zip_path = dataset_dir / zip_file
    extract_dir = dataset_dir / zip_file.replace('.zip', '')

    if extract_dir.exists():
        print(f"Directory {extract_dir.name} already exists, skipping unzip")
        return extract_dir

    if not zip_path.exists():
        print(f"Warning: {zip_file} not found")
        return None

    print(f"Unzipping {zip_file}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

    print(f"Extracted to {extract_dir}")
    return extract_dir


def get_unique_cases(folder_path: Path) -> Set[str]:
    """Extract unique case IDs from folder."""
    cases = set()
    if not folder_path.exists():
        return cases

    for file in folder_path.glob("*_img.png"):
        case_id = file.stem.replace("_img", "")
        cases.add(case_id)

    return cases


def convert_instance_to_semantic(instance_mask: np.ndarray) -> np.ndarray:
    """
    Convert instance segmentation mask to semantic segmentation mask.

    Instance mask: each cell/object has a unique ID (1, 2, 3, ...)
    Semantic mask: background=0, all cells=1

    Args:
        instance_mask: numpy array with instance IDs

    Returns:
        semantic_mask: numpy array with 0=background, 1=foreground (cell)
    """
    semantic_mask = (instance_mask > 0).astype(np.uint8)
    return semantic_mask


def convert_to_grayscale(image_array: np.ndarray) -> np.ndarray:
    """
    Convert RGB/multi-channel images to grayscale.

    WHY GRAYSCALE FOR CELL SEGMENTATION?
    ====================================
    Cell and nuclei segmentation relies on morphological features (shape, size, edges)
    rather than color information. Converting to grayscale is standard practice because:

    1. BIOLOGICAL REALITY: Fluorescence microscopy typically captures intensity in a
       single wavelength (e.g., DAPI for nuclei, GFP for cells). Even when multi-channel
       images exist, nnUNet works best with intensity-based morphology.

    2. COMPUTATIONAL EFFICIENCY: Single-channel images reduce model complexity and
       memory requirements while maintaining segmentation quality. nnUNet doesn't need
       color information to learn cell boundaries.

    3. COMMUNITY STANDARD: Cellpose, nnUNet, and medical imaging literature all use
       grayscale for cell segmentation. RGB channels add noise without improving results.

    4. DATA CONSISTENCY: PNG images loaded as RGB introduce artificial channel data.
       Converting ensures all images have consistent, meaningful intensity channels.

    CONVERSION METHOD: Standard luminosity formula (0.299*R + 0.587*G + 0.114*B)
    preserves perceived brightness and works well for microscopy images.

    Args:
        image_array: RGB image as (H, W, 3) or grayscale as (H, W)

    Returns:
        grayscale_image: Single-channel image as (H, W) uint8
    """
    # If already grayscale (2D), return as-is
    if len(image_array.shape) == 2:
        return image_array.astype(np.uint8)

    # Convert RGB to grayscale using standard luminosity formula
    # This preserves perceived brightness better than simple averaging
    if image_array.shape[2] == 3:  # RGB
        grayscale = (
            0.299 * image_array[:, :, 0] +  # Red channel weight
            0.587 * image_array[:, :, 1] +  # Green channel weight (most sensitive to human eye)
            0.114 * image_array[:, :, 2]    # Blue channel weight
        )
        return grayscale.astype(np.uint8)

    # For RGBA or other formats, just take first channel or average
    if image_array.shape[2] >= 3:
        grayscale = (
            0.299 * image_array[:, :, 0] +
            0.587 * image_array[:, :, 1] +
            0.114 * image_array[:, :, 2]
        )
        return grayscale.astype(np.uint8)

    # Fallback: single channel
    return image_array[:, :, 0].astype(np.uint8)


def copy_and_rename_files(
    src_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Optional[Path],
    case_ids: Set[str],
    file_ending: str = ".png",
    is_test: bool = False
) -> int:
    """
    Copy and rename files from source to destination directories.
    Convert instance masks to semantic masks (training data only).
    Convert RGB images to grayscale for optimal cell segmentation.

    Args:
        src_dir: Source directory with images and masks
        dst_images_dir: Destination for images
        dst_labels_dir: Destination for labels (None for test data)
        case_ids: Set of case identifiers
        file_ending: File extension
        is_test: Whether this is test data (no label conversion needed)

    Returns:
        Number of cases processed
    """
    count = 0

    for case_id in sorted(case_ids):
        img_src = src_dir / f"{case_id}_img.png"

        if not img_src.exists():
            print(f"Warning: Image not found for case {case_id}")
            continue

        # Load image and convert to grayscale
        image_array = np.array(Image.open(img_src))
        grayscale_image = convert_to_grayscale(image_array)

        # Save grayscale image in nnUNet format
        img_dst = dst_images_dir / f"{case_id}_0000{file_ending}"
        Image.fromarray(grayscale_image).save(img_dst)

        if not is_test and dst_labels_dir is not None:
            mask_src = src_dir / f"{case_id}_masks.png"

            if not mask_src.exists():
                print(f"Warning: Mask not found for case {case_id}")
                continue

            instance_mask = np.array(Image.open(mask_src))
            semantic_mask = convert_instance_to_semantic(instance_mask)

            mask_dst = dst_labels_dir / f"{case_id}{file_ending}"
            Image.fromarray(semantic_mask).save(mask_dst)

        count += 1
        if count % 50 == 0:
            print(f"  Processed {count} cases...")

    return count


def create_dataset_json(
    output_dir: Path,
    num_training: int,
    file_ending: str = ".png"
):
    """
    Create dataset.json file with semantic segmentation format.

    Uses single grayscale channel for optimal cell segmentation performance.
    """

    dataset_json = {
        "channel_names": {
            "0": "Grayscale"  # Single-channel grayscale for morphological features
        },
        "labels": {
            "background": 0,
            "cell": 1
        },
        "numTraining": num_training,
        "file_ending": file_ending
    }

    json_path = output_dir / "dataset.json"
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nCreated dataset.json with {num_training} training cases")
    print("Channel: 1x Grayscale (converted from RGB)")
    print("Semantic classes: background (0), cell (1)")
    return json_path


def main():
    """Main conversion function."""
    script_dir = Path(__file__).parent

    print("=" * 70)
    print("nnUNet Dataset Format Converter - Cellpose Semantic Segmentation")
    print("=" * 70)

    print("\n[Step 1/3] Unzipping data...")
    print("-" * 70)

    train_zip = "train.zip"
    test_zip = "test.zip"

    train_dir = unzip_data(script_dir, train_zip)
    test_dir = unzip_data(script_dir, test_zip)

    if train_dir is None or not train_dir.exists():
        print("Error: Could not unzip training data")
        return

    print("\n[Step 2/3] Creating output directories...")
    print("-" * 70)

    images_tr_dir = script_dir / "imagesTr"
    labels_tr_dir = script_dir / "labelsTr"
    images_ts_dir = script_dir / "imagesTs"

    images_tr_dir.mkdir(exist_ok=True)
    labels_tr_dir.mkdir(exist_ok=True)
    images_ts_dir.mkdir(exist_ok=True)

    print("Created directories:")
    print(f"  - {images_tr_dir.name}")
    print(f"  - {labels_tr_dir.name}")
    print(f"  - {images_ts_dir.name}")

    print("\n[Step 3/3] Converting data to nnUNet format...")
    print("-" * 70)

    print("\nProcessing training data:")
    train_cases = get_unique_cases(train_dir)
    print(f"Found {len(train_cases)} training cases")

    if train_cases:
        num_training = copy_and_rename_files(
            train_dir,
            images_tr_dir,
            labels_tr_dir,
            train_cases,
            is_test=False
        )

        print(f"Successfully converted {num_training} training cases")

        create_dataset_json(script_dir, num_training)
    else:
        print("Error: No training cases found")
        return

    if test_dir and test_dir.exists():
        print("\nProcessing test data:")
        test_cases = get_unique_cases(test_dir)
        print(f"Found {len(test_cases)} test cases")

        if test_cases:
            num_test = copy_and_rename_files(
                test_dir,
                images_ts_dir,
                None,
                test_cases,
                is_test=True
            )
            print(f"Successfully converted {num_test} test cases")
    else:
        print("\nNote: Test data not found or not extracted")

    print("\n" + "=" * 70)
    print("Conversion Complete!")
    print("=" * 70)

    print("\nDirectory structure:")
    print(f"  imagesTr/   - {len(list(images_tr_dir.glob('*_0000.png')))} training images (grayscale)")
    print(f"  labelsTr/   - {len(list(labels_tr_dir.glob('*.png')))} training semantic masks")
    print(f"  imagesTs/   - {len(list(images_ts_dir.glob('*_0000.png')))} test images (grayscale)")
    print("  dataset.json - metadata file")

    print("\nImage format:")
    print("  - 1 channel: Grayscale (converted from RGB for optimal segmentation)")
    print("  - Reason: Cell morphology depends on intensity, not color")
    print("  - Standard: Aligns with Cellpose and medical imaging best practices")

    print("\nSemantic classes:")
    print("  - 0: background")
    print("  - 1: cell")

    print("\nNext steps:")
    print("  1. Set nnUNet environment variable (if not already set):")
    print("     export nnUNet_raw=/path/to/nnUNet_raw")
    print("  2. Run preprocessing:")
    print("     nnUNetv2_plan_and_preprocess -d 3")
    print("  3. Train model:")
    print("     nnUNetv2_train 3 2d 0")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
