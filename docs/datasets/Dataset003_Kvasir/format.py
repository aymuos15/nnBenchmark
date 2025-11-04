#!/usr/bin/env python3
"""
nnUNet Dataset Format Converter for Kvasir-SEG

Converts Kvasir-SEG polyp segmentation data to nnUNet format.

This script:
1. Unzips archive.zip containing the Kvasir-SEG dataset
2. Reads train.txt and val.txt for the predefined split
3. Converts and organizes files into nnUNet format
4. Keeps RGB images (3 channels) for colonoscopy segmentation
5. Normalizes masks to binary format (background=0, polyp=1)
6. Creates dataset.json metadata file

Dataset Split:
- Training: 880 images (88%) → imagesTr/, labelsTr/
- Validation: 120 images (12%) → imagesTs/, labelsTs/
"""

import os
import json
import shutil
import zipfile
from pathlib import Path
from typing import Set, List, Optional
import numpy as np
from PIL import Image


def unzip_data(dataset_dir: Path, zip_file: str = "archive.zip") -> Optional[Path]:
    """
    Unzip the Kvasir-SEG dataset archive.

    Args:
        dataset_dir: Path to dataset directory
        zip_file: Name of zip file (default: 'archive.zip')

    Returns:
        Path to extracted Kvasir-SEG directory or None if failed
    """
    zip_path = dataset_dir / zip_file

    if not zip_path.exists():
        print(f"Error: {zip_file} not found at {zip_path}")
        return None

    # The archive extracts to Kvasir-SEG/Kvasir-SEG/
    kvasir_dir = dataset_dir / "Kvasir-SEG" / "Kvasir-SEG"

    if kvasir_dir.exists():
        print(f"Kvasir-SEG directory already exists, skipping unzip")
        return kvasir_dir

    print(f"Unzipping {zip_file}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

    if kvasir_dir.exists():
        print(f"Extracted to {kvasir_dir}")
        return kvasir_dir
    else:
        print(f"Error: Expected directory {kvasir_dir} not found after extraction")
        return None


def read_split_file(split_file: Path) -> List[str]:
    """
    Read train.txt or val.txt to get list of image IDs.

    Args:
        split_file: Path to split file (train.txt or val.txt)

    Returns:
        List of image ID stems (without extension)
    """
    if not split_file.exists():
        print(f"Warning: Split file {split_file} not found")
        return []

    with open(split_file, 'r') as f:
        # Read lines and strip whitespace
        image_ids = [line.strip() for line in f if line.strip()]

    return image_ids


def normalize_mask(mask_array: np.ndarray) -> np.ndarray:
    """
    Normalize mask to binary format.

    Kvasir-SEG masks are typically:
    - Background: 0 (black)
    - Polyp: 255 (white)

    We normalize to:
    - Background: 0
    - Polyp: 1

    Args:
        mask_array: numpy array with mask values

    Returns:
        binary_mask: numpy array with 0=background, 1=polyp
    """
    # Convert any non-zero value to 1
    binary_mask = (mask_array > 0).astype(np.uint8)
    return binary_mask


def process_and_copy_files(
    images_dir: Path,
    masks_dir: Path,
    image_ids: List[str],
    dst_images_dir: Path,
    dst_labels_dir: Path,
    start_case_id: int = 0,
    file_ending: str = ".png"
) -> int:
    """
    Process and copy files from source to destination directories.
    Keep RGB images (3 channels) for colonoscopy segmentation.
    Normalize masks to binary format.

    Args:
        images_dir: Source directory with images
        masks_dir: Source directory with masks
        image_ids: List of image ID stems to process
        dst_images_dir: Destination for images
        dst_labels_dir: Destination for labels
        start_case_id: Starting case ID for nnUNet naming (default: 0)
        file_ending: File extension

    Returns:
        Number of cases successfully processed
    """
    count = 0

    for idx, image_id in enumerate(sorted(image_ids)):
        # Source files (original Kvasir naming)
        img_src = images_dir / f"{image_id}.jpg"
        mask_src = masks_dir / f"{image_id}.jpg"

        if not img_src.exists():
            print(f"Warning: Image not found: {img_src.name}")
            continue

        if not mask_src.exists():
            print(f"Warning: Mask not found: {mask_src.name}")
            continue

        # nnUNet case ID (sequential numbering)
        case_id = str(start_case_id + idx).zfill(3)

        # Load RGB image and split into separate channels
        image = Image.open(img_src)

        # Convert to RGB if not already (in case of RGBA or other formats)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy array to split channels
        image_array = np.array(image)

        # Save each channel as a separate file (nnUNet multi-channel format)
        # Channel 0: Red
        r_channel = image_array[:, :, 0]
        img_dst_r = dst_images_dir / f"{case_id}_0000{file_ending}"
        Image.fromarray(r_channel).save(img_dst_r)

        # Channel 1: Green
        g_channel = image_array[:, :, 1]
        img_dst_g = dst_images_dir / f"{case_id}_0001{file_ending}"
        Image.fromarray(g_channel).save(img_dst_g)

        # Channel 2: Blue
        b_channel = image_array[:, :, 2]
        img_dst_b = dst_images_dir / f"{case_id}_0002{file_ending}"
        Image.fromarray(b_channel).save(img_dst_b)

        # Load mask and normalize to binary
        mask = Image.open(mask_src)

        # Convert to numpy array and handle different mask formats
        mask_array = np.array(mask)

        # If mask is RGB/RGBA, convert to grayscale first
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]  # Take first channel

        # Normalize to binary (0=background, 1=polyp)
        binary_mask = normalize_mask(mask_array)

        # Save mask in nnUNet format (no suffix, just case ID)
        mask_dst = dst_labels_dir / f"{case_id}{file_ending}"
        Image.fromarray(binary_mask).save(mask_dst)

        count += 1
        if count % 50 == 0:
            print(f"  Processed {count}/{len(image_ids)} cases...")

    return count


def create_dataset_json(
    output_dir: Path,
    num_training: int,
    file_ending: str = ".png"
):
    """
    Create dataset.json file for Kvasir-SEG polyp segmentation.

    Uses RGB (3 channels) for colonoscopy images, as color information
    can be important for distinguishing polyps from surrounding tissue.
    """

    dataset_json = {
        "channel_names": {
            "0": "R",  # Red channel
            "1": "G",  # Green channel
            "2": "B"   # Blue channel
        },
        "labels": {
            "background": 0,
            "polyp": 1
        },
        "numTraining": num_training,
        "file_ending": file_ending
    }

    json_path = output_dir / "dataset.json"
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nCreated dataset.json with {num_training} training cases")
    print(f"Channels: 3x RGB (Red, Green, Blue)")
    print(f"Semantic classes: background (0), polyp (1)")
    return json_path


def main():
    """Main conversion function."""
    script_dir = Path(__file__).parent

    print("=" * 70)
    print("nnUNet Dataset Format Converter - Kvasir-SEG Polyp Segmentation")
    print("=" * 70)

    print("\n[Step 1/4] Unzipping data...")
    print("-" * 70)

    kvasir_dir = unzip_data(script_dir, "archive.zip")

    if kvasir_dir is None or not kvasir_dir.exists():
        print("Error: Could not extract Kvasir-SEG data")
        return

    # Define source directories
    images_dir = kvasir_dir / "images"
    masks_dir = kvasir_dir / "masks"

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return

    if not masks_dir.exists():
        print(f"Error: Masks directory not found: {masks_dir}")
        return

    print("\n[Step 2/4] Reading train/val split...")
    print("-" * 70)

    # Read split files from the dataset root directory
    train_file = script_dir / "train.txt"
    val_file = script_dir / "val.txt"

    train_ids = read_split_file(train_file)
    val_ids = read_split_file(val_file)

    if not train_ids:
        print("Error: No training image IDs found")
        return

    if not val_ids:
        print("Warning: No validation image IDs found")

    print(f"Found {len(train_ids)} training images")
    print(f"Found {len(val_ids)} validation images")
    print(f"Total: {len(train_ids) + len(val_ids)} images")

    print("\n[Step 3/4] Creating output directories...")
    print("-" * 70)

    images_tr_dir = script_dir / "imagesTr"
    labels_tr_dir = script_dir / "labelsTr"
    images_ts_dir = script_dir / "imagesTs"
    labels_ts_dir = script_dir / "labelsTs"

    images_tr_dir.mkdir(exist_ok=True)
    labels_tr_dir.mkdir(exist_ok=True)
    images_ts_dir.mkdir(exist_ok=True)
    labels_ts_dir.mkdir(exist_ok=True)

    print(f"Created directories:")
    print(f"  - {images_tr_dir.name}/")
    print(f"  - {labels_tr_dir.name}/")
    print(f"  - {images_ts_dir.name}/")
    print(f"  - {labels_ts_dir.name}/")

    print("\n[Step 4/4] Converting data to nnUNet format...")
    print("-" * 70)

    # Process training data
    print("\nProcessing training data:")
    num_training = process_and_copy_files(
        images_dir,
        masks_dir,
        train_ids,
        images_tr_dir,
        labels_tr_dir,
        start_case_id=0
    )
    print(f"Successfully converted {num_training} training cases")

    # Process validation data (goes to test folders in nnUNet)
    if val_ids:
        print("\nProcessing validation data:")
        num_val = process_and_copy_files(
            images_dir,
            masks_dir,
            val_ids,
            images_ts_dir,
            labels_ts_dir,
            start_case_id=0
        )
        print(f"Successfully converted {num_val} validation cases")

    # Create dataset.json
    create_dataset_json(script_dir, num_training)

    print("\n" + "=" * 70)
    print("Conversion Complete!")
    print("=" * 70)

    print("\nDirectory structure:")
    num_train_cases = len(list(images_tr_dir.glob('*_0000.png')))
    num_val_cases = len(list(images_ts_dir.glob('*_0000.png')))
    print(f"  imagesTr/   - {num_train_cases * 3} files ({num_train_cases} cases × 3 channels)")
    print(f"  labelsTr/   - {len(list(labels_tr_dir.glob('*.png')))} training binary masks")
    print(f"  imagesTs/   - {num_val_cases * 3} files ({num_val_cases} cases × 3 channels)")
    print(f"  labelsTs/   - {len(list(labels_ts_dir.glob('*.png')))} validation binary masks")
    print(f"  dataset.json - metadata file")

    print("\nImage format:")
    print("  - 3 channels: RGB (Red, Green, Blue)")
    print("  - Each channel saved as separate file: *_0000.png (R), *_0001.png (G), *_0002.png (B)")
    print("  - Reason: Color information aids polyp detection in colonoscopy")
    print("  - Resolution: Variable (from original Kvasir-SEG dataset)")

    print("\nSemantic classes:")
    print("  - 0: background")
    print("  - 1: polyp")

    print("\nDataset split:")
    print(f"  - Training: {num_training} images (88%)")
    if val_ids:
        print(f"  - Validation: {num_val} images (12%)")

    print("\nNext steps:")
    print("  1. Verify the conversion looks correct")
    print("  2. Set nnUNet environment variable (if not already set):")
    print("     export nnUNet_raw=/path/to/nnUNet_raw")
    print("  3. Run preprocessing:")
    print("     nnUNetv2_plan_and_preprocess -d 3")
    print("  4. Train model:")
    print("     nnUNetv2_train 3 2d 0")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
