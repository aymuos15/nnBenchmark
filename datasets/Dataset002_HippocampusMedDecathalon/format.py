#!/usr/bin/env python3
"""
nnUNet Dataset Format Converter for Hippocampus (Medical Decathlon Task04)

Converts Medical Decathlon hippocampus data to nnUNet format.

This script:
1. Extracts Task04_Hippocampus.tar
2. Renames files to nnUNet naming convention (adds _0000 channel suffix to images)
3. Reorganizes files into nnUNet format
4. Creates simplified dataset.json metadata file

The data is already in semantic segmentation format with 3 classes:
- 0: background
- 1: Anterior hippocampus
- 2: Posterior hippocampus

Data format: 3D MRI volumes in NIfTI (.nii.gz) format
"""

import json
import shutil
import tarfile
from pathlib import Path
from typing import Optional, Set


def extract_tar(dataset_dir: Path, tar_file: str) -> Optional[Path]:
    """
    Extract a tar archive.

    Args:
        dataset_dir: Path to dataset directory
        tar_file: Name of tar file (e.g., 'Task04_Hippocampus.tar')

    Returns:
        Path to extracted directory or None if failed
    """
    tar_path = dataset_dir / tar_file
    extract_name = tar_file.replace(".tar", "")
    extract_dir = dataset_dir / extract_name

    if extract_dir.exists():
        print(f"Directory {extract_dir.name} already exists, skipping extraction")
        return extract_dir

    if not tar_path.exists():
        print(f"Warning: {tar_file} not found")
        return None

    print(f"Extracting {tar_file}...")
    with tarfile.open(tar_path, "r") as tar_ref:
        tar_ref.extractall(dataset_dir)

    print(f"Extracted to {extract_dir}")
    return extract_dir


def get_unique_cases(folder_path: Path, prefix: str = "hippocampus_") -> Set[str]:
    """
    Extract unique case IDs from folder.

    Args:
        folder_path: Path to folder containing .nii.gz files
        prefix: Filename prefix to look for

    Returns:
        Set of case IDs (e.g., {'001', '002', ...})
    """
    cases = set()
    if not folder_path.exists():
        return cases

    for file in folder_path.glob(f"{prefix}*.nii.gz"):
        # Extract case ID: hippocampus_001.nii.gz -> 001
        case_id = file.stem.replace(".nii", "").replace(prefix, "")
        cases.add(case_id)

    return cases


def copy_and_rename_files(
    src_images_dir: Path,
    src_labels_dir: Optional[Path],
    dst_images_dir: Path,
    dst_labels_dir: Optional[Path],
    case_ids: Set[str],
    prefix: str = "hippocampus_",
    file_ending: str = ".nii.gz",
) -> int:
    """
    Copy and rename files from source to destination directories.
    Adds _0000 channel suffix to image filenames for nnUNet compatibility.

    Args:
        src_images_dir: Source directory for images
        src_labels_dir: Source directory for labels (None for test data)
        dst_images_dir: Destination for images
        dst_labels_dir: Destination for labels (None for test data)
        case_ids: Set of case identifiers
        prefix: Filename prefix
        file_ending: File extension

    Returns:
        Number of cases processed
    """
    count = 0

    for case_id in sorted(case_ids):
        img_src = src_images_dir / f"{prefix}{case_id}{file_ending}"

        if not img_src.exists():
            print(f"Warning: Image not found for case {case_id}")
            continue

        # Add _0000 channel suffix for nnUNet
        img_dst = dst_images_dir / f"{prefix}{case_id}_0000{file_ending}"

        # Copy the NIfTI file directly (no conversion needed)
        shutil.copy2(img_src, img_dst)

        # Copy label if this is training data
        if src_labels_dir is not None and dst_labels_dir is not None:
            label_src = src_labels_dir / f"{prefix}{case_id}{file_ending}"

            if not label_src.exists():
                print(f"Warning: Label not found for case {case_id}")
                continue

            # Labels don't need channel suffix
            label_dst = dst_labels_dir / f"{prefix}{case_id}{file_ending}"
            shutil.copy2(label_src, label_dst)

        count += 1
        if count % 50 == 0:
            print(f"  Processed {count} cases...")

    return count


def create_dataset_json(
    output_dir: Path, num_training: int, file_ending: str = ".nii.gz"
):
    """
    Create simplified dataset.json file for nnUNet.

    Adapts from Medical Decathlon format to nnUNet's expected format.
    """

    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "Anterior": 1, "Posterior": 2},
        "numTraining": num_training,
        "file_ending": file_ending,
        "name": "Hippocampus",
        "description": "Left and right hippocampus segmentation from MRI",
        "reference": "Vanderbilt University Medical Center - Medical Decathlon Task04",
        "licence": "CC-BY-SA 4.0",
    }

    json_path = output_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nCreated dataset.json with {num_training} training cases")
    print("Modality: MRI (3D volumes)")
    print("Semantic classes: background (0), Anterior (1), Posterior (2)")
    return json_path


def main():
    """Main conversion function."""
    script_dir = Path(__file__).parent

    print("=" * 70)
    print("nnUNet Dataset Format Converter - Hippocampus (Medical Decathlon)")
    print("=" * 70)

    print("\n[Step 1/3] Extracting data...")
    print("-" * 70)

    tar_file = "Task04_Hippocampus.tar"

    extracted_dir = extract_tar(script_dir, tar_file)

    if extracted_dir is None or not extracted_dir.exists():
        print("Error: Could not extract data")
        return

    # Medical Decathlon structure
    src_images_tr = extracted_dir / "imagesTr"
    src_labels_tr = extracted_dir / "labelsTr"
    src_images_ts = extracted_dir / "imagesTs"

    print("\n[Step 2/3] Creating output directories...")
    print("-" * 70)

    # nnUNet output structure (at dataset root, not in extracted folder)
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
    train_cases = get_unique_cases(src_images_tr)
    print(f"Found {len(train_cases)} training cases")

    if train_cases:
        num_training = copy_and_rename_files(
            src_images_tr, src_labels_tr, images_tr_dir, labels_tr_dir, train_cases
        )

        print(f"Successfully converted {num_training} training cases")

        create_dataset_json(script_dir, num_training)
    else:
        print("Error: No training cases found")
        return

    if src_images_ts.exists():
        print("\nProcessing test data:")
        test_cases = get_unique_cases(src_images_ts)
        print(f"Found {len(test_cases)} test cases")

        if test_cases:
            num_test = copy_and_rename_files(
                src_images_ts,
                None,  # No labels for test data
                images_ts_dir,
                None,  # No labels for test data
                test_cases,
            )
            print(f"Successfully converted {num_test} test cases")
    else:
        print("\nNote: Test data not found in extracted archive")

    print("\n" + "=" * 70)
    print("Conversion Complete!")
    print("=" * 70)

    print("\nDirectory structure:")
    print(
        f"  imagesTr/   - {len(list(images_tr_dir.glob('*_0000.nii.gz')))} training images (3D MRI)"
    )
    print(
        f"  labelsTr/   - {len(list(labels_tr_dir.glob('*.nii.gz')))} training semantic masks"
    )
    print(
        f"  imagesTs/   - {len(list(images_ts_dir.glob('*_0000.nii.gz')))} test images (3D MRI)"
    )
    print("  dataset.json - metadata file")

    print("\nFile format:")
    print("  - NIfTI (.nii.gz) - 3D volumetric MRI data")
    print("  - Images: hippocampus_XXX_0000.nii.gz (channel suffix added)")
    print("  - Labels: hippocampus_XXX.nii.gz (multi-class semantic)")

    print("\nSemantic classes:")
    print("  - 0: background")
    print("  - 1: Anterior hippocampus")
    print("  - 2: Posterior hippocampus")

    print("\nNext steps:")
    print("  1. Set nnUNet environment variable (if not already set):")
    print("     export nnUNet_raw=/path/to/nnUNet_raw")
    print("  2. Run preprocessing:")
    print("     nnUNetv2_plan_and_preprocess -d 2")
    print("  3. Train model:")
    print("     nnUNetv2_train 2 3d_fullres 0")
    print("     (Note: Use 3d_fullres for 3D medical imaging data)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
