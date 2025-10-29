#!/usr/bin/env python3
"""
Copy exact case assignments from splits_final.json to splits.json format.

Usage:
    python copy_splits.py <splits_final_path> <splits_output_path>

Example:
    python copy_splits.py /path/to/splits_final.json /path/to/splits.json
"""

import json
import sys
from pathlib import Path


def convert_splits(splits_final_path: str, splits_output_path: str) -> None:
    """
    Convert splits_final.json format to splits.json format.

    Args:
        splits_final_path: Path to input splits_final.json
        splits_output_path: Path to output splits.json
    """
    # Read splits_final.json
    print(f"Reading: {splits_final_path}")
    with open(splits_final_path, 'r') as f:
        splits_final = json.load(f)

    # Validate input format
    if not isinstance(splits_final, list):
        raise ValueError("splits_final.json must be a list of fold dictionaries")

    # Convert to splits.json format
    splits = {}
    for fold_idx, fold_data in enumerate(splits_final):
        if 'train' not in fold_data or 'val' not in fold_data:
            raise ValueError(f"Fold {fold_idx} missing 'train' or 'val' key")

        # Convert case IDs to filenames (add _0000.nii.gz suffix)
        train_files = [f"{case_id}_0000.nii.gz" for case_id in fold_data['train']]
        val_files = [f"{case_id}_0000.nii.gz" for case_id in fold_data['val']]

        # Create fold entry
        splits[f"fold_{fold_idx}"] = {
            "train": train_files,
            "val": val_files
        }

        print(f"  Fold {fold_idx}: {len(train_files)} train, {len(val_files)} val")

    # Write output
    print(f"\nWriting: {splits_output_path}")
    with open(splits_output_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"✓ Successfully converted {len(splits)} folds")

    # Summary
    total_train = sum(len(splits[f"fold_{i}"]["train"]) for i in range(len(splits)))
    total_val = sum(len(splits[f"fold_{i}"]["val"]) for i in range(len(splits)))
    print("\nSummary:")
    print(f"  Total folds: {len(splits)}")
    print(f"  Total train samples: {total_train}")
    print(f"  Total val samples: {total_val}")
    print(f"  Avg train per fold: {total_train // len(splits)}")
    print(f"  Avg val per fold: {total_val // len(splits)}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    splits_final_path = sys.argv[1]
    splits_output_path = sys.argv[2]

    # Validate input file exists
    if not Path(splits_final_path).exists():
        print(f"Error: Input file not found: {splits_final_path}")
        sys.exit(1)

    # Create output directory if needed
    output_dir = Path(splits_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        convert_splits(splits_final_path, splits_output_path)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
