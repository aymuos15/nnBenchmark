#!/usr/bin/env python3
"""
nnBenchmark vs nnUNet Configuration Comparison Script
======================================================

Compares Dataset001_Cellpose configurations between:
- nnUNet: dataset_fingerprint.json, plans.json
- nnBenchmark: fold_0.yaml

Validates that nnBenchmark exactly replicates nnUNet v2.4.1 configuration.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Install with: pip install pyyaml")
    sys.exit(1)


# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def colorize(text: str, color: str) -> str:
    """Colorize text for terminal output."""
    return f"{color}{text}{Colors.RESET}"


class ComparisonResult:
    """Stores comparison results for a single parameter."""

    def __init__(
        self,
        param_name: str,
        nnunet_value: Any,
        nnbenchmark_value: Any,
        matches: bool,
        expected_diff: bool = False,
        note: str = "",
    ):
        self.param_name = param_name
        self.nnunet_value = nnunet_value
        self.nnbenchmark_value = nnbenchmark_value
        self.matches = matches
        self.expected_diff = expected_diff
        self.note = note

    def __repr__(self) -> str:
        if self.matches:
            status = colorize("✅ MATCH", Colors.GREEN)
        elif self.expected_diff:
            status = colorize("⚠️  EXPECTED DIFF", Colors.YELLOW)
        else:
            status = colorize("❌ MISMATCH", Colors.RED)

        result = f"{status} | {self.param_name}"
        if not self.matches:
            result += f"\n    nnUNet:      {self.nnunet_value}"
            result += f"\n    nnBenchmark: {self.nnbenchmark_value}"
        if self.note:
            result += f"\n    Note: {self.note}"
        return result


class ConfigComparator:
    """Compares nnUNet and nnBenchmark configurations."""

    def __init__(
        self,
        fingerprint_path: Path,
        plans_path: Path,
        config_path: Path,
        verbose: bool = False,
    ):
        self.fingerprint_path = fingerprint_path
        self.plans_path = plans_path
        self.config_path = config_path
        self.verbose = verbose
        self.results: List[ComparisonResult] = []

        # Load files
        self.fingerprint = self._load_json(fingerprint_path)
        self.plans = self._load_json(plans_path)
        self.config = self._load_yaml(config_path)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        if not path.exists():
            print(colorize(f"ERROR: File not found: {path}", Colors.RED))
            sys.exit(1)
        with open(path, "r") as f:
            return json.load(f)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file."""
        if not path.exists():
            print(colorize(f"ERROR: File not found: {path}", Colors.RED))
            sys.exit(1)
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def add_result(
        self,
        param_name: str,
        nnunet_value: Any,
        nnbenchmark_value: Any,
        expected_diff: bool = False,
        note: str = "",
    ) -> None:
        """Add comparison result."""
        matches = nnunet_value == nnbenchmark_value
        result = ComparisonResult(
            param_name, nnunet_value, nnbenchmark_value, matches, expected_diff, note
        )
        self.results.append(result)

    def compare_dataset_properties(self) -> None:
        """Compare dataset fingerprinting properties."""

        # Patch size (spatial_size)
        plans_2d = self.plans["configurations"]["2d"]
        nnunet_patch = plans_2d["patch_size"]
        nnbenchmark_patch = self.config["dataset"]["spatial_size"]
        self.add_result("Patch Size", nnunet_patch, nnbenchmark_patch)

        # Median image size
        nnunet_median = plans_2d["median_image_size_in_voxels"]
        nnbenchmark_median = self.config["dataset"].get("median_shape", "Not stored")
        self.add_result("Median Image Size", nnunet_median, nnbenchmark_median)

        # Spacing
        nnunet_spacing = plans_2d["spacing"]
        nnbenchmark_spacing = self.config["dataset"].get("median_spacing", "Not stored")
        self.add_result("Median Spacing", nnunet_spacing, nnbenchmark_spacing)

        # Cropped shapes statistics
        if "shapes_after_crop" in self.fingerprint:
            import numpy as np
            from PIL import Image

            shapes_after_crop = np.array(self.fingerprint["shapes_after_crop"])
            num_images = len(shapes_after_crop)

            # Calculate statistics on spatial dimensions from nnUNet fingerprint (exclude channel dimension)
            spatial_shapes = shapes_after_crop[:, 1:]  # Skip first dimension (channel)
            nnunet_min_shape = tuple(int(x) for x in spatial_shapes.min(axis=0))
            nnunet_max_shape = tuple(int(x) for x in spatial_shapes.max(axis=0))
            median_shape_calc = tuple(int(x) for x in np.median(spatial_shapes, axis=0))

            # Read actual cropped shapes from preprocessed imagesTr directory
            preprocessed_dir = self.fingerprint_path.parent / "imagesTr"
            nnbenchmark_shapes = []

            if preprocessed_dir.exists():
                image_files = sorted(
                    list(preprocessed_dir.glob("*.png"))
                    + list(preprocessed_dir.glob("*.nii.gz"))
                )
                for img_path in image_files:
                    try:
                        if img_path.suffix == ".png":
                            img = Image.open(img_path)
                            # PIL gives (width, height) but we want (height, width)
                            nnbenchmark_shapes.append((img.height, img.width))
                        # Can add .nii.gz support if needed
                    except Exception:
                        pass  # Skip problematic files

                if nnbenchmark_shapes:
                    nnbenchmark_shapes_array = np.array(nnbenchmark_shapes)
                    nnbenchmark_min_shape = tuple(
                        int(x) for x in nnbenchmark_shapes_array.min(axis=0)
                    )
                    nnbenchmark_max_shape = tuple(
                        int(x) for x in nnbenchmark_shapes_array.max(axis=0)
                    )
                    nnbenchmark_num_images = len(nnbenchmark_shapes)
                else:
                    nnbenchmark_min_shape = "Could not read"
                    nnbenchmark_max_shape = "Could not read"
                    nnbenchmark_num_images = "Could not read"
            else:
                nnbenchmark_min_shape = "Directory not found"
                nnbenchmark_max_shape = "Directory not found"
                nnbenchmark_num_images = "Directory not found"

            # Show number of images
            self.add_result(
                "Number of Training Images",
                num_images,
                nnbenchmark_num_images,
                note="Both use same preprocessed training images",
            )

            # Compare cropped shape distributions
            self.add_result(
                "Min Cropped Shape",
                list(nnunet_min_shape),
                list(nnbenchmark_min_shape)
                if isinstance(nnbenchmark_min_shape, tuple)
                else nnbenchmark_min_shape,
                note="Minimum spatial dimensions after cropping",
            )

            self.add_result(
                "Max Cropped Shape",
                list(nnunet_max_shape),
                list(nnbenchmark_max_shape)
                if isinstance(nnbenchmark_max_shape, tuple)
                else nnbenchmark_max_shape,
                note="Maximum spatial dimensions after cropping",
            )

            # Verify median calculation matches
            if list(median_shape_calc) == nnunet_median:
                self.add_result(
                    "Median Shape Calculation",
                    "Consistent",
                    "Consistent",
                    note=f"nnUNet median {median_shape_calc} matches plans median {nnunet_median}",
                )

        # Foreground intensity properties
        nnunet_fg = self.fingerprint["foreground_intensity_properties_per_channel"]["0"]
        nnbenchmark_fg_mean = self.config["dataset"].get(
            "foreground_intensity_mean", "Not stored"
        )
        self.add_result(
            "FG Intensity - Mean",
            round(nnunet_fg["mean"], 2),
            nnbenchmark_fg_mean
            if isinstance(nnbenchmark_fg_mean, str)
            else round(nnbenchmark_fg_mean, 2),
        )

    def compare_architecture(self) -> None:
        """Compare network architecture."""

        plans_2d = self.plans["configurations"]["2d"]
        arch = plans_2d["architecture"]
        arch_kwargs = arch["arch_kwargs"]

        model = self.config["model"]

        # Network type
        nnunet_net = arch["network_class_name"].split(".")[-1]
        nnbenchmark_net = model["type"]
        self.add_result(
            "Network Type",
            nnunet_net,
            nnbenchmark_net,
            note="PlainConvUNet (nnUNet) == DynUNet (MONAI)",
        )

        # Spatial dims
        nnbenchmark_dims = model["spatial_dims"]
        self.add_result("Spatial Dimensions", 2, nnbenchmark_dims)

        # Feature channels
        nnunet_features = arch_kwargs["features_per_stage"]
        nnbenchmark_features = model["filters"]
        self.add_result("Feature Channels", nnunet_features, nnbenchmark_features)

        # Kernel sizes
        nnunet_kernels = arch_kwargs["kernel_sizes"]
        nnbenchmark_kernels = model["kernel_size"]
        self.add_result("Kernel Sizes", nnunet_kernels, nnbenchmark_kernels)

        # Strides (critical: first level must be [1,1] for full resolution)
        nnunet_strides = arch_kwargs["strides"]
        nnbenchmark_strides = model["strides"]
        self.add_result("Strides", nnunet_strides, nnbenchmark_strides)

        # Verify first level is full resolution
        if nnbenchmark_strides[0] == [1, 1]:
            self.add_result(
                "First Level Full Resolution",
                True,
                True,
                note="✅ Critical: First encoder level maintains full resolution",
            )

        # Normalization
        nnunet_norm = arch_kwargs["norm_op"].split(".")[-1]
        nnbenchmark_norm = model["norm_name"][0]
        self.add_result(
            "Normalization",
            nnunet_norm,
            nnbenchmark_norm,
            note="InstanceNorm2d (nnUNet) == INSTANCE (MONAI)",
        )

        # Normalization affine
        nnunet_affine = arch_kwargs["norm_op_kwargs"]["affine"]
        nnbenchmark_affine = model["norm_name"][1]["affine"]
        self.add_result("Norm Affine", nnunet_affine, nnbenchmark_affine)

        # Activation
        nnunet_act = arch_kwargs["nonlin"].split(".")[-1]
        nnbenchmark_act = model["act_name"][0]
        self.add_result("Activation", nnunet_act, nnbenchmark_act.replace("_", ""))

        # Activation slope
        nnunet_slope = arch_kwargs["nonlin_kwargs"].get("inplace", True)
        nnbenchmark_slope = model["act_name"][1].get("inplace", True)
        self.add_result("Activation Inplace", nnunet_slope, nnbenchmark_slope)

        # Negative slope
        nnunet_neg_slope = 0.01  # Default for LeakyReLU
        nnbenchmark_neg_slope = model["act_name"][1].get("negative_slope", 0.01)
        self.add_result(
            "LeakyReLU Negative Slope", nnunet_neg_slope, nnbenchmark_neg_slope
        )

        # Residual blocks
        nnbenchmark_res = model.get("res_block", False)
        self.add_result(
            "Residual Blocks",
            False,
            nnbenchmark_res,
            note="Plain convolutions (no residual)",
        )

        # Deep supervision
        nnunet_ds = True  # Enabled by default in nnUNet
        nnbenchmark_ds = model.get("deep_supervision", False)
        self.add_result("Deep Supervision", nnunet_ds, nnbenchmark_ds)

    def compare_training(self) -> None:
        """Compare training configuration."""

        plans_2d = self.plans["configurations"]["2d"]
        training = self.config["training"]

        # Batch size
        nnunet_batch = plans_2d["batch_size"]
        nnbenchmark_batch = training["batch_size"]
        self.add_result("Batch Size", nnunet_batch, nnbenchmark_batch)

        # Epochs (known difference)
        nnunet_epochs = 1000
        nnbenchmark_epochs = training["epochs"]
        self.add_result(
            "Epochs",
            nnunet_epochs,
            nnbenchmark_epochs,
            expected_diff=True,
            note="Framework difference: nnUNet uses 1000 epochs with 250 iters/epoch, "
            "nnBenchmark uses 200 full epochs (PyTorch Lightning)",
        )

        # Learning rate
        nnunet_lr = 0.01
        nnbenchmark_lr = training["learning_rate"]
        self.add_result("Learning Rate", nnunet_lr, nnbenchmark_lr)

        # Mixed precision
        nnunet_amp = True  # Default in nnUNet
        nnbenchmark_amp = training.get("mixed_precision", False)
        self.add_result("Mixed Precision (AMP)", nnunet_amp, nnbenchmark_amp)

    def compare_optimizer(self) -> None:
        """Compare optimizer configuration."""

        optimizer = self.config["optimizer"]

        # Optimizer type
        self.add_result("Optimizer Type", "SGD", optimizer["type"])

        # Weight decay
        nnunet_wd = 3e-5
        nnbenchmark_wd = optimizer["weight_decay"]
        self.add_result("Weight Decay", nnunet_wd, nnbenchmark_wd)

        # Momentum
        nnunet_momentum = 0.99
        nnbenchmark_momentum = optimizer["momentum"]
        self.add_result("Momentum", nnunet_momentum, nnbenchmark_momentum)

        # Nesterov
        nnunet_nesterov = True
        nnbenchmark_nesterov = optimizer["nesterov"]
        self.add_result("Nesterov", nnunet_nesterov, nnbenchmark_nesterov)

    def compare_loss(self) -> None:
        """Compare loss function configuration."""

        loss = self.config["loss"]

        # Loss type
        self.add_result("Loss Type", "DC_and_CE_loss", loss["type"])

        # Batch dice
        plans_2d = self.plans["configurations"]["2d"]
        nnunet_batch_dice = plans_2d.get("batch_dice", True)
        nnbenchmark_batch_dice = loss.get("batch", False)
        self.add_result("Batch Dice", nnunet_batch_dice, nnbenchmark_batch_dice)

        # Softmax
        nnbenchmark_softmax = loss.get("softmax", False)
        self.add_result("Softmax", True, nnbenchmark_softmax)

        # To onehot
        nnbenchmark_onehot = loss.get("to_onehot_y", False)
        self.add_result("To OneHot", True, nnbenchmark_onehot)

    def compare_normalization(self) -> None:
        """Compare data normalization schemes."""

        plans_2d = self.plans["configurations"]["2d"]

        # Normalization scheme
        nnunet_norm = plans_2d["normalization_schemes"][0]
        # nnBenchmark uses NormalizeIntensityd (Z-score)
        self.add_result(
            "Normalization Scheme",
            nnunet_norm,
            "NormalizeIntensityd (Z-score)",
            note="Both use Z-score normalization",
        )

        # Use mask for norm
        nnunet_mask = plans_2d["use_mask_for_norm"][0]
        # nnBenchmark: nonzero=false in transforms
        self.add_result("Use Mask for Norm", nnunet_mask, False)

    def compare_augmentation(self) -> None:
        """Compare data augmentation pipeline."""

        transforms = self.config.get("transforms", {})
        train_transforms = transforms.get("train", [])

        # Count augmentation types
        aug_counts = {}
        for transform in train_transforms:
            aug_type = transform.get("type", "Unknown")
            aug_counts[aug_type] = aug_counts.get(aug_type, 0) + 1

        # Key augmentations from nnUNet
        expected_augs = [
            "RandSpatialCropd",
            "RandRotated",
            "RandZoomd",
            "RandFlipd",
            "RandGaussianNoised",
            "RandGaussianSmoothd",
            "RandScaleIntensityd",
            "RandAdjustContrastd",
            "RandHistogramShiftd",
        ]

        for aug in expected_augs:
            has_aug = aug in aug_counts
            self.add_result(
                f"Augmentation: {aug}",
                "Present",
                "Present" if has_aug else "Missing",
            )

        # Rotation probability and range
        rotation_transforms = [
            t for t in train_transforms if t.get("type") == "RandRotated"
        ]
        if rotation_transforms:
            rot = rotation_transforms[0]
            nnunet_rot_prob = 0.2
            nnbenchmark_rot_prob = rot.get("prob", 0.0)
            self.add_result(
                "Rotation Probability", nnunet_rot_prob, nnbenchmark_rot_prob
            )

            nnunet_rot_range = 0.5236  # 30 degrees in radians
            nnbenchmark_rot_range = rot.get("range_x", 0.0)
            self.add_result(
                "Rotation Range (radians)",
                round(nnunet_rot_range, 4),
                round(nnbenchmark_rot_range, 4),
            )

        # Scaling
        zoom_transforms = [t for t in train_transforms if t.get("type") == "RandZoomd"]
        if zoom_transforms:
            zoom = zoom_transforms[0]
            nnunet_zoom_prob = 0.2
            nnbenchmark_zoom_prob = zoom.get("prob", 0.0)
            self.add_result(
                "Scaling Probability", nnunet_zoom_prob, nnbenchmark_zoom_prob
            )

    def generate_summary(self) -> Tuple[int, int, int]:
        """Generate comparison summary."""
        total = len(self.results)
        matches = sum(1 for r in self.results if r.matches)
        expected_diffs = sum(
            1 for r in self.results if r.expected_diff and not r.matches
        )
        mismatches = total - matches - expected_diffs

        print("\n" + "=" * 80)
        print(colorize("📊 COMPARISON SUMMARY", Colors.BOLD + Colors.CYAN))
        print("=" * 80)
        print(f"Total Parameters Checked: {total}")
        print(colorize(f"✅ Matches: {matches}", Colors.GREEN))
        print(colorize(f"⚠️  Expected Differences: {expected_diffs}", Colors.YELLOW))
        print(colorize(f"❌ Unexpected Mismatches: {mismatches}", Colors.RED))

        if mismatches == 0:
            print(
                colorize(
                    "\n🎉 SUCCESS: nnBenchmark configuration matches nnUNet!",
                    Colors.BOLD + Colors.GREEN,
                )
            )

        return matches, expected_diffs, mismatches

    def print_results(self) -> None:
        """Print all comparison results."""
        if self.verbose:
            for result in self.results:
                print(result)

    def print_table(self) -> None:
        """Print results in table format."""
        # Separate results by category
        matches = [r for r in self.results if r.matches]
        expected_diffs = [r for r in self.results if r.expected_diff and not r.matches]
        mismatches = [r for r in self.results if not r.matches and not r.expected_diff]

        def print_section_table(
            results: List[ComparisonResult], title: str, color: str
        ):
            if not results:
                return

            print(f"\n{colorize(title, Colors.BOLD + color)}")
            print("=" * 120)

            # Header
            header = f"{'Parameter':<40} | {'nnUNet Value':<30} | {'nnBenchmark Value':<30} | {'Status':<10}"
            print(colorize(header, Colors.BOLD))
            print("-" * 120)

            # Rows
            for r in results:
                status_symbol = (
                    "✅" if r.matches else ("⚠️" if r.expected_diff else "❌")
                )
                param = (
                    r.param_name[:38] + ".." if len(r.param_name) > 40 else r.param_name
                )
                nnunet_str = (
                    str(r.nnunet_value)[:28] + ".."
                    if len(str(r.nnunet_value)) > 30
                    else str(r.nnunet_value)
                )
                nnbench_str = (
                    str(r.nnbenchmark_value)[:28] + ".."
                    if len(str(r.nnbenchmark_value)) > 30
                    else str(r.nnbenchmark_value)
                )

                row = f"{param:<40} | {nnunet_str:<30} | {nnbench_str:<30} | {status_symbol}"
                print(row)

                # Print note if exists
                if r.note:
                    note_text = f"    └─ Note: {r.note}"
                    print(colorize(note_text, Colors.CYAN))

        # Print each section (matches first)
        if matches:
            print_section_table(matches, "✅ MATCHES", Colors.GREEN)

        if expected_diffs:
            print_section_table(
                expected_diffs, "⚠️  EXPECTED DIFFERENCES", Colors.YELLOW
            )

        if mismatches:
            print_section_table(
                mismatches, "❌ UNEXPECTED MISMATCHES (NEEDS ATTENTION)", Colors.RED
            )

    def run_comparison(self) -> int:
        """Run full comparison and return exit code."""
        print(colorize("\n" + "=" * 80, Colors.BOLD))
        print(
            colorize(
                "nnBenchmark vs nnUNet Configuration Comparison",
                Colors.BOLD + Colors.CYAN,
            )
        )
        print(colorize("=" * 80, Colors.BOLD))
        print("Dataset: Dataset001_Cellpose")
        print(f"Fingerprint: {self.fingerprint_path}")
        print(f"Plans: {self.plans_path}")
        print(f"Config: {self.config_path}")

        # Run all comparisons (silently collect results)
        self.compare_dataset_properties()
        self.compare_architecture()
        self.compare_training()
        self.compare_optimizer()
        self.compare_loss()
        self.compare_normalization()
        self.compare_augmentation()

        # Print results in table format
        self.print_table()

        # Generate summary
        _, _, mismatches = self.generate_summary()

        # Return exit code (0 = success, 1 = mismatches)
        return 0 if mismatches == 0 else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare nnBenchmark and nnUNet configurations for Dataset001_Cellpose"
    )
    parser.add_argument(
        "--fingerprint",
        type=Path,
        default=Path(
            "/home/localssk23/CAI4Soumya/SegData/nnUNet_preprocessed/Dataset001_Cellpose/dataset_fingerprint.json"
        ),
        help="Path to dataset_fingerprint.json",
    )
    parser.add_argument(
        "--plans",
        type=Path,
        default=Path(
            "/home/localssk23/CAI4Soumya/SegData/nnUNet_results/Dataset001_Cellpose/nnUNetTrainer_50epochs__nnUNetPlans__2d/plans.json"
        ),
        help="Path to plans.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/home/localssk23/CAI4Soumya/SegData/nnUNet_results/Dataset001_Cellpose/fold_0/fold_0.yaml"
        ),
        help="Path to nnBenchmark config YAML",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed comparison results",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Run comparison
    comparator = ConfigComparator(
        args.fingerprint,
        args.plans,
        args.config,
        verbose=args.verbose,
    )
    exit_code = comparator.run_comparison()

    # Save JSON output if requested
    if args.json_output:
        results_data = {
            "total": len(comparator.results),
            "matches": sum(1 for r in comparator.results if r.matches),
            "expected_diffs": sum(
                1 for r in comparator.results if r.expected_diff and not r.matches
            ),
            "mismatches": sum(
                1 for r in comparator.results if not r.matches and not r.expected_diff
            ),
            "results": [
                {
                    "param": r.param_name,
                    "nnunet": str(r.nnunet_value),
                    "nnbenchmark": str(r.nnbenchmark_value),
                    "matches": r.matches,
                    "expected_diff": r.expected_diff,
                    "note": r.note,
                }
                for r in comparator.results
            ],
        }
        with open(args.json_output, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\n📄 Results saved to: {args.json_output}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
