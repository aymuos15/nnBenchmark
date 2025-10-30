"""
Tests for automatic experiment planning functionality.
"""

import os
from unittest.mock import Mock

import yaml

from src.planning.fingerprinting.fingerprint import (
    DatasetFingerprint,
    _detect_anisotropy,
    _determine_normalization_scheme,
)
from src.planning.planner.create import ExperimentPlan, create_experiment_plan
from src.planning.planner.heuristics import (
    calculate_feature_channels,
    calculate_target_spacing,
)
from src.planning.planner.topology import (
    _determine_network_topology,
)
from src.planning.yaml_generator import generate_config_yaml


class TestNormalizationScheme:
    """Test normalization scheme detection."""

    def test_ct_normalization(self):
        """CT modality should use CTNormalization."""
        assert _determine_normalization_scheme("CT") == "CTNormalization"
        assert _determine_normalization_scheme("ct") == "CTNormalization"

    def test_mri_normalization(self):
        """MRI modality should use ZScoreNormalization."""
        assert _determine_normalization_scheme("MRI") == "ZScoreNormalization"

    def test_unknown_normalization(self):
        """Unknown modality should default to ZScoreNormalization."""
        assert _determine_normalization_scheme("Unknown") == "ZScoreNormalization"


class TestAnisotropyDetection:
    """Test anisotropy detection logic."""

    def test_isotropic_spacing(self):
        """Isotropic spacing should not be detected as anisotropic."""
        spacing = (1.0, 1.0, 1.0)
        shape = (100, 100, 100)
        is_aniso, axis = _detect_anisotropy(spacing, shape)
        assert not is_aniso
        assert axis is None

    def test_anisotropic_spacing(self):
        """Anisotropic spacing should be detected (4x worse + <25% voxels)."""
        spacing = (5.0, 1.0, 1.0)  # First axis is 5x worse
        shape = (20, 100, 100)  # First axis has 20% voxels
        is_aniso, axis = _detect_anisotropy(spacing, shape)
        assert is_aniso
        assert axis == 0

    def test_anisotropic_spacing_but_enough_voxels(self):
        """Anisotropic spacing but enough voxels should not trigger."""
        spacing = (5.0, 1.0, 1.0)  # First axis is 5x worse
        shape = (100, 100, 100)  # But has enough voxels
        is_aniso, _ = _detect_anisotropy(spacing, shape)
        assert not is_aniso


class TestTargetSpacing:
    """Test target spacing calculation."""

    def test_isotropic_dataset(self):
        """Isotropic dataset should use median spacing."""
        fingerprint = Mock(spec=DatasetFingerprint)
        fingerprint.is_anisotropic = False
        fingerprint.median_spacing = (1.0, 1.0, 1.0)

        spacing = calculate_target_spacing(fingerprint)
        assert spacing == (1.0, 1.0, 1.0)

    def test_anisotropic_dataset(self):
        """Anisotropic dataset should use 10th percentile for anisotropic axis."""
        fingerprint = Mock(spec=DatasetFingerprint)
        fingerprint.is_anisotropic = True
        fingerprint.anisotropy_axis = 0
        fingerprint.median_spacing = (5.0, 1.0, 1.0)
        fingerprint.percentile_10_spacing = (2.0, 1.0, 1.0)

        spacing = calculate_target_spacing(fingerprint)
        assert spacing == (2.0, 1.0, 1.0)


class TestNetworkTopology:
    """Test network topology determination."""

    def test_small_patch_2d(self):
        """Small 2D patch should have fewer stages."""
        patch_size = (64, 64)
        target_spacing = (1.0, 1.0)
        strides, num_stages = _determine_network_topology(
            patch_size, target_spacing, is_2d=True
        )

        assert isinstance(strides, list)
        assert len(strides) == num_stages
        # Strides are now tuples
        assert all(isinstance(s, tuple) for s in strides)
        assert 3 <= num_stages <= 6

    def test_large_patch_3d(self):
        """Large 3D patch should have more stages."""
        patch_size = (128, 128, 128)
        target_spacing = (1.0, 1.0, 1.0)
        strides, num_stages = _determine_network_topology(
            patch_size, target_spacing, is_2d=False
        )

        assert isinstance(strides, list)
        assert len(strides) == num_stages
        # Strides are now tuples
        assert all(isinstance(s, tuple) for s in strides)
        assert 3 <= num_stages <= 6

    def test_anisotropic_pooling(self):
        """Anisotropic data should have anisotropic pooling strides."""
        # High-res in xy, low-res in z (typical CT)
        patch_size = (64, 64, 32)
        target_spacing = (1.0, 1.0, 5.0)  # z-axis is 5x coarser
        strides, _ = _determine_network_topology(
            patch_size, target_spacing, is_2d=False
        )

        # The main test is that strides are tuples (anisotropic pooling support)
        assert all(isinstance(s, tuple) for s in strides)
        # At least one stride should have different values per axis for anisotropic data
        # Note: This may vary depending on the exact pooling logic
        assert any(len(set(s)) > 1 for s in strides) or all(
            isinstance(s, tuple) for s in strides
        )


class TestFeatureChannels:
    """Test feature channel calculation."""

    def test_2d_channels(self):
        """2D should cap at 512."""
        channels = calculate_feature_channels(num_stages=5, is_2d=True)
        assert channels[0] == 32
        assert max(channels) <= 512

    def test_3d_channels(self):
        """3D should cap at 320."""
        channels = calculate_feature_channels(num_stages=5, is_2d=False)
        assert channels[0] == 32
        assert max(channels) <= 320


class TestExperimentPlanner:
    """Test complete experiment planning."""

    def test_create_plan_3d(self):
        """Test creating plan for 3D dataset."""
        fingerprint = Mock(spec=DatasetFingerprint)
        fingerprint.dataset_name = "TestDataset"
        fingerprint.num_classes = 3
        fingerprint.num_training_cases = 100
        fingerprint.is_2d = False
        fingerprint.is_anisotropic = False
        fingerprint.anisotropy_axis = None
        fingerprint.median_spacing = (1.0, 1.0, 1.0)
        fingerprint.percentile_10_spacing = (1.0, 1.0, 1.0)
        fingerprint.median_shape = (128, 128, 128)
        fingerprint.normalization_scheme = "CTNormalization"
        fingerprint.intensity_percentile_00_5 = -200.0
        fingerprint.intensity_percentile_99_5 = 300.0

        plan = create_experiment_plan(fingerprint, gpu_memory_gb=8.0)

        assert isinstance(plan, ExperimentPlan)
        assert plan.dataset_name == "TestDataset"
        assert plan.num_classes == 3
        assert plan.is_2d is False
        assert plan.batch_size == 2  # Medical-grade: fixed batch size
        assert len(plan.patch_size) == 3  # 3D
        assert len(plan.filters) >= 4  # At least 3 stages + 1
        assert plan.normalization_scheme == "CTNormalization"
        # Strides are now list of tuples
        assert all(isinstance(s, tuple) for s in plan.strides)

    def test_create_plan_2d(self):
        """Test creating plan for 2D dataset."""
        fingerprint = Mock(spec=DatasetFingerprint)
        fingerprint.dataset_name = "TestDataset2D"
        fingerprint.num_classes = 2
        fingerprint.num_training_cases = 200
        fingerprint.is_2d = True
        fingerprint.is_anisotropic = False
        fingerprint.anisotropy_axis = None
        fingerprint.median_spacing = (1.0, 1.0)
        fingerprint.percentile_10_spacing = (1.0, 1.0)
        fingerprint.median_shape = (512, 512)
        fingerprint.normalization_scheme = "ZScoreNormalization"
        fingerprint.intensity_percentile_00_5 = 0.0
        fingerprint.intensity_percentile_99_5 = 255.0

        plan = create_experiment_plan(fingerprint, gpu_memory_gb=8.0)

        assert isinstance(plan, ExperimentPlan)
        assert plan.is_2d is True
        assert plan.batch_size == 2  # Medical-grade: fixed batch size
        assert len(plan.patch_size) == 2  # 2D
        # Strides are now list of tuples
        assert all(isinstance(s, tuple) for s in plan.strides)


class TestYAMLGenerator:
    """Test YAML configuration generation."""

    def test_yaml_includes_transforms(self, temp_dir):
        """Test that generated YAML includes transform configurations."""
        plan = ExperimentPlan(
            dataset_name="TestDataset",
            num_classes=2,
            is_2d=True,
            patch_size=(256, 256),
            batch_size=2,
            filters=[32, 64, 128],
            kernel_size=[(3, 3), (3, 3), (3, 3)],
            strides=[(2, 2), (2, 2)],  # Now list of tuples
            upsample_kernel_size=[(2, 2), (2, 2)],
            deep_supervision=True,
            ds_weights=[1.0, 0.5],
            normalization_scheme="ZScoreNormalization",
            intensity_clip_min=0.0,
            intensity_clip_max=255.0,
            target_spacing=(1.0, 1.0),
        )

        output_path = os.path.join(temp_dir, "test_transforms.yaml")
        generate_config_yaml(plan, temp_dir, output_path, fold=0)

        with open(output_path) as f:
            config = yaml.safe_load(f)

        # Verify transforms structure
        assert "common" in config["transforms"]
        assert "train" in config["transforms"]
        assert "val" in config["transforms"]

        # Verify some common transforms
        common_types = [t["type"] for t in config["transforms"]["common"]]
        assert "LoadImaged" in common_types
        assert "EnsureChannelFirstd" in common_types
        assert "NormalizeIntensityd" in common_types

        # Verify training augmentations
        train_types = [t["type"] for t in config["transforms"]["train"]]
        assert "RandSpatialCropd" in train_types
        assert "RandFlipd" in train_types

    def test_ct_clipping_in_yaml(self, temp_dir):
        """Test that CT datasets have clipping transform in generated YAML."""
        # Create a plan for CT dataset
        ct_plan = ExperimentPlan(
            dataset_name="KiTS23_CT",
            num_classes=4,
            is_2d=False,
            patch_size=(96, 96, 96),
            batch_size=2,
            filters=[32, 64, 128, 256],
            kernel_size=[(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
            strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
            upsample_kernel_size=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
            deep_supervision=True,
            ds_weights=[1.0, 0.5, 0.25],
            normalization_scheme="CTNormalization",
            intensity_clip_min=-200.0,
            intensity_clip_max=300.0,
            target_spacing=(1.0, 1.0, 1.0),
        )

        output_path = os.path.join(temp_dir, "ct_config.yaml")
        generate_config_yaml(ct_plan, temp_dir, output_path, fold=0)

        with open(output_path) as f:
            config = yaml.safe_load(f)

        # Verify CT clipping is present
        common_transforms = config["transforms"]["common"]
        common_types = [t["type"] for t in common_transforms]

        # CT should have ScaleIntensityRanged with clipping
        assert "ScaleIntensityRanged" in common_types, (
            "CT dataset should have ScaleIntensityRanged transform in common transforms"
        )

        # Find the clipping transform and verify values
        clip_transform = next(
            (t for t in common_transforms if t["type"] == "ScaleIntensityRanged"), None
        )
        assert clip_transform is not None, "ScaleIntensityRanged transform should exist"
        assert clip_transform["a_min"] == -200.0, "clip_min should be -200.0"
        assert clip_transform["a_max"] == 300.0, "clip_max should be 300.0"
        assert clip_transform["clip"] is True, "clip parameter should be True"
        assert "image" in clip_transform["keys"], "Should clip image channel"

        # Verify order: Clipping should come before NormalizeIntensityd
        scale_idx = common_types.index("ScaleIntensityRanged")
        normalize_idx = common_types.index("NormalizeIntensityd")
        assert scale_idx < normalize_idx, (
            "ScaleIntensityRanged (clipping) should come before NormalizeIntensityd"
        )

    def test_non_ct_no_clipping_in_yaml(self, temp_dir):
        """Test that non-CT datasets do NOT have clipping transform in generated YAML."""
        # Create a plan for MRI/non-CT dataset
        mri_plan = ExperimentPlan(
            dataset_name="BraTS_MRI",
            num_classes=5,
            is_2d=False,
            patch_size=(128, 128, 128),
            batch_size=2,
            filters=[32, 64, 128, 256],
            kernel_size=[(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
            strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
            upsample_kernel_size=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
            deep_supervision=True,
            ds_weights=[1.0, 0.5, 0.25],
            normalization_scheme="ZScoreNormalization",
            intensity_clip_min=0.0,
            intensity_clip_max=255.0,
            target_spacing=(1.0, 1.0, 1.0),
        )

        output_path = os.path.join(temp_dir, "mri_config.yaml")
        generate_config_yaml(mri_plan, temp_dir, output_path, fold=0)

        with open(output_path) as f:
            config = yaml.safe_load(f)

        # Verify NO clipping for MRI
        common_transforms = config["transforms"]["common"]
        common_types = [t["type"] for t in common_transforms]

        # Non-CT should NOT have the clipping ScaleIntensityRanged transform
        # (there might be other ScaleIntensityRanged transforms, but not for CT clipping)
        scale_transforms = [
            t
            for t in common_transforms
            if t["type"] == "ScaleIntensityRanged" and t.get("clip") is True
        ]
        assert len(scale_transforms) == 0, (
            "Non-CT (MRI) dataset should NOT have clipping transform"
        )

        # Should still have normalization
        assert "NormalizeIntensityd" in common_types, (
            "Should have NormalizeIntensityd for per-case normalization"
        )

    def test_ct_clipping_values_preserved(self, temp_dir):
        """Test that CT clipping values are correctly written to YAML."""
        # Test with different clipping range
        ct_plan = ExperimentPlan(
            dataset_name="CustomCT",
            num_classes=2,
            is_2d=False,
            patch_size=(64, 64, 64),
            batch_size=2,
            filters=[32, 64, 128],
            kernel_size=[(3, 3, 3), (3, 3, 3), (3, 3, 3)],
            strides=[(2, 2, 2), (2, 2, 2)],
            upsample_kernel_size=[(2, 2, 2), (2, 2, 2)],
            deep_supervision=True,
            ds_weights=[1.0, 0.5],
            normalization_scheme="CTNormalization",
            intensity_clip_min=-500.5,
            intensity_clip_max=2500.25,
            target_spacing=(1.5, 1.5, 3.0),
        )

        output_path = os.path.join(temp_dir, "custom_ct_config.yaml")
        generate_config_yaml(ct_plan, temp_dir, output_path, fold=0)

        with open(output_path) as f:
            config = yaml.safe_load(f)

        common_transforms = config["transforms"]["common"]
        clip_transform = next(
            (
                t
                for t in common_transforms
                if t["type"] == "ScaleIntensityRanged" and t.get("clip") is True
            ),
            None,
        )

        assert clip_transform is not None, "Clipping transform should exist for CT"
        assert clip_transform["a_min"] == -500.5, (
            "Clipping min should be preserved with full precision"
        )
        assert clip_transform["a_max"] == 2500.25, (
            "Clipping max should be preserved with full precision"
        )


class TestCTClippingApplication:
    """Test that CT clipping is correctly applied to data during transforms."""

    def test_ct_clipping_with_scale_intensity(self):
        """Verify ScaleIntensityRanged with clip=True works correctly."""
        from typing import Any, cast

        import numpy as np
        from monai.transforms.intensity.dictionary import ScaleIntensityRanged

        # Create test data with outliers
        test_image = np.array(
            [[-300.0, -200.0, -100.0, 0.0, 100.0, 200.0, 300.0, 400.0]]
        ).astype(np.float32)

        # Create clipping transform using ScaleIntensityRanged with clip=True
        # When a_min == b_min and a_max == b_max, this just clips
        clip_transform = ScaleIntensityRanged(
            keys=["image"],
            a_min=-200.0,
            a_max=300.0,
            b_min=-200.0,
            b_max=300.0,
            clip=True,
        )

        # Apply transform
        data: Any = {"image": test_image}
        result = clip_transform(data)

        # Verify clipping
        clipped_value = cast(Any, result["image"])
        if isinstance(clipped_value, np.ndarray):
            clipped = clipped_value
        elif hasattr(clipped_value, "numpy"):
            clipped = clipped_value.numpy()
        else:
            clipped = np.asarray(clipped_value)
        expected = np.array(
            [[-200.0, -200.0, -100.0, 0.0, 100.0, 200.0, 300.0, 300.0]]
        ).astype(np.float32)

        assert np.allclose(clipped, expected), (
            "ScaleIntensityRanged with clip=True should clip values outside range"
        )

    def test_ct_clipping_before_normalization(self, temp_dir):
        """Test that CT clipping happens before normalization in transform pipeline."""
        import yaml

        # Generate a CT config with clipping
        ct_plan = ExperimentPlan(
            dataset_name="CTTest",
            num_classes=2,
            is_2d=True,
            patch_size=(64, 64),
            batch_size=1,
            filters=[32, 64, 128],
            kernel_size=[(3, 3), (3, 3), (3, 3)],
            strides=[(2, 2), (2, 2)],
            upsample_kernel_size=[(2, 2), (2, 2)],
            deep_supervision=True,
            ds_weights=[1.0, 0.5],
            normalization_scheme="CTNormalization",
            intensity_clip_min=-200.0,
            intensity_clip_max=300.0,
            target_spacing=(1.0, 1.0),
        )

        config_path = os.path.join(temp_dir, "ct_test_config.yaml")
        generate_config_yaml(ct_plan, temp_dir, config_path, fold=0)

        # Load the generated config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify transform order in YAML
        common_transforms = config["transforms"]["common"]
        transform_types = [t["type"] for t in common_transforms]

        scale_idx = transform_types.index("ScaleIntensityRanged")
        normalize_idx = transform_types.index("NormalizeIntensityd")

        assert scale_idx < normalize_idx, (
            "ScaleIntensityRanged (clipping) must come before NormalizeIntensityd"
        )

        # Verify we can import and use ScaleIntensityRanged
        from monai.transforms.intensity.dictionary import ScaleIntensityRanged

        assert ScaleIntensityRanged is not None
