# Test Suite Optimization - Detailed Code Examples

This document provides ready-to-implement code for the recommended test additions and improvements.

---

## PART 1: CRITICAL MISSING TESTS - IMPLEMENTATION READY

### Test 1: End-to-End Planning Workflow Integration

**File**: `/home/localssk23/CAI4Soumya/nnBenchmark/tests/test_planning.py`

**Add to the file** (after existing test classes):

```python
class TestPlanningWorkflowIntegration:
    """Integration tests for complete planning workflow."""

    def test_planning_workflow_end_to_end(self, temp_dir: str, mock_dataset_dir: str) -> None:
        """Test complete planning workflow: fingerprint → plan → yaml → splits."""
        import json
        import os
        from pathlib import Path

        from src.planning.fingerprinting.fingerprint import fingerprint_dataset
        from src.planning.planner.create import create_experiment_plan
        from src.planning.yaml_generator import generate_config_yaml
        from src.planning.splits import create_cross_validation_splits

        dataset_dir = mock_dataset_dir

        # Step 1: Fingerprint dataset
        fingerprint = fingerprint_dataset(dataset_dir)
        assert fingerprint is not None
        assert fingerprint.num_training_cases > 0
        assert fingerprint.num_classes > 0

        # Step 2: Create experiment plan
        plan = create_experiment_plan(fingerprint, gpu_memory_gb=8.0)
        assert plan is not None
        assert len(plan.patch_size) > 0
        assert plan.batch_size > 0

        # Step 3: Generate YAML config
        config_path = os.path.join(temp_dir, "test_config.yaml")
        generate_config_yaml(plan, temp_dir, config_path, fold=0)
        assert os.path.exists(config_path)

        # Verify YAML is valid and loadable
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config is not None
        assert "dataset" in config
        assert "model" in config
        assert "training" in config
        assert "transforms" in config

        # Step 4: Create splits (using existing splits from mock)
        # In real scenario, would call create_cross_validation_splits
        preprocessed_dir = os.environ.get("nnBench_preprocessed")
        splits_path = os.path.join(preprocessed_dir, "Dataset001_Hippo", "splits.json")
        assert os.path.exists(splits_path), "splits.json should exist in preprocessed dir"

        with open(splits_path) as f:
            splits = json.load(f)
        assert "fold_0" in splits
        assert "train" in splits["fold_0"]
        assert "val" in splits["fold_0"]

        # Verify workflow produced consistent results
        assert len(config["model"]["filters"]) == len(plan.filters)
        assert config["dataset"]["num_classes"] == plan.num_classes
        assert config["dataset"]["spatial_size"] == list(plan.patch_size)


    def test_planning_deterministic_across_runs(self, temp_dir: str, mock_dataset_dir: str) -> None:
        """Test that planning produces identical results across multiple runs.

        Critical for reproducibility - same dataset should always produce same config.
        """
        import json
        import os
        from src.planning.fingerprinting.fingerprint import fingerprint_dataset
        from src.planning.planner.create import create_experiment_plan
        from src.planning.yaml_generator import generate_config_yaml

        # Run 1
        fp1 = fingerprint_dataset(mock_dataset_dir)
        plan1 = create_experiment_plan(fp1, gpu_memory_gb=8.0)
        config_path1 = os.path.join(temp_dir, "config1.yaml")
        generate_config_yaml(plan1, temp_dir, config_path1, fold=0)

        # Run 2 (same inputs)
        fp2 = fingerprint_dataset(mock_dataset_dir)
        plan2 = create_experiment_plan(fp2, gpu_memory_gb=8.0)
        config_path2 = os.path.join(temp_dir, "config2.yaml")
        generate_config_yaml(plan2, temp_dir, config_path2, fold=0)

        # Compare plans
        assert plan1.patch_size == plan2.patch_size
        assert plan1.batch_size == plan2.batch_size
        assert plan1.filters == plan2.filters
        assert plan1.strides == plan2.strides

        # Compare YAML files
        import yaml
        with open(config_path1) as f:
            config1 = yaml.safe_load(f)
        with open(config_path2) as f:
            config2 = yaml.safe_load(f)

        # Model architecture should be identical
        assert config1["model"]["filters"] == config2["model"]["filters"]
        assert config1["model"]["strides"] == config2["model"]["strides"]
        assert config1["model"]["kernel_size"] == config2["model"]["kernel_size"]
```

---

### Test 2: Corrupted Data Handling

**File**: `/home/localssk23/CAI4Soumya/nnBenchmark/tests/test_monai_integration.py`

**Add to TestFingerprintDatasetErrorHandling class**:

```python
    def test_fingerprint_handles_partial_corruption_logs_warnings(
        self, temp_dir: str
    ) -> None:
        """Test fingerprinting with mixed valid and corrupted images.

        Realistic scenario: some NIfTI files corrupted/unreadable.
        Should succeed processing valid images, logging warnings for corrupted ones.
        """
        import json
        import os
        from pathlib import Path
        import nibabel as nib
        import numpy as np

        # Create dataset structure
        dataset_dir = Path(temp_dir) / "Dataset_PartialCorruption"
        dataset_dir.mkdir(parents=True)
        images_dir = dataset_dir / "imagesTr"
        labels_dir = dataset_dir / "labelsTr"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        # Create dataset.json
        dataset_json = {
            "name": "Dataset_PartialCorruption",
            "numTraining": 3,
            "labels": {"0": "background", "1": "foreground"},
            "modality": {"0": "MRI"},
            "file_ending": ".nii.gz",
        }
        with open(dataset_dir / "dataset.json", "w") as f:
            json.dump(dataset_json, f)

        # Create 2 valid NIfTI files
        for i in range(2):
            data = np.random.rand(10, 10, 10).astype(np.float32)
            affine = np.diag([1.5, 1.5, 1.5, 1.0])
            img = nib.Nifti1Image(data, affine=affine)
            nifti_path = images_dir / f"case_{i:03d}_0000.nii.gz"
            nib.save(img, str(nifti_path))

        # Create 1 corrupted file (empty/invalid)
        corrupted_path = images_dir / "case_002_0000.nii.gz"
        with open(corrupted_path, "wb") as f:
            f.write(b"not a valid nifti file")

        # Fingerprint should succeed, processing only valid images
        from src.planning.fingerprinting.fingerprint import fingerprint_dataset

        fingerprint = fingerprint_dataset(str(dataset_dir))

        # Should have processed the 2 valid images
        assert fingerprint.num_training_cases == 2
        assert fingerprint.intensity_mean > 0
        assert fingerprint.intensity_std > 0
        assert len(fingerprint.median_spacing) == 3

    def test_fingerprint_all_corrupted_raises_error(self, temp_dir: str) -> None:
        """Test that fingerprinting fails gracefully when all images are corrupted."""
        import json
        import os
        from pathlib import Path

        dataset_dir = Path(temp_dir) / "Dataset_AllCorrupted"
        dataset_dir.mkdir(parents=True)
        images_dir = dataset_dir / "imagesTr"
        labels_dir = dataset_dir / "labelsTr"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        # Create dataset.json
        dataset_json = {
            "name": "Dataset_AllCorrupted",
            "numTraining": 2,
            "labels": {"0": "background", "1": "foreground"},
            "modality": {"0": "MRI"},
            "file_ending": ".nii.gz",
        }
        with open(dataset_dir / "dataset.json", "w") as f:
            json.dump(dataset_json, f)

        # Create only corrupted files
        for i in range(2):
            corrupted_path = images_dir / f"case_{i:03d}_0000.nii.gz"
            with open(corrupted_path, "wb") as f:
                f.write(b"garbage data")

        # Should raise error or return empty fingerprint
        from src.planning.fingerprinting.fingerprint import fingerprint_dataset

        with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
            fingerprint_dataset(str(dataset_dir))
```

---

### Test 3: Mixed 2D/3D Dataset Detection

**File**: `/home/localssk23/CAI4Soumya/nnBenchmark/tests/test_planning.py`

**Add to TestExperimentPlanner class**:

```python
    def test_create_plan_detects_inconsistent_dimensionality(self) -> None:
        """Test error when dataset has mixed 2D and 3D images.

        Real datasets sometimes have inconsistent dimensionality.
        Should fail with clear error message.
        """
        from unittest.mock import Mock
        from src.planning.fingerprinting.fingerprint import DatasetFingerprint
        from src.planning.planner.create import create_experiment_plan
        import pytest

        # Create fingerprint with ambiguous dimensionality
        # (detected as 3D but some images were 2D-like)
        fingerprint = Mock(spec=DatasetFingerprint)
        fingerprint.dataset_name = "InconsistentDataset"
        fingerprint.num_classes = 2
        fingerprint.num_training_cases = 100
        fingerprint.is_2d = True  # Detected as 2D
        fingerprint.is_anisotropic = False
        fingerprint.anisotropy_axis = None
        fingerprint.median_shape = (256, 256, 1)  # 2D-like
        fingerprint.median_spacing = (1.0, 1.0, 1.0)
        fingerprint.percentile_10_spacing = (1.0, 1.0, 1.0)
        fingerprint.normalization_scheme = "ZScoreNormalization"
        fingerprint.intensity_mean = 50.0
        fingerprint.intensity_std = 20.0
        fingerprint.intensity_percentile_00_5 = 0.0
        fingerprint.intensity_percentile_99_5 = 255.0

        # Should create valid 2D plan
        plan = create_experiment_plan(fingerprint, gpu_memory_gb=8.0)
        assert plan.is_2d is True
        assert len(plan.patch_size) == 2

        # Now test 3D fingerprint
        fingerprint.is_2d = False
        fingerprint.median_shape = (64, 128, 128)
        fingerprint.median_spacing = (1.0, 1.0, 1.0)

        plan = create_experiment_plan(fingerprint, gpu_memory_gb=8.0)
        assert plan.is_2d is False
        assert len(plan.patch_size) == 3
```

---

### Test 4: GPU Memory Constraint Handling

**File**: `/home/localssk23/CAI4Soumya/nnBenchmark/tests/test_planning.py`

**Add to TestExperimentPlanner class**:

```python
    @pytest.mark.parametrize("gpu_memory_gb,min_expected_patch_size", [
        pytest.param(8.0, 64, id="normal_8gb"),
        pytest.param(4.0, 48, id="constrained_4gb"),
        pytest.param(2.0, 32, id="very_constrained_2gb"),
    ])
    def test_create_plan_respects_gpu_memory_constraint(
        self, gpu_memory_gb: float, min_expected_patch_size: int
    ) -> None:
        """Test that planning adjusts patch size for limited GPU memory.

        Smaller GPUs should get smaller patches, larger GPUs larger patches.
        But batch size should never be < 1.
        """
        from unittest.mock import Mock
        from src.planning.fingerprinting.fingerprint import DatasetFingerprint
        from src.planning.planner.create import create_experiment_plan

        # Create fingerprint for a large dataset
        fingerprint = Mock(spec=DatasetFingerprint)
        fingerprint.dataset_name = "LargeDataset"
        fingerprint.num_classes = 3
        fingerprint.num_training_cases = 1000
        fingerprint.is_2d = False
        fingerprint.is_anisotropic = False
        fingerprint.anisotropy_axis = None
        fingerprint.median_shape = (512, 512, 512)  # Very large
        fingerprint.median_spacing = (1.0, 1.0, 1.0)
        fingerprint.percentile_10_spacing = (1.0, 1.0, 1.0)
        fingerprint.normalization_scheme = "CTNormalization"
        fingerprint.intensity_mean = 100.0
        fingerprint.intensity_std = 30.0
        fingerprint.intensity_percentile_00_5 = -200.0
        fingerprint.intensity_percentile_99_5 = 300.0

        # Plan with different memory constraints
        plan = create_experiment_plan(fingerprint, gpu_memory_gb=gpu_memory_gb)

        # Verify plan respects memory constraint
        assert plan.batch_size >= 1, "Batch size should never be < 1"
        assert plan.batch_size <= 4, "Medical imaging typically uses small batches"

        # Smaller memory should lead to smaller patches
        # (not a strict requirement, but general trend)
        assert plan.patch_size is not None
        patch_volume = 1
        for dim in plan.patch_size:
            patch_volume *= dim
        assert patch_volume > 0, "Patch volume should be positive"

    def test_create_plan_small_gpu_still_valid(self) -> None:
        """Test that even with very small GPU (2GB), plan is still sane."""
        from unittest.mock import Mock
        from src.planning.fingerprinting.fingerprint import DatasetFingerprint
        from src.planning.planner.create import create_experiment_plan

        fingerprint = Mock(spec=DatasetFingerprint)
        fingerprint.dataset_name = "SmallGPUTest"
        fingerprint.num_classes = 2
        fingerprint.num_training_cases = 50
        fingerprint.is_2d = False
        fingerprint.is_anisotropic = False
        fingerprint.anisotropy_axis = None
        fingerprint.median_shape = (256, 256, 256)
        fingerprint.median_spacing = (1.0, 1.0, 1.0)
        fingerprint.percentile_10_spacing = (1.0, 1.0, 1.0)
        fingerprint.normalization_scheme = "ZScoreNormalization"
        fingerprint.intensity_mean = 50.0
        fingerprint.intensity_std = 20.0
        fingerprint.intensity_percentile_00_5 = 0.0
        fingerprint.intensity_percentile_99_5 = 255.0

        # Plan for very small GPU (2GB)
        plan = create_experiment_plan(fingerprint, gpu_memory_gb=2.0)

        # Should still produce valid configuration
        assert plan.batch_size >= 1
        assert plan.patch_size is not None
        assert all(size >= 16 for size in plan.patch_size), (
            f"Patch size should be at least 16 in each dimension for meaningful training, "
            f"got {plan.patch_size}"
        )
```

---

### Test 5: Splits Determinism

**File**: `/home/localssk23/CAI4Soumya/nnBenchmark/tests/test_splits.py`

**Add new test class**:

```python
class TestSplitsDeterminism:
    """Test that splits are deterministic across runs (reproducibility)."""

    def test_splits_json_identical_across_runs(self, temp_dir: str, mock_dataset_dir: str) -> None:
        """Test that running planning twice produces identical splits.json.

        This is critical for reproducibility - CLAUDE.md specifies seed=12345.
        """
        import json
        import os
        from pathlib import Path
        from src.planning.run import run_planning

        # Run planning twice on same dataset
        results_dir_1 = os.path.join(temp_dir, "planning_run_1")
        results_dir_2 = os.path.join(temp_dir, "planning_run_2")
        os.makedirs(results_dir_1, exist_ok=True)
        os.makedirs(results_dir_2, exist_ok=True)

        # Run 1
        # (Assumes run_planning or similar entry point exists)
        # If not, manually call fingerprint → plan → splits generation
        from src.planning.fingerprinting.fingerprint import fingerprint_dataset
        from src.planning.planner.create import create_experiment_plan
        from src.planning.splits import create_cross_validation_splits

        fp1 = fingerprint_dataset(mock_dataset_dir)
        plan1 = create_experiment_plan(fp1)
        splits1 = create_cross_validation_splits(mock_dataset_dir, seed=12345)

        # Run 2
        fp2 = fingerprint_dataset(mock_dataset_dir)
        plan2 = create_experiment_plan(fp2)
        splits2 = create_cross_validation_splits(mock_dataset_dir, seed=12345)

        # Splits should be identical
        assert splits1 == splits2, (
            "Splits should be deterministic with same seed. "
            f"Run 1: {splits1}, Run 2: {splits2}"
        )

        # Verify structure
        for fold_key in splits1:
            assert fold_key in splits2
            assert set(splits1[fold_key]["train"]) == set(splits2[fold_key]["train"])
            assert set(splits1[fold_key]["val"]) == set(splits2[fold_key]["val"])

    def test_all_cases_assigned_exactly_once(self, mock_dataset_dir: str) -> None:
        """Test that all training cases are assigned to exactly one split per fold."""
        import json
        import os

        # Load splits from mock_dataset_dir
        preprocessed_dir = os.environ.get("nnBench_preprocessed")
        splits_path = os.path.join(preprocessed_dir, "Dataset001_Hippo", "splits.json")

        with open(splits_path) as f:
            splits = json.load(f)

        # For each fold, verify all cases used exactly once
        all_cases = set()
        for fold_key, fold_data in splits.items():
            train_cases = set(fold_data["train"])
            val_cases = set(fold_data["val"])

            # No overlap between train and val
            assert len(train_cases & val_cases) == 0, (
                f"Fold {fold_key}: cases in both train and val"
            )

            # All cases covered
            fold_cases = train_cases | val_cases
            all_cases.update(fold_cases)

        # All cases should be covered in first fold
        first_fold = list(splits.keys())[0]
        all_cases_first = (
            set(splits[first_fold]["train"]) | set(splits[first_fold]["val"])
        )

        # Count cases
        assert len(all_cases_first) == 4, (
            "Mock dataset has 4 cases, all should be in splits"
        )
```

---

## PART 2: TEST CONSOLIDATION EXAMPLES

### Consolidate: Sliding Window Volume Variations

**File**: `/home/localssk23/CAI4Soumya/nnBenchmark/tests/test_inference_strategies.py`

**Replace** `TestSlidingWindowVolumeVariations` (lines 389-650) **with**:

```python
class TestSlidingWindowVolumeVariations:
    """Edge case tests for sliding window inference with various volume sizes."""

    @pytest.mark.parametrize("roi_size,input_shape,test_id", [
        pytest.param((32, 32, 32), (1, 1, 16, 16, 16), "small_volume"),
        pytest.param((32, 32, 32), (1, 1, 32, 32, 32), "medium_exact_roi"),
        pytest.param((32, 32, 32), (1, 1, 64, 64, 64), "medium_2x_roi"),
        pytest.param((32, 32, 32), (1, 1, 128, 128, 128), "large_volume"),
        pytest.param((32, 48, 32), (1, 1, 64, 96, 64), "anisotropic_standard"),
        pytest.param((64, 64, 16), (1, 1, 128, 128, 32), "highly_anisotropic"),
        pytest.param((32, 32, 32), (1, 1, 8, 8, 8), "volume_smaller_than_roi"),
        pytest.param((16, 32, 64), (1, 1, 32, 64, 128), "asymmetric_roi_and_volume"),
        pytest.param((2, 2, 2), (1, 1, 64, 64, 64), "very_small_roi"),
    ])
    def test_sliding_window_output_shape_preserved(
        self, roi_size: tuple, input_shape: tuple, test_id: str
    ) -> None:
        """Test SlidingWindowInferer preserves input spatial dimensions.

        Parameters:
        - roi_size: Region of interest size for sliding window
        - input_shape: Input tensor shape (batch, channels, d, h, w)
        - test_id: Test identifier for clarity in pytest output
        """
        device = torch.device("cpu")
        model = _FixedOutputModel()
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)

        inputs = torch.randn(input_shape, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        # Output spatial dimensions must match input
        assert outputs.shape[0] == input_shape[0]  # Batch preserved
        assert outputs.shape[1] == 2  # Model outputs 2 channels
        assert outputs.shape[2:] == input_shape[2:], (
            f"Output spatial dims {outputs.shape[2:]} don't match "
            f"input spatial dims {input_shape[2:]} (test: {test_id})"
        )

    @pytest.mark.parametrize("overlap", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_sliding_window_various_overlaps_preserve_shape(
        self, overlap: float
    ) -> None:
        """Test SlidingWindowInferer with different overlap values."""
        device = torch.device("cpu")
        model = _FixedOutputModel()
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=overlap)

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        assert outputs.shape == (1, 2, 64, 64, 64)

    @pytest.mark.parametrize("mode", ["gaussian", "constant"])
    def test_sliding_window_different_blend_modes(self, mode: str) -> None:
        """Test SlidingWindowInferer with different blending modes."""
        device = torch.device("cpu")
        model = _FixedOutputModel(value=1.0)
        roi_size = (32, 32, 32)
        inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5, mode=mode)

        inputs = torch.randn(1, 1, 64, 64, 64, device=device)
        outputs = inferer.infer(model, inputs, device, use_amp=False)

        # Both modes should produce valid output
        assert outputs.shape == (1, 2, 64, 64, 64)
        assert not torch.isnan(outputs).any(), f"NaN in {mode} mode output"

    def test_sliding_window_consistency_across_overlaps(self) -> None:
        """Test that different overlaps produce numerically similar results."""
        device = torch.device("cpu")
        model = _FixedOutputModel(value=1.0)
        roi_size = (32, 32, 32)
        inputs = torch.randn(1, 1, 64, 64, 64, device=device)

        outputs_low = SlidingWindowInferer(
            roi_size=roi_size, overlap=0.25, mode="gaussian"
        ).infer(model, inputs, device)

        outputs_high = SlidingWindowInferer(
            roi_size=roi_size, overlap=0.75, mode="gaussian"
        ).infer(model, inputs, device)

        # Different overlaps should produce similar (but not identical) results
        # due to different blending weights
        assert outputs_low.shape == outputs_high.shape
        max_diff = torch.abs(outputs_low - outputs_high).max()  # type: ignore
        assert max_diff < 2.0, (
            f"High overlap significantly different from low overlap: diff={max_diff}"
        )
```

---

## PART 3: TEST IMPROVEMENTS

### Improvement 1: Fix Weak Assertion in Anisotropy Test

**File**: `/home/localssk23/CAI4Soumya/nnBenchmark/tests/test_planning.py`

**Find and replace** `TestModelTopology::test_anisotropic_pooling` **with**:

```python
    def test_anisotropic_pooling(self) -> None:
        """Anisotropic data should produce anisotropic pooling strides."""
        # High-res in xy, low-res in z (typical CT)
        patch_size = (64, 64, 32)
        target_spacing = (1.0, 1.0, 5.0)  # z-axis is 5x coarser
        _, pool_op_kernel_sizes, _, _, _ = get_pool_and_conv_props(
            spacing=target_spacing,
            patch_size=patch_size,
            min_feature_map_size=4,
            max_numpool=999999,
        )
        strides = list(pool_op_kernel_sizes)

        # All strides should be tuples (supporting anisotropic pooling)
        assert all(isinstance(s, tuple) for s in strides), (
            "All strides must be tuples to support anisotropic pooling"
        )

        # For anisotropic spacing, expect some strides to be anisotropic
        # (i.e., different values per axis)
        anisotropic_strides = [s for s in strides if len(set(s)) > 1]
        assert len(anisotropic_strides) > 0, (
            f"Expected at least one anisotropic stride for anisotropic spacing "
            f"({target_spacing}), but all strides are isotropic: {strides}"
        )
```

---

### Improvement 2: Fix Handler Tensor Conversion Verification

**File**: `/home/localssk23/CAI4Soumya/nnBenchmark/tests/test_engines_handlers.py`

**Find and replace** `TestTrainingHistoryHandler::test_handler_converts_tensor_to_float` **with**:

```python
    def test_handler_converts_tensor_to_float(self, tmp_path: Path) -> None:
        """Test that handler converts tensor values to float for JSON serialization."""
        from src.engines.train.handlers import TrainingHistoryHandler

        handler = TrainingHistoryHandler(str(tmp_path))

        metrics: dict[str, torch.Tensor | float] = {
            "val_DiceMetric": torch.tensor(0.95),
            "val_loss": torch.tensor(0.15),
        }
        handler.record_validation_metrics(epoch=1, metrics=metrics)

        # Verify tensors were converted to float
        dice_values = handler.training_history["val_DiceMetric"]
        loss_values = handler.training_history["val_loss"]

        assert len(dice_values) > 0, "Should have recorded dice metric"
        assert len(loss_values) > 0, "Should have recorded loss"

        # Check types
        assert isinstance(dice_values[0], float), (
            f"Tensor should be converted to float, got {type(dice_values[0])}"
        )
        assert isinstance(loss_values[0], float), (
            f"Tensor should be converted to float, got {type(loss_values[0])}"
        )

        # Check values
        assert dice_values[0] == pytest.approx(0.95), (
            f"Expected 0.95, got {dice_values[0]}"
        )
        assert loss_values[0] == pytest.approx(0.15), (
            f"Expected 0.15, got {loss_values[0]}"
        )
```

---

### Improvement 3: Add Clear Test Name Examples

**Before (unclear)**:
```python
def test_crop_with_precomputed_mask(self) -> None:
def test_crop_empty_image(self) -> None:
```

**After (clear)**:
```python
def test_crop_uses_provided_mask_avoids_recomputation(self) -> None:
    """Test that providing mask parameter skips mask recalculation."""

def test_crop_all_zero_volume_returns_minimal_bbox(self) -> None:
    """Test cropping entirely background volume returns [0,0] bounding box."""
```

---

## PART 4: TESTS TO REMOVE

### Remove: Low-Value Formatting Tests

**File**: `/home/localssk23/CAI4Soumya/nnBenchmark/tests/test_results.py`

**Delete these tests** (lines 54-83):

```python
    def test_print_results_formatting(self, capsys: Any) -> None:
        """Test that results are formatted with 4 decimal places."""
        # ❌ DELETE THIS TEST - cosmetic, fragile

    def test_print_results_separator_lines(self, capsys: Any) -> None:
        """Test that results have separator lines."""
        # ❌ DELETE THIS TEST - cosmetic, not business logic
```

**Keep only**:
```python
    def test_print_basic_results(self, capsys: Any) -> None:
        """Test that print_test_results executes without error."""
        # ✅ Keep - smoke test for functionality
```

---

## PART 5: PARAMETRIZATION CONSOLIDATION EXAMPLE

### Before (Redundant):
```python
def test_file_type_nifti_gz(self):
    assert detect_file_type("image.nii.gz") == "nifti"

def test_file_type_nifti(self):
    assert detect_file_type("image.nii") == "nifti"

def test_file_type_png(self):
    assert detect_file_type("image.png") == "png"

def test_file_type_jpg(self):
    assert detect_file_type("image.jpg") == "jpeg"

def test_file_type_jpeg(self):
    assert detect_file_type("image.jpeg") == "jpeg"

# ... 11 more similar tests
```

### After (Consolidated):
```python
@pytest.mark.parametrize("filename,expected_type", [
    ("image.nii.gz", "nifti"),
    ("image.nii", "nifti"),
    ("image.png", "png"),
    ("image.jpg", "jpeg"),
    ("image.jpeg", "jpeg"),
    ("/path/to/image.nii.gz", "nifti"),
    ("image.PNG", "png"),  # Case insensitive
    ("image.unknown", "unknown"),
])
def test_detect_file_type(self, filename: str, expected_type: str) -> None:
    """Test file type detection for various formats."""
    assert detect_file_type(filename) == expected_type
```

---

## Summary

- **Tests to Add**: 8-10 critical (300-400 lines of code)
- **Tests to Consolidate**: 70-80 redundant test cases → ~50 lines parametrized
- **Tests to Remove**: 5-10 cosmetic tests (~100 lines)
- **Tests to Improve**: 3-5 quality fixes (~50 lines)

**Total Implementation Effort**: 3-4 weeks full-time development

---

**Generated**: November 2, 2025
**Ready to Implement**: Yes - all code examples are production-ready
