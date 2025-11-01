# Planning

Automatic configuration generation from dataset analysis.

## What is It?

A workflow that analyzes your dataset and automatically generates a complete training configuration, including model architecture, hyperparameters, and data preprocessing parameters.

## Why Use It?

1. **Reproducibility** - Deterministic configuration generation from dataset properties
2. **Hardware optimization** - Automatic GPU memory, CPU cores, and caching strategy detection
3. **Dataset-adaptive** - Architecture and preprocessing parameters derived from actual data statistics
4. **No manual tuning** - Eliminates guesswork in choosing patch sizes, batch sizes, and network topology

## The 5-Step Workflow

### Step 0: Preprocessing
Crops images to nonzero regions using binary foreground masks and bounding boxes. Reduces dataset size by 25-50% for brain MRI, minimal reduction for organ CT. Saves preprocessed images to `preprocessed/<dataset_name>/imagesTr/` and `preprocessed/<dataset_name>/labelsTr/`.

**Factors considered**:
- **[Adaptive]** Binary mask creation from non-zero voxels
  - Logical OR across all channels to detect foreground
  - Channel dimension detection (3D: C,H,W,D or 2D: C,H,W)
  - Source: `src/preprocessing/cropping.py::create_nonzero_mask()`
- **[Constant]** Morphological hole-filling algorithm
  - Uses scipy `binary_fill_holes` for continuous mask
  - Fills small holes in foreground region
  - Source: `src/preprocessing/cropping.py::create_nonzero_mask()`
- **[Adaptive]** Bounding box calculation per case
  - Min/max indices per dimension from binary mask
  - Returns exclusive upper bound (numpy slicing format)
  - Source: `src/preprocessing/cropping.py::get_bbox_from_mask()`
- **[Adaptive]** Cropping both images and labels to same bbox
  - Handles multi-channel and single-channel formats
  - Validates shape matching between image and segmentation
  - Source: `src/preprocessing/cropping.py::crop_to_nonzero()`
- **[Adaptive]** Metadata preservation
  - Original shape, cropped shape, voxel spacing
  - Affine transforms for NIfTI files
  - Source: `src/planning/fingerprinting/prepare_dataset.py::preprocess_and_crop_dataset()`

### Step 1: Fingerprinting
Parallel analysis of preprocessed images to extract statistical properties.

**Factors computed**:
- **[Adaptive]** Shape statistics across all cases
  - Median, 10th/90th percentiles per dimension
  - Channel-first format handling (C,H,W,D or C,H,W)
  - Source: `src/planning/fingerprinting/fingerprint.py::fingerprint_dataset()`
- **[Adaptive]** Spacing statistics across all cases
  - Median, 10th/90th percentiles per axis
  - Extracted from NIfTI metadata or defaulted to [1.0, 1.0] for PNG/JPG
  - Source: `src/planning/fingerprinting/fingerprint.py::fingerprint_dataset()`
- **[Adaptive]** Intensity statistics from foreground voxels only
  - Mean, std, 0.5th/99.5th percentiles pooled across all cases
  - Uses segmentation masks to extract foreground regions
  - Source: `src/planning/fingerprinting/fingerprint.py::fingerprint_dataset()`
- **[Constant]** Foreground voxel sampling strategy
  - Fixed 10,000 samples per case for memory efficiency
  - Random sampling with fixed seed (12345) for reproducibility
  - Source: `src/planning/fingerprinting/loading.py::load_image_properties()`
- **[Adaptive]** Dimensionality detection (2D vs 3D)
  - Based on number of spatial dimensions (excludes channel dim)
  - 2D: shape [C,H,W], 3D: shape [C,H,W,D]
  - Source: `src/planning/fingerprinting/fingerprint.py::fingerprint_dataset()`
- **[Constant + Adaptive]** Anisotropy detection
  - Constant thresholds: spacing ratio >3× AND voxel count <0.25 (25%)
  - Adaptive: compares worst axis spacing to better axes
  - Source: `src/planning/fingerprinting/spacing.py::detect_anisotropy()`
- **[Constant]** Normalization scheme determination
  - CT channel → CTNormalization (percentile clipping)
  - Other channels → ZScoreNormalization (per-case)
  - Source: `src/planning/fingerprinting/metadata.py::determine_normalization_scheme()`
- **[Adaptive]** Unique class value scanning from labels
  - Samples up to 50 label files to find unique values
  - num_classes = max_label_value + 1
  - Source: `src/planning/fingerprinting/metadata.py::scan_unique_label_values()`

### Step 2: Experiment Planning
Calculates all model architecture and training parameters from fingerprint using 8-step process.

**Factors calculated**:
- **[Constant + Adaptive]** Target spacing calculation
  - Adaptive: Uses median spacing by default
  - Adaptive: For anisotropic axis, uses 10th percentile spacing
  - Constant: Anisotropy detection threshold >3× (aniso_threshold=3.0)
  - Source: `src/planning/planner/heuristics.py::calculate_target_spacing()`
- **[Constant + Adaptive]** Initial patch size normalization
  - Constant: 3D formula uses 256³ normalization constant
  - Constant: 2D formula uses 2048² normalization constant
  - Adaptive: Scaled by target spacing and clipped to median shape
  - Source: `src/planning/planner/sizing.py::calculate_initial_patch_size()`
- **[Constant + Adaptive]** Network topology determination
  - Constant: min_feature_map_size = 4 (bottleneck constraint)
  - Constant: Spacing ratio threshold <2× for pooling eligibility
  - Constant: Pooling kernel sizes = 2 or 1 per axis
  - Adaptive: num_pool_per_axis based on spacing ratios and patch size
  - Adaptive: strides per axis determined iteratively
  - Constant: conv_kernel_sizes = 3×3×3 after axis becomes eligible
  - Source: `src/planning/planner/topology.py::get_pool_and_conv_props()`
- **[Constant + Adaptive]** Feature channels per stage
  - Constant: base_features = 32 (UNet_base_num_features)
  - Constant: Doubling per level (2^i multiplier)
  - Constant: max_features = 512 for 2D, 320 for 3D
  - Adaptive: Number of stages from topology algorithm
  - Source: `src/planning/planner/heuristics.py::calculate_feature_channels()`
- **[Adaptive]** Upsample kernel sizes
  - Inverse of encoder strides (skips first identity stride)
  - Symmetric decoder architecture
  - Source: `src/planning/planner/create.py::create_experiment_plan()`
- **[Constant]** Deep supervision weights
  - Exponential decay formula: 2^(-i) for output i
  - deep_supr_num = 1 (DynUNet default)
  - Source: `src/planning/planner/heuristics.py::calculate_deep_supervision_weights()`
- **[Adaptive]** Dataset voxel count estimation
  - num_cases × product(median_shape)
  - Used for batch size 5% coverage cap
  - Source: `src/planning/planner/create.py::create_experiment_plan()`
- **[Constant + Adaptive]** Batch size calculation
  - Constant: UNet_reference_val = 560M for 3D, 85M for 2D
  - Constant: reference_batch_size = 2 for 3D, 12 for 2D
  - Constant: reference_GPU = 8 GB
  - Constant: max_dataset_covered = 5%
  - Constant: min_batch_size = 2
  - Adaptive: Scaled by GPU memory and patch complexity
  - Source: `src/planning/planner/sizing.py::calculate_batch_size()`
- **[Constant + Adaptive]** Intensity normalization ranges
  - Constant: CT uses 0.5th/99.5th percentile clipping
  - Constant: MRI/PET uses per-case z-score (no clipping)
  - Adaptive: Percentile values computed from dataset
  - Source: `src/planning/planner/create.py::create_experiment_plan()`

### Step 3: YAML Generation
Writes complete training configuration by converting ExperimentPlan to YAML with all components and resource optimization.

**Parameters written**:
- **[Adaptive]** DynUNet architecture parameters
  - spatial_dims (2D or 3D from fingerprint)
  - Constant: in_channels=1 (single channel)
  - out_channels (num_classes from fingerprint)
  - filters, strides, kernel_size, upsample_kernel_size (from Step 2)
  - Constant: deep_supervision=true, deep_supr_num=1
  - ds_weights (from Step 2)
  - Constant: norm_name=INSTANCE with affine=true
  - Constant: act_name=leakyrelu with negative_slope=0.01
  - Constant: res_block=false
  - Source: `src/planning/yaml_generator.py::_write_model_config()`
- **[Adaptive]** UNet nested config (auto-derived)
  - channels (matches DynUNet filters)
  - strides (simplified: skip identity stride, take first element)
  - Constant: num_res_units=2
  - Source: `src/planning/yaml_generator.py::_derive_unet_params_from_dynunet()`
- **[Constant]** Training hyperparameters
  - epochs=200, learning_rate=0.01, val_interval=5
  - checkpoint_metric=DiceMetric, mixed_precision=true
  - Source: `src/planning/yaml_generator.py::_write_training_config()`
- **[Adaptive]** Batch size from Step 2
  - Source: `src/planning/yaml_generator.py::_write_training_config()`
- **[Constant]** Optimizer settings (SGD)
  - weight_decay=0.00003, momentum=0.99, nesterov=true
  - Source: `src/planning/yaml_generator.py::_write_optimizer_config()`
- **[Constant]** Loss configuration (DiceCELoss)
  - to_onehot_y=true, softmax=true, batch=true
  - Source: `src/planning/yaml_generator.py::_write_loss_config()`
- **[Constant]** Metrics configuration
  - DiceMetric (include_background=false, reduction=mean_batch)
  - SurfaceDiceMetric (tolerance thresholds for surface evaluation)
  - Source: `src/planning/yaml_generator.py::_write_metrics_config()`
- **[Constant]** Transforms pipeline (full augmentation)
  - RandCropByPosNegLabeld, RandAffined, RandGaussianNoised, RandGaussianSmoothd
  - RandScaleIntensityd, RandShiftIntensityd, RandFlipd, spacing resampling
  - Source: `src/planning/yaml_generator.py::_write_transforms_config()`
- **[Adaptive]** System resources optimization
  - num_workers from CPU detection (conservative/balanced/aggressive strategies)
  - cache_enabled based on dataset_size_mb vs available_ram_gb
  - cache_rate calculated as fraction (0.0-1.0), persistent_workers flag
  - Source: `src/planning/fingerprinting/resources.py::calculate_optimal_workers()`, `::calculate_cache_strategy()`

### Step 4: Cross-Validation Splits
Generates k-fold cross-validation splits with deterministic seeding for reproducibility.

**Parameters configured**:
- **[Adaptive]** Case identifier extraction
  - Scans imagesTr directory for all *_0000.* files
  - Extracts case IDs by removing channel suffix
  - Validates against dataset.json training entries
  - Source: `src/planning/splits.py::extract_case_identifiers()`
- **[Constant]** K-fold splitting configuration
  - n_folds=5 (5-fold cross-validation)
  - Source: `src/planning/splits.py::create_splits()`
- **[Constant]** Random seed for reproducibility
  - seed=12345 (fixed seed ensures deterministic splits)
  - Source: `src/planning/splits.py::create_splits()`
- **[Adaptive]** Stratification option
  - stratified=False by default
  - Can be enabled for balanced class distribution per fold
  - Uses sklearn StratifiedKFold when enabled
  - Source: `src/planning/splits.py::create_splits()`, `::get_labels_for_stratification()`
- **[Adaptive]** Train/validation case assignment
  - Each fold: 80% train, 20% validation
  - Case IDs assigned to 'train' and 'val' lists per fold
  - Source: `src/planning/splits.py::create_splits()`
- **[Adaptive]** Output location
  - Saved to preprocessed/<dataset_name>/splits.json
  - JSON format with fold indices and case assignments
  - Source: `src/planning/run.py::run_planning()`

## Usage

```bash
nnBench.plan --dataset /path/to/Dataset001_Name --gpu-memory 24
```

**Required**:
- `--dataset`: Path to dataset directory containing `imagesTr/` and `labelsTr/`

**Optional**:
- `--gpu-memory`: GPU VRAM in GB (default: auto-detect)
- `--fold`: Specific fold number (default: all folds)
- `--output`: Config output path (default: `configs/dataset_name.yaml`)
- `--num-workers`: Parallel workers for fingerprinting (default: auto-detect)

## Outputs

Generated files:
- `preprocessed/<dataset_name>/imagesTr/` - Cropped training images
- `preprocessed/<dataset_name>/labelsTr/` - Cropped training labels
- `<dataset_dir>/dataset.json` - Dataset metadata and channel information
- `preprocessed/<dataset_name>/fingerprint.json` - Statistical properties from Step 1
- `preprocessed/<dataset_name>/splits.json` - Cross-validation fold assignments
- `results/<dataset_name>/fold_0/fold_0.yaml` - Complete training configuration

## Implementation

- **Entry point**: `src/planning/cli.py` - Command-line interface
- **Orchestrator**: `src/planning/run.py` - 5-step workflow coordinator
- **Fingerprinting**: `src/planning/fingerprinting/` - Dataset analysis
- **Planning**: `src/planning/planner/` - Architecture calculation heuristics
- **Config generation**: `src/planning/yaml_generator.py` - YAML writer
- **Splits**: `src/planning/splits.py` - Cross-validation generation
