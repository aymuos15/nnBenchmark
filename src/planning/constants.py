"""
Central registry of all planning constants.

This module provides a single source of truth for all constants used throughout
the planning workflow. Constants are organized by category and can be referenced
by both code and documentation verification tests.

Usage:
    from src.planning.constants import PLANNING_CONSTANTS

    # Use in code
    num_samples = PLANNING_CONSTANTS.FOREGROUND_SAMPLES_PER_CASE
    seed = PLANNING_CONSTANTS.RANDOM_SEED
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningConstants:
    """
    All constants used in the planning workflow.

    Categories:
    - Fingerprinting: Dataset analysis constants
    - Patch Size: Normalization constants for patch size calculation
    - Model Topology: Constants for architecture determination
    - Feature Channels: Base values and caps for channel counts
    - Batch Size: Reference values for batch size calculation
    - Training: Hyperparameters for training configuration
    - Splits: Cross-validation constants
    """

    # ========================================================================
    # Fingerprinting
    # ========================================================================
    FOREGROUND_SAMPLES_PER_CASE: int = 10000
    RANDOM_SEED: int = 12345
    ANISOTROPY_THRESHOLD: float = 3.0  # Threshold for anisotropy detection
    ANISOTROPY_VOXEL_RATIO: float = 0.25  # Minimum voxel ratio for anisotropy

    # ========================================================================
    # Patch Size Normalization
    # ========================================================================
    PATCH_NORM_3D: int = 256  # 256³ normalization constant
    PATCH_NORM_2D: int = 2048  # 2048² normalization constant

    # ========================================================================
    # Model Topology
    # ========================================================================
    MIN_FEATURE_MAP_SIZE: int = 4  # Bottleneck constraint

    # ========================================================================
    # Feature Channels
    # ========================================================================
    BASE_FEATURES: int = 32  # Starting number of features
    MAX_FEATURES_2D: int = 512  # Cap for 2D architectures
    MAX_FEATURES_3D: int = 320  # Cap for 3D architectures

    # ========================================================================
    # Batch Size Calculation
    # ========================================================================
    UNET_REFERENCE_VAL_3D: int = 560_000_000  # Reference complexity 3D
    UNET_REFERENCE_VAL_2D: int = 85_000_000  # Reference complexity 2D
    UNET_REFERENCE_CORRESP_GB: int = 8  # Reference GPU memory
    UNET_REFERENCE_CORRESP_BS_3D: int = 2  # Reference batch size 3D
    UNET_REFERENCE_CORRESP_BS_2D: int = 12  # Reference batch size 2D
    MAX_DATASET_COVERED: float = 0.05  # Max 5% dataset per batch
    UNET_MIN_BATCH_SIZE: int = 2  # Minimum allowed batch size

    # ========================================================================
    # Training Hyperparameters
    # ========================================================================
    EPOCHS: int = 1000
    NUM_ITERATIONS_PER_EPOCH: int = 250  # nnU-Net default
    LEARNING_RATE: float = 0.01
    VAL_INTERVAL: int = 1
    WEIGHT_DECAY: float = 0.00003
    MOMENTUM: float = 0.99
    NESTEROV: bool = True

    # ========================================================================
    # Cross-Validation Splits
    # ========================================================================
    N_FOLDS: int = 5  # 5-fold cross-validation


# Singleton instance
PLANNING_CONSTANTS = PlanningConstants()
