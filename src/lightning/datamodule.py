"""
LightningDataModule for segmentation datasets.
Wraps existing data utilities (get_data_dicts, build_transforms) with Lightning interface.
"""

from __future__ import annotations

from monai.data.dataset import CacheDataset, Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.utils.builders import build_transforms
from src.utils.data import get_data_dicts


class SegmentationDataModule(LightningDataModule):
    """
    Lightning DataModule for medical image segmentation.

    Handles data loading, transforms, and DataLoader creation for train/val splits.
    Automatically handles distributed sampling when using DDP.
    """

    def __init__(self, cfg: dict, data_dir: str, fold: int):
        """
        Initialize DataModule.

        Args:
            cfg: Configuration dictionary
            data_dir: Path to dataset directory
            fold: Fold number for cross-validation
        """
        super().__init__()
        self.cfg = cfg
        self.data_dir = data_dir
        self.fold = fold

        # Datasets (initialized in setup)
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None

        # Track whether we're using CacheDataset (affects persistent_workers)
        self._using_cache = False

    def _should_use_persistent_workers(self) -> bool:
        """
        Determine if persistent_workers should be enabled.

        CacheDataset requires persistent_workers=False.
        Otherwise, enable persistent_workers only if num_workers > 0.

        Returns:
            True if persistent_workers should be enabled, False otherwise
        """
        if self._using_cache:
            return False
        return self.cfg["training"]["num_workers"] > 0

    def setup(self, stage: str | None = None) -> None:
        """
        Setup datasets and transforms.
        Called automatically by Lightning before training/validation.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Get data dicts using existing utility
        train_data, val_data = get_data_dicts(self.data_dir, self.fold)

        # Build transforms using existing builders
        train_transforms = build_transforms(self.cfg, mode="train")
        val_transforms = build_transforms(self.cfg, mode="val")

        # Check if caching is enabled in config
        cache_config = self.cfg.get("dataset", {}).get("cache", {})
        use_cache = cache_config.get("enabled", False)
        cache_rate = cache_config.get("cache_rate", 1.0)

        if use_cache:
            # Use CacheDataset for faster training
            # Caches deterministic transforms (LoadImaged, ScaleIntensityRanged, etc.)
            # Random transforms (RandCropd, RandFlipd) still executed every iteration
            self.train_ds = CacheDataset(
                data=train_data,
                transform=train_transforms,
                cache_rate=cache_rate,
                num_workers=self.cfg["training"]["num_workers"],
            )
            # Only create validation dataset if there's validation data
            if val_data:
                self.val_ds = CacheDataset(
                    data=val_data,
                    transform=val_transforms,
                    cache_rate=cache_rate,
                    num_workers=self.cfg["training"]["num_workers"],
                )
            else:
                self.val_ds = None
            self._using_cache = True
        else:
            # Use basic Dataset (no caching)
            self.train_ds = Dataset(data=train_data, transform=train_transforms)
            # Only create validation dataset if there's validation data
            self.val_ds = (
                Dataset(data=val_data, transform=val_transforms) if val_data else None
            )
            self._using_cache = False

    def train_dataloader(self) -> DataLoader:
        """
        Create training DataLoader.
        Lightning handles distributed sampling automatically.

        Returns:
            DataLoader for training
        """
        if self.train_ds is None:
            raise RuntimeError(
                "train_dataloader called before setup initialized train dataset"
            )

        return DataLoader(
            self.train_ds,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=self.cfg["training"]["num_workers"],
            persistent_workers=self._should_use_persistent_workers(),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        """
        Create validation DataLoader.

        Returns validation DataLoader if validation data exists, None otherwise.
        This allows training without validation (e.g., when fold=-1).
        """
        if self.val_ds is None:
            return None

        return DataLoader(
            self.val_ds,
            batch_size=1,  # Validation uses batch_size=1 for full volumes
            num_workers=self.cfg["training"]["num_workers"],
            persistent_workers=self._should_use_persistent_workers(),
            pin_memory=True,
        )
