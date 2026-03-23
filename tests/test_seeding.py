"""
Tests for centralised seeding functionality.

Verifies that:
1. set_random_seeds() sets seeds correctly for all libraries
2. get_seed_from_config() extracts seeds with correct priority
3. enable_cuda_determinism() controls CUDA determinism
4. Reproducibility is achieved with the same seed
5. Different seeds produce different results
"""


from typing import Any

import numpy as np
import torch

from src.utils.seeding import (
    enable_cuda_determinism,
    get_seed_from_config,
    set_random_seeds,
)


class TestSetRandomSeeds:
    """Test set_random_seeds() function."""

    def test_set_seeds_with_same_seed_produces_same_results(self) -> None:
        """Verify that same seed produces identical random sequences."""
        seed = 42

        # First sequence
        set_random_seeds(seed)
        torch_values_1 = [torch.randn(3).tolist() for _ in range(3)]
        np_values_1 = [np.random.randn(3).tolist() for _ in range(3)]

        # Reset and generate again
        set_random_seeds(seed)
        torch_values_2 = [torch.randn(3).tolist() for _ in range(3)]
        np_values_2 = [np.random.randn(3).tolist() for _ in range(3)]

        # Should be identical
        for v1, v2 in zip(torch_values_1, torch_values_2):
            for a, b in zip(v1, v2):
                assert abs(a - b) < 1e-6

        for v1, v2 in zip(np_values_1, np_values_2):
            for a, b in zip(v1, v2):
                assert abs(a - b) < 1e-10

    def test_set_seeds_with_different_seed_produces_different_results(self) -> None:
        """Verify that different seeds produce different random sequences."""
        seed1, seed2 = 42, 123

        # First sequence
        set_random_seeds(seed1)
        torch_values_1 = torch.randn(10)
        np_values_1 = np.random.randn(10)

        # Different seed sequence
        set_random_seeds(seed2)
        torch_values_2 = torch.randn(10)
        np_values_2 = np.random.randn(10)

        # Should be different
        assert not torch.allclose(torch_values_1, torch_values_2)
        assert not np.allclose(np_values_1, np_values_2)

    def test_set_seeds_torch_cuda_available(self) -> None:
        """Test that set_random_seeds handles CUDA gracefully."""
        seed = 42
        # Should not raise an error even if CUDA is available or not
        set_random_seeds(seed)
        assert torch.initial_seed() == seed

    def test_reproducibility_across_modules(self) -> None:
        """Verify reproducibility with sequences from different modules."""
        seed = 42

        set_random_seeds(seed)
        torch_val = torch.randn(1).item()
        np_val = np.random.randn()

        set_random_seeds(seed)
        torch_val_2 = torch.randn(1).item()
        np_val_2 = np.random.randn()

        assert abs(torch_val - torch_val_2) < 1e-6
        assert abs(np_val - np_val_2) < 1e-10


class TestGetSeedFromConfig:
    """Test get_seed_from_config() function."""

    def test_get_seed_from_top_level(self) -> None:
        """Test extracting seed from top-level config key."""
        cfg = {"seed": 123}
        assert get_seed_from_config(cfg) == 123

    def test_get_seed_from_training_section(self) -> None:
        """Test extracting seed from training section."""
        cfg = {"training": {"seed": 456}}
        assert get_seed_from_config(cfg) == 456

    def test_get_seed_from_inference_section(self) -> None:
        """Test extracting seed from inference section."""
        cfg = {"inference": {"seed": 789}}
        assert get_seed_from_config(cfg) == 789

    def test_priority_top_level_over_training(self) -> None:
        """Test that top-level seed has priority over training."""
        cfg = {"seed": 100, "training": {"seed": 200}}
        assert get_seed_from_config(cfg) == 100

    def test_priority_training_over_inference(self) -> None:
        """Test that training seed has priority over inference."""
        cfg = {"training": {"seed": 200}, "inference": {"seed": 300}}
        assert get_seed_from_config(cfg) == 200

    def test_priority_order(self) -> None:
        """Test the complete priority order: top-level > training > inference > default."""
        # Priority 1: Top-level
        assert (
            get_seed_from_config(
                {"seed": 1, "training": {"seed": 2}, "inference": {"seed": 3}}
            )
            == 1
        )

        # Priority 2: Training (without top-level)
        assert (
            get_seed_from_config({"training": {"seed": 2}, "inference": {"seed": 3}})
            == 2
        )

        # Priority 3: Inference (without top-level or training)
        assert get_seed_from_config({"inference": {"seed": 3}}) == 3

        # Priority 4: Default (no seed in config)
        assert get_seed_from_config({}) == 12345

    def test_default_seed_value(self) -> None:
        """Test that default seed is 12345."""
        cfg = {}
        assert get_seed_from_config(cfg) == 12345

    def test_handles_empty_config_sections(self) -> None:
        """Test handling of empty config sections."""
        cfg = {"training": {}, "inference": {}}
        assert get_seed_from_config(cfg) == 12345

    def test_handles_none_values(self) -> None:
        """Test handling of None values in config."""
        cfg = {"seed": None, "training": {"seed": None}, "inference": {"seed": None}}
        assert get_seed_from_config(cfg) == 12345


class TestEnableCudaDeterminism:
    """Test enable_cuda_determinism() function."""

    def test_enable_cuda_determinism_true(self) -> None:
        """Test enabling CUDA determinism."""
        enable_cuda_determinism(deterministic=True)
        if torch.cuda.is_available():
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False

    def test_enable_cuda_determinism_false(self) -> None:
        """Test disabling CUDA determinism (allows cudnn auto-tuning)."""
        enable_cuda_determinism(deterministic=False)
        if torch.cuda.is_available():
            assert torch.backends.cudnn.deterministic is False
            assert torch.backends.cudnn.benchmark is True

    def test_enable_cuda_determinism_default(self) -> None:
        """Test that default is False (for better performance)."""
        enable_cuda_determinism()
        if torch.cuda.is_available():
            assert torch.backends.cudnn.deterministic is False
            assert torch.backends.cudnn.benchmark is True


class TestIntegration:
    """Integration tests for seeding workflow."""

    def test_full_seeding_workflow(self) -> None:
        """Test complete seeding workflow as used in train/test."""
        cfg: dict[str, Any] = {
            "seed": 42,
            "training": {"epochs": 2},
            "dataset": {"fold": 0},
        }

        # Extract seed and set it
        seed = get_seed_from_config(cfg)
        set_random_seeds(seed)
        enable_cuda_determinism(deterministic=True)

        # Generate some random values
        torch_val = torch.randn(5)
        np_val = np.random.randn(5)

        # Reset with same config
        seed = get_seed_from_config(cfg)
        set_random_seeds(seed)
        enable_cuda_determinism(deterministic=True)

        # Should get identical results
        torch_val_2 = torch.randn(5)
        np_val_2 = np.random.randn(5)

        assert torch.allclose(torch_val, torch_val_2)
        assert np.allclose(np_val, np_val_2)

    def test_seeding_with_splits_reproducibility(self) -> None:
        """Test that seeding ensures reproducible splits."""
        from sklearn.model_selection import KFold

        data = list(range(10))
        seed = 42

        # Generate splits with seed
        set_random_seeds(seed)
        kfold1 = KFold(n_splits=3, shuffle=True, random_state=seed)
        splits1 = list(kfold1.split(data))

        # Generate splits again with same seed
        set_random_seeds(seed)
        kfold2 = KFold(n_splits=3, shuffle=True, random_state=seed)
        splits2 = list(kfold2.split(data))

        # Splits should be identical
        assert len(splits1) == len(splits2)
        for (train1, val1), (train2, val2) in zip(splits1, splits2):
            assert np.array_equal(train1, train2)
            assert np.array_equal(val1, val2)
