"""Tests for tensor cache preprocessing and loading."""

import torch
from pathlib import Path

from src.transforms.tensor_loading import LoadPreprocessedTensord


def test_save_and_load_tensor_cache(tmp_path: Path):
    """Test that .pt files can be saved and loaded correctly."""
    pt_path = tmp_path / "test_case.pt"

    image = torch.randn(1, 40, 56, 40)
    label = torch.randint(0, 3, (1, 40, 56, 40))

    torch.save({"image": image, "label": label}, pt_path)

    # Load with the same settings as LoadPreprocessedTensord
    cached = torch.load(pt_path, weights_only=False, mmap=True)
    assert torch.equal(cached["image"], image)
    assert torch.equal(cached["label"], label)


def test_load_preprocessed_tensord(tmp_path: Path):
    """Test LoadPreprocessedTensord transform loads .pt and populates keys."""
    pt_path = tmp_path / "case_001.pt"

    image = torch.randn(1, 40, 56, 40)
    label = torch.randint(0, 3, (1, 40, 56, 40))
    torch.save({"image": image, "label": label}, pt_path)

    transform = LoadPreprocessedTensord(keys=("image", "label"))
    data = {"tensor_cache": str(pt_path), "extra_key": "preserved"}

    result = transform(data)

    assert torch.equal(result["image"], image)
    assert torch.equal(result["label"], label)
    assert result["extra_key"] == "preserved"


def test_load_preserves_dtype(tmp_path: Path):
    """Test that loaded tensors preserve their original dtype."""
    pt_path = tmp_path / "dtype_test.pt"

    image = torch.randn(1, 10, 10, 10, dtype=torch.float32)
    label = torch.randint(0, 3, (1, 10, 10, 10), dtype=torch.int16)
    torch.save({"image": image, "label": label}, pt_path)

    transform = LoadPreprocessedTensord(keys=("image", "label"))
    result = transform({"tensor_cache": str(pt_path)})

    assert result["image"].dtype == torch.float32
    assert result["label"].dtype == torch.int16
