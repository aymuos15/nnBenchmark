"""Offline preprocessing: save common-transform outputs as .pt tensors.

After planning generates the YAML config, this module applies the
deterministic common transforms (LoadImage, Normalize, Pad, ToTensor)
once and saves the results as .pt files with memory-mapped access support.

During training, cached tensors are loaded via torch.load(mmap=True),
skipping NIfTI loading and common transforms entirely.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from monai import transforms
from tqdm import tqdm

from src.utils.files import extract_case_id


def preprocess_to_tensors(config_path: str, data_dir: str) -> None:
    """Apply common transforms to all cases and save as .pt files.

    Called during planning, after YAML config generation.

    Args:
        config_path: Path to the generated fold YAML config.
        data_dir: Raw dataset directory (used to resolve preprocessed paths).
    """
    from src.config.load import load_config
    from src.config.paths import get_preprocessed_root
    from src.engines.setup import _instantiate_component
    from src.utils.data import _build_case_to_paths_mapping

    cfg = load_config(config_path)
    dataset_name = Path(data_dir).name
    preprocessed_dir = get_preprocessed_root() / dataset_name
    tensors_dir = preprocessed_dir / "tensorsTr"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    # Build common transforms pipeline
    common_transform_list = []
    for t_cfg in cfg["transforms"]["common"]:
        common_transform_list.append(_instantiate_component(dict(t_cfg)))
    common_transforms = transforms.Compose(common_transform_list)

    # Get all case paths
    case_to_paths = _build_case_to_paths_mapping(data_dir)

    existing = sum(1 for c in case_to_paths for _ in [1] if (tensors_dir / f"{extract_case_id(c, remove_channel_suffix=True)}.pt").exists())
    if existing == len(case_to_paths):
        logger.debug("All tensor cache files already exist, skipping")
        return

    logger.info(f"Preprocessing {len(case_to_paths)} cases to tensor cache...")

    for case_name, paths in tqdm(case_to_paths.items(), desc="Tensor cache"):
        case_id = extract_case_id(case_name, remove_channel_suffix=True)
        output_path = tensors_dir / f"{case_id}.pt"

        if output_path.exists():
            continue

        data_dict = {"image": paths["image"], "label": paths["label"]}
        result = common_transforms(data_dict)

        torch.save(
            {"image": result["image"].contiguous(), "label": result["label"].contiguous()},
            output_path,
        )

    logger.info(f"Tensor cache saved to {tensors_dir}")
