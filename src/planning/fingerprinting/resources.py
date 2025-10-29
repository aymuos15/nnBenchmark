"""
System resource detection and optimization module.

Automatically detects CPU, RAM, and GPU resources and recommends
optimal configuration parameters for training and data loading.
"""

from __future__ import annotations

from dataclasses import dataclass

import psutil
import torch
from loguru import logger


@dataclass
class SystemResources:
    """Detected system resources and recommendations."""

    # CPU
    cpu_count: int
    cpu_logical_count: int
    recommended_workers: int

    # RAM
    total_ram_gb: float
    available_ram_gb: float
    usable_ram_gb: float  # 70% of available for safety

    # GPU
    gpu_available: bool
    gpu_memory_gb: float
    gpu_name: str

    # Recommendations
    num_workers: int
    cache_enabled: bool
    cache_rate: float
    persistent_workers: bool


def detect_cpu_cores() -> tuple[int, int]:
    """
    Detect physical and logical CPU cores.

    Returns:
        Tuple of (physical_cores, logical_cores)
    """
    try:
        physical = psutil.cpu_count(logical=False) or 1
        logical = psutil.cpu_count(logical=True) or 1
        return physical, logical
    except Exception as e:
        logger.warning(f"Failed to detect CPU cores: {e}. Defaulting to 1 core.")
        return 1, 1


def detect_system_ram() -> tuple[float, float]:
    """
    Detect total and available system RAM.

    Returns:
        Tuple of (total_ram_gb, available_ram_gb)
    """
    try:
        total = psutil.virtual_memory().total / (1024**3)
        available = psutil.virtual_memory().available / (1024**3)
        return round(total, 1), round(available, 1)
    except Exception as e:
        logger.warning(f"Failed to detect system RAM: {e}. Defaulting to 16 GB.")
        return 16.0, 12.0


def detect_gpu_memory() -> tuple[bool, float, str]:
    """
    Detect GPU availability and memory.

    Returns:
        Tuple of (gpu_available, gpu_memory_gb, gpu_name)
    """
    if not torch.cuda.is_available():
        return False, 0.0, "None"

    try:
        # Get GPU properties
        props = torch.cuda.get_device_properties(0)
        memory_gb = props.total_memory / (1024**3)
        gpu_name = props.name

        # Use 80% of available memory as conservative estimate
        usable_memory = memory_gb * 0.8

        return True, round(usable_memory, 1), gpu_name
    except Exception as e:
        logger.warning(f"Failed to detect GPU memory: {e}. GPU will not be used.")
        return False, 0.0, "None"


def calculate_optimal_workers(
    cpu_count: int,
    strategy: str = "aggressive",
) -> int:
    """
    Calculate optimal number of DataLoader workers based on CPU cores.

    Args:
        cpu_count: Number of logical CPU cores
        strategy: One of 'aggressive', 'balanced', or 'conservative'
            - aggressive: 75-80% of cores, up to 16 workers
            - balanced: 50-60% of cores, up to 12 workers
            - conservative: 25-50% of cores, up to 8 workers

    Returns:
        Recommended number of workers
    """
    if strategy == "aggressive":
        # Use 75% of cores, but leave at least 2 cores free, cap at 16
        workers = max(1, min(int(cpu_count * 0.75), cpu_count - 2, 16))
    elif strategy == "balanced":
        # Use 50% of cores, leave 3+ cores free, cap at 12
        workers = max(1, min(int(cpu_count * 0.5), cpu_count - 3, 12))
    elif strategy == "conservative":
        # Use 25% of cores or half, whichever is smaller, cap at 8
        workers = max(1, min(int(cpu_count * 0.25), cpu_count // 2, 8))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return int(workers)


def calculate_cache_strategy(
    dataset_size_mb: float,
    available_ram_gb: float,
) -> tuple[bool, float]:
    """
    Calculate intelligent caching strategy based on dataset size vs available RAM.

    Args:
        dataset_size_mb: Estimated dataset size in MB
        available_ram_gb: Available system RAM in GB

    Returns:
        Tuple of (cache_enabled, cache_rate)
        - cache_rate: 0.0 (no cache) to 1.0 (full cache)
    """
    dataset_size_gb = dataset_size_mb / 1024
    usable_ram_gb = (
        available_ram_gb * 0.7
    )  # Safety margin: only use 70% of available RAM

    # Calculate what percentage of dataset we can cache
    if dataset_size_gb == 0:
        return False, 0.0

    dataset_ratio = dataset_size_gb / usable_ram_gb

    # Three-tier intelligent caching system
    if dataset_ratio < 0.2:
        # Small dataset (< 20% of usable RAM): full cache
        return True, 1.0
    elif dataset_ratio < 0.6:
        # Medium dataset (20-60% of usable RAM): partial cache
        # Scale cache_rate inversely with dataset ratio
        # At 0.2: cache_rate=0.8, at 0.6: cache_rate=0.3
        cache_rate = max(0.3, 1.0 - (dataset_ratio / 0.8))
        return True, round(cache_rate, 2)
    else:
        # Large dataset (> 60% of usable RAM): no cache
        return False, 0.0


def get_gpu_memory_for_planning(fallback_gb: float = 8.0) -> float:
    """
    Get GPU memory for planning workflow.

    Returns just the usable memory value with a sensible default
    for nnU-Net compatibility. This is the planning-specific interface
    to the more general detect_gpu_memory().

    Args:
        fallback_gb: Default memory in GB if GPU unavailable (default: 8.0 for nnU-Net)

    Returns:
        Usable GPU memory in GB
    """
    gpu_available, gpu_memory_gb, _ = detect_gpu_memory()
    return gpu_memory_gb if gpu_available else fallback_gb


def get_system_resources(
    gpu_memory_gb_override: float | None = None,
    num_workers_strategy: str = "aggressive",
    dataset_size_mb: float = 0.0,
) -> SystemResources:
    """
    Detect all system resources and calculate optimal configuration.

    Args:
        gpu_memory_gb_override: Override GPU memory detection with this value
        num_workers_strategy: Strategy for calculating optimal workers
        dataset_size_mb: Dataset size in MB for intelligent caching decisions

    Returns:
        SystemResources object with all detected resources and recommendations
    """
    # Detect hardware
    physical_cores, logical_cores = detect_cpu_cores()
    total_ram_gb, available_ram_gb = detect_system_ram()
    gpu_available, gpu_memory_gb, gpu_name = detect_gpu_memory()

    # Override GPU memory if specified
    if gpu_memory_gb_override is not None:
        gpu_memory_gb = gpu_memory_gb_override

    # Calculate recommendations
    recommended_workers = calculate_optimal_workers(
        logical_cores, strategy=num_workers_strategy
    )
    usable_ram_gb = available_ram_gb * 0.7
    cache_enabled, cache_rate = calculate_cache_strategy(
        dataset_size_mb, available_ram_gb
    )

    # persistent_workers is beneficial when num_workers > 0 and not using cache dataset
    # (CacheDataset requires persistent_workers=False)
    persistent_workers = recommended_workers > 0 and not cache_enabled

    return SystemResources(
        cpu_count=physical_cores,
        cpu_logical_count=logical_cores,
        recommended_workers=recommended_workers,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        usable_ram_gb=usable_ram_gb,
        gpu_available=gpu_available,
        gpu_memory_gb=gpu_memory_gb,
        gpu_name=gpu_name,
        num_workers=recommended_workers,
        cache_enabled=cache_enabled,
        cache_rate=cache_rate,
        persistent_workers=persistent_workers,
    )
