"""Albumentations-based offline dataset augmentation for YOLOMatic."""
from __future__ import annotations

from yolomatic.augmentation.engine import AugmentationStats, SplitConfig, run_augmentation
from yolomatic.augmentation.profiles import (
    BUILT_IN_PROFILES,
    PROFILES_DIR,
    AugmentationProfile,
    clone_profile,
    delete_profile,
    list_profiles,
    load_profile,
    save_profile,
)
from yolomatic.augmentation.transforms import TRANSFORM_GROUPS, get_params_for_transform

__all__ = [
    "AugmentationProfile",
    "AugmentationStats",
    "BUILT_IN_PROFILES",
    "PROFILES_DIR",
    "SplitConfig",
    "TRANSFORM_GROUPS",
    "clone_profile",
    "delete_profile",
    "get_params_for_transform",
    "list_profiles",
    "load_profile",
    "run_augmentation",
    "save_profile",
]
