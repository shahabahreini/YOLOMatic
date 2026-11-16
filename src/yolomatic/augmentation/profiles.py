"""Augmentation profile YAML schema, CRUD operations, and built-in profiles."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROFILES_DIR = Path("configs") / "augmentation_profiles"


@dataclass
class AugmentationProfile:
    name: str
    description: str
    multiplier: int
    seed: int
    include_originals: bool
    transforms: list[dict[str, Any]]
    created_at: str
    modified_at: str
    schema_version: int = 1


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def profile_path(name: str, profiles_dir: Path = PROFILES_DIR) -> Path:
    return profiles_dir / f"{name}.yaml"


def list_profiles(profiles_dir: Path = PROFILES_DIR) -> list[str]:
    """Return sorted list of user-saved profile names (YAML stems)."""
    if not profiles_dir.exists():
        return []
    return sorted(p.stem for p in profiles_dir.glob("*.yaml"))


def load_profile(name: str, profiles_dir: Path = PROFILES_DIR) -> AugmentationProfile:
    """Load a profile from disk, falling back to built-ins if not on disk."""
    path = profile_path(name, profiles_dir)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return AugmentationProfile(
            name=data.get("name", name),
            description=data.get("description", ""),
            multiplier=int(data.get("multiplier", 3)),
            seed=int(data.get("seed", 42)),
            include_originals=bool(data.get("include_originals", True)),
            transforms=data.get("transforms", []),
            created_at=data.get("created_at", _now()),
            modified_at=data.get("modified_at", _now()),
            schema_version=int(data.get("schema_version", 1)),
        )
    if name in BUILT_IN_PROFILES:
        return BUILT_IN_PROFILES[name]
    raise FileNotFoundError(f"Profile '{name}' not found in {profiles_dir} or built-ins.")


def save_profile(profile: AugmentationProfile, profiles_dir: Path = PROFILES_DIR) -> None:
    """Persist a profile to disk as YAML."""
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile.modified_at = _now()
    data = {
        "schema_version": profile.schema_version,
        "name": profile.name,
        "description": profile.description,
        "multiplier": profile.multiplier,
        "seed": profile.seed,
        "include_originals": profile.include_originals,
        "created_at": profile.created_at,
        "modified_at": profile.modified_at,
        "transforms": profile.transforms,
    }
    with open(profile_path(profile.name, profiles_dir), "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def delete_profile(name: str, profiles_dir: Path = PROFILES_DIR) -> None:
    """Remove a profile YAML from disk. Does not affect built-ins."""
    path = profile_path(name, profiles_dir)
    if path.exists():
        path.unlink()


def clone_profile(
    source_name: str,
    new_name: str,
    profiles_dir: Path = PROFILES_DIR,
) -> AugmentationProfile:
    """Copy a profile (from disk or built-ins) under a new name."""
    src = load_profile(source_name, profiles_dir)
    now = _now()
    return AugmentationProfile(
        name=new_name,
        description=src.description,
        multiplier=src.multiplier,
        seed=src.seed,
        include_originals=src.include_originals,
        transforms=[dict(t) for t in src.transforms],
        created_at=now,
        modified_at=now,
        schema_version=src.schema_version,
    )


def ensure_builtin_profiles(profiles_dir: Path = PROFILES_DIR) -> None:
    """Seed built-in profiles to disk if the directory is empty or missing."""
    profiles_dir.mkdir(parents=True, exist_ok=True)
    for name, profile in BUILT_IN_PROFILES.items():
        dest = profile_path(name, profiles_dir)
        if not dest.exists():
            save_profile(profile, profiles_dir)


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

_VEGETATION_TRANSFORMS = [
    {"name": "D4",                       "enabled": True,  "p": 1.0},
    {"name": "Rotate",                   "enabled": True,  "p": 0.5,
     "limit_low": -15, "limit_high": 15, "border_mode": 4},
    {"name": "RandomBrightnessContrast", "enabled": True,  "p": 0.7,
     "brightness_limit_low": -0.2, "brightness_limit_high": 0.2,
     "contrast_limit_low": -0.2,   "contrast_limit_high": 0.2},
    {"name": "CLAHE",                    "enabled": True,  "p": 0.4,
     "clip_limit_low": 1.0, "clip_limit_high": 4.0},
    {"name": "GaussNoise",               "enabled": True,  "p": 0.3,
     "std_range_low": 0.012, "std_range_high": 0.028},
    {"name": "GaussianBlur",             "enabled": True,  "p": 0.2,
     "blur_limit_low": 3, "blur_limit_high": 7},
    {"name": "ElasticTransform",         "enabled": True,  "p": 0.3,
     "alpha": 1.0, "sigma": 50.0},
    {"name": "Perspective",              "enabled": True,  "p": 0.3,
     "scale_low": 0.05, "scale_high": 0.1},
    {"name": "RandomShadow",             "enabled": True,  "p": 0.3},
    {"name": "CoarseDropout",            "enabled": True,  "p": 0.2,
     "num_holes_low": 1, "num_holes_high": 8,
     "hole_height_low": 8, "hole_height_high": 32,
     "hole_width_low": 8, "hole_width_high": 32},
    {"name": "ImageCompression",         "enabled": True,  "p": 0.2,
     "quality_low": 70, "quality_high": 95},
    {"name": "HueSaturationValue",       "enabled": False, "p": 0.5},
    {"name": "MotionBlur",               "enabled": False, "p": 0.2},
    {"name": "HorizontalFlip",           "enabled": False, "p": 0.5},
    {"name": "VerticalFlip",             "enabled": False, "p": 0.5},
    {"name": "RandomSunFlare",           "enabled": False, "p": 0.1},
    {"name": "RandomFog",                "enabled": False, "p": 0.1},
    {"name": "ISONoise",                 "enabled": False, "p": 0.2},
    {"name": "Sharpen",                  "enabled": False, "p": 0.2},
    {"name": "GridDistortion",           "enabled": False, "p": 0.2},
    {"name": "OpticalDistortion",        "enabled": False, "p": 0.2},
    {"name": "Equalize",                 "enabled": False, "p": 0.2},
    {"name": "Solarize",                 "enabled": False, "p": 0.1},
    {"name": "ToGray",                   "enabled": False, "p": 0.1},
    {"name": "Posterize",                "enabled": False, "p": 0.1},
    {"name": "RGBShift",                 "enabled": False, "p": 0.2},
    {"name": "MedianBlur",               "enabled": False, "p": 0.2},
    {"name": "Downscale",                "enabled": False, "p": 0.1},
    {"name": "Affine",                   "enabled": False, "p": 0.3},
    {"name": "ShiftScaleRotate",         "enabled": False, "p": 0.3},
]

_GENERAL_TRANSFORMS = [
    {"name": "HorizontalFlip",           "enabled": True,  "p": 0.5},
    {"name": "VerticalFlip",             "enabled": False, "p": 0.5},
    {"name": "RandomBrightnessContrast", "enabled": True,  "p": 0.5,
     "brightness_limit_low": -0.15, "brightness_limit_high": 0.15,
     "contrast_limit_low": -0.15,   "contrast_limit_high": 0.15},
    {"name": "GaussNoise",               "enabled": True,  "p": 0.3,
     "std_range_low": 0.009, "std_range_high": 0.022},
    {"name": "GaussianBlur",             "enabled": False, "p": 0.2,
     "blur_limit_low": 3, "blur_limit_high": 5},
    {"name": "D4",                       "enabled": False, "p": 1.0},
    {"name": "Rotate",                   "enabled": False, "p": 0.3,
     "limit_low": -10, "limit_high": 10, "border_mode": 4},
    {"name": "CLAHE",                    "enabled": False, "p": 0.3,
     "clip_limit_low": 1.0, "clip_limit_high": 4.0},
    {"name": "ElasticTransform",         "enabled": False, "p": 0.2,
     "alpha": 1.0, "sigma": 50.0},
    {"name": "CoarseDropout",            "enabled": False, "p": 0.2,
     "num_holes_low": 1, "num_holes_high": 4,
     "hole_height_low": 8, "hole_height_high": 32,
     "hole_width_low": 8, "hole_width_high": 32},
    {"name": "ImageCompression",         "enabled": False, "p": 0.2,
     "quality_low": 75, "quality_high": 100},
    {"name": "HueSaturationValue",       "enabled": False, "p": 0.4},
    {"name": "Perspective",              "enabled": False, "p": 0.2,
     "scale_low": 0.05, "scale_high": 0.1},
    {"name": "Sharpen",                  "enabled": False, "p": 0.2},
    {"name": "Equalize",                 "enabled": False, "p": 0.2},
    {"name": "RGBShift",                 "enabled": False, "p": 0.2},
    {"name": "MotionBlur",               "enabled": False, "p": 0.2},
    {"name": "MedianBlur",               "enabled": False, "p": 0.2},
]

_MINIMAL_TRANSFORMS = [
    {"name": "D4", "enabled": True, "p": 1.0},
    {"name": "HorizontalFlip", "enabled": False, "p": 0.5},
    {"name": "VerticalFlip",   "enabled": False, "p": 0.5},
    {"name": "RandomBrightnessContrast", "enabled": False, "p": 0.3,
     "brightness_limit_low": -0.1, "brightness_limit_high": 0.1,
     "contrast_limit_low": -0.1,   "contrast_limit_high": 0.1},
    {"name": "GaussNoise", "enabled": False, "p": 0.2,
     "var_limit_low": 5.0, "var_limit_high": 20.0},
]

_EPOCH = "2026-05-14T12:00:00"

BUILT_IN_PROFILES: dict[str, AugmentationProfile] = {
    "vegetation_aerial_optimal": AugmentationProfile(
        name="vegetation_aerial_optimal",
        description=(
            "Optimized for QGIS NIR aerial imagery — rotation-invariant "
            "single-class vegetation/tree segmentation. D4 symmetry + "
            "radiometric variation + atmospheric effects."
        ),
        multiplier=3,
        seed=42,
        include_originals=True,
        transforms=_VEGETATION_TRANSFORMS,
        created_at=_EPOCH,
        modified_at=_EPOCH,
    ),
    "general_detection": AugmentationProfile(
        name="general_detection",
        description=(
            "Conservative profile suitable for most object detection datasets. "
            "Horizontal flip, brightness/contrast variation, and light Gaussian noise."
        ),
        multiplier=2,
        seed=0,
        include_originals=True,
        transforms=_GENERAL_TRANSFORMS,
        created_at=_EPOCH,
        modified_at=_EPOCH,
    ),
    "minimal": AugmentationProfile(
        name="minimal",
        description=(
            "Safe universal baseline: D4 symmetry only. "
            "Applies one of the 8 dihedral symmetry operations per image. "
            "Zero risk of annotation corruption."
        ),
        multiplier=2,
        seed=0,
        include_originals=True,
        transforms=_MINIMAL_TRANSFORMS,
        created_at=_EPOCH,
        modified_at=_EPOCH,
    ),
}
