from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RFDETRVariant:
    model: str
    class_name: str
    task: str
    size: str
    resolution: int
    map_5095: float | None
    latency_ms: float | None
    params_m: float | None
    license: str
    plus: bool = False


RFDETR_VARIANTS: dict[str, RFDETRVariant] = {
    "RF-DETR-Nano": RFDETRVariant(
        "RF-DETR-Nano", "RFDETRNano", "detection", "N", 384, 48.4, 2.3, 30.5, "Apache-2.0"
    ),
    "RF-DETR-Small": RFDETRVariant(
        "RF-DETR-Small", "RFDETRSmall", "detection", "S", 512, 53.0, 3.5, 32.1, "Apache-2.0"
    ),
    "RF-DETR-Medium": RFDETRVariant(
        "RF-DETR-Medium", "RFDETRMedium", "detection", "M", 576, 54.7, 4.4, 33.7, "Apache-2.0"
    ),
    "RF-DETR-Large": RFDETRVariant(
        "RF-DETR-Large", "RFDETRLarge", "detection", "L", 704, 56.5, 6.8, 33.9, "Apache-2.0"
    ),
    "RF-DETR-XLarge": RFDETRVariant(
        "RF-DETR-XLarge", "RFDETRXLarge", "detection", "XL", 700, 58.6, 11.5, 126.4, "PML-1.0", True
    ),
    "RF-DETR-2XLarge": RFDETRVariant(
        "RF-DETR-2XLarge", "RFDETR2XLarge", "detection", "2XL", 880, 60.1, 17.2, 126.9, "PML-1.0", True
    ),
    "RF-DETR-Seg-Nano": RFDETRVariant(
        "RF-DETR-Seg-Nano", "RFDETRSegNano", "segmentation", "N", 312, None, None, None, "Apache-2.0"
    ),
    "RF-DETR-Seg-Small": RFDETRVariant(
        "RF-DETR-Seg-Small", "RFDETRSegSmall", "segmentation", "S", 384, None, None, None, "Apache-2.0"
    ),
    "RF-DETR-Seg-Medium": RFDETRVariant(
        "RF-DETR-Seg-Medium", "RFDETRSegMedium", "segmentation", "M", 432, None, None, None, "Apache-2.0"
    ),
    "RF-DETR-Seg-Large": RFDETRVariant(
        "RF-DETR-Seg-Large", "RFDETRSegLarge", "segmentation", "L", 504, None, None, None, "Apache-2.0"
    ),
    "RF-DETR-Seg-XLarge": RFDETRVariant(
        "RF-DETR-Seg-XLarge", "RFDETRSegXLarge", "segmentation", "XL", 624, None, None, None, "Apache-2.0"
    ),
    "RF-DETR-Seg-2XLarge": RFDETRVariant(
        "RF-DETR-Seg-2XLarge", "RFDETRSeg2XLarge", "segmentation", "2XL", 768, None, None, None, "Apache-2.0"
    ),
    # Keypoint/pose is a preview architecture (RFDETRKeypointPreview), pretrained
    # on COCO person keypoints. Training accepts COCO keypoint JSON only.
    "RF-DETR-Keypoint": RFDETRVariant(
        "RF-DETR-Keypoint", "RFDETRKeypointPreview", "pose", "Preview", 640, None, None, None, "Apache-2.0"
    ),
}


RFDETR_FAMILY_KEYS = {"rfdetr", "rfdetr-seg", "rfdetr-pose"}


def is_rfdetr_model(value: str | None) -> bool:
    return bool(value) and str(value).lower().startswith("rf-detr")


def get_rfdetr_variant(model_choice: str) -> RFDETRVariant:
    try:
        return RFDETR_VARIANTS[model_choice]
    except KeyError as error:
        raise ValueError(f"Unknown RF-DETR model variant: {model_choice}") from error


def rfdetr_table_rows(task: str | None = None) -> list[dict[str, object]]:
    rows = []
    for variant in RFDETR_VARIANTS.values():
        if task is not None and variant.task != task:
            continue
        rows.append(
            {
                "Model": variant.model,
                "Input Size": variant.resolution,
                "mAPval 50-95": "-" if variant.map_5095 is None else variant.map_5095,
                "Latency (ms)": "-" if variant.latency_ms is None else variant.latency_ms,
                "params (M)": "-" if variant.params_m is None else variant.params_m,
                "License": variant.license,
            }
        )
    return rows


_SIZE_TOKENS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("2xlarge", "2xl"), "2XLarge"),
    (("xlarge", "xl"), "XLarge"),
    (("large",), "Large"),
    (("medium",), "Medium"),
    (("small",), "Small"),
    (("nano",), "Nano"),
)


def _class_from_checkpoint_args(weight_path: Path) -> str | None:
    """Identify the variant from the args RF-DETR stores inside its checkpoint."""
    try:
        from yolomatic.utils.ml_dependencies import import_torch

        torch = import_torch()
        checkpoint = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - missing torch, foreign or corrupt checkpoint
        return None
    args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if args is None:
        return None
    get = args.get if isinstance(args, dict) else (lambda key, default=None: getattr(args, key, default))
    resolution = get("resolution")
    if resolution is None:
        return None
    task = "segmentation" if get("segmentation_head", False) else "detection"
    matches = [v.class_name for v in RFDETR_VARIANTS.values() if v.task == task and v.resolution == int(resolution)]
    return matches[0] if len(matches) == 1 else None


def infer_rfdetr_class(weight_path: Path | str, hint: str = "") -> str:
    """Return the RF-DETR model class name (e.g. ``RFDETRSegLarge``) for a checkpoint.

    Checkpoint metadata wins. The name-based fallback only looks at the run
    folder, the file name and ``hint``, split into whole tokens: matching
    substrings of the full path picked ``XLarge`` for ``/pixel_data/`` and the
    Seg class for any path under ``/segments/``.
    """
    path = Path(weight_path)
    from_checkpoint = _class_from_checkpoint_args(path) if path.is_file() else None
    if from_checkpoint:
        return from_checkpoint

    import re

    names = [path.stem, path.parent.name, path.parent.parent.name, hint]
    tokens = {token for name in names for token in re.split(r"[^a-z0-9]+", name.lower()) if token}
    prefix = "RFDETRSeg" if tokens & {"seg", "segment", "segmentation"} else "RFDETR"
    for aliases, size in _SIZE_TOKENS:
        if tokens.intersection(aliases):
            return f"{prefix}{size}"
    return f"{prefix}Medium"
