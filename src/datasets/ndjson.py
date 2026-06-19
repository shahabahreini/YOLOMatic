from __future__ import annotations

from typing import Any


def pose_metadata_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return validated Ultralytics pose metadata, inferring kpt_shape when needed."""
    header = next((row for row in rows if row.get("type") == "dataset"), {})
    raw_shape = header.get("kpt_shape")
    kpt_shape: list[int] | None = None
    if isinstance(raw_shape, (list, tuple)) and len(raw_shape) == 2:
        try:
            candidate = [int(raw_shape[0]), int(raw_shape[1])]
        except (TypeError, ValueError):
            candidate = []
        if len(candidate) == 2 and candidate[0] > 0 and candidate[1] in (2, 3):
            kpt_shape = candidate

    pose_values: list[list[Any]] = []
    for row in rows:
        annotations = row.get("annotations")
        if not isinstance(annotations, dict):
            continue
        pose_values.extend(
            ann
            for ann in (annotations.get("pose") or [])
            if isinstance(ann, list) and len(ann) > 5
        )

    if kpt_shape is None:
        lengths = {len(ann) - 5 for ann in pose_values[:50]}
        if len(lengths) == 1:
            value_count = next(iter(lengths))
            samples = [ann[5:] for ann in pose_values[:50]]
            if value_count % 3 == 0 and all(
                value in (0, 1, 2) for sample in samples for value in sample[2::3]
            ):
                kpt_shape = [value_count // 3, 3]
            elif value_count % 2 == 0 and value_count % 3 != 0:
                kpt_shape = [value_count // 2, 2]
    if kpt_shape is None:
        raise ValueError(
            "Pose NDJSON is missing a valid kpt_shape and it could not be inferred."
        )

    expected_length = 5 + kpt_shape[0] * kpt_shape[1]
    if not any(len(ann) == expected_length for ann in pose_values):
        raise ValueError(
            f"Pose NDJSON has no annotations matching kpt_shape {kpt_shape}; "
            f"expected {expected_length} values per pose annotation."
        )

    raw_names = header.get("kpt_names") or header.get("keypoint_names")
    if isinstance(raw_names, dict):
        raw_names = next(
            (value for value in raw_names.values() if isinstance(value, list)), None
        )
    kpt_names = (
        [str(value) for value in raw_names] if isinstance(raw_names, list) else []
    )
    if len(kpt_names) != kpt_shape[0]:
        kpt_names = [f"kpt_{idx}" for idx in range(kpt_shape[0])]

    skeleton = header.get("skeleton")
    if not isinstance(skeleton, list):
        skeleton = []
    flip_idx = header.get("flip_idx")
    if not isinstance(flip_idx, list) or len(flip_idx) != kpt_shape[0]:
        flip_idx = None
    return {
        "kpt_shape": kpt_shape,
        "kpt_names": kpt_names,
        "skeleton": skeleton,
        "flip_idx": flip_idx,
    }
