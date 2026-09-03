"""Tests for Roboflow pre-flight dataset structural validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.datasets.validate_roboflow import (
    TASK_INSTANCE_SEG,
    TASK_OBJECT_DETECT,
    TASK_POSE,
    TASK_SEMANTIC_SEG,
    _is_valid_image_header,
    map_yolo_task_to_roboflow,
    validate_roboflow_dataset,
)


def _create_minimal_png(path: Path) -> None:
    # Minimal 1x1 PNG header + data
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa74U\xa6\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.write_bytes(png_bytes)


def test_task_mapping():
    assert map_yolo_task_to_roboflow("segment") == TASK_INSTANCE_SEG
    assert map_yolo_task_to_roboflow("detect") == TASK_OBJECT_DETECT
    assert map_yolo_task_to_roboflow("semantic") == TASK_SEMANTIC_SEG
    assert map_yolo_task_to_roboflow("pose") == TASK_POSE


def test_valid_image_header(tmp_path: Path):
    valid_png = tmp_path / "test.png"
    _create_minimal_png(valid_png)
    assert _is_valid_image_header(valid_png) is True

    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"corrupt header content")
    assert _is_valid_image_header(corrupt_file) is False


def test_validate_well_formed_segmentation_dataset(tmp_path: Path):
    ds_dir = tmp_path / "test_seg_ds"
    (ds_dir / "images" / "train").mkdir(parents=True)
    (ds_dir / "images" / "val").mkdir(parents=True)
    (ds_dir / "labels" / "train").mkdir(parents=True)
    (ds_dir / "labels" / "val").mkdir(parents=True)

    # data.yaml
    data_yaml = {
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["tree", "bush"],
        "task": "segment",
    }
    (ds_dir / "data.yaml").write_text(yaml.dump(data_yaml))

    # Train sample 1: valid polygon with 4 points (8 coordinates)
    _create_minimal_png(ds_dir / "images" / "train" / "img1.png")
    (ds_dir / "labels" / "train" / "img1.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")

    # Train sample 2: clean negative background (0-byte label)
    _create_minimal_png(ds_dir / "images" / "train" / "img2.png")
    (ds_dir / "labels" / "train" / "img2.txt").write_text("")

    # Val sample 1
    _create_minimal_png(ds_dir / "images" / "val" / "img_val1.png")
    (ds_dir / "labels" / "val" / "img_val1.txt").write_text("1 0.2 0.2 0.8 0.2 0.8 0.8\n")

    res = validate_roboflow_dataset(ds_dir)
    assert res.is_valid is True
    assert res.detected_task == TASK_INSTANCE_SEG
    assert res.num_classes == 2
    assert res.total_images == 3
    assert res.total_annotated == 2
    assert res.total_negatives == 1
    assert len(res.errors) == 0


def test_validate_detection_with_invalid_rows(tmp_path: Path):
    ds_dir = tmp_path / "test_det_ds"
    (ds_dir / "images" / "train").mkdir(parents=True)
    (ds_dir / "labels" / "train").mkdir(parents=True)

    data_yaml = {
        "train": "images/train",
        "nc": 1,
        "names": ["car"],
        "task": "detect",
    }
    (ds_dir / "data.yaml").write_text(yaml.dump(data_yaml))

    # Image with out-of-range class ID
    _create_minimal_png(ds_dir / "images" / "train" / "bad_class.png")
    (ds_dir / "labels" / "train" / "bad_class.txt").write_text("5 0.5 0.5 0.2 0.2\n")

    # Image with unnormalized coords (> 1.0)
    _create_minimal_png(ds_dir / "images" / "train" / "bad_coords.png")
    (ds_dir / "labels" / "train" / "bad_coords.txt").write_text("0 1.5 0.5 0.2 0.2\n")

    # Image with non-positive dimension
    _create_minimal_png(ds_dir / "images" / "train" / "bad_dim.png")
    (ds_dir / "labels" / "train" / "bad_dim.txt").write_text("0 0.5 0.5 0.0 0.2\n")

    res = validate_roboflow_dataset(ds_dir)
    assert res.is_valid is False
    assert len(res.errors) > 0
    assert any("class index 5 outside valid range" in err for err in res.errors)
    assert any("box coordinates outside normalized" in err for err in res.errors)
    assert any("non-positive box dimension" in err for err in res.errors)


def test_validate_segmentation_degenerate_polygons(tmp_path: Path):
    ds_dir = tmp_path / "test_degen_ds"
    (ds_dir / "images" / "train").mkdir(parents=True)
    (ds_dir / "labels" / "train").mkdir(parents=True)

    data_yaml = {
        "train": "images/train",
        "nc": 1,
        "names": ["leaf"],
        "task": "segment",
    }
    (ds_dir / "data.yaml").write_text(yaml.dump(data_yaml))

    # Polygon with only 2 vertices (4 coordinates) - invalid for polygon
    _create_minimal_png(ds_dir / "images" / "train" / "img_degen.png")
    (ds_dir / "labels" / "train" / "img_degen.txt").write_text("0 0.1 0.1 0.5 0.5\n")

    res = validate_roboflow_dataset(ds_dir)
    assert res.is_valid is False
    assert any("requires at least 3 points" in err for err in res.errors)
