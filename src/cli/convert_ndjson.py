from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml
from rich.live import Live
from rich.panel import Panel

from src.datasets.ndjson import pose_metadata_from_rows
from src.datasets.prepare import slugify
from src.utils.cli import (
    NAV_BACK,
    ParameterDefinition,
    clear_screen,
    console,
    format_path,
    get_parameter_value_input,
    get_user_choice,
    print_stylized_header,
    render_summary_panel,
)

OUTPUT_FORMATS = ("YOLO Detection", "YOLO Segmentation", "YOLO Pose", "COCO", "COCO Pose")
WIZARD_STEPS = ["Source", "Format", "Output", "Confirm", "Convert"]


@dataclass
class ConversionStats:
    source_path: str
    output_path: str
    output_format: str
    total_rows: int
    converted_images: int = 0
    total_annotations: int = 0
    classes: list[str] = field(default_factory=list)
    skipped_rows: int = 0
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _wizard_kwargs(step_index: int) -> dict[str, Any]:
    return {"wizard_steps": WIZARD_STEPS, "wizard_current_step": step_index}


def _discover_ndjson_files(root: Path = Path(".")) -> list[Path]:
    candidates: list[Path] = []
    for search_root, pattern in ((root, "*.ndjson"), (root / "datasets", "**/*.ndjson")):
        if not search_root.exists():
            continue
        for path in search_root.glob(pattern):
            if path.is_file() and not path.name.startswith("."):
                candidates.append(path)
    return sorted(set(candidates), key=lambda path: str(path.resolve()).lower())


def _display_path(path: Path, root: Path = Path(".")) -> str:
    try:
        return format_path(path.relative_to(root))
    except ValueError:
        return format_path(path)


def _ndjson_description(path: Path) -> str:
    try:
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
        row_count = len(rows)
        first = rows[0] if rows else {}
        if first.get("type") == "dataset":
            source_type = "Ultralytics Platform NDJSON"
            classes = first.get("class_names") or first.get("names") or {}
        elif isinstance(first.get("data_row"), dict):
            source_type = "Labelbox NDJSON"
            classes = {}
        else:
            source_type = "NDJSON"
            classes = {}
        class_count = len(classes) if isinstance(classes, (dict, list)) else 0
        return (
            f"[bold cyan]{path.name}[/bold cyan]\n\n"
            f"Format: [yellow]{source_type}[/yellow]\n"
            f"Rows: [yellow]{row_count}[/yellow]\n"
            f"Classes: [yellow]{class_count or 'detected while converting'}[/yellow]\n\n"
            f"[dim]{path}[/dim]"
        )
    except Exception as exc:
        return (
            f"[bold cyan]{path.name}[/bold cyan]\n\n"
            f"[yellow]Could not inspect file:[/yellow] {exc}\n\n"
            f"[dim]{path}[/dim]"
        )


def _read_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid NDJSON at line {line_no}: {exc}") from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _download_image(url: str, output_path: Path, timeout: int = 15) -> tuple[bool, str | None]:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return True, None
    except Exception as exc:
        return False, type(exc).__name__


def _dedupe_filename(name: str, used: set[str]) -> str:
    candidate = Path(name).name or "image.jpg"
    if candidate not in used:
        used.add(candidate)
        return candidate
    stem = Path(candidate).stem
    suffix = Path(candidate).suffix or ".jpg"
    idx = 2
    while True:
        renamed = f"{stem}_{idx:04d}{suffix}"
        if renamed not in used:
            used.add(renamed)
            return renamed
        idx += 1


def _extract_labelbox_objects(row: dict[str, Any]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for project in row.get("projects", {}).values():
        for label in project.get("labels", []):
            objects.extend(label.get("annotations", {}).get("objects", []))
    return objects


def _ultralytics_class_names_from_header(rows: list[dict[str, Any]]) -> list[str]:
    """Pull class names from the Ultralytics-platform NDJSON `dataset` header row."""
    for row in rows:
        if row.get("type") != "dataset":
            continue
        raw = row.get("class_names") or row.get("names") or {}
        if isinstance(raw, dict):
            names_by_id: dict[int, str] = {}
            for key, value in raw.items():
                try:
                    names_by_id[int(key)] = str(value)
                except (TypeError, ValueError):
                    continue
            if names_by_id:
                max_id = max(names_by_id)
                return [names_by_id.get(idx, f"class_{idx}") for idx in range(max_id + 1)]
        if isinstance(raw, list):
            return [str(name) for name in raw]
    return []


def _extract_ultralytics_objects(
    row: dict[str, Any],
    img_w: int,
    img_h: int,
    class_names: list[str],
    pose_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert Ultralytics-platform NDJSON annotations to Labelbox-shaped pixel-space objects."""
    payload = row.get("annotations")
    if not isinstance(payload, dict):
        return []
    objects: list[dict[str, Any]] = []

    def _name_for(class_id: int) -> str:
        if 0 <= class_id < len(class_names):
            return class_names[class_id]
        return f"class_{class_id}"

    for raw_segment in payload.get("segments") or []:
        if not isinstance(raw_segment, list) or len(raw_segment) < 7:
            continue
        try:
            class_id = int(float(raw_segment[0]))
            coords = [float(v) for v in raw_segment[1:]]
        except (TypeError, ValueError):
            continue
        if len(coords) < 6 or len(coords) % 2 != 0:
            continue
        points = [{"x": coords[i] * img_w, "y": coords[i + 1] * img_h} for i in range(0, len(coords), 2)]
        objects.append({"name": _name_for(class_id), "polygon": points})

    for raw_box in payload.get("boxes") or payload.get("bboxes") or []:
        if not isinstance(raw_box, list) or len(raw_box) < 5:
            continue
        try:
            class_id = int(float(raw_box[0]))
            xc, yc, w, h = (float(v) for v in raw_box[1:5])
        except (TypeError, ValueError):
            continue
        objects.append({
            "name": _name_for(class_id),
            "bounding_box": {
                "left": (xc - w / 2) * img_w,
                "top": (yc - h / 2) * img_h,
                "width": w * img_w,
                "height": h * img_h,
            },
        })

    if pose_meta:
        kpt_shape = pose_meta["kpt_shape"]
        expected_length = 5 + int(kpt_shape[0]) * int(kpt_shape[1])
        for raw_pose in payload.get("pose") or []:
            if not isinstance(raw_pose, list) or len(raw_pose) != expected_length:
                continue
            try:
                class_id = int(float(raw_pose[0]))
                xc, yc, width, height = (float(value) for value in raw_pose[1:5])
                keypoints = [float(value) for value in raw_pose[5:]]
            except (TypeError, ValueError):
                continue
            objects.append({
                "name": _name_for(class_id),
                "bounding_box": {
                    "left": (xc - width / 2) * img_w,
                    "top": (yc - height / 2) * img_h,
                    "width": width * img_w,
                    "height": height * img_h,
                },
                "keypoints": keypoints,
            })

    return objects


def _class_id(classes: dict[str, int], name: str | None) -> int:
    key = str(name or "unnamed")
    if key not in classes:
        classes[key] = len(classes)
    return classes[key]


def _bbox_from_polygon(points: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    left = min(xs)
    top = min(ys)
    return left, top, max(xs) - left, max(ys) - top


def _bbox_to_yolo(left: float, top: float, width: float, height: float, img_w: int, img_h: int) -> str:
    return (
        f"{(left + width / 2) / img_w:.6f} "
        f"{(top + height / 2) / img_h:.6f} "
        f"{width / img_w:.6f} "
        f"{height / img_h:.6f}"
    )


def _rect_segment(left: float, top: float, width: float, height: float, img_w: int, img_h: int) -> list[float]:
    x1 = left / img_w
    y1 = top / img_h
    x2 = (left + width) / img_w
    y2 = (top + height) / img_h
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _write_yolo_label(
    label_path: Path,
    objects: list[dict[str, Any]],
    classes: dict[str, int],
    output_format: str,
    img_w: int,
    img_h: int,
    pose_meta: dict[str, Any] | None = None,
) -> int:
    lines: list[str] = []
    for obj in objects:
        cls = _class_id(classes, obj.get("name"))
        if output_format == "YOLO Pose":
            bbox = obj.get("bounding_box")
            keypoints = obj.get("keypoints")
            if not isinstance(bbox, dict) or not isinstance(keypoints, list) or pose_meta is None:
                continue
            expected = int(pose_meta["kpt_shape"][0]) * int(pose_meta["kpt_shape"][1])
            if len(keypoints) != expected:
                continue
            box = _bbox_to_yolo(
                float(bbox["left"]),
                float(bbox["top"]),
                float(bbox["width"]),
                float(bbox["height"]),
                img_w,
                img_h,
            )
            lines.append(f"{cls} {box} " + " ".join(f"{float(value):.6f}" for value in keypoints))
            continue
        if "bounding_box" in obj:
            bbox = obj["bounding_box"]
            left = float(bbox["left"])
            top = float(bbox["top"])
            width = float(bbox["width"])
            height = float(bbox["height"])
            if output_format == "YOLO Detection":
                lines.append(f"{cls} {_bbox_to_yolo(left, top, width, height, img_w, img_h)}")
            else:
                lines.append(f"{cls} " + " ".join(f"{value:.6f}" for value in _rect_segment(left, top, width, height, img_w, img_h)))
        elif "polygon" in obj:
            points = obj["polygon"] or []
            if len(points) < 3:
                continue
            if output_format == "YOLO Detection":
                left, top, width, height = _bbox_from_polygon(points)
                lines.append(f"{cls} {_bbox_to_yolo(left, top, width, height, img_w, img_h)}")
            else:
                coords: list[float] = []
                for point in points:
                    coords.extend([float(point["x"]) / img_w, float(point["y"]) / img_h])
                lines.append(f"{cls} " + " ".join(f"{value:.6f}" for value in coords))
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def _append_coco_annotations(
    coco_images: list[dict[str, Any]],
    coco_annotations: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    classes: dict[str, int],
    filename: str,
    img_w: int,
    img_h: int,
    next_annotation_id: int,
    pose_meta: dict[str, Any] | None = None,
) -> int:
    image_id = len(coco_images) + 1
    coco_images.append({"id": image_id, "file_name": filename, "width": img_w, "height": img_h})
    annotation_id = next_annotation_id
    for obj in objects:
        if pose_meta is not None and not isinstance(obj.get("keypoints"), list):
            continue
        cls = _class_id(classes, obj.get("name")) + 1
        segmentation: list[list[float]] = []
        if "polygon" in obj:
            points = obj["polygon"] or []
            if len(points) < 3:
                continue
            flat_points: list[float] = []
            for point in points:
                flat_points.extend([float(point["x"]), float(point["y"])])
            left, top, width, height = _bbox_from_polygon(points)
            segmentation = [flat_points]
        elif "bounding_box" in obj:
            bbox = obj["bounding_box"]
            left = float(bbox["left"])
            top = float(bbox["top"])
            width = float(bbox["width"])
            height = float(bbox["height"])
        else:
            continue
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": cls,
            "segmentation": segmentation,
            "bbox": [left, top, width, height],
            "area": max(0.0, width * height),
            "iscrowd": 0,
        }
        if pose_meta is not None:
            kpt_count, kpt_ndim = (int(value) for value in pose_meta["kpt_shape"])
            raw_keypoints = obj["keypoints"]
            if len(raw_keypoints) != kpt_count * kpt_ndim:
                continue
            keypoints: list[float | int] = []
            visible = 0
            for offset in range(0, len(raw_keypoints), kpt_ndim):
                x = float(raw_keypoints[offset]) * img_w
                y = float(raw_keypoints[offset + 1]) * img_h
                visibility = int(float(raw_keypoints[offset + 2])) if kpt_ndim == 3 else 2
                keypoints.extend([x, y, visibility])
                if visibility > 0:
                    visible += 1
            annotation["keypoints"] = keypoints
            annotation["num_keypoints"] = visible
        coco_annotations.append(annotation)
        annotation_id += 1
    return annotation_id


def _row_image_url(row: dict[str, Any]) -> str | None:
    data_row = row.get("data_row")
    if isinstance(data_row, dict) and data_row.get("row_data"):
        return str(data_row["row_data"])
    if row.get("type") == "image" and row.get("url"):
        return str(row["url"])
    return None


def _row_filename(row: dict[str, Any], url: str, fallback_idx: int) -> str:
    data_row = row.get("data_row") if isinstance(row.get("data_row"), dict) else {}
    return Path(str(data_row.get("global_key") or row.get("file") or Path(urlparse(url).path).name or f"row_{fallback_idx:06d}.jpg")).name


def _row_split(row: dict[str, Any]) -> tuple[str, bool]:
    raw = str(row.get("split") or "").strip().lower()
    aliases = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
    return aliases.get(raw, "train"), raw not in aliases


def convert_ndjson_to_format(
    ndjson_path: Path,
    output_format: str,
    output_dir: Path,
    *,
    max_workers: int = 10,
    overwrite: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ConversionStats:
    started = time.time()
    if output_format == "YOLO":
        output_format = "YOLO Detection"
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}")
    if not ndjson_path.exists() or not ndjson_path.is_file():
        raise FileNotFoundError(f"NDJSON file not found: {ndjson_path}")

    rows = _read_ndjson(ndjson_path)
    pose_output = output_format in {"YOLO Pose", "COCO Pose"}
    # Detect Ultralytics-platform NDJSON (presence of a dataset header row OR normalized image annotations).
    is_ultralytics = any(row.get("type") == "dataset" for row in rows) or any(
        row.get("type") == "image" and isinstance(row.get("annotations"), dict)
        and (
            row["annotations"].get("segments")
            or row["annotations"].get("boxes")
            or row["annotations"].get("bboxes")
            or row["annotations"].get("pose")
        )
        for row in rows
    )
    if pose_output and not is_ultralytics:
        raise ValueError("Pose conversion currently requires an Ultralytics-platform NDJSON export.")
    pose_meta = pose_metadata_from_rows(rows) if pose_output else None

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels"
    if output_format.startswith("YOLO"):
        labels_dir.mkdir(parents=True, exist_ok=True)

    stats = ConversionStats(
        source_path=str(ndjson_path.resolve()),
        output_path=str(output_dir.resolve()),
        output_format=output_format,
        total_rows=len(rows),
    )
    ultralytics_class_names = _ultralytics_class_names_from_header(rows) if is_ultralytics else []

    classes: dict[str, int] = {name: idx for idx, name in enumerate(ultralytics_class_names)}
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    next_annotation_id = 1
    coco_splits: dict[str, dict[str, Any]] = {}
    used_names: set[str] = set()

    if progress_callback:
        progress_callback(0, len(rows), "Scanning NDJSON rows...")

    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        for idx, row in enumerate(rows, start=1):
            url = _row_image_url(row)
            if not url:
                if row.get("type") != "dataset":
                    stats.warnings.append(f"Skipped row {idx}: missing image URL")
                    stats.skipped_rows += 1
                continue
            filename = _dedupe_filename(_row_filename(row, url, idx), used_names)
            split, used_fallback = _row_split(row)
            if pose_output and used_fallback:
                stats.warnings.append(f"Row {idx}: missing or unknown split; defaulted to train")
            image_path = images_dir / split / filename if pose_output else images_dir / filename
            futures[executor.submit(_download_image, url, image_path)] = (idx, row, filename, image_path, split)

        total_downloads = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            idx, row, filename, image_path, split = futures[future]
            ok, reason = future.result()
            if not ok:
                stats.warnings.append(f"Skipped row {idx}: image download failed ({reason or 'unknown error'})")
                stats.skipped_rows += 1
                if progress_callback:
                    progress_callback(completed, total_downloads, f"Converting rows... {completed}/{total_downloads}")
                continue
            try:
                from PIL import Image

                with Image.open(image_path) as img:
                    img_w, img_h = img.size
            except Exception as exc:
                stats.warnings.append(f"Skipped row {idx}: downloaded image could not be opened ({type(exc).__name__})")
                stats.skipped_rows += 1
                if progress_callback:
                    progress_callback(completed, total_downloads, f"Converting rows... {completed}/{total_downloads}")
                continue

            if is_ultralytics:
                objects = _extract_ultralytics_objects(
                    row,
                    int(img_w),
                    int(img_h),
                    ultralytics_class_names,
                    pose_meta,
                )
            else:
                objects = _extract_labelbox_objects(row)
            if pose_output:
                annotations = row.get("annotations")
                raw_pose = annotations.get("pose") or [] if isinstance(annotations, dict) else []
                valid_pose_count = sum(1 for obj in objects if isinstance(obj.get("keypoints"), list))
                if len(raw_pose) != valid_pose_count:
                    stats.warnings.append(
                        f"Row {idx}: skipped {len(raw_pose) - valid_pose_count} malformed pose annotation(s)"
                    )
            if output_format.startswith("YOLO"):
                stats.total_annotations += _write_yolo_label(
                    (labels_dir / split if pose_output else labels_dir) / f"{Path(filename).stem}.txt",
                    objects,
                    classes,
                    output_format,
                    int(img_w),
                    int(img_h),
                    pose_meta,
                )
            else:
                state = coco_splits.setdefault(
                    split,
                    {"images": [], "annotations": [], "next_annotation_id": 1},
                ) if pose_output else {
                    "images": coco_images,
                    "annotations": coco_annotations,
                    "next_annotation_id": next_annotation_id,
                }
                previous_id = int(state["next_annotation_id"])
                new_annotation_id = _append_coco_annotations(
                    state["images"],
                    state["annotations"],
                    objects,
                    classes,
                    str(Path("images") / split / filename) if pose_output else filename,
                    int(img_w),
                    int(img_h),
                    previous_id,
                    pose_meta,
                )
                state["next_annotation_id"] = new_annotation_id
                if pose_output:
                    coco_splits[split] = state
                else:
                    next_annotation_id = new_annotation_id
                stats.total_annotations += new_annotation_id - previous_id
            stats.converted_images += 1
            if progress_callback:
                progress_callback(completed, total_downloads, f"Converting rows... {completed}/{total_downloads}")

    class_names = [name for name, _ in sorted(classes.items(), key=lambda item: item[1])]
    stats.classes = class_names
    if output_format.startswith("YOLO"):
        data_yaml = {
            "path": str(output_dir.resolve()),
            "nc": len(class_names),
            "names": class_names,
        }
        if output_format == "YOLO Pose":
            for split in ("train", "val", "test"):
                if (images_dir / split).is_dir():
                    data_yaml[split] = f"images/{split}"
            data_yaml["task"] = "pose"
            data_yaml["kpt_shape"] = pose_meta["kpt_shape"]
            if pose_meta.get("flip_idx") is not None:
                data_yaml["flip_idx"] = pose_meta["flip_idx"]
        else:
            data_yaml["train"] = "images"
            data_yaml["task"] = "segment" if output_format == "YOLO Segmentation" else "detect"
        (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    else:
        categories = []
        for idx, name in enumerate(class_names):
            category: dict[str, Any] = {"id": idx + 1, "name": name}
            if pose_output:
                category["keypoints"] = pose_meta["kpt_names"]
                category["skeleton"] = pose_meta["skeleton"]
            categories.append(category)
        if pose_output:
            annotations_dir = output_dir / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)
            for split, state in sorted(coco_splits.items()):
                payload = {
                    "images": state["images"],
                    "annotations": state["annotations"],
                    "categories": categories,
                }
                (annotations_dir / f"instances_{split}.json").write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
        else:
            payload = {"images": coco_images, "annotations": coco_annotations, "categories": categories}
            (output_dir / "annotations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    stats.elapsed_seconds = time.time() - started
    manifest = stats.to_dict()
    manifest["max_workers"] = max_workers
    manifest["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (output_dir / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return stats


def _select_source() -> Path | None:
    clear_screen()
    print_stylized_header("Convert Dataset Format")
    detected = _discover_ndjson_files()
    labels = [_display_path(path) for path in detected]
    path_by_label = dict(zip(labels, detected, strict=True))
    choice = get_user_choice(
        [*labels, "Enter Manual Path", "Back"] if detected else ["Enter Manual Path", "Back"],
        allow_back=True,
        title="Select NDJSON Source",
        text="Choose a detected export or enter a path manually:",
        descriptions={label: _ndjson_description(path) for label, path in path_by_label.items()},
        breadcrumbs=["YOLOmatic", "Convert Dataset", "Source"],
        tip="Detected files include project-root .ndjson exports and any .ndjson under datasets/.",
        **_wizard_kwargs(0),
    )
    if choice in (NAV_BACK, "Back"):
        return None
    if choice != "Enter Manual Path":
        return path_by_label[choice]

    raw = get_parameter_value_input(
        ParameterDefinition(
            "ndjson_path",
            "source",
            "",
            "str",
            "Path to NDJSON file",
            "Enter an absolute path or a path relative to the project root.",
        ),
        "",
    )
    if raw in (None, NAV_BACK):
        return None
    path = Path(str(raw)).expanduser()
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".ndjson":
        console.print(Panel(f"[bold red]NDJSON file not found or invalid:[/bold red] {path}", border_style="red"))
        input("\nPress Enter to continue...")
        return None
    return path


def _select_output_format(source: Path) -> str | None:
    choice = get_user_choice(
        [*OUTPUT_FORMATS, "Back"],
        allow_back=True,
        title="Output Format",
        text="Choose the converted dataset format:",
        descriptions={
            "YOLO Detection": "Writes YOLO box labels and converts polygons to tight boxes.",
            "YOLO Segmentation": "Writes YOLO polygon labels and converts boxes to rectangle polygons.",
            "YOLO Pose": "Preserves Ultralytics pose boxes and keypoints in split-aware YOLO labels.",
            "COCO": "Writes images/ plus annotations.json with COCO instances.",
            "COCO Pose": "Writes split-aware COCO annotations with boxes, keypoints, and pose metadata.",
        },
        breadcrumbs=["YOLOmatic", "Convert Dataset", "Format"],
        status_fields={"Source": str(source)},
        **_wizard_kwargs(1),
    )
    return None if choice in (NAV_BACK, "Back") else choice


def _collect_output_settings(source: Path, output_format: str) -> tuple[Path, int, bool] | None:
    default_name = f"{slugify(source.stem)}_{slugify(output_format)}"
    raw_output = get_parameter_value_input(
        ParameterDefinition(
            "output_dir",
            "output",
            str(Path("datasets") / default_name),
            "str",
            "Output directory",
            "Converted files will be written here. Existing non-empty directories require overwrite.",
        ),
        str(Path("datasets") / default_name),
    )
    if raw_output in (None, NAV_BACK):
        return None
    raw_workers = get_parameter_value_input(
        ParameterDefinition(
            "max_workers",
            "performance",
            10,
            "int",
            "Download workers",
            "Concurrent image downloads for NDJSON rows.",
            min_value=1,
            max_value=64,
        ),
        10,
    )
    if raw_workers in (None, NAV_BACK):
        return None
    output_dir = Path(str(raw_output)).expanduser()
    overwrite = False
    if output_dir.exists() and any(output_dir.iterdir()):
        choice = get_user_choice(
            ["Choose Different Folder", "Overwrite Existing Folder", "Back"],
            title="Output Exists",
            text=f"{output_dir} already exists and is not empty.",
            descriptions={
                "Choose Different Folder": "Return to output settings and enter another path.",
                "Overwrite Existing Folder": "Delete the existing output directory before converting.",
            },
            breadcrumbs=["YOLOmatic", "Convert Dataset", "Output"],
            **_wizard_kwargs(2),
        )
        if choice in (NAV_BACK, "Back"):
            return None
        if choice == "Choose Different Folder":
            return _collect_output_settings(source, output_format)
        overwrite = True
    return output_dir, int(raw_workers), overwrite


def _run_with_progress(ndjson_path: Path, output_format: str, output_dir: Path, max_workers: int, overwrite: bool) -> ConversionStats | None:
    progress_state: dict[str, Any] = {"done": 0, "total": 0, "message": "Initializing..."}

    def callback(current: int, total: int, message: str) -> None:
        progress_state.update({"done": current, "total": total, "message": message})

    stats_holder: list[ConversionStats] = []
    errors: list[Exception] = []

    import threading

    done = threading.Event()

    def worker() -> None:
        try:
            stats_holder.append(
                convert_ndjson_to_format(
                    ndjson_path,
                    output_format,
                    output_dir,
                    max_workers=max_workers,
                    overwrite=overwrite,
                    progress_callback=callback,
                )
            )
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    with Live(refresh_per_second=4) as live:
        while not done.is_set():
            total = progress_state["total"]
            current = progress_state["done"]
            pct = f"{current / total * 100:.0f}%" if total else "..."
            live.update(
                Panel(
                    f"{progress_state['message']}\n\n[cyan]{current}/{total}[/cyan]",
                    title=f"[cyan]Converting Dataset [{pct}][/cyan]",
                    border_style="cyan",
                )
            )
            time.sleep(0.25)
    thread.join()
    if errors:
        console.print(Panel(f"[bold red]Dataset conversion failed:[/bold red]\n\n{type(errors[0]).__name__}: {errors[0]}", border_style="red"))
        return None
    return stats_holder[0]


def _render_result(stats: ConversionStats) -> None:
    try:
        elapsed = f"{float(stats.elapsed_seconds):.1f}s"
    except (TypeError, ValueError):
        elapsed = str(stats.elapsed_seconds)

    render_summary_panel(
        "Converted Dataset",
        {
            "Output Path": stats.output_path,
            "Output Format": stats.output_format,
            "Rows": stats.total_rows,
            "Images": stats.converted_images,
            "Annotations": stats.total_annotations,
            "Classes": f"{len(stats.classes)} ({', '.join(stats.classes[:6])}{'...' if len(stats.classes) > 6 else ''})",
            "Skipped": stats.skipped_rows,
            "Warnings": len(stats.warnings),
            "Time": elapsed,
        },
    )
    if stats.warnings:
        preview = "\n".join(f"- {warning}" for warning in stats.warnings[:8])
        if len(stats.warnings) > 8:
            preview += f"\n[dim]... and {len(stats.warnings) - 8} more in conversion_manifest.json[/dim]"
        console.print(Panel(preview, title="[yellow]Warnings[/yellow]", border_style="yellow"))


def main() -> None:
    source = _select_source()
    if source is None:
        return
    output_format = _select_output_format(source)
    if output_format is None:
        return
    output_settings = _collect_output_settings(source, output_format)
    if output_settings is None:
        return
    output_dir, max_workers, overwrite = output_settings

    clear_screen()
    print_stylized_header("Convert Dataset Format - Confirm")
    render_summary_panel(
        "Conversion Plan",
        {
            "Source": source,
            "Output Format": output_format,
            "Output": format_path(output_dir, max_chars=72),
            "Workers": max_workers,
            "Overwrite": "yes" if overwrite else "no",
        },
    )
    confirm = get_user_choice(
        ["Start Conversion", "Back"],
        allow_back=True,
        title="Confirm",
        text="Review the conversion plan:",
        breadcrumbs=["YOLOmatic", "Convert Dataset", "Confirm"],
        **_wizard_kwargs(3),
    )
    if confirm in (NAV_BACK, "Back"):
        return
    stats = _run_with_progress(source, output_format, output_dir, max_workers, overwrite)
    if stats is not None:
        _render_result(stats)
    input("\nPress Enter to return to main menu...")


if __name__ == "__main__":
    main()
