"""SAM 3.1 inference wizard — Auto, Text-prompted, and Box-prompted modes."""
from __future__ import annotations

import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rich.live import Live
from rich.panel import Panel

from src.utils.cli import (
    NAV_BACK,
    clear_screen,
    console,
    expected_error_panel,
    get_parameter_value_input,
    get_user_choice,
    print_stylized_header,
    render_summary_panel,
)
from src.utils.ml_dependencies import MLDependencyError, check_hf_auth, import_sam_transformers
from src.utils.project import is_sam_checkpoint, project_root

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

_HF_PRETRAINED = {
    "facebook/sam3.1  (pretrained — Object Multiplex)": "facebook/sam3.1",
    "facebook/sam3    (pretrained — base)": "facebook/sam3",
}

_MODE_DESCRIPTIONS = {
    "Auto (segment everything)": (
        "[bold cyan]Auto Segmentation[/bold cyan]\n\n"
        "SAM's built-in detector automatically finds and segments every object "
        "instance in each image — no prompts required.\n\n"
        "[dim]Best for:[/dim]  exploratory annotation, generating pseudo-labels, "
        "quality inspection of a new dataset."
    ),
    "Text-prompted": (
        "[bold cyan]Text-Prompted Segmentation[/bold cyan]\n\n"
        "Provide one or more concept labels (e.g. 'vegetation, tree, shrub'). "
        "SAM 3.1's open-vocabulary DETR detector finds all instances of those "
        "concepts and segments them.\n\n"
        "[dim]Best for:[/dim]  targeted extraction of specific object classes "
        "across a large image batch."
    ),
    "Box-prompted (from YOLO detections)": (
        "[bold cyan]Box-Prompted Segmentation[/bold cyan]\n\n"
        "Point to a folder of YOLO detection .txt files. Each bounding box is "
        "converted to a pixel-space prompt and passed to SAM, which produces a "
        "precise instance mask.\n\n"
        "[dim]Best for:[/dim]  upgrading an existing detection dataset to "
        "segmentation-quality masks. Pairs well with YOLOv11 or YOLO26 detectors."
    ),
}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _ensure_hf_auth() -> str | None:
    """Check for a HuggingFace token; guide user to authenticate if missing."""
    token = check_hf_auth()
    if token:
        return token

    clear_screen()
    print_stylized_header("SAM 3.1 — HuggingFace Authentication")
    console.print(Panel(
        "[bold yellow]HuggingFace Authentication Required[/bold yellow]\n\n"
        "SAM 3.1 is a [italic]gated[/italic] model. To access it you must:\n\n"
        "  [bold]1.[/bold]  Create a free account at  huggingface.co\n"
        "  [bold]2.[/bold]  Accept Meta's terms at    huggingface.co/facebook/sam3.1\n"
        "  [bold]3.[/bold]  Create an access token at huggingface.co/settings/tokens\n\n"
        "Then either set  [bold cyan]HF_TOKEN=hf_xxxx[/bold cyan]  in your shell "
        "environment, or run  [bold cyan]huggingface-cli login[/bold cyan], "
        "or enter the token below.",
        border_style="yellow",
        padding=(1, 2),
    ))

    raw = get_parameter_value_input(
        name="HF_TOKEN",
        current_value="",
        value_type="str",
        description="Paste your HuggingFace access token (starts with hf_)",
    )
    if raw is None or raw == NAV_BACK:
        return None
    token_str = str(raw).strip()
    if token_str.startswith("hf_"):
        os.environ["HF_TOKEN"] = token_str
        return token_str

    console.print("[yellow]  Token does not start with 'hf_' — skipping.[/yellow]")
    return None


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def _find_local_sam_checkpoints() -> list[Path]:
    root = project_root()
    sam_dir = root / "runs" / "sam3.1"
    if not sam_dir.exists():
        return []
    return sorted(
        [d for d in sam_dir.rglob("checkpoint-*") if is_sam_checkpoint(d)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _select_model(token: str) -> str | None:
    """Return a HuggingFace model ID or local checkpoint path string."""
    root = project_root()
    options: list[str] = list(_HF_PRETRAINED.keys())
    descriptions: dict[str, str] = {
        "facebook/sam3.1  (pretrained — Object Multiplex)": (
            "[bold cyan]SAM 3.1[/bold cyan]  [green]● Latest[/green]\n\n"
            "873M parameters. Object Multiplex shared memory — 7× faster "
            "multi-object throughput compared to SAM 3.\n\n"
            "[dim]Downloads ~1.7 GB from HuggingFace on first use.[/dim]"
        ),
        "facebook/sam3    (pretrained — base)": (
            "[bold cyan]SAM 3[/bold cyan]\n\n"
            "848M parameters. Predecessor to SAM 3.1 without Object Multiplex."
        ),
    }

    local_ckpts = _find_local_sam_checkpoints()
    for ckpt in local_ckpts[:5]:
        try:
            label = f"Local: {ckpt.relative_to(root)} (fine-tuned)"
        except ValueError:
            label = f"Local: {ckpt.name} (fine-tuned)"
        options.append(label)
        descriptions[label] = (
            f"[bold cyan]{ckpt.name}[/bold cyan]  [green]● Local fine-tune[/green]\n\n"
            f"[dim]Path:[/dim]  {ckpt}\n"
            f"[dim]Modified:[/dim]  "
            + datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        )

    options.append("Enter custom HuggingFace model ID...")
    descriptions["Enter custom HuggingFace model ID..."] = (
        "Type any HuggingFace model ID such as 'facebook/sam3.1' "
        "or the path to a local fine-tuned checkpoint directory."
    )

    clear_screen()
    choice = get_user_choice(
        options,
        allow_back=True,
        title="Select SAM Model",
        text="Choose a pretrained model or a locally fine-tuned checkpoint:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "SAM Segment", "Model"],
    )
    if choice in (NAV_BACK, "Back"):
        return None

    if choice in _HF_PRETRAINED:
        return _HF_PRETRAINED[choice]

    if choice == "Enter custom HuggingFace model ID...":
        raw = get_parameter_value_input(
            name="model_id",
            current_value="facebook/sam3.1",
            value_type="str",
            description="HuggingFace model ID or local path",
        )
        if raw is None or raw == NAV_BACK:
            return None
        return str(raw).strip()

    # Local checkpoint — extract path
    for ckpt in local_ckpts:
        try:
            label = f"Local: {ckpt.relative_to(root)} (fine-tuned)"
        except ValueError:
            label = f"Local: {ckpt.name} (fine-tuned)"
        if choice == label:
            return str(ckpt)
    return None


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

def _select_mode() -> str | None:
    modes = list(_MODE_DESCRIPTIONS.keys())
    clear_screen()
    choice = get_user_choice(
        modes,
        allow_back=True,
        title="Select Inference Mode",
        text="How do you want to prompt SAM?",
        descriptions=_MODE_DESCRIPTIONS,
        breadcrumbs=["YOLOmatic", "SAM Segment", "Mode"],
    )
    return None if choice in (NAV_BACK, "Back") else choice


# ---------------------------------------------------------------------------
# Mode-specific configuration
# ---------------------------------------------------------------------------

def _configure_text_mode() -> list[str] | None:
    """Return a list of concept labels to segment."""
    raw = get_parameter_value_input(
        name="concepts",
        current_value="vegetation",
        value_type="str",
        description="Comma-separated concept labels (e.g. 'vegetation, tree, shrub')",
    )
    if raw is None or raw == NAV_BACK:
        return None
    return [c.strip() for c in str(raw).split(",") if c.strip()]


def _configure_box_mode() -> Path | None:
    """Return path to directory containing YOLO detection .txt files."""
    raw = get_parameter_value_input(
        name="detections_dir",
        current_value="",
        value_type="str",
        description="Path to folder containing YOLO detection .txt files",
    )
    if raw is None or raw == NAV_BACK:
        return None
    p = Path(str(raw).strip())
    if not p.exists():
        console.print(f"[yellow]  Directory not found: {p}[/yellow]")
        return None
    return p


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

def _select_source() -> Path | None:
    source_choice = get_user_choice(
        ["Single Image", "Folder (batch)"],
        allow_back=True,
        title="Select Input Source",
        text="Process a single image or an entire folder?",
        descriptions={
            "Single Image": "Run SAM on one image file.",
            "Folder (batch)": "Run SAM on every image in a directory.",
        },
        breadcrumbs=["YOLOmatic", "SAM Segment", "Source"],
    )
    if source_choice in (NAV_BACK, "Back"):
        return None

    raw = get_parameter_value_input(
        name="source_path",
        current_value="",
        value_type="str",
        description=(
            "Path to image file" if source_choice == "Single Image"
            else "Path to folder containing images"
        ),
    )
    if raw is None or raw == NAV_BACK:
        return None
    p = Path(str(raw).strip())
    if not p.exists():
        console.print(f"[yellow]  Path not found: {p}[/yellow]")
        return None
    return p


# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------

def _configure_output() -> Path | None:
    default = str(project_root() / "output" / "sam_results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    raw = get_parameter_value_input(
        name="output_dir",
        current_value=default,
        value_type="str",
        description="Directory where PNG overlays, COCO JSON, and YOLO .txt files will be saved",
    )
    if raw is None or raw == NAV_BACK:
        return None
    return Path(str(raw).strip())


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def _masks_to_yolo_txt(masks: list[np.ndarray], img_w: int, img_h: int) -> list[str]:
    """Convert binary masks to YOLO segmentation polygon lines (class 0)."""
    lines: list[str] = []
    for mask in masks:
        try:
            import cv2
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            pts = max(contours, key=cv2.contourArea).reshape(-1, 2)
            if len(pts) < 3:
                continue
            norm = [f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in pts]
            lines.append("0 " + " ".join(norm))
        except ImportError:
            pass
    return lines


def _save_outputs(
    image_path: Path,
    masks: list[np.ndarray],
    output_dir: Path,
    texts: list[str] | None = None,
) -> None:
    """Save PNG overlay, append to COCO JSON, write YOLO .txt."""
    from PIL import Image as PILImage, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    img = PILImage.open(image_path).convert("RGB")
    img_w, img_h = img.size

    coco_anns: list[dict] = []
    for i, mask in enumerate(masks):
        if mask.dtype != bool:
            mask = mask > 0.5
        mask_bool = mask.astype(bool)
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            continue

        # Build overlay layer
        colour = (50, 205, 50, 80) if texts is None else (30, 144, 255, 80)
        overlay = PILImage.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        ov_arr = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        ov_arr[ys, xs] = colour
        overlay = PILImage.fromarray(ov_arr, "RGBA")
        img = img.convert("RGBA")
        img = PILImage.alpha_composite(img, overlay).convert("RGB")

        area = int(mask_bool.sum())
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        coco_anns.append({
            "id": i + 1,
            "image_id": image_path.stem,
            "category_id": 0,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "area": area,
            "iscrowd": 0,
            "segmentation": [],
        })

    stem = image_path.stem
    img.save(output_dir / f"{stem}_sam.png")

    # Append to COCO JSON
    coco_file = output_dir / "annotations.coco.json"
    if coco_file.exists():
        existing = json.loads(coco_file.read_text())
    else:
        existing = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "object"}],
        }
    existing["images"].append({
        "id": image_path.stem, "file_name": image_path.name,
        "width": img_w, "height": img_h,
    })
    existing["annotations"].extend(coco_anns)
    coco_file.write_text(json.dumps(existing, indent=2))

    # YOLO txt
    valid_masks = [m.astype(bool) for m in masks if m.astype(bool).sum() > 0]
    yolo_lines = _masks_to_yolo_txt(valid_masks, img_w, img_h)
    if yolo_lines:
        (output_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))


# ---------------------------------------------------------------------------
# Inference runners
# ---------------------------------------------------------------------------

def _load_model_and_processor(model_id: str, token: str):
    transformers = import_sam_transformers()
    console.print(f"[dim]  Loading SAM model: {model_id}[/dim]")
    local_path = Path(model_id)
    load_kw: dict = {} if local_path.exists() else {"token": token}
    model = transformers.Sam3TrackerModel.from_pretrained(model_id, **load_kw)
    processor = transformers.Sam3TrackerProcessor.from_pretrained(model_id, **load_kw)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device


def _run_auto(image_path: Path, model, processor, device: str, output_dir: Path) -> int:
    from transformers import pipeline as hf_pipeline
    generator = hf_pipeline(
        "mask-generation",
        model=model,
        feature_extractor=processor,
        device=0 if device == "cuda" else -1,
    )
    outputs = generator(str(image_path), points_per_batch=64)
    masks = [np.array(m) for m in outputs.get("masks", [])]
    _save_outputs(image_path, masks, output_dir)
    return len(masks)


def _run_text(image_path: Path, texts: list[str], model, processor, device: str, output_dir: Path) -> int:
    from PIL import Image as PILImage
    img = PILImage.open(image_path).convert("RGB")
    inputs = processor(images=img, text=texts, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    masks_tensor = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )
    masks = []
    if masks_tensor:
        m = masks_tensor[0]
        scores = outputs.iou_scores.cpu()
        for obj_idx in range(m.shape[0]):
            best = int(scores[0, obj_idx].argmax()) if scores.ndim >= 3 else 0
            masks.append(m[obj_idx, best].numpy().astype(bool))
    _save_outputs(image_path, masks, output_dir, texts=texts)
    return len(masks)


def _load_yolo_boxes(txt_path: Path, img_w: int, img_h: int) -> list[list[float]]:
    """Load YOLO detection .txt and return xyxy pixel boxes."""
    boxes: list[list[float]] = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append([x1, y1, x2, y2])
        except ValueError:
            continue
    return boxes


def _run_box(
    image_path: Path,
    detections_dir: Path,
    model,
    processor,
    device: str,
    output_dir: Path,
) -> int:
    from PIL import Image as PILImage
    img = PILImage.open(image_path).convert("RGB")
    img_w, img_h = img.size
    txt_path = detections_dir / (image_path.stem + ".txt")
    boxes = _load_yolo_boxes(txt_path, img_w, img_h)
    if not boxes:
        return 0
    inputs = processor(images=img, input_boxes=[[boxes]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    masks_tensor = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )
    masks = []
    if masks_tensor:
        m = masks_tensor[0]
        scores = outputs.iou_scores.cpu()
        for obj_idx in range(m.shape[0]):
            best = int(scores[0, obj_idx].argmax()) if scores.ndim >= 3 else 0
            masks.append(m[obj_idx, best].numpy().astype(bool))
    _save_outputs(image_path, masks, output_dir)
    return len(masks)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _run_batch(
    model_id: str,
    mode: str,
    mode_config: object,
    source: Path,
    output_dir: Path,
    token: str,
) -> None:
    images: list[Path] = (
        [source] if source.is_file()
        else sorted(p for p in source.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    )
    if not images:
        console.print("[yellow]  No images found.[/yellow]")
        return

    model, processor, device = _load_model_and_processor(model_id, token)
    total = len(images)
    total_masks = 0
    errors = 0
    log_lines: list[str] = []
    done = threading.Event()

    def _worker() -> None:
        nonlocal total_masks, errors
        for i, img_path in enumerate(images, 1):
            try:
                if mode == "Auto (segment everything)":
                    n = _run_auto(img_path, model, processor, device, output_dir)
                elif mode == "Text-prompted":
                    n = _run_text(img_path, mode_config, model, processor, device, output_dir)
                else:
                    n = _run_box(img_path, mode_config, model, processor, device, output_dir)
                total_masks += n
                log_lines.append(f"  [{i}/{total}] {img_path.name} → {n} masks")
            except Exception as exc:
                errors += 1
                log_lines.append(f"  [{i}/{total}] [red]ERROR[/red] {img_path.name}: {exc}")
        done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    with Live(refresh_per_second=4) as live:
        while not done.is_set():
            lines = log_lines[-20:]
            live.update(Panel(
                "\n".join(lines) or "[dim]Starting...[/dim]",
                title="SAM Inference",
                border_style="cyan",
            ))
            time.sleep(0.25)
        # Final update
        lines = log_lines[-20:]
        live.update(Panel(
            "\n".join(lines) or "[dim]Done.[/dim]",
            title="SAM Inference",
            border_style="cyan",
        ))
    thread.join()

    console.print(
        Panel(
            f"[bold green]Done.[/bold green]  "
            f"{total_masks} masks generated across {total} images.  "
            f"{errors} errors.\n\n"
            f"[dim]Outputs saved to:[/dim]  {output_dir}",
            border_style="green",
            padding=(1, 2),
        )
    )


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------

def _confirm(model_id: str, mode: str, mode_config: object, source: Path, output_dir: Path) -> bool:
    rows: dict[str, str] = {
        "Model": model_id,
        "Mode": mode,
        "Source": str(source),
        "Output Dir": str(output_dir),
    }
    if mode == "Text-prompted" and isinstance(mode_config, list):
        rows["Concepts"] = ", ".join(mode_config)
    elif mode.startswith("Box-prompted") and mode_config is not None:
        rows["Detections Dir"] = str(mode_config)

    clear_screen()
    print_stylized_header("SAM Segment — Confirm")
    console.print(render_summary_panel(rows, title="Ready to Run"))

    choice = get_user_choice(
        ["Start SAM Inference", "Cancel"],
        title="Confirm",
        text="Review settings above, then start:",
        descriptions={
            "Start SAM Inference": "[green]Begin segmentation.[/green]",
            "Cancel": "[dim]Return to main menu.[/dim]",
        },
        breadcrumbs=["YOLOmatic", "SAM Segment", "Confirm"],
    )
    return choice == "Start SAM Inference"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    clear_screen()
    print_stylized_header("SAM 3.1 Segmentation")

    token = _ensure_hf_auth()
    if not token:
        console.print("[yellow]  Authentication cancelled. Returning to main menu.[/yellow]")
        input("\nPress Enter to continue...")
        return

    model_id = _select_model(token)
    if model_id is None:
        return

    mode = _select_mode()
    if mode is None:
        return

    mode_config: object = None
    if mode == "Text-prompted":
        mode_config = _configure_text_mode()
        if mode_config is None:
            return
    elif mode.startswith("Box-prompted"):
        mode_config = _configure_box_mode()
        if mode_config is None:
            return

    source = _select_source()
    if source is None:
        return

    output_dir = _configure_output()
    if output_dir is None:
        return

    if not _confirm(model_id, mode, mode_config, source, output_dir):
        return

    try:
        _run_batch(model_id, mode, mode_config, source, output_dir, token)
    except MLDependencyError as exc:
        console.print(Panel(
            f"[bold red]Dependency error:[/bold red] {exc}",
            border_style="red", padding=(1, 2),
        ))
    except Exception as exc:
        console.print(Panel(
            f"[bold red]Error:[/bold red] {exc}",
            border_style="red", padding=(1, 2),
        ))

    input("\nPress Enter to return to main menu...")
