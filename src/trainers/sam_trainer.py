"""SAM 3.1 fine-tuning trainer using HuggingFace Trainer API."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel

from src.utils.ml_dependencies import MLDependencyError, check_hf_auth, import_sam_transformers

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _SAMDataset(torch.utils.data.Dataset):
    """COCO-format instance segmentation dataset for SAM fine-tuning.

    Each item yields: pixel_values, input_boxes, ground_truth_masks.
    Only images that have at least one annotation with a polygon mask are
    included.
    """

    def __init__(self, image_dir: Path, ann_file: Path, processor, input_size: int = 1008):
        self.image_dir = image_dir
        self.processor = processor
        self.input_size = input_size

        with ann_file.open() as f:
            coco = json.load(f)

        id_to_path: dict[int, Path] = {
            img["id"]: image_dir / img["file_name"]
            for img in coco.get("images", [])
        }
        id_to_size: dict[int, tuple[int, int]] = {
            img["id"]: (img["width"], img["height"])
            for img in coco.get("images", [])
        }
        id_to_anns: dict[int, list[dict]] = {img_id: [] for img_id in id_to_path}
        for ann in coco.get("annotations", []):
            if ann["image_id"] in id_to_anns:
                id_to_anns[ann["image_id"]].append(ann)

        self.samples: list[dict[str, Any]] = []
        for img_id, anns in id_to_anns.items():
            valid_anns = [a for a in anns if a.get("bbox") and len(a["bbox"]) == 4]
            if not valid_anns:
                continue
            self.samples.append({
                "path": id_to_path[img_id],
                "size": id_to_size.get(img_id, (640, 640)),
                "anns": valid_anns,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image as PILImage

        sample = self.samples[idx]
        img = PILImage.open(sample["path"]).convert("RGB")
        w, h = sample["size"]

        # Build ground-truth masks and boxes from annotations
        gt_masks: list[np.ndarray] = []
        boxes: list[list[float]] = []
        for ann in sample["anns"]:
            bbox = ann["bbox"]  # COCO format: x, y, w, h
            box_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            boxes.append(box_xyxy)

            segs = ann.get("segmentation", [])
            if segs and isinstance(segs[0], list) and len(segs[0]) >= 6:
                mask = self._poly_to_mask(segs[0], w, h)
            else:
                # Fall back to bounding-box mask
                mask = np.zeros((h, w), dtype=bool)
                x1, y1, x2, y2 = int(box_xyxy[0]), int(box_xyxy[1]), int(box_xyxy[2]), int(box_xyxy[3])
                mask[y1:y2, x1:x2] = True
            gt_masks.append(mask)

        # Processor encodes image + boxes for SAM
        inputs = self.processor(
            images=img,
            input_boxes=[[boxes]],
            return_tensors="pt",
        )

        gt_tensor = torch.as_tensor(np.stack(gt_masks), dtype=torch.float32)
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_boxes": inputs["input_boxes"].squeeze(0),
            "ground_truth_masks": gt_tensor,
            "original_sizes": inputs.get("original_sizes", torch.tensor([[h, w]])).squeeze(0),
        }

    @staticmethod
    def _poly_to_mask(poly: list[float], w: int, h: int) -> np.ndarray:
        try:
            import cv2
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            return mask.astype(bool)
        except ImportError:
            return np.zeros((h, w), dtype=bool)


def _find_ann_file(dataset_dir: Path, split: str) -> Path | None:
    """Locate COCO annotation file for a given split."""
    candidates = [
        dataset_dir / f"_annotations.coco.json",
        dataset_dir / f"{split}/_annotations.coco.json",
        dataset_dir / "annotations" / f"instances_{split}.json",
        dataset_dir / "annotations" / f"instances_{split}2017.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in (dataset_dir / split).glob("*.json") if (dataset_dir / split).exists() else []:
        return candidate
    for candidate in dataset_dir.glob("*.json"):
        return candidate
    return None


def _find_image_dir(dataset_dir: Path, split: str) -> Path:
    for candidate in [dataset_dir / split / "images", dataset_dir / split, dataset_dir / "images"]:
        if candidate.exists():
            return candidate
    return dataset_dir


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_sam_model_and_processor(config: dict) -> tuple:
    """Load Sam3TrackerModel and processor from HuggingFace Hub or local checkpoint."""
    transformers = import_sam_transformers()

    hf_token = check_hf_auth()
    if hf_token is None:
        raise MLDependencyError(
            "HuggingFace token not found.\n"
            "Set the HF_TOKEN environment variable or run 'huggingface-cli login' "
            "to authenticate before fine-tuning SAM 3.1."
        )

    model_id = config["model"]["base_model"]
    console.print(f"[dim]  Loading processor from {model_id}...[/dim]")
    processor = transformers.Sam3TrackerProcessor.from_pretrained(model_id, token=hf_token)
    console.print(f"[dim]  Loading model from {model_id}...[/dim]")
    model = transformers.Sam3TrackerModel.from_pretrained(model_id, token=hf_token)
    return model, processor


def _freeze_image_encoder(model) -> int:
    """Freeze SAM image encoder (ViT). Returns number of frozen parameters."""
    frozen = 0
    for name, param in model.named_parameters():
        if "vision_encoder" in name or "image_encoder" in name:
            param.requires_grad = False
            frozen += param.numel()
    return frozen


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_mask_metrics(eval_preds):
    """Compute mean IoU over predicted vs ground-truth masks."""
    logits, labels = eval_preds
    preds = torch.sigmoid(torch.tensor(logits)) > 0.5
    labels_t = torch.tensor(labels).bool()
    intersection = (preds & labels_t).float().sum((-2, -1))
    union = (preds | labels_t).float().sum((-2, -1))
    iou = (intersection / (union + 1e-6)).mean().item()
    return {"mean_iou": iou}


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate_fn(batch: list[dict]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_boxes": torch.stack([b["input_boxes"] for b in batch]),
        "ground_truth_masks": [b["ground_truth_masks"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SAMFinetuneTrainer:
    """Fine-tunes SAM 3.1 mask decoder + prompt encoder using HuggingFace Trainer."""

    def __init__(self, config: dict):
        self.config = config

    def train(self) -> Path:
        from transformers import TrainingArguments, Trainer

        model, processor = _load_sam_model_and_processor(self.config)

        if self.config["model"].get("freeze_image_encoder", True):
            frozen = _freeze_image_encoder(model)
            console.print(f"[dim]  Frozen {frozen:,} image-encoder parameters.[/dim]")

        dataset_dir = Path(self.config["dataset"]["base_dir"])
        input_size = self.config["dataset"].get("input_size", 1008)

        train_ann = _find_ann_file(dataset_dir, "train")
        val_ann = _find_ann_file(dataset_dir, "val")

        if train_ann is None:
            raise FileNotFoundError(
                f"No training annotation file found in {dataset_dir}. "
                "Expected COCO JSON under train/ or root."
            )

        train_dataset = _SAMDataset(_find_image_dir(dataset_dir, "train"), train_ann, processor, input_size)
        val_dataset = (
            _SAMDataset(_find_image_dir(dataset_dir, "val"), val_ann, processor, input_size)
            if val_ann is not None
            else None
        )

        console.print(
            f"  Dataset: {len(train_dataset)} training samples"
            + (f", {len(val_dataset)} validation samples" if val_dataset else "")
        )

        tr = self.config["training"]
        output_cfg = self.config["output"]
        output_dir = str(Path(output_cfg["output_dir"]) / output_cfg["run_name"])

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=tr["epochs"],
            per_device_train_batch_size=tr["batch_size"],
            per_device_eval_batch_size=tr["batch_size"],
            learning_rate=tr["learning_rate"],
            weight_decay=tr["weight_decay"],
            warmup_steps=tr["warmup_steps"],
            save_steps=tr["save_steps"],
            eval_steps=tr["save_steps"],
            max_grad_norm=tr["max_grad_norm"],
            fp16=tr.get("fp16", True) and torch.cuda.is_available(),
            eval_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=val_dataset is not None,
            logging_steps=50,
            report_to="none",
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=_collate_fn,
        )

        console.print("\n[bold cyan]Starting SAM 3.1 fine-tuning...[/bold cyan]")
        trainer.train()
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        return Path(output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_file: str | Path) -> None:
    """Entry point called by yolo_trainer.py dispatch."""
    import yaml

    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"SAM config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text())

    console.print(
        Panel(
            f"[bold]SAM 3.1 Fine-tuning[/bold]\n\n"
            f"Base model: [cyan]{config['model']['base_model']}[/cyan]\n"
            f"Dataset:    [cyan]{config['dataset']['base_dir']}[/cyan]\n"
            f"Epochs:     [cyan]{config['training']['epochs']}[/cyan]\n"
            f"Strategy:   [cyan]{config['model'].get('fine_tune_strategy', 'decoder_only')}[/cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    trainer = SAMFinetuneTrainer(config)
    try:
        output_dir = trainer.train()
        console.print(
            Panel(
                f"[bold green]Fine-tuning complete.[/bold green]\n\n"
                f"Checkpoint saved to:\n[cyan]{output_dir}[/cyan]",
                border_style="green",
                padding=(1, 2),
            )
        )
    except MLDependencyError as exc:
        console.print(
            Panel(
                f"[bold red]Dependency error:[/bold red] {exc}",
                border_style="red",
                padding=(1, 2),
            )
        )
        raise
