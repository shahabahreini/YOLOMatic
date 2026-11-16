"""SAM 3.1 fine-tuning trainer using HuggingFace Trainer API."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.panel import Panel

from yolomatic.cli.result import CommandResult
from yolomatic.ui.theme import CONSOLE
from yolomatic.utils.logging_setup import log_exception
from yolomatic.utils.ml_dependencies import (
    MLDependencyError,
    check_hf_auth,
    import_sam_transformers,
)

console = CONSOLE
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

        with ann_file.open(encoding="utf-8") as f:
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
            polygons = [seg for seg in segs if isinstance(seg, list) and len(seg) >= 6] if isinstance(segs, list) else []
            if polygons:
                mask = self._polys_to_mask(polygons, w, h)
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
        pixel_values = inputs["pixel_values"].squeeze(0)
        input_boxes = inputs["input_boxes"].reshape(-1, 4).float()

        # Align ground truth with what the model sees: the processor resizes the
        # image (longest side) and pads it to a square, and SAM predicts masks for
        # that padded canvas. Reproduce the same resize + pad on the masks.
        padded_h, padded_w = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
        reshaped = inputs.get("reshaped_input_sizes")
        if reshaped is not None:
            resized_h, resized_w = (int(v) for v in torch.as_tensor(reshaped).reshape(-1)[:2])
        else:
            scale = min(padded_h / h, padded_w / w)
            resized_h, resized_w = round(h * scale), round(w * scale)
        gt_tensor = torch.zeros((len(gt_masks), padded_h, padded_w), dtype=torch.uint8)
        stacked = torch.as_tensor(np.stack(gt_masks).astype(np.uint8))[:, None].float()
        resized = torch.nn.functional.interpolate(stacked, size=(resized_h, resized_w), mode="nearest")
        gt_tensor[:, :resized_h, :resized_w] = resized[:, 0].to(torch.uint8)
        return {
            "pixel_values": pixel_values,
            "input_boxes": input_boxes,
            "ground_truth_masks": gt_tensor,
        }

    @staticmethod
    def _poly_to_mask(poly: list[float], w: int, h: int) -> np.ndarray:
        return _SAMDataset._polys_to_mask([poly], w, h)

    @staticmethod
    def _polys_to_mask(polys: list[list[float]], w: int, h: int) -> np.ndarray:
        """Rasterize every polygon part of an annotation (not just the first)."""
        from yolomatic.utils.ml_dependencies import import_cv2

        cv2 = import_cv2()  # raises MLDependencyError with an install hint
        mask = np.zeros((h, w), dtype=np.uint8)
        parts = [np.array(poly, dtype=np.float32).reshape(-1, 2).round().astype(np.int32) for poly in polys]
        cv2.fillPoly(mask, parts, 1)
        return mask.astype(bool)


_SPLIT_ALIASES = {"val": ("val", "valid", "validation"), "train": ("train",), "test": ("test",)}


def _find_ann_file(dataset_dir: Path, split: str) -> Path | None:
    """Locate the COCO annotation file for ``split``.

    Split-specific files win. A single root-level ``_annotations.coco.json`` is
    only used for training: it used to be returned for "val" as well, so the
    model was validated on its own training data. Roboflow's ``valid/`` folder
    is recognised for "val".
    """
    aliases = _SPLIT_ALIASES.get(split, (split,))
    candidates: list[Path] = []
    for name in aliases:
        candidates += [
            dataset_dir / name / "_annotations.coco.json",
            dataset_dir / "annotations" / f"instances_{name}.json",
            dataset_dir / "annotations" / f"instances_{name}2017.json",
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for name in aliases:
        folder = dataset_dir / name
        if folder.is_dir():
            found = sorted(folder.glob("*.json"))
            if found:
                return found[0]
    if split == "train":
        root_file = dataset_dir / "_annotations.coco.json"
        if root_file.exists():
            return root_file
        found = sorted(dataset_dir.glob("*.json"))
        return found[0] if found else None
    return None


def _find_image_dir(dataset_dir: Path, split: str) -> Path:
    for name in _SPLIT_ALIASES.get(split, (split,)):
        for candidate in (dataset_dir / name / "images", dataset_dir / name):
            if candidate.exists():
                return candidate
    for candidate in [dataset_dir / "images"]:
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
    return _freeze_modules(model, ("vision_encoder", "image_encoder"))


def _freeze_modules(model, name_markers: tuple[str, ...]) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        if any(marker in name for marker in name_markers):
            param.requires_grad = False
            frozen += param.numel()
    return frozen


# fine_tune_strategy -> parameter-name markers to freeze.
FINE_TUNE_STRATEGIES: dict[str, tuple[str, ...]] = {
    "decoder_only": ("vision_encoder", "image_encoder", "prompt_encoder"),
    "prompt_and_decoder": ("vision_encoder", "image_encoder"),
    "full": (),
}


def apply_fine_tune_strategy(model, config: dict) -> tuple[str, int]:
    """Freeze parameters according to ``model.fine_tune_strategy``.

    The strategy used to be printed but ignored (only the image encoder flag
    was honoured).
    """
    model_cfg = config.get("model", {})
    strategy = str(model_cfg.get("fine_tune_strategy") or "").strip().lower()
    if strategy not in FINE_TUNE_STRATEGIES:
        strategy = "prompt_and_decoder" if model_cfg.get("freeze_image_encoder", True) else "full"
    return strategy, _freeze_modules(model, FINE_TUNE_STRATEGIES[strategy])


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def _select_low_res_masks(pred_masks: torch.Tensor) -> torch.Tensor:
    """Return predicted mask logits as ``(B, N, H, W)``."""
    if pred_masks.ndim == 5:  # (B, N, num_multimask, H, W)
        return pred_masks[:, :, 0]
    if pred_masks.ndim == 4:
        return pred_masks
    raise ValueError(f"Unexpected pred_masks shape {tuple(pred_masks.shape)}")


def mask_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """BCE + Dice over ``(K, H, W)`` mask logits and binary targets."""
    if logits.numel() == 0:
        return logits.sum() * 0.0
    targets = targets.float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits).flatten(1)
    flat = targets.flatten(1)
    dice = 1 - (2 * (probs * flat).sum(1) + eps) / (probs.sum(1) + flat.sum(1) + eps)
    return bce + dice.mean()


def mask_iou(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = torch.sigmoid(logits) > 0.5
    truth = targets.bool()
    intersection = (preds & truth).float().sum((-2, -1))
    union = (preds | truth).float().sum((-2, -1))
    return intersection / union.clamp_min(1e-6)


def sam_batch_loss(model, inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Run SAM on a padded batch; return ``(loss, per-mask IoU, outputs)``."""
    gt = inputs["ground_truth_masks"]      # (B, Nmax, H, W) uint8 on the padded canvas
    valid = inputs["box_valid"]            # (B, Nmax) bool
    outputs = model(
        pixel_values=inputs["pixel_values"],
        input_boxes=inputs["input_boxes"],
        multimask_output=False,
    )
    logits = _select_low_res_masks(outputs.pred_masks)
    targets = torch.nn.functional.interpolate(gt.float(), size=logits.shape[-2:], mode="nearest")
    logits, targets = logits[valid], targets[valid]
    return mask_loss(logits, targets), mask_iou(logits.detach(), targets), outputs


def _compute_mask_metrics(eval_preds):
    """Compute mean IoU over predicted vs ground-truth masks."""
    logits, labels = eval_preds
    return {"mean_iou": mask_iou(torch.as_tensor(logits), torch.as_tensor(labels)).mean().item()}


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Pad boxes and masks to the largest object count in the batch.

    ``torch.stack`` on the raw boxes failed whenever images in a batch had
    different numbers of objects (any batch_size > 1 on real data).
    """
    max_boxes = max(item["input_boxes"].shape[0] for item in batch)
    height, width = batch[0]["ground_truth_masks"].shape[-2:]
    boxes = torch.zeros((len(batch), max_boxes, 4), dtype=torch.float32)
    boxes[..., 2:] = 1.0  # harmless dummy boxes for padding slots
    masks = torch.zeros((len(batch), max_boxes, height, width), dtype=torch.uint8)
    valid = torch.zeros((len(batch), max_boxes), dtype=torch.bool)
    for index, item in enumerate(batch):
        count = item["input_boxes"].shape[0]
        boxes[index, :count] = item["input_boxes"]
        masks[index, :count] = item["ground_truth_masks"]
        valid[index, :count] = True
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_boxes": boxes,
        "ground_truth_masks": masks,
        "box_valid": valid,
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def _make_hf_trainer_class():
    from transformers import Trainer

    class SAMMaskTrainer(Trainer):
        """HF Trainer with a SAM mask loss.

        Plain ``Trainer`` passed ``ground_truth_masks`` to ``model.forward``,
        which neither accepts it nor returns a loss, so training stopped on the
        first step.
        """

        _eval_ious: list[float]

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, ious, outputs = sam_batch_loss(model, inputs)
            if not model.training:
                self._eval_ious.extend(ious.tolist())
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            # The default step calls model(**inputs) when the batch has no
            # "labels", which SAM's forward rejects; evaluate through our loss.
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss = self.compute_loss(model, inputs)
            return loss.detach(), None, None

        def evaluate(self, *args, **kwargs):
            self._eval_ious = []
            metrics = super().evaluate(*args, **kwargs)
            if self._eval_ious:
                metrics["eval_mean_iou"] = float(np.mean(self._eval_ious))
                self.log({"eval_mean_iou": metrics["eval_mean_iou"]})
            return metrics

    return SAMMaskTrainer


class SAMFinetuneTrainer:
    """Fine-tunes SAM 3.1 mask decoder + prompt encoder using HuggingFace Trainer."""

    def __init__(self, config: dict):
        self.config = config

    def train(self) -> Path:
        from transformers import TrainingArguments

        model, processor = _load_sam_model_and_processor(self.config)

        strategy, frozen = apply_fine_tune_strategy(model, self.config)
        console.print(f"[dim]  Strategy {strategy}: froze {frozen:,} parameters.[/dim]")

        dataset_dir = Path(self.config["dataset"]["base_dir"])
        input_size = self.config["dataset"].get("input_size", 1008)

        train_ann = _find_ann_file(dataset_dir, "train")
        val_ann = _find_ann_file(dataset_dir, "val")
        if val_ann is not None and train_ann is not None and val_ann.resolve() == train_ann.resolve():
            val_ann = None

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
        if len(train_dataset) == 0:
            raise ValueError(f"No annotated training images found via {train_ann}.")

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
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            prediction_loss_only=True,
            logging_steps=50,
            report_to="none",
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )

        trainer = _make_hf_trainer_class()(
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

def main(config_file: str | Path) -> CommandResult:
    """Entry point called by yolo_trainer.py dispatch."""
    import yaml

    config_path = Path(config_file)
    if not config_path.exists():
        return CommandResult.failed(f"SAM config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

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

    try:
        output_dir = SAMFinetuneTrainer(config).train()
    except KeyboardInterrupt:
        return CommandResult.cancelled("SAM fine-tuning interrupted.")
    except MLDependencyError as exc:
        return CommandResult.failed(f"Dependency error: {exc}")
    except Exception as exc:  # noqa: BLE001
        log_path = log_exception(exc, "SAM fine-tuning")
        return CommandResult.failed(
            f"SAM fine-tuning failed: {exc}",
            hint=f"Full traceback in {log_path}" if log_path else None,
        )
    return CommandResult.ok("SAM fine-tuning complete.", outputs={"Checkpoint": output_dir})
