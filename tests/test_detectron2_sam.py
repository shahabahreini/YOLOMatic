"""Phase-6 regression tests: Detectron2 training/prediction and SAM fine-tuning/prediction."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Detectron2 (detectron2 itself is not installed in CI; use small fakes)
# ---------------------------------------------------------------------------

class _FakeCfg(SimpleNamespace):
    def __init__(self, meta_architecture: str = "GeneralizedRCNN") -> None:
        super().__init__(
            MODEL=SimpleNamespace(
                META_ARCHITECTURE=meta_architecture,
                ROI_HEADS=SimpleNamespace(NUM_CLASSES=80, SCORE_THRESH_TEST=0.05, BATCH_SIZE_PER_IMAGE=512),
                RETINANET=SimpleNamespace(NUM_CLASSES=80, SCORE_THRESH_TEST=0.05),
                DEVICE="cpu",
                WEIGHTS="",
            ),
            DATASETS=SimpleNamespace(TRAIN=(), TEST=()),
            DATALOADER=SimpleNamespace(NUM_WORKERS=0),
            SOLVER=SimpleNamespace(IMS_PER_BATCH=2, BASE_LR=0.1, MAX_ITER=1, CHECKPOINT_PERIOD=1),
            TEST=SimpleNamespace(EVAL_PERIOD=0),
            INPUT=SimpleNamespace(),
            OUTPUT_DIR="",
        )
        self.merged: list[str] = []

    def merge_from_file(self, path: str) -> None:
        self.merged.append(path)

    def dump(self) -> str:
        return f"MODEL:\n  META_ARCHITECTURE: {self.MODEL.META_ARCHITECTURE}\n"


class _FakeCatalog:
    def __init__(self) -> None:
        self.names: set[str] = set()

    def list(self) -> list[str]:
        return sorted(self.names)

    def remove(self, name: str) -> None:
        self.names.discard(name)


def _d2_symbols(meta_architecture: str = "GeneralizedRCNN") -> dict:
    catalog = _FakeCatalog()

    def register(name, _meta, _ann, _root):
        if name in catalog.names:
            raise AssertionError(f"Dataset '{name}' is already registered!")
        catalog.names.add(name)

    return {
        "model_zoo": SimpleNamespace(get_config_file=lambda p: p, get_checkpoint_url=lambda p: f"zoo://{p}"),
        "get_cfg": lambda: _FakeCfg(meta_architecture),
        "register_coco_instances": register,
        "DefaultTrainer": type("DefaultTrainer", (), {"__init__": lambda self, cfg: None}),
        "DatasetCatalog": catalog,
        "MetadataCatalog": _FakeCatalog(),
        "COCOEvaluator": lambda name, output_dir: ("evaluator", name, output_dir),
    }


def _d2_config(root: Path) -> dict:
    ann = root / "annotations" / "instances_train.json"
    ann.parent.mkdir(parents=True, exist_ok=True)
    ann.write_text("{}", encoding="utf-8")
    (root / "annotations" / "instances_val.json").write_text("{}", encoding="utf-8")
    return {
        "settings": {"auto_download_pretrained": True, "task": "detection"},
        "detectron2": {"config_path": "COCO-Detection/retinanet_R_50_FPN_3x.yaml", "weights_url": "x"},
        "dataset": {"name": "leaves", "prepared_dir": str(root), "classes": ["leaf", "stem"]},
        "training": {"output_dir": str(root / "runs"), "score_thresh_test": 0.4},
    }


class Detectron2TrainingTest(unittest.TestCase):
    def test_retinanet_class_count_and_idempotent_registration(self) -> None:
        from yolomatic.trainers import detectron2_trainer as d2

        symbols = _d2_symbols("RetinaNet")
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(d2, "_import_detectron2_training_symbols", return_value=symbols):
            config = _d2_config(Path(tmp))
            cfg = d2.build_detectron2_cfg(config)
            # Regression: RetinaNet kept its default 80 classes.
            self.assertEqual(cfg.MODEL.RETINANET.NUM_CLASSES, 2)
            self.assertEqual(cfg.MODEL.ROI_HEADS.NUM_CLASSES, 2)
            self.assertEqual(cfg.MODEL.RETINANET.SCORE_THRESH_TEST, 0.4)
            self.assertGreater(cfg.TEST.EVAL_PERIOD, 0)
            # Regression: building twice in one process raised "already registered".
            d2.build_detectron2_cfg(config)

    def test_trainer_has_coco_evaluator_and_run_metadata_is_written(self) -> None:
        from yolomatic.trainers import detectron2_trainer as d2

        trainer_cls = d2.make_trainer_class(_d2_symbols())
        cfg = _FakeCfg()
        cfg.OUTPUT_DIR = "/runs/x"
        evaluator = trainer_cls.build_evaluator(cfg, "val_ds")
        self.assertEqual(evaluator[:2], ("evaluator", "val_ds"))

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            d2.write_run_metadata(run_dir, cfg, _d2_config(Path(tmp)))
            meta = json.loads((run_dir / d2.RUN_METADATA_FILE).read_text(encoding="utf-8"))
            self.assertEqual(meta["classes"], ["leaf", "stem"])
            self.assertTrue((run_dir / "config.yaml").is_file())


class Detectron2PredictTest(unittest.TestCase):
    def test_predict_config_is_rebuilt_from_the_training_run(self) -> None:
        # Regression: prediction used a bare get_cfg() (wrong architecture), forced
        # CPU, ignored the threshold and discarded the outputs.
        from yolomatic.cli import predict
        from yolomatic.trainers import detectron2_trainer as d2

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "detectron2" / "2026-01-01"
            cfg = _FakeCfg("RetinaNet")
            d2.write_run_metadata(run_dir, cfg, _d2_config(Path(tmp)))
            weight = run_dir / "model_final.pth"
            weight.write_bytes(b"w")
            self.assertTrue(predict.is_detectron2_weight(weight))
            with patch.object(predict, "import_torch", return_value=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))):
                built, names = predict.build_detectron2_predict_cfg(weight, 0.6, lambda: _FakeCfg("RetinaNet"), None)
            self.assertEqual(built.merged, [str(run_dir / "config.yaml")])
            self.assertEqual(names, ["leaf", "stem"])
            self.assertEqual(built.MODEL.ROI_HEADS.SCORE_THRESH_TEST, 0.6)
            self.assertEqual(built.MODEL.RETINANET.SCORE_THRESH_TEST, 0.6)
            self.assertEqual(built.MODEL.WEIGHTS, str(weight))

    def test_rfdetr_checkpoint_is_not_mistaken_for_detectron2(self) -> None:
        from yolomatic.cli import predict

        self.assertFalse(predict.is_detectron2_weight(Path("/data/retinanet_notes/rfdetr/checkpoint.pth")))
        self.assertTrue(predict.is_detectron2_weight(Path("/x/runs/detectron2/old/model_final.pth")))

    def test_instances_are_serialised(self) -> None:
        from yolomatic.cli.predict import detectron2_instances_to_records

        instances = MagicMock()
        instances.has.side_effect = lambda name: True
        instances.pred_boxes.tensor.tolist.return_value = [[1.0, 2.0, 3.0, 4.0]]
        instances.scores.tolist.return_value = [0.9]
        instances.pred_classes.tolist.return_value = [1]
        records = detectron2_instances_to_records(instances, ["leaf", "stem"])
        self.assertEqual(records, [{"class_id": 1, "class_name": "stem", "score": 0.9, "box_xyxy": [1.0, 2.0, 3.0, 4.0]}])


# ---------------------------------------------------------------------------
# SAM fine-tuning
# ---------------------------------------------------------------------------

class _FakeSam(torch.nn.Module):
    """Minimal stand-in with SAM's call signature: predicts a mask per box."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_encoder = torch.nn.Conv2d(3, 1, 1)
        self.prompt_encoder = torch.nn.Linear(4, 4)
        self.mask_decoder = torch.nn.Conv2d(1, 1, 1)

    def forward(self, pixel_values, input_boxes, multimask_output=True):
        feats = self.mask_decoder(self.vision_encoder(pixel_values))           # (B, 1, H, W)
        low = torch.nn.functional.interpolate(feats, size=(8, 8))              # (B, 1, 8, 8)
        n = input_boxes.shape[1]
        scale = self.prompt_encoder(input_boxes).mean(-1)[..., None, None]     # (B, N, 1, 1)
        pred = low.expand(-1, n, -1, -1) + scale                                # (B, N, 8, 8)
        return SimpleNamespace(pred_masks=pred[:, :, None])                     # (B, N, 1, 8, 8)


class _FakeProcessor:
    def __call__(self, images, input_boxes, return_tensors):
        w, h = images.size
        scale = 16 / max(w, h)
        return {
            "pixel_values": torch.zeros((1, 3, 16, 16)),
            "input_boxes": torch.tensor(input_boxes, dtype=torch.float32) * scale,
            "reshaped_input_sizes": torch.tensor([[round(h * scale), round(w * scale)]]),
        }


def _coco_dataset(root: Path, split_dirs=("train", "valid")) -> None:
    from PIL import Image

    for split in split_dirs:
        folder = root / split
        folder.mkdir(parents=True)
        Image.new("RGB", (32, 16)).save(folder / "a.jpg")
        Image.new("RGB", (32, 16)).save(folder / "b.jpg")
        annotations = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 10, 8], "segmentation": [[2, 2, 12, 2, 12, 10, 2, 10]]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [1, 1, 5, 5], "segmentation": [[1, 1, 6, 1, 6, 6]]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [10, 2, 8, 8], "segmentation": []},
        ]
        (folder / "_annotations.coco.json").write_text(json.dumps({
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 32, "height": 16},
                {"id": 2, "file_name": "b.jpg", "width": 32, "height": 16},
            ],
            "annotations": annotations,
            "categories": [{"id": 1, "name": "leaf"}],
        }), encoding="utf-8")


class SamTrainerTest(unittest.TestCase):
    def test_annotation_lookup_uses_split_files_and_never_validates_on_train(self) -> None:
        from yolomatic.trainers.sam_trainer import _find_ann_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _coco_dataset(root)
            self.assertEqual(_find_ann_file(root, "train"), root / "train" / "_annotations.coco.json")
            self.assertEqual(_find_ann_file(root, "val"), root / "valid" / "_annotations.coco.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "_annotations.coco.json").write_text("{}", encoding="utf-8")
            self.assertEqual(_find_ann_file(root, "train"), root / "_annotations.coco.json")
            self.assertIsNone(_find_ann_file(root, "val"))

    def test_dataset_aligns_masks_and_collate_pads_variable_object_counts(self) -> None:
        from yolomatic.trainers.sam_trainer import _collate_fn, _SAMDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _coco_dataset(root, ("train",))
            dataset = _SAMDataset(root / "train", root / "train" / "_annotations.coco.json", _FakeProcessor())
            items = [dataset[0], dataset[1]]
        self.assertEqual(items[0]["ground_truth_masks"].shape, (1, 16, 16))
        self.assertEqual(items[1]["ground_truth_masks"].shape, (2, 16, 16))
        # 32x16 image -> resized 16x8 then padded: rows 8.. stay empty.
        self.assertEqual(int(items[0]["ground_truth_masks"][:, 8:].sum()), 0)
        batch = _collate_fn(items)  # regression: torch.stack failed on 1 vs 2 boxes
        self.assertEqual(batch["input_boxes"].shape, (2, 2, 4))
        self.assertEqual(batch["box_valid"].tolist(), [[True, False], [True, True]])

    def test_loss_is_differentiable_and_ignores_padding(self) -> None:
        from yolomatic.trainers.sam_trainer import mask_loss, sam_batch_loss

        perfect = mask_loss(torch.full((1, 4, 4), 20.0), torch.ones((1, 4, 4)))
        wrong = mask_loss(torch.full((1, 4, 4), -20.0), torch.ones((1, 4, 4)))
        self.assertLess(float(perfect), float(wrong))

        model = _FakeSam()
        inputs = {
            "pixel_values": torch.zeros((1, 3, 16, 16)),
            "input_boxes": torch.ones((1, 2, 4)),
            "ground_truth_masks": torch.ones((1, 2, 16, 16), dtype=torch.uint8),
            "box_valid": torch.tensor([[True, False]]),
        }
        loss, ious, _ = sam_batch_loss(model, inputs)
        loss.backward()
        self.assertEqual(ious.shape, (1,))
        self.assertIsNotNone(model.mask_decoder.weight.grad)

    def test_fine_tune_strategy_freezes_the_right_modules(self) -> None:
        from yolomatic.trainers.sam_trainer import apply_fine_tune_strategy

        model = _FakeSam()
        strategy, frozen = apply_fine_tune_strategy(model, {"model": {"fine_tune_strategy": "decoder_only"}})
        self.assertEqual(strategy, "decoder_only")
        self.assertGreater(frozen, 0)
        self.assertFalse(model.vision_encoder.weight.requires_grad)
        self.assertFalse(model.prompt_encoder.weight.requires_grad)
        self.assertTrue(model.mask_decoder.weight.requires_grad)

    def test_one_real_trainer_step(self) -> None:
        # Regression: plain HF Trainer could not compute a loss for SAM batches.
        from transformers import TrainingArguments

        from yolomatic.trainers.sam_trainer import _collate_fn, _make_hf_trainer_class, _SAMDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _coco_dataset(root)
            processor = _FakeProcessor()
            train = _SAMDataset(root / "train", root / "train" / "_annotations.coco.json", processor)
            val = _SAMDataset(root / "valid", root / "valid" / "_annotations.coco.json", processor)
            args = TrainingArguments(
                output_dir=str(root / "out"), max_steps=1, per_device_train_batch_size=2,
                per_device_eval_batch_size=2, report_to="none", remove_unused_columns=False,
                dataloader_num_workers=0, prediction_loss_only=True, use_cpu=True,
                save_strategy="no", eval_strategy="no", logging_steps=1,
            )
            trainer = _make_hf_trainer_class()(
                model=_FakeSam(), args=args, train_dataset=train, eval_dataset=val, data_collator=_collate_fn,
            )
            result = trainer.train()
            metrics = trainer.evaluate()
        self.assertTrue(np.isfinite(result.training_loss))
        self.assertIn("eval_loss", metrics)
        self.assertIn("eval_mean_iou", metrics)


# ---------------------------------------------------------------------------
# SAM prediction output
# ---------------------------------------------------------------------------

class SamOutputTest(unittest.TestCase):
    def test_coco_output_is_valid_and_keeps_classes(self) -> None:
        from PIL import Image

        from yolomatic.cli.sam_predict import SamOutputWriter, _save_outputs

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "images"
            for sub in ("a", "b"):  # same filename in two folders
                (source / sub).mkdir(parents=True)
                Image.new("RGB", (20, 10)).save(source / sub / "img.jpg")
            out = Path(tmp) / "out"
            writer = SamOutputWriter(out, source, {0: "leaf", 3: "fruit"})
            mask = np.zeros((10, 20), dtype=bool)
            mask[2:6, 3:9] = True
            _save_outputs(source / "a" / "img.jpg", [mask], writer, class_ids=[3])
            _save_outputs(source / "b" / "img.jpg", [mask, mask], writer, class_ids=[0, 3])
            coco_path = writer.close()
            coco = json.loads(coco_path.read_text(encoding="utf-8"))
            files = sorted(p.name for p in out.iterdir())

        self.assertEqual([img["id"] for img in coco["images"]], [1, 2])
        self.assertEqual([ann["id"] for ann in coco["annotations"]], [1, 2, 3])
        self.assertEqual([ann["image_id"] for ann in coco["annotations"]], [1, 2, 2])
        self.assertEqual([ann["category_id"] for ann in coco["annotations"]], [4, 1, 4])
        self.assertTrue(all(ann["segmentation"] and len(ann["segmentation"][0]) >= 6 for ann in coco["annotations"]))
        self.assertEqual({c["id"]: c["name"] for c in coco["categories"]}, {1: "leaf", 4: "fruit"})
        self.assertIn("a__img_sam.png", files)
        self.assertIn("b__img_sam.png", files)
        self.assertIn("a__img.txt", files)


if __name__ == "__main__":
    unittest.main()
