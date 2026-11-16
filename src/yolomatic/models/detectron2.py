from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detectron2Variant:
    model: str
    config_path: str
    task: str
    weights_url: str
    family: str


DETECTRON2_VARIANTS: dict[str, Detectron2Variant] = {
    "Faster R-CNN R50-FPN 3x": Detectron2Variant(
        "Faster R-CNN R50-FPN 3x",
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "detection",
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "detectron2",
    ),
    "RetinaNet R50-FPN 3x": Detectron2Variant(
        "RetinaNet R50-FPN 3x",
        "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
        "detection",
        "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
        "detectron2",
    ),
    "Mask R-CNN R50-FPN 3x": Detectron2Variant(
        "Mask R-CNN R50-FPN 3x",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "segmentation",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "detectron2-seg",
    ),
}


def is_detectron2_model(value: str | None) -> bool:
    return bool(value) and str(value) in DETECTRON2_VARIANTS


def get_detectron2_variant(model_choice: str) -> Detectron2Variant:
    try:
        return DETECTRON2_VARIANTS[model_choice]
    except KeyError as error:
        raise ValueError(f"Unknown Detectron2 model variant: {model_choice}") from error


def detectron2_table_rows(task: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in DETECTRON2_VARIANTS.values():
        if task is not None and variant.task != task:
            continue
        rows.append(
            {
                "Model": variant.model,
                "Task": variant.task.title(),
                "Backbone": "R50-FPN",
                "Config": variant.config_path,
            }
        )
    return rows
