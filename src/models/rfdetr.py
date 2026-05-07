from __future__ import annotations

from dataclasses import dataclass


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
}


RFDETR_FAMILY_KEYS = {"rfdetr", "rfdetr-seg"}


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
