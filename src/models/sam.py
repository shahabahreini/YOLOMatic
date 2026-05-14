from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SAMVariant:
    model: str
    hf_model_id: str
    task: str
    params_b: float
    description: str
    family: str = "sam3.1"


SAM_VARIANTS: dict[str, SAMVariant] = {
    "SAM 3.1": SAMVariant(
        model="SAM 3.1",
        hf_model_id="facebook/sam3.1",
        task="segmentation",
        params_b=0.873,
        description=(
            "SAM 3.1 with Object Multiplex — shared memory for simultaneous "
            "multi-object tracking at 7× faster throughput."
        ),
        family="sam3.1",
    ),
    "SAM 3": SAMVariant(
        model="SAM 3",
        hf_model_id="facebook/sam3",
        task="segmentation",
        params_b=0.848,
        description=(
            "SAM 3 base — open-vocabulary concept segmentation without "
            "Object Multiplex. Predecessor to SAM 3.1."
        ),
        family="sam3.1",
    ),
}


def is_sam_model(value: str | None) -> bool:
    return bool(value) and str(value) in SAM_VARIANTS


def get_sam_variant(model_choice: str) -> SAMVariant:
    try:
        return SAM_VARIANTS[model_choice]
    except KeyError as error:
        raise ValueError(f"Unknown SAM model variant: {model_choice}") from error


def sam_table_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in SAM_VARIANTS.values():
        rows.append(
            {
                "Model": variant.model,
                "Task": variant.task.title(),
                "Params (B)": variant.params_b,
                "HuggingFace ID": variant.hf_model_id,
                "Notes": "Gated — HF token required",
            }
        )
    return rows
