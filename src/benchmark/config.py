from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkConfig:
    weights: list[Path]
    validation_dir: Path
    annotations_file: Path | None = None
    output_dir: Path = field(default_factory=lambda: Path("output/benchmark_reports"))
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    device: str = "auto"
    generate_thumbnails: bool = True
    max_thumbnail_size: int = 224
    open_in_browser: bool = False
