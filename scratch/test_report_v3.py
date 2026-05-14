import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Any
import numpy as np

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from benchmark.report import write_benchmark_report

@dataclass
class PredObject:
    conf: float
    cls: int
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None

@dataclass
class GTObject:
    cls: int
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    area: float = 0.0

@dataclass
class ImageResult:
    image_id: int
    image_path: Path
    task: str
    gt_count: int
    pred_count: int
    tp: int
    fp: int
    fn: int
    matched_ious: list[float]
    precision: float
    recall: float
    f1: float
    dominant_bucket: str = "medium"
    raw_preds: list[PredObject] = field(default_factory=list)
    raw_gts: list[GTObject] = field(default_factory=list)
    mean_iou: float = 0.0

@dataclass
class SizeBucketMetrics:
    name: str
    count: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    map50: float = 0.0

@dataclass
class ModelMetrics:
    weights_path: Path
    task: str
    precision: float
    recall: float
    f1: float
    map50: float
    map75: float
    map50_95: float
    small: SizeBucketMetrics
    medium: SizeBucketMetrics
    large: SizeBucketMetrics
    per_image: list[ImageResult]

@dataclass
class Config:
    validation_dir: Path = Path("datasets/val")
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    generate_thumbnails: bool = False
    max_thumbnail_size: int = 256

@dataclass
class BenchmarkResult:
    models: List[ModelMetrics]
    config: Config

def generate_fake_data():
    models = []
    # Generate 25 models to test scrolling
    for i in range(25):
        m_path = Path(f"runs/train/exp{i}/weights/best.pt")
        per_image = []
        for j in range(20):
            tp = np.random.randint(5, 20)
            fp = np.random.randint(0, 5)
            fn = np.random.randint(0, 5)
            gt_count = tp + fn
            pred_count = tp + fp
            
            per_image.append(ImageResult(
                image_id=j,
                image_path=Path(f"image_{j}.jpg"),
                task="detect",
                gt_count=gt_count,
                pred_count=pred_count,
                tp=tp,
                fp=fp,
                fn=fn,
                matched_ious=[0.8]*tp,
                precision=tp/(tp+fp) if (tp+fp)>0 else 0,
                recall=tp/(tp+fn) if (tp+fn)>0 else 0,
                f1=0.8,
                dominant_bucket="medium",
                raw_preds=[PredObject(0.9, 0, (0,0,10,10))],
                raw_gts=[GTObject(0, (0,0,10,10), area=100.0)],
                mean_iou=0.8
            ))
        
        models.append(ModelMetrics(
            weights_path=m_path,
            task="detect",
            precision=np.random.uniform(0.6, 0.95),
            recall=np.random.uniform(0.6, 0.95),
            f1=np.random.uniform(0.6, 0.95),
            map50=np.random.uniform(0.6, 0.95),
            map75=np.random.uniform(0.5, 0.8),
            map50_95=np.random.uniform(0.4, 0.7),
            small=SizeBucketMetrics(name="small", map50=np.random.uniform(0.3, 0.6)),
            medium=SizeBucketMetrics(name="medium", map50=np.random.uniform(0.6, 0.85)),
            large=SizeBucketMetrics(name="large", map50=np.random.uniform(0.8, 0.98)),
            per_image=per_image
        ))
    
    return BenchmarkResult(models=models, config=Config())

if __name__ == "__main__":
    result = generate_fake_data()
    output_dir = Path("output/fake_benchmark_v3")
    report_path = write_benchmark_report(result, output_dir)
    print(f"Report generated at: {report_path.absolute()}")
