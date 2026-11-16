"""Long-form model family descriptions shown in the Configure Model menu."""
from __future__ import annotations

MODEL_DESCRIPTIONS: dict[str, str] = {
    "yolo26-sem": (
            "[bold cyan]YOLO26 Semantic[/bold cyan]  [green]● Per-pixel segmentation — 2025[/green]\n\n"
            "Dense [bold]per-pixel[/bold] class prediction — no boxes, no instances. "
            "Pick this over [bold]-seg[/bold] when you want one mask for the whole "
            "class (canopy cover, land cover) rather than separately counted objects.\n\n"
            "Trains on ordinary YOLO polygon labels: they are rasterized into masks "
            "on the fly, so an existing [bold]-seg[/bold] dataset works unchanged. "
            "Loss is BCE/CE + Dice; the metric is [bold]mIoU[/bold], not mAP.\n\n"
            "Note the head predicts at stride 8 and upsamples 8x, so fine boundaries "
            "benefit from a larger [bold]imgsz[/bold] more than from a larger model. "
            "Weights: [dim]yolo26<n/s/m/l/x>-sem.pt[/dim] (Cityscapes-pretrained)."
        ),
        "yolo26-pose": (
            "[bold cyan]YOLO26 Pose[/bold cyan]  [green]● Keypoint detection — 2025[/green]\n\n"
            "Latest-generation pose estimation. Trains on YOLO keypoint datasets "
            "(data.yaml with [bold]kpt_shape[/bold] + per-object keypoints). "
            "Weights: [dim]yolo26<n/s/m/l/x>-pose.pt[/dim]."
        ),
        "yolov11-pose": (
            "[bold cyan]YOLO11 Pose[/bold cyan]  [green]● Keypoint detection[/green]\n\n"
            "Mature, well-supported pose family with official COCO-Pose weights "
            "across n/s/m/l/x. Solid default for custom keypoint datasets."
        ),
        "yolov8-pose": (
            "[bold cyan]YOLOv8 Pose[/bold cyan]  [green]● Keypoint detection[/green]\n\n"
            "Stable, widely-deployed pose family. Choose when you want maximum "
            "ecosystem/tooling compatibility for keypoint models."
        ),
        "detectron2": (
            "[bold cyan]Detectron2[/bold cyan]  [green]● Optional native COCO detection[/green]\n\n"
            "Faster R-CNN and RetinaNet variants using Detectron2's model zoo. "
            "Detectron2 is imported only when you train or predict with this family."
        ),
        "sam3.1": (
            "[bold cyan]SAM 3.1[/bold cyan]  [green]● Foundation Segmentation — 2025[/green]\n\n"
            "[bold]Architecture[/bold]\n"
            "  • Meta's Segment Anything Model 3.1 (Object Multiplex)\n"
            "  • Unified DETR detector + memory-based tracker\n"
            "  • Shared memory for simultaneous multi-object segmentation\n"
            "  • 7× faster multi-object throughput vs SAM 3\n\n"
            "[bold]Capabilities[/bold]\n"
            "  • Open-vocabulary text prompting ('vegetation', 'person')\n"
            "  • Auto mask generation — segment everything without prompts\n"
            "  • Point and bounding box prompted segmentation\n"
            "  • Video object segmentation and multi-object tracking\n\n"
            "[bold]Parameters[/bold]  873M\n"
            "[bold]Input Size[/bold]  1008×1008\n"
            "[bold]HuggingFace[/bold]  facebook/sam3.1\n\n"
            "[dim]Gated model — requires HuggingFace account and "
            "Meta's terms agreement.[/dim]\n\n"
            "[bold]Best for[/bold]\n"
            "  High-quality instance masks, zero-shot segmentation, "
            "annotation generation for new domains"
        ),
        "detectron2-seg": (
            "[bold cyan]Detectron2 Segmentation[/bold cyan]  [green]● Optional native COCO masks[/green]\n\n"
            "Mask R-CNN instance segmentation with COCO annotations. YOLO polygon "
            "datasets are converted into cached COCO manifests when needed."
        ),
        "rfdetr": (
            "[bold cyan]RF-DETR[/bold cyan]  [green]● Transformer detection[/green]\n\n"
            "Real-time DETR-style object detection with automatic pretrained "
            "weight download. Core models are Apache-2.0; XL and 2XL require "
            "RF-DETR Plus licensing."
        ),
        "rfdetr-seg": (
            "[bold cyan]RF-DETR-Seg[/bold cyan]  [green]● Transformer segmentation[/green]\n\n"
            "Instance segmentation variants using RF-DETR's segmentation model "
            "classes. Pretrained weights are downloaded automatically on first use."
        ),
        "rfdetr-pose": (
            "[bold cyan]RF-DETR-Keypoint[/bold cyan]  [yellow]● Transformer pose (preview)[/yellow]\n\n"
            "[bold yellow]Preview architecture[/bold yellow] (RFDETRKeypointPreview), "
            "pretrained on COCO person keypoints. Training requires [bold]COCO "
            "keypoint JSON[/bold]; YOLO pose datasets are auto-converted to cached "
            "COCO manifests. Pretrained weights download automatically on first use."
        ),
        "yolo26": (
            "[bold cyan]YOLO26[/bold cyan]  [green]● Latest — 2026[/green]\n\n"
            "[bold]Architecture[/bold]\n"
            "  • End-to-end NMS-free inference (no post-processing step)\n"
            "  • DFL removed — simpler export, wider edge compatibility\n"
            "  • MuSGD optimizer (hybrid SGD + Muon, inspired by LLM training)\n"
            "  • ProgLoss + STAL loss for better small-object accuracy\n\n"
            "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
            "  • mAP:    40.9 (nano)  →  57.5 (xlarge)\n"
            "  • Params: 2.4M (nano)  →  55.7M (xlarge)\n"
            "  • Speed:  1.7 ms T4 TensorRT (nano)  |  CPU ONNX not published\n\n"
            "[bold]Best for[/bold]\n"
            "  Edge devices, IoT, robotics, CPU-only and mobile deployments"
        ),
        "yolo26-seg": (
            "[bold cyan]YOLO26-Seg[/bold cyan]  [green]● Latest — 2026[/green]\n\n"
            "[bold]Architecture[/bold]\n"
            "  • Same NMS-free, DFL-removed, MuSGD base as YOLO26\n"
            "  • Instance segmentation head for pixel-level boundary detection\n"
            "  • Edge-optimized — fast CPU ONNX inference among seg models\n\n"
            "[bold]Benchmarks[/bold]  [dim](COCO val2017, segmentation)[/dim]\n"
            "  • mAP box: 33.9 (nano)  →  47.0 (xlarge)\n"
            "  • Params:  2.7M (nano)  →  62.8M (xlarge)\n"
            "  • Speed:   53 ms CPU ONNX (nano)  |  2.1 ms T4 TensorRT (nano)\n\n"
            "[bold]Best for[/bold]\n"
            "  Pixel-level detection on resource-constrained or edge hardware"
        ),
        "yolov12": (
            "[bold cyan]YOLOv12[/bold cyan]  [yellow]● Research — 2025[/yellow]\n\n"
            "[bold]Architecture[/bold]\n"
            "  • Area Attention mechanism — large receptive field, attention-based\n"
            "  • R-ELAN (Residual Efficient Layer Aggregation Networks)\n"
            "  • Optional FlashAttention for memory-efficient training\n"
            "  • Higher peak accuracy than YOLO11 at cost of stability\n\n"
            "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
            "  • mAP:    40.6 (nano)  →  55.2 (xlarge)\n"
            "  • Params: 2.6M (nano)  →  59.1M (xlarge)\n"
            "  • Speed:  1.64 ms T4 TensorRT (nano)  |  CPU ONNX not published\n\n"
            "[bold]Recommendation[/bold]\n"
            "  [bold red]Not recommended for production[/bold red] — training instability and "
            "high GPU memory consumption. Use YOLO11 or YOLO26 for production."
        ),
        "yolov12-seg": (
            "[bold cyan]YOLOv12-Seg[/bold cyan]  [yellow]● Research — 2025[/yellow]\n\n"
            "[bold]Architecture[/bold]\n"
            "  • Attention-centric YOLO12 base with segmentation head\n"
            "  • Area Attention + R-ELAN backbone\n\n"
            "[bold]Benchmarks[/bold]  [dim](COCO val2017)[/dim]\n"
            "  • Mask mAP: not yet officially published by Ultralytics\n"
            "  • Speed:    not yet officially published by Ultralytics\n\n"
            "[bold]Recommendation[/bold]\n"
            "  [bold red]Not recommended for production[/bold red] — inherits YOLO12 "
            "instability. Use YOLO11-seg or YOLO26-seg instead."
        ),
        "yolov11": (
            "[bold cyan]YOLOv11[/bold cyan]  [green]● Stable — 2024[/green]\n\n"
            "[bold]Architecture[/bold]\n"
            "  • Improved backbone and neck over YOLOv8\n"
            "  • 22% fewer parameters than YOLOv8m with higher mAP\n"
            "  • Supports all tasks: Detect, Segment, Classify, Pose, OBB\n\n"
            "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
            "  • mAP:    39.5 (nano)  →  54.7 (xlarge)\n"
            "  • Params: 2.6M (nano)  →  56.9M (xlarge)\n"
            "  • Speed:  56 ms CPU ONNX (nano)  |  1.5 ms T4 TensorRT (nano)\n\n"
            "[bold]Best for[/bold]\n"
            "  Production, enterprise, mission-critical applications"
        ),
        "yolov11-seg": (
            "[bold cyan]YOLOv11-Seg[/bold cyan]  [green]● Stable — 2024[/green]\n\n"
            "[bold]Architecture[/bold]\n"
            "  • YOLO11 backbone with segmentation head\n"
            "  • Proven training stability across diverse datasets\n\n"
            "[bold]Benchmarks[/bold]  [dim](COCO val2017, segmentation)[/dim]\n"
            "  • mAP box:  38.9 (nano)  →  54.7 (xlarge)\n"
            "  • mAP mask: 32.0 (nano)  →  43.8 (xlarge)\n"
            "  • Params:   2.9M (nano)  →  62.1M (xlarge)\n"
            "  • Speed:    66 ms CPU ONNX (nano)  |  2.9 ms T4 TensorRT (nano)\n\n"
            "[bold]Best for[/bold]\n"
            "  Production segmentation requiring reliability and full benchmark data"
        ),
        "yolov10": (
            "[bold cyan]YOLOv10[/bold cyan]  [dim]● Mature[/dim]\n\n"
            "[bold]Architecture[/bold]\n"
            "  • Anchor-free, NMS-free inference (pre-YOLO26 pioneer)\n"
            "  • 6 size variants: N / S / M / B / L / X\n"
            "  • Dual-head design: one for training, one for inference\n\n"
            "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
            "  • mAP:     38.5 (N)  →  54.4 (X)\n"
            "  • Latency: 1.84 ms (N)  →  10.70 ms (X) T4 TensorRT\n\n"
            "[bold]Recommendation[/bold]\n"
            "  Prefer YOLO11 or YOLO26 for new projects. Use when an existing "
            "pipeline is already built on YOLOv10."
        ),
        "yolov9": (
            "[bold cyan]YOLOv9[/bold cyan]  [dim]● Mature[/dim]\n\n"
            "[bold]Architecture[/bold]\n"
            "  • Programmable Gradient Information (PGI) — preserves full\n"
            "    information through deep network layers\n"
            "  • Generalised Efficient Layer Aggregation Network (GELAN)\n"
            "  • 5 variants: t / s / m / c / e\n\n"
            "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
            "  • mAP:    38.3 (tiny)  →  55.6 (extra-large)\n"
            "  • Params: 2.0M (tiny)  →  58.1M (extra-large)\n\n"
            "[bold]Best for[/bold]\n"
            "  When PGI gradient stability is required or a YOLOv9 checkpoint "
            "is already available"
        ),
        "yolov9-seg": (
            "[bold cyan]YOLOv9-Seg[/bold cyan]  [dim]● Mature[/dim]\n\n"
            "[bold]Architecture[/bold]\n"
            "  • PGI-based backbone with segmentation head\n"
            "  • Same GELAN feature aggregation as YOLOv9 detection\n"
            "  • 5 variants: t / s / m / c / e\n\n"
            "[bold]Benchmarks[/bold]  [dim](COCO val2017)[/dim]\n"
            "  • Mask mAP: not officially published by Ultralytics\n\n"
            "[bold]Best for[/bold]\n"
            "  Segmentation when PGI gradient properties are desired "
            "or an existing YOLOv9-seg checkpoint is in use"
        ),
        "yolov8": (
            "[bold cyan]YOLOv8[/bold cyan]  [dim]● Mature — 2023[/dim]\n\n"
            "[bold]Architecture[/bold]\n"
            "  • Anchor-free, decoupled detection head\n"
            "  • Industry-standard baseline — extensively documented\n"
            "  • Broadest third-party tool and framework support\n"
            "  • 5 variants: n / s / m / l / x\n\n"
            "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
            "  • mAP:    37.3 (nano)  →  53.9 (xlarge)\n"
            "  • Params: 3.2M (nano)  →  68.2M (xlarge)\n"
            "  • Speed:  80 ms CPU ONNX (nano)  |  0.99 ms A100 TensorRT (nano)\n\n"
            "[bold]Best for[/bold]\n"
            "  Legacy compatibility, existing YOLOv8 pipelines, or when "
            "maximum ecosystem support is required"
        ),
        "yolov8-seg": (
            "[bold cyan]YOLOv8-Seg[/bold cyan]  [dim]● Mature — 2023[/dim]\n\n"
            "[bold]Architecture[/bold]\n"
            "  • YOLOv8 backbone with segmentation head\n"
            "  • Most widely supported segmentation baseline\n"
            "  • 5 variants: n / s / m / l / x\n\n"
            "[bold]Benchmarks[/bold]  [dim](COCO val2017, segmentation)[/dim]\n"
            "  • mAP box:  36.7 (nano)  →  53.4 (xlarge)\n"
            "  • mAP mask: 30.5 (nano)  →  43.4 (xlarge)\n"
            "  • Params:   3.4M (nano)  →  71.8M (xlarge)\n"
            "  • Speed:    96 ms CPU ONNX (nano)  |  1.21 ms A100 TensorRT (nano)\n\n"
            "[bold]Best for[/bold]\n"
            "  Production segmentation requiring maximum ecosystem compatibility"
        ),
        "yolox": (
            "[bold cyan]YOLOX[/bold cyan]  [dim]● Mature[/dim]\n\n"
            "[bold]Architecture[/bold]\n"
            "  • Anchor-free with decoupled classification/regression head\n"
            "  • Simpler training setup than anchor-based predecessors\n"
            "  • 4 variants: S / M / L / X\n\n"
            "[bold]Benchmarks[/bold]  [dim](COCO val2017, detection)[/dim]\n"
            "  • mAP:    40.5 (S)  →  51.1 (X)\n"
            "  • Params: 9.0M (S)  →  99.1M (X)\n"
            "  • FPS:    102 (S)  →  58 (X)  [dim](V100 GPU)[/dim]\n\n"
            "[bold]Best for[/bold]\n"
            "  When a clean anchor-free baseline and training stability "
            "are the primary requirements"
        ),
}
