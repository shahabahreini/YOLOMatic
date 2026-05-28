---
description: Compare YOLOmatic with Ultralytics CLI, Roboflow, and hosted YOLO training platforms.
---

# Comparison

| Capability | YOLOmatic | Ultralytics CLI | Roboflow | YOLO Hub-style tools |
| --- | --- | --- | --- | --- |
| Interactive terminal wizard | Yes | Limited | Web UI | Web UI |
| Multiple YOLO generations | Yes | Yes | Export/deploy focused | Varies |
| Native RF-DETR training | Yes | No | Deploy focused | Varies |
| SAM 3.1 workflows | Yes | No | Limited | Varies |
| Detectron2 training | Yes | No | No | Varies |
| Hardware-aware config generation | Yes | Manual | Abstracted | Abstracted |
| Labelbox/Ultralytics NDJSON conversion | Yes | No | Import focused | Varies |
| Local benchmark HTML reports | Yes | Partial metrics | Hosted analytics | Varies |
| Roboflow upload/deploy | Yes | No | Native platform | No |
| Fully local workflow | Yes | Yes | No | No |

YOLOmatic is best when you want local control, repeatable configs, terminal UX,
and support for several model families in one project. Hosted platforms are
better when team collaboration, browser labeling, and managed infrastructure are
the primary requirements.

Related pages: [Models](guides/models.md), [Benchmarking](guides/benchmarking.md), [Cloud upload](guides/cloud-upload.md).
