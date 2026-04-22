# YOLO26, YOLO12, YOLO11 Integration Guide

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd YOLOMatic

# Sync the existing project environment with uv
uv sync
```

After installation you can run the CLI via the `yolomatic` entrypoints managed by `uv`.

## Versioning

The project version is stored in `pyproject.toml` and controlled with `uv`.
Use `uv version` or `uv version --bump <patch|minor|major>` to adjust it.

### First Run

```bash
# Start the interactive model selection
uv run yolomatic

# Follow the prompts to:
# 1. Select a model (recommend YOLO26)
# 2. Choose variant (n/s/m/l/x)
# 3. Select dataset
# 4. Review configuration

# Start training
uv run yolomatic-train

# Run predictions with the TUI
uv run yolomatic-predict

# Or run prediction directly with explicit arguments
uv run yolomatic-predict --mode single --weight runs/segment/train/weights/best.pt --source /path/to/image.jpg

# Monitor training logs
uv run yolomatic-tensorboard
```

---

## Model Selection Guide

### YOLO26 - Latest Edge Optimizer 🚀

**Best For**: Edge devices, CPU inference, real-time mobile apps

**Advantages**:

- ⚡ 43% faster CPU inference
- 🎯 Higher accuracy on small objects
- 📦 Smaller model sizes
- 🚀 NMS-free end-to-end inference
- 🔋 Perfect for IoT and robotics

**Performance**:
| Size | mAP | Params | Speed (CPU) |
|------|-----|--------|------------|
| Nano | 40.9 | 2.4M | Fastest |
| Small | 48.6 | 9.5M | Very Fast |
| Medium | 53.1 | 20.4M | Fast |
| Large | 55.0 | 24.8M | Moderate |
| XLarge | 57.5 | 55.7M | High Quality |

**When to Choose**:

- ✅ Deploying on edge devices
- ✅ Need fast CPU inference
- ✅ Memory constrained
- ✅ Mobile/IoT applications

---

### YOLO12 - Attention-Centric Research Model

**Best For**: Research, benchmarking, attention mechanism studies

**Key Features**:

- 🧠 Area Attention Mechanism
- 🔗 R-ELAN architecture
- 💡 FlashAttention support (optional)
- 📊 Highest accuracy potential

**Performance**:
| Size | mAP | Params | Speed |
|------|-----|--------|-------|
| Nano | 40.6 | 2.6M | Moderate |
| Small | 48.0 | 9.3M | Moderate |
| Medium | 52.5 | 20.2M | Moderate |
| Large | 53.7 | 26.4M | Slower |
| XLarge | 55.2 | 59.1M | Slower |

⚠️ **Warnings**:

- ❌ Training instability
- ❌ Higher memory consumption
- ❌ Slower CPU throughput
- ❌ Not recommended for production

**When to Choose**:

- ✅ Research purposes
- ✅ Academic benchmarking
- ✅ GPU-rich environments
- ✅ Attention mechanism studies
- ❌ Production deployments

---

### YOLO11 - Production-Ready Standard

**Best For**: Production systems, enterprise deployments, balanced performance

**Advantages**:

- ✅ Proven stability
- ✅ 22% fewer params than YOLOv8m (with higher accuracy)
- ✅ Fast CPU inference
- ✅ All task types supported
- ✅ Excellent documentation

**Performance**:
| Size | mAP | Params | Speed (CPU) | Speed (GPU) |
|------|-----|--------|------------|------------|
| Nano | 39.5 | 2.6M | 56ms | 1.5ms |
| Small | 47.0 | 9.4M | 90ms | 2.5ms |
| Medium | 51.5 | 20.1M | 183ms | 4.7ms |
| Large | 53.4 | 25.3M | 239ms | 6.2ms |
| XLarge | 54.7 | 56.9M | 463ms | 11.3ms |

**When to Choose**:

- ✅ Production deployments
- ✅ Enterprise systems
- ✅ Mission-critical applications
- ✅ Need proven stability
- ✅ Mixed CPU/GPU environments

---

## Feature Comparison

### Computer Vision Tasks

All three models support:

- ✅ **Object Detection**: Identify and locate objects
- ✅ **Instance Segmentation**: Detect and delineate boundaries
- ✅ **Image Classification**: Categorize images
- ✅ **Pose Estimation**: Detect keypoints
- ✅ **OBB**: Oriented object detection (rotated)

### Operational Modes

All three models support:

- ✅ **Inference**: Run predictions
- ✅ **Validation**: Evaluate performance
- ✅ **Training**: Fine-tune on custom data
- ✅ **Export**: Deploy to production formats

---

## Detailed Specifications

### Architecture Comparison

| Aspect                 | YOLO26            | YOLO12            | YOLO11      |
| ---------------------- | ----------------- | ----------------- | ----------- |
| **Base Architecture**  | End-to-end        | Attention-centric | CNN-based   |
| **NMS**                | None (end-to-end) | Traditional       | Traditional |
| **Optimization**       | Edge-focused      | Attention-focused | Balanced    |
| **CPU Performance**    | ⭐⭐⭐⭐⭐        | ⭐⭐              | ⭐⭐⭐      |
| **GPU Performance**    | ⭐⭐⭐⭐          | ⭐⭐⭐⭐          | ⭐⭐⭐⭐    |
| **Training Stability** | Excellent         | Variable          | Excellent   |
| **Memory Usage**       | Low               | High              | Moderate    |

### Feature Set

| Feature                           | YOLO26  | YOLO12        | YOLO11  |
| --------------------------------- | ------- | ------------- | ------- |
| **DFL (Distribution Focal Loss)** | Removed | Present       | Present |
| **NMS-Free Inference**            | ✅      | ❌            | ❌      |
| **MuSGD Optimizer**               | ✅      | ❌            | ❌      |
| **Area Attention**                | ❌      | ✅            | ❌      |
| **R-ELAN**                        | ❌      | ✅            | ❌      |
| **FlashAttention**                | ❌      | ✅ (optional) | ❌      |
| **Multi-scale Proto**             | ✅      | ❌            | ❌      |
| **RLE for Pose**                  | ✅      | ❌            | ❌      |

---

## Training Configuration

### Default Training Parameters (applies to all models)

```yaml
training:
  epochs: 150 # Number of training epochs
  imgsz: 640 # Input image size
  batch: 16 # Batch size
  cache: true # Cache dataset
  workers: 8 # Data loading workers
  label_smoothing: 0.1 # Label smoothing
  close_mosaic: 50 # Disable mosaic augmentation

  # Augmentation
  hsv_h: 0.015 # HSV-Hue augmentation
  hsv_s: 0.7 # HSV-Saturation augmentation
  hsv_v: 0.4 # HSV-Value augmentation
  fliplr: 0.5 # Flip left-right probability
  device: auto # Device (cuda/mps/cpu)
```

### Model-Specific Recommendations

**YOLO26**:

```python
# Optimize for edge deployment
batch: 8-16 (reduce for edge)
epochs: 100-150
imgsz: 480-640 (smaller for edge)
device: "cpu"  # Test on CPU
```

**YOLO12**:

```python
# Research configuration
batch: 32-64 (larger for attention)
epochs: 150-200
imgsz: 640
device: "cuda"  # Requires GPU
# Note: May have training instability
```

**YOLO11**:

```python
# Production configuration
batch: 16-32
epochs: 100-150
imgsz: 640
device: "auto"  # Use best available
```

---

## Performance Benchmarks

### Accuracy (mAP on COCO val2017)

**Best Overall**:

- 🥇 YOLO26x: 57.5 mAP
- 🥈 YOLO12x: 55.2 mAP
- 🥉 YOLO11x: 54.7 mAP

**Best Nano Model**:

- 🥇 YOLO26n: 40.9 mAP
- 🥈 YOLO12n: 40.6 mAP
- 🥉 YOLO11n: 39.5 mAP

### Speed (CPU ONNX)

**Fastest**:

1. YOLO26n (Estimated ~40ms)
2. YOLO11n (56.1 ± 0.8ms)
3. YOLO12n (Slower, exact TBD)

### Parameter Efficiency

**Smallest Model**:

- YOLO26n: 2.4M params
- YOLO12n: 2.6M params
- YOLO11n: 2.6M params

**Largest Model**:

- YOLO26x: 55.7M params
- YOLO12x: 59.1M params
- YOLO11x: 56.9M params

---

## Deployment Scenarios

### Scenario 1: Mobile App (Phone/Tablet)

**Recommended**: YOLO26n or YOLO26s

```python
# Use tiny model for fast inference
model = YOLO("yolo26n.pt")
# Expected: 40-50ms per frame on modern phones
```

### Scenario 2: Edge Device (RPi, Jetson)

**Recommended**: YOLO26s or YOLO26m

```python
# Medium model for balance
model = YOLO("yolo26s.pt")
# CPU inference optimized
```

### Scenario 3: Enterprise Server

**Recommended**: YOLO11l or YOLO11x

```python
# Larger model for accuracy
model = YOLO("yolo11l.pt")
# GPU-accelerated inference
```

### Scenario 4: Research Lab

**Recommended**: YOLO12m or YOLO12l

```python
# Attention-based for study
model = YOLO("yolo12m.pt")
# High accuracy with GPU
```

---

## Export Options

All models support export to multiple formats:

```python
# ONNX (universal, recommended)
model.export(format="onnx")

# TensorRT (NVIDIA GPUs)
model.export(format="engine")

# CoreML (Apple devices)
model.export(format="coreml")

# TensorFlow Lite (Mobile)
model.export(format="tflite")

# OpenVINO (Intel)
model.export(format="openvino")
```

---

## Troubleshooting

### YOLO26 Issues

**"Model not found"**

```bash
# Ultralytics will auto-download from hub
# Ensure internet connection is active
```

**"Slow on CPU"**

- Try smaller variant (n or s)
- Increase batch size if memory allows
- Check CPU load

**"Memory exceeded"**

- Reduce image size (imgsz)
- Reduce batch size
- Use smaller model variant

### YOLO12 Issues

**"Training instability"**

- Normal for YOLO12
- Reduce learning rate
- Use gradient clipping
- Consider YOLO11 or YOLO26 instead

**"High memory usage"**

- Expected with attention layers
- Reduce batch size significantly
- Use smaller variant
- Ensure sufficient GPU VRAM

**"FlashAttention compilation fails"**

- Not required - fallback is automatic
- Optional optimization only

### YOLO11 Issues

**"Inference slow"**

- Use TensorRT export for GPU
- Reduce image size
- Check GPU utilization

**"Training convergence issues"**

- Adjust learning rate
- Verify dataset format
- Check class balance

---

## Additional Resources

### Documentation

- [YOLO26 Official Docs](https://docs.ultralytics.com/models/yolo26/)
- [YOLO12 Official Docs](https://docs.ultralytics.com/models/yolo12/)
- [YOLO11 Official Docs](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics Hub](https://hub.ultralytics.com/)

### Community

- [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)
- [Discussions](https://github.com/ultralytics/ultralytics/discussions)
- [Discord Community](https://discord.com/invite/ultralytics)

### Related Files

- See `README.md` for installation instructions
- See `MODELS.md` for detailed model comparison
- See `INTEGRATION_SUMMARY.md` for implementation details

---

## Summary Table

| Use Case       | Recommended      | Why                         |
| -------------- | ---------------- | --------------------------- |
| Mobile App     | YOLO26n/s        | Fastest, smallest           |
| Edge Device    | YOLO26m          | Balanced speed/accuracy     |
| IoT/Robotics   | YOLO26 (any)     | CPU optimized, 43% faster   |
| Production Web | YOLO11l          | Stable, proven, accurate    |
| Enterprise     | YOLO11x          | Best accuracy, robust       |
| Research       | YOLO12           | Cutting-edge architecture   |
| Benchmarking   | YOLO26 vs YOLO11 | Compare edge vs traditional |
| Learning       | Any              | All well-documented         |

---

**Last Updated**: January 18, 2026  
**YOLOMatic Version**: 2.0  
**Status**: ✅ Fully Supported and Integrated
