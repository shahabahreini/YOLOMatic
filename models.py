model_data_dict = {
    "yolov11": [
        {
            "Model": "YOLO11n",
            "Input Size": 640,
            "mAPval 50-95": 39.5,
            "Speed CPU ONNX (ms)": "56.1 ± 0.8",
            "Speed T4 TensorRT (ms)": "1.5 ± 0.0",
            "params (M)": 2.6,
            "FLOPs (B)": 6.5,
        },
        {
            "Model": "YOLO11s",
            "Input Size": 640,
            "mAPval 50-95": 47.0,
            "Speed CPU ONNX (ms)": "90.0 ± 1.2",
            "Speed T4 TensorRT (ms)": "2.5 ± 0.0",
            "params (M)": 9.4,
            "FLOPs (B)": 21.5,
        },
        {
            "Model": "YOLO11m",
            "Input Size": 640,
            "mAPval 50-95": 51.5,
            "Speed CPU ONNX (ms)": "183.2 ± 2.0",
            "Speed T4 TensorRT (ms)": "4.7 ± 0.1",
            "params (M)": 20.1,
            "FLOPs (B)": 68.0,
        },
        {
            "Model": "YOLO11l",
            "Input Size": 640,
            "mAPval 50-95": 53.4,
            "Speed CPU ONNX (ms)": "238.6 ± 1.4",
            "Speed T4 TensorRT (ms)": "6.2 ± 0.1",
            "params (M)": 25.3,
            "FLOPs (B)": 86.9,
        },
        {
            "Model": "YOLO11x",
            "Input Size": 640,
            "mAPval 50-95": 54.7,
            "Speed CPU ONNX (ms)": "462.8 ± 6.7",
            "Speed T4 TensorRT (ms)": "11.3 ± 0.2",
            "params (M)": 56.9,
            "FLOPs (B)": 194.9,
        },
    ],
    "yolov10": [
        {
            "Model": "YOLOv10-N",
            "Input Size": 640,
            "mAPval 50-95": 38.5,
            "FLOPs (G)": 6.7,
            "Latency (ms)": 1.84,
        },
        {
            "Model": "YOLOv10-S",
            "Input Size": 640,
            "mAPval 50-95": 46.3,
            "FLOPs (G)": 21.6,
            "Latency (ms)": 2.49,
        },
        {
            "Model": "YOLOv10-M",
            "Input Size": 640,
            "mAPval 50-95": 51.1,
            "FLOPs (G)": 59.1,
            "Latency (ms)": 4.74,
        },
        {
            "Model": "YOLOv10-B",
            "Input Size": 640,
            "mAPval 50-95": 52.5,
            "FLOPs (G)": 92.0,
            "Latency (ms)": 5.74,
        },
        {
            "Model": "YOLOv10-L",
            "Input Size": 640,
            "mAPval 50-95": 53.2,
            "FLOPs (G)": 120.3,
            "Latency (ms)": 7.28,
        },
        {
            "Model": "YOLOv10-X",
            "Input Size": 640,
            "mAPval 50-95": 54.4,
            "FLOPs (G)": 160.4,
            "Latency (ms)": 10.70,
        },
    ],
    "yolov9": [
        {
            "Model": "YOLOv9t",
            "Input Size": 640,
            "mAPval 50-95": 38.3,
            "mAPval 50": 53.1,
            "params (M)": 2.0,
            "FLOPs (B)": 7.7,
        },
        {
            "Model": "YOLOv9s",
            "Input Size": 640,
            "mAPval 50-95": 46.8,
            "mAPval 50": 63.4,
            "params (M)": 7.2,
            "FLOPs (B)": 26.7,
        },
        {
            "Model": "YOLOv9m",
            "Input Size": 640,
            "mAPval 50-95": 51.4,
            "mAPval 50": 68.1,
            "params (M)": 20.1,
            "FLOPs (B)": 76.8,
        },
        {
            "Model": "YOLOv9c",
            "Input Size": 640,
            "mAPval 50-95": 53.0,
            "mAPval 50": 70.2,
            "params (M)": 25.5,
            "FLOPs (B)": 102.8,
        },
        {
            "Model": "YOLOv9e",
            "Input Size": 640,
            "mAPval 50-95": 55.6,
            "mAPval 50": 72.8,
            "params (M)": 58.1,
            "FLOPs (B)": 192.5,
        },
    ],
    "yolov8": [
        {
            "Model": "YOLOv8n",
            "Input Size": 640,
            "mAPval 50-95": 37.3,
            "Speed CPU ONNX (ms)": 80.4,
            "Speed A100 TensorRT (ms)": 0.99,
            "params (M)": 3.2,
            "FLOPs (B)": 8.7,
        },
        {
            "Model": "YOLOv8s",
            "Input Size": 640,
            "mAPval 50-95": 44.9,
            "Speed CPU ONNX (ms)": 128.4,
            "Speed A100 TensorRT (ms)": 1.20,
            "params (M)": 11.2,
            "FLOPs (B)": 28.6,
        },
        {
            "Model": "YOLOv8m",
            "Input Size": 640,
            "mAPval 50-95": 50.2,
            "Speed CPU ONNX (ms)": 234.7,
            "Speed A100 TensorRT (ms)": 1.83,
            "params (M)": 25.9,
            "FLOPs (B)": 78.9,
        },
        {
            "Model": "YOLOv8l",
            "Input Size": 640,
            "mAPval 50-95": 52.9,
            "Speed CPU ONNX (ms)": 375.2,
            "Speed A100 TensorRT (ms)": 2.39,
            "params (M)": 43.7,
            "FLOPs (B)": 165.2,
        },
        {
            "Model": "YOLOv8x",
            "Input Size": 640,
            "mAPval 50-95": 53.9,
            "Speed CPU ONNX (ms)": 479.1,
            "Speed A100 TensorRT (ms)": 3.53,
            "params (M)": 68.2,
            "FLOPs (B)": 257.8,
        },
    ],
    "yolov12": [
        {
            "Model": "YOLO12n",
            "Input Size": 640,
            "mAPval 50-95": 40.6,
            "Speed CPU ONNX (ms)": "-",
            "Speed T4 TensorRT (ms)": "1.64",
            "params (M)": 2.6,
            "FLOPs (B)": 6.5,
            "Comparison": "+2.1%/-9% (vs. YOLOv10n)",
        },
        {
            "Model": "YOLO12s",
            "Input Size": 640,
            "mAPval 50-95": 48.0,
            "Speed CPU ONNX (ms)": "-",
            "Speed T4 TensorRT (ms)": "2.61",
            "params (M)": 9.3,
            "FLOPs (B)": 21.4,
            "Comparison": "+0.1%/+42% (vs. RT-DETRv2)",
        },
        {
            "Model": "YOLO12m",
            "Input Size": 640,
            "mAPval 50-95": 52.5,
            "Speed CPU ONNX (ms)": "-",
            "Speed T4 TensorRT (ms)": "4.86",
            "params (M)": 20.2,
            "FLOPs (B)": 67.5,
            "Comparison": "+1.0%/-3% (vs. YOLO11m)",
        },
        {
            "Model": "YOLO12l",
            "Input Size": 640,
            "mAPval 50-95": 53.7,
            "Speed CPU ONNX (ms)": "-",
            "Speed T4 TensorRT (ms)": "6.77",
            "params (M)": 26.4,
            "FLOPs (B)": 88.9,
            "Comparison": "+0.4%/-8% (vs. YOLO11l)",
        },
        {
            "Model": "YOLO12x",
            "Input Size": 640,
            "mAPval 50-95": 55.2,
            "Speed CPU ONNX (ms)": "-",
            "Speed T4 TensorRT (ms)": "11.79",
            "params (M)": 59.1,
            "FLOPs (B)": 199.0,
            "Comparison": "+0.6%/-4% (vs. YOLO11x)",
        },
    ],
    "yolox": [
        {
            "Model": "YOLOX-S",
            "Input Size": 640,
            "mAPval 50-95": "40.5%",
            "params (M)": 9.0,
            "FLOPs (G)": 26.8,
            "FPS": 102,
        },
        {
            "Model": "YOLOX-M",
            "Input Size": 640,
            "mAPval 50-95": "46.9%",
            "params (M)": 25.3,
            "FLOPs (G)": 73.8,
            "FPS": 81,
        },
        {
            "Model": "YOLOX-L",
            "Input Size": 640,
            "mAPval 50-95": "49.7%",
            "params (M)": 54.2,
            "FLOPs (G)": 155.6,
            "FPS": 69,
        },
        {
            "Model": "YOLOX-X",
            "Input Size": 640,
            "mAPval 50-95": "51.1%",
            "params (M)": 99.1,
            "FLOPs (G)": 281.9,
            "FPS": 58,
        },
    ],
    "yolo_nas": [
        {
            "Model": "YOLO-NAS-S",
            "Input Size": 640,
            "mAPval 50-95": 47.5,
            "Speed CPU ONNX (ms)": "90.2",
            "Speed T4 TensorRT (ms)": "2.4",
            "params (M)": 9.3,
            "FLOPs (B)": 21.3,
        },
        {
            "Model": "YOLO-NAS-M",
            "Input Size": 640,
            "mAPval 50-95": 51.4,
            "Speed CPU ONNX (ms)": "182.8",
            "Speed T4 TensorRT (ms)": "4.6",
            "params (M)": 20.0,
            "FLOPs (B)": 67.5,
        },
        {
            "Model": "YOLO-NAS-L",
            "Input Size": 640,
            "mAPval 50-95": 53.2,
            "Speed CPU ONNX (ms)": "237.9",
            "Speed T4 TensorRT (ms)": "6.1",
            "params (M)": 25.2,
            "FLOPs (B)": 86.5,
        },
    ],
}
