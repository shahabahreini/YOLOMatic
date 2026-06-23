(function () {
  const base = "https://shahabahreini.github.io/YOLOMatic/";
  const path = window.location.pathname.replace(/\/$/, "");
  const schemas = [];

  const software = {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    "name": "YOLOmatic",
    "applicationCategory": "DeveloperApplication",
    "operatingSystem": "Linux, macOS, Windows",
    "programmingLanguage": "Python",
    "url": base,
    "codeRepository": "https://github.com/shahabahreini/YOLOMatic",
    "license": "https://www.apache.org/licenses/LICENSE-2.0",
    "author": {
      "@type": "Person",
      "name": "Shahab Bahreini Jangjoo"
    },
    "offers": {
      "@type": "Offer",
      "price": "0",
      "priceCurrency": "USD"
    }
  };

  if (path.endsWith("/YOLOMatic") || path === "") {
    schemas.push(software);
  }

  if (path.endsWith("/faq")) {
    schemas.push({
      "@context": "https://schema.org",
      "@type": "FAQPage",
      "mainEntity": [
        ["What is YOLOmatic?", "YOLOmatic is a Python 3.12 CLI/TUI for configuring, training, fine-tuning, predicting, benchmarking, augmenting, converting, monitoring, and uploading computer-vision models."],
        ["Who is YOLOmatic for?", "YOLOmatic is for practitioners, researchers, and teams that want repeatable local computer-vision training workflows."],
        ["Is YOLOmatic a hosted training platform?", "No. YOLOmatic runs locally, while optionally integrating with services such as Roboflow, ClearML, HuggingFace, and TensorBoard."],
        ["Which model families does it support?", "It supports YOLO26, YOLOv12, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOX, RF-DETR, SAM 3.1, and Detectron2."],
        ["Is YOLOmatic only for YOLO?", "No. It also supports native RF-DETR, SAM 3.1 workflows, Detectron2, dataset conversion, augmentation, benchmarking, and upload helpers."],
        ["Is a GPU required?", "No, but a CUDA GPU is strongly recommended for training. CPU and Apple Silicon MPS fallbacks are supported."],
        ["What Python version does YOLOmatic require?", "YOLOmatic targets Python 3.12."],
        ["Does YOLOmatic support Windows?", "Yes. Windows is supported, including CUDA-capable systems."],
        ["Does YOLOmatic support macOS?", "Yes. macOS can use CPU or Apple Silicon MPS; NVIDIA CUDA is not available on macOS."],
        ["Does YOLOmatic support RF-DETR?", "Yes. RF-DETR configs route to the native RF-DETR trainer and use .pth checkpoints."],
        ["Can YOLOmatic fine-tune RF-DETR?", "Yes. YOLOmatic discovers RF-DETR .pth checkpoints and writes fine-tuning configs."],
        ["Does YOLOmatic support SAM 3.1?", "Yes. YOLOmatic supports SAM 3.1 auto, text-prompted, box-prompted segmentation, and COCO-format fine-tuning."],
        ["Does YOLOmatic support Detectron2?", "Yes. It supports Detectron2 training flows for Faster R-CNN, RetinaNet, and Mask R-CNN style configurations."],
        ["What dataset formats does YOLOmatic understand?", "YOLOmatic works with YOLO folders, COCO JSON, Labelbox NDJSON, and Ultralytics-platform NDJSON."],
        ["Can YOLOmatic convert Labelbox NDJSON?", "Yes. YOLOmatic converts Labelbox exports into YOLO or COCO formats with concurrent image downloads."],
        ["Can YOLOmatic convert Ultralytics-platform NDJSON?", "Yes. It reads normalized annotations.segments, annotations.boxes, and annotations.pose from Ultralytics-platform image rows, including YOLO Pose and COCO Pose output."],
        ["Can YOLOmatic split datasets?", "Yes. It supports random, class-balanced, and smart-balanced dataset splitting."],
        ["What is smart-balanced splitting?", "Smart-balanced splitting preserves rare classes first, then fills remaining split capacity with deterministic random tie-breaking."],
        ["Can YOLOmatic augment datasets offline?", "Yes. It includes Albumentations-powered offline augmentation with reusable profiles and YOLO/COCO output support."],
        ["How do I generate a training config?", "Run uv run yolomatic, choose Configure Model, select a model and dataset, then save the generated YAML."],
        ["How does YOLOmatic choose the trainer?", "The smart training router dispatches configs to Ultralytics YOLO, native RF-DETR, SAM 3.1, or Detectron2."],
        ["Can YOLOmatic run batch prediction?", "Yes. Folder prediction is supported with progress display and worker-based parallelism where applicable."],
        ["Can YOLOmatic benchmark trained models?", "Yes. It benchmarks Ultralytics YOLO checkpoints and exports, including ONNX and TensorRT engines, on validation data and generates an HTML report."],
        ["Which benchmark metrics are included?", "Reports include mAP, F1, per-image rankings, confidence inspection, thumbnails, and UMAP vector scatter plots."],
        ["Can YOLOmatic upload trained models to Roboflow?", "Yes. It uploads YOLO checkpoints and deploys RF-DETR checkpoints through the Roboflow upload workflow."],
        ["Does YOLOmatic support ClearML?", "Yes. ClearML tracking is optional and training can continue without it."],
        ["Where should secrets go?", "Secrets should live in .env or shell environment variables, not in committed configs or docs."],
        ["How should I cite YOLOmatic?", "Use CITATION.cff for GitHub citation metadata or CITATION.bib for BibTeX."]
      ].map(function (qa) {
        return {
          "@type": "Question",
          "name": qa[0],
          "acceptedAnswer": {"@type": "Answer", "text": qa[1]}
        };
      })
    });
  }

  if (path.endsWith("/getting-started/quickstart") || path.endsWith("/getting-started/first-training-run")) {
    schemas.push({
      "@context": "https://schema.org",
      "@type": "HowTo",
      "name": document.title,
      "step": [
        {"@type": "HowToStep", "name": "Install YOLOmatic", "text": "Install with uv or sync the repository environment."},
        {"@type": "HowToStep", "name": "Launch the TUI", "text": "Run uv run yolomatic and choose a workflow."},
        {"@type": "HowToStep", "name": "Train or evaluate", "text": "Generate a config, train, predict, benchmark, or upload artifacts."}
      ]
    });
  }

  if (path.includes("/guides/") || path.includes("/reference/")) {
    schemas.push({
      "@context": "https://schema.org",
      "@type": "TechArticle",
      "headline": document.title,
      "author": {"@type": "Person", "name": "Shahab Bahreini Jangjoo"},
      "publisher": {"@type": "Organization", "name": "YOLOmatic"},
      "url": window.location.href
    });
  }

  schemas.forEach(function (schema) {
    const script = document.createElement("script");
    script.type = "application/ld+json";
    script.text = JSON.stringify(schema);
    document.head.appendChild(script);
  });
})();
