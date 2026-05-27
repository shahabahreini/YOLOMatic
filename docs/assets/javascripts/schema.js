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
        ["What is YOLOmatic?", "YOLOmatic is a Python CLI/TUI for configuring, training, fine-tuning, predicting, benchmarking, and uploading computer-vision models."],
        ["Which model families does it support?", "It supports YOLO26, YOLOv12, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOX, RF-DETR, SAM 3.1, and Detectron2."],
        ["Does YOLOmatic train RF-DETR?", "Yes. RF-DETR configs route to the native RF-DETR trainer and use .pth checkpoints."],
        ["Can it convert Labelbox NDJSON?", "Yes. YOLOmatic converts Labelbox and Ultralytics-platform NDJSON exports into YOLO or COCO formats."],
        ["Is a GPU required?", "No, but a CUDA GPU is strongly recommended for training. CPU and Apple Silicon MPS fallbacks are supported."]
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
