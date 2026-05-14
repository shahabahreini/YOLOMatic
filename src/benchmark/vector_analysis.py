"""Feature extraction and dimensionality reduction for the vector scatter plot."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_features(image_result, image_path: Path) -> np.ndarray:
    """Return a 12-dim feature vector for one image result."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean_r = float(arr[:, :, 0].mean())
        mean_g = float(arr[:, :, 1].mean())
        mean_b = float(arr[:, :, 2].mean())
        luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        std_lum = float(luminance.std())
    except Exception:
        mean_r = mean_g = mean_b = std_lum = 0.5

    gt_count = max(image_result.gt_count, 1)
    tp = image_result.tp
    fp = image_result.fp
    fn = image_result.fn
    total = tp + fp + fn or 1

    return np.array([
        mean_r,
        mean_g,
        mean_b,
        std_lum,
        float(image_result.f1),
        float(image_result.precision),
        float(image_result.recall),
        min(tp / total, 1.0),
        min(fp / total, 1.0),
        min(fn / total, 1.0),
        float(image_result.mean_iou),
        min(gt_count / 20.0, 1.0),
    ], dtype=np.float32)


def project_umap(features: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Project N×12 features to N×2. Tries UMAP first, falls back to PCA."""
    if len(features) < 4:
        # Too few points for meaningful projection
        return features[:, :2] if features.shape[1] >= 2 else np.zeros((len(features), 2))

    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, len(features) - 1),
            min_dist=min_dist,
            random_state=None,
            n_jobs=-1,
        )
        return reducer.fit_transform(features).astype(np.float32)
    except ImportError:
        pass
    except Exception:
        pass

    # sklearn PCA fallback
    try:
        from sklearn.decomposition import PCA
        n_components = min(2, features.shape[0], features.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        projected = pca.fit_transform(features)
        if projected.shape[1] < 2:
            projected = np.hstack([projected, np.zeros((len(projected), 2 - projected.shape[1]))])
        return projected.astype(np.float32)
    except ImportError:
        pass

    # Manual 2-component PCA (no external deps)
    X = features.astype(np.float64)
    X -= X.mean(axis=0)
    cov = np.cov(X.T)
    if cov.ndim < 2:
        cov = np.array([[cov, 0], [0, 0]])
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1][:2]
    components = vecs[:, idx]
    projected = X @ components
    if projected.shape[1] < 2:
        projected = np.hstack([projected, np.zeros((len(projected), 2 - projected.shape[1]))])
    return projected.astype(np.float32)


def build_vector_data(model_metrics, thumbnail_fn=None) -> dict:
    """Build lists of x, y, f1, meta for the scatter plot."""
    results = model_metrics.per_image
    if not results:
        return {}

    features = np.stack([
        extract_features(r, r.image_path)
        for r in results
    ])

    coords = project_umap(features)

    data = {
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "f1": [r.f1 for r in results],
        "precision": [r.precision for r in results],
        "recall": [r.recall for r in results],
        "tp": [r.tp for r in results],
        "fp": [r.fp for r in results],
        "fn": [r.fn for r in results],
        "mean_iou": [r.mean_iou for r in results],
        "gt_count": [r.gt_count for r in results],
        "pred_count": [r.pred_count for r in results],
        "image_name": [r.image_path.name for r in results],
        "image_id": [r.image_id for r in results],
        "thumbnails": [],
    }

    if thumbnail_fn is not None:
        for r in results:
            try:
                b64 = thumbnail_fn(r)
            except Exception:
                b64 = ""
            data["thumbnails"].append(b64)

    return data
