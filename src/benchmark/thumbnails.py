"""Thumbnail generation with GT/prediction overlay."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np


_GREY_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x00\x00\x00\x00:~\x9bU\x00\x00\x00\nIDATx\x9cc\xf8\x0f\x00\x00"
    b"\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _placeholder_b64() -> str:
    return base64.b64encode(_GREY_PNG_1x1).decode()


def generate_thumbnail(
    image_path: Path,
    gt_objects: list,
    pred_objects: list,
    size: int = 224,
    task: str = "detection",
) -> bytes:
    """Return PNG bytes of a resized image with GT (green) and pred (red) overlays."""
    try:
        from PIL import Image, ImageDraw
        import cv2

        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        img_resized = img.resize((size, size), Image.LANCZOS)
        draw = ImageDraw.Draw(img_resized, "RGBA")
        sx, sy = size / orig_w, size / orig_h

        # Draw GT objects (green outline)
        for gt in gt_objects:
            if task == "segmentation" and gt.mask is not None:
                _draw_mask_outline(draw, gt.mask, orig_w, orig_h, sx, sy, color=(0, 200, 0, 180))
            else:
                x1, y1, x2, y2 = gt.box_xyxy
                draw.rectangle(
                    [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                    outline=(0, 200, 0, 220), width=2,
                )

        # Draw predicted objects (red outline)
        for pred in pred_objects:
            if task == "segmentation" and pred.mask is not None:
                _draw_mask_outline(draw, pred.mask, orig_w, orig_h, sx, sy, color=(220, 0, 0, 180))
            else:
                x1, y1, x2, y2 = pred.box_xyxy
                draw.rectangle(
                    [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                    outline=(220, 0, 0, 220), width=2,
                )

        buf = io.BytesIO()
        img_resized.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception:
        return _GREY_PNG_1x1


def _draw_mask_outline(draw, mask: np.ndarray, orig_w: int, orig_h: int,
                       sx: float, sy: float, color: tuple) -> None:
    try:
        import cv2
        from PIL import Image

        m = mask.astype(np.uint8) * 255
        if m.shape != (orig_h, orig_w):
            m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts = [(int(pt[0][0] * sx), int(pt[0][1] * sy)) for pt in contour]
            if len(pts) >= 2:
                draw.polygon(pts, outline=color)
    except Exception:
        pass


def encode_thumbnail(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode()


def make_thumbnail_b64(
    image_path: Path,
    gt_objects: list,
    pred_objects: list,
    size: int = 224,
    task: str = "detection",
) -> str:
    try:
        png = generate_thumbnail(image_path, gt_objects, pred_objects, size, task)
        return encode_thumbnail(png)
    except Exception:
        return _placeholder_b64()
