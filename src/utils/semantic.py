"""Pixel conventions for dense semantic-segmentation masks.

Ultralytics feeds semantic masks straight into ``CrossEntropyLoss(ignore_index=255)``,
so every pixel must hold a **raw class index** in ``[0, nc)`` — anything else trips a
device-side ``t >= 0 && t < n_classes`` assert during training.

Background needs an index of its own, and Ultralytics picks it differently depending
on the class count (see ``ultralytics.data.utils.add_polygon_background``):

* ``nc == 1`` — binary. Background is ``0`` and the single foreground class is ``1``;
  the model emits one channel and trains with BCE. ``nc`` stays 1.
* ``nc > 1``  — an explicit ``background`` class is appended at index ``nc``,
  foreground classes keep their own ids, and ``nc`` becomes ``nc + 1``.

Both the augmentation writer and the benchmark's ground-truth rasterizer go through
here so that a mask written by one is scored correctly by the other.
"""
from __future__ import annotations

BACKGROUND_CLASS_NAME = "background"
IGNORE_INDEX = 255


def semantic_background_index(num_classes: int) -> int:
    """Pixel value that means "background" for a dataset with ``num_classes`` classes."""
    return 0 if num_classes <= 1 else int(num_classes)


def semantic_pixel_value(class_id: int, num_classes: int) -> int:
    """Pixel value representing ``class_id`` in a dense mask."""
    return int(class_id) + 1 if num_classes <= 1 else int(class_id)


def semantic_class_names(class_names: list[str]) -> list[str]:
    """Class list as it must appear in data.yaml for a dense-mask dataset.

    Multi-class datasets gain an explicit trailing ``background`` entry; binary
    datasets keep their single name, with background left implicit at pixel 0.
    """
    names = list(class_names)
    if len(names) <= 1:
        return names
    return names + [BACKGROUND_CLASS_NAME]


def semantic_max_pixel_value(num_classes: int) -> int:
    """Largest legal (non-ignore) pixel value for a dataset with ``num_classes`` classes."""
    return 1 if num_classes <= 1 else int(num_classes)
