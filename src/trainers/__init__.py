# trainers package
from .yolo_trainer import main as yolo_main
from .nas_trainer import main as nas_main

__all__ = ["yolo_main", "nas_main"]
