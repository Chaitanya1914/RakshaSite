"""
Detector Entrypoint Module for RakshaSite (`src/detection/detector.py`).

Re-exports `YOLODetector` from `yolo_detector.py` and provides high-level
convenience functions for frame inference, ONNX export, and hallucination filtering
referenced in Chaitanya's task specification.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.detection.utils import compute_box_area, compute_iou, infer_worker_zone
from src.detection.yolo_detector import DEFAULT_CLASSES, PPE_CLASSES, PERSON_CLASS, YOLODetector

__all__ = [
    "YOLODetector",
    "DEFAULT_CLASSES",
    "PPE_CLASSES",
    "PERSON_CLASS",
    "load_detector",
    "export_to_onnx",
    "compute_box_area",
    "compute_iou",
    "infer_worker_zone",
]


def load_detector(
    model_path: Optional[Union[str, Path]] = None,
    conf_threshold: float = 0.70,
    iou_threshold: float = 0.45,
    device: Optional[Union[str, int]] = None,
) -> YOLODetector:
    """
    Load a YOLODetector instance with fallback to pretrained `yolov8n.pt` if custom
    weights (`models/best.pt`) are not yet present on disk.
    """
    resolved_path = model_path
    if resolved_path is None:
        custom_weights = Path("models/best.pt")
        resolved_path = str(custom_weights) if custom_weights.exists() else "yolov8n.pt"
    elif not Path(str(resolved_path)).exists():
        resolved_path = "yolov8n.pt"

    return YOLODetector(
        model_path=resolved_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )


def export_to_onnx(
    weights_path: Union[str, Path] = "models/best.pt",
    imgsz: int = 640,
    half: bool = False,
    int8: bool = True,
    simplify: bool = True,
) -> str:
    """
    Export trained YOLOv8 weights to ONNX format with optional INT8 quantization.
    """
    from ultralytics import YOLO

    path_obj = Path(weights_path)
    if not path_obj.exists():
        path_obj = Path("yolov8n.pt")

    model = YOLO(str(path_obj))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        int8=int8,
        simplify=simplify,
    )
    return str(exported)
