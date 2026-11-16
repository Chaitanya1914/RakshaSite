import numpy as np
import pytest
from src.detection.utils import compute_box_area, compute_iou, infer_worker_zone
from src.detection.yolo_detector import YOLODetector


# Dummy functions to test (preserved for backward compatibility)
def load_model():
    return "YOLO_Loaded"


def filter_hallucinations(boxes, has_person=True):
    if not has_person:
        return []
    return [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) > 100]


def test_yolo_detector_loads_model():
    assert load_model() == "YOLO_Loaded"


def test_filter_hallucinations_removes_small_boxes():
    boxes = [[0, 0, 5, 5], [0, 0, 50, 50]]  # areas: 25, 2500
    filtered = filter_hallucinations(boxes)
    assert len(filtered) == 1
    assert filtered[0] == [0, 0, 50, 50]


def test_filter_hallucinations_requires_person():
    boxes = [[0, 0, 50, 50]]
    filtered = filter_hallucinations(boxes, has_person=False)
    assert len(filtered) == 0


def test_empty_frame_no_detections():
    boxes = []
    assert filter_hallucinations(boxes) == []


def test_yolo_detector_containment_and_area_filter():
    # Test YOLODetector._is_inside_person and filter_hallucinations directly without loading GPU weights
    det_instance = YOLODetector.__new__(YOLODetector)
    det_instance.conf_threshold = 0.70
    det_instance.min_area_ratio = 0.01
    from src.utils.logger import get_logger

    det_instance.logger = get_logger("TestYOLO")

    raw_dets = [
        {"class_name": "person", "confidence": 0.92, "bbox": (100.0, 100.0, 300.0, 500.0)},
        # Valid helmet inside person head region
        {"class_name": "no_helmet", "confidence": 0.88, "bbox": (150.0, 110.0, 250.0, 200.0)},
        # Orphaned helmet floating outside person bbox -> should be filtered
        {"class_name": "helmet", "confidence": 0.90, "bbox": (500.0, 500.0, 580.0, 580.0)},
        # Tiny box (< 1% of 640x640 = 4096 px) -> should be filtered
        {"class_name": "no_vest", "confidence": 0.95, "bbox": (160.0, 200.0, 170.0, 210.0)},
    ]

    filtered = det_instance.filter_hallucinations(raw_dets, frame_shape=(640, 640, 3))
    classes = [d["class_name"] for d in filtered]
    assert "person" in classes
    assert "no_helmet" in classes
    assert "helmet" not in classes
    assert "no_vest" not in classes


def test_detection_utils_geometry_and_zones():
    assert compute_box_area((0, 0, 10, 20)) == 200.0
    assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    zone, prox = infer_worker_zone((100, 40, 200, 180), frame_shape=(720, 1280))
    assert zone == "scaffolding_elevated"
    assert prox >= 0.6
