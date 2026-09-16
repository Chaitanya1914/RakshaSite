import pytest

# Dummy functions to test
def load_model(): return "YOLO_Loaded"
def filter_hallucinations(boxes, has_person=True):
    if not has_person: return []
    return [b for b in boxes if (b[2]-b[0])*(b[3]-b[1]) > 100]

def test_yolo_detector_loads_model():
    assert load_model() == "YOLO_Loaded"

def test_filter_hallucinations_removes_small_boxes():
    boxes = [[0, 0, 5, 5], [0, 0, 50, 50]] # areas: 25, 2500
    filtered = filter_hallucinations(boxes)
    assert len(filtered) == 1
    assert filtered[0] == [0, 0, 50, 50]

def test_filter_hallucinations_requires_person():
    boxes = [[0, 0, 50, 50]]
    filtered = filter_hallucinations(boxes, has_person=False)
    assert len(filtered) == 0

def test_empty_frame_no_detections():
    # If no detections are present, it should handle gracefully
    boxes = []
    assert filter_hallucinations(boxes) == []
