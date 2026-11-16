"""
Unit tests for scratch/evaluate_ppe_metrics.py
Tests class remapping, label parsing, hardware detection, baseline extraction,
and JSON schema validation for real_val_metrics.json and detection_experiments.json.
"""

import json
from pathlib import Path
import pytest

from scratch.evaluate_ppe_metrics import (
    CPP_TO_BEST_MAP,
    BEST_TO_CANONICAL,
    CANONICAL_CLASSES,
    VIOLATION_CLASSES,
    POSITIVE_CLASSES,
    box_iou,
    box_overlap_fraction,
    is_in_head_region,
    evaluate_worker_violations,
    remap_class_id,
    get_canonical_name,
    remap_label_content,
    load_training_baseline,
    detect_hardware_environment,
    validate_real_val_metrics_schema,
    validate_detection_experiments_schema,
)



def test_cpp_to_best_mapping():
    """Verify that construction-ppe classes map to the correct best.pt classes."""
    assert remap_class_id(0) == 0    # helmet -> Hardhat
    assert remap_class_id(2) == 7    # vest -> Safety Vest
    assert remap_class_id(5) == 5    # Person -> Person
    assert remap_class_id(6) == 2    # no_helmet -> NO-Hardhat
    assert remap_class_id(11) == 4   # no_vest -> NO-Safety Vest

    # Excluded classes must return None
    excluded = [1, 3, 4, 7, 8, 9, 10, 99]
    for cls_id in excluded:
        assert remap_class_id(cls_id) is None


def test_best_to_canonical_mapping():
    """Verify that best.pt class IDs map to canonical safety monitoring names."""
    assert get_canonical_name(0) == "helmet"
    assert get_canonical_name(2) == "no_helmet"
    assert get_canonical_name(4) == "no_vest"
    assert get_canonical_name(5) == "person"
    assert get_canonical_name(7) == "vest"

    # Non-monitored classes must return None
    for non_monitored in [1, 3, 6, 8, 9, 100]:
        assert get_canonical_name(non_monitored) is None


def test_canonical_classes_definition():
    """Verify standard canonical class sets."""
    assert set(CANONICAL_CLASSES) == {"person", "helmet", "no_helmet", "vest", "no_vest"}
    assert set(VIOLATION_CLASSES) == {"no_helmet", "no_vest"}
    assert set(POSITIVE_CLASSES) == {"person", "helmet", "vest"}


def test_remap_label_content_basic():
    """Verify that multi-line labels are correctly filtered and remapped."""
    content = (
        "0 0.50 0.50 0.10 0.10\n"    # helmet -> 0
        "1 0.20 0.20 0.05 0.05\n"    # gloves -> excluded
        "6 0.30 0.30 0.15 0.15\n"    # no_helmet -> 2
        "11 0.60 0.60 0.20 0.20\n"   # no_vest -> 4
        "3 0.80 0.80 0.10 0.10\n"    # boots -> excluded
        "5 0.50 0.50 0.40 0.80\n"    # Person -> 5
        "2 0.50 0.60 0.30 0.40\n"    # vest -> 7
    )
    remapped, valid_count, counts = remap_label_content(content)
    assert valid_count == 5
    lines = remapped.strip().splitlines()
    assert len(lines) == 5

    assert lines[0].startswith("0 ")
    assert lines[1].startswith("2 ")
    assert lines[2].startswith("4 ")
    assert lines[3].startswith("5 ")
    assert lines[4].startswith("7 ")

    assert counts["helmet"] == 1
    assert counts["no_helmet"] == 1
    assert counts["no_vest"] == 1
    assert counts["person"] == 1
    assert counts["vest"] == 1


def test_remap_label_content_float_class_tokens():
    """Verify that float class tokens (e.g. '0.0') are safely parsed."""
    content = "0.0 0.5 0.5 0.2 0.2\n6.0 0.1 0.1 0.2 0.2\n"
    remapped, count, counts = remap_label_content(content)
    assert count == 2
    assert counts["helmet"] == 1
    assert counts["no_helmet"] == 1


def test_remap_label_content_empty_and_whitespace():
    """Verify handling of empty or whitespace strings."""
    remapped, count, counts = remap_label_content("   \n\n  \t  \n")
    assert count == 0
    assert remapped == ""
    assert all(v == 0 for v in counts.values())


def test_load_training_baseline_existing():
    """Verify baseline loading from actual training_plots/results.csv."""
    csv_path = Path("models/training_plots/results.csv")
    baseline = load_training_baseline(csv_path)
    assert baseline["precision"] == 0.8999
    assert baseline["recall"] == 0.7311
    assert baseline["map50"] == 0.8088
    assert baseline["map50_95"] == 0.5071
    assert baseline["training_epochs"] == 100


def test_load_training_baseline_missing():
    """Verify fallback baseline loading when file is absent."""
    missing_path = Path("non_existent_results.csv")
    baseline = load_training_baseline(missing_path)
    assert baseline["precision"] == 0.8999
    assert baseline["map50"] == 0.8088


def test_detect_hardware_environment():
    """Verify hardware environment detection structure and justification."""
    hw = detect_hardware_environment()
    assert hw["device"] in ("cpu", "cuda", "cuda:0", 0)
    assert "cpu_model" in hw
    assert "torch_version" in hw
    assert "retraining_decision" in hw
    assert hw["retraining_decision"] in ("KEEP_BASELINE", "ADOPT_FINETUNED", "FINE_TUNE")
    assert isinstance(hw["justification"], list)
    assert len(hw["justification"]) >= 3


def test_validate_real_val_metrics_schema_success():
    """Verify schema validation succeeds on valid mock structure."""
    valid_data = {
        "model_metadata": {
            "weights_file": "models/best.pt",
            "architecture": "YOLOv8n",
            "model_classes": ["Hardhat", "Person"],
        },
        "baseline_metrics_roboflow_100epochs": {"map50": 0.8088},
        "held_out_evaluation": {
            "image_count": 200,
            "precision": 0.6925,
            "map50": 0.562,
            "per_class_metrics": {},
        },
        "hardware_and_performance": {
            "device": "cpu",
            "torch_version": "2.10.0+cpu",
            "inference_speed_ms": {"total_latency_ms": 54.3},
        },
        "baseline_decision": {
            "decision": "KEEP_BASELINE",
            "justification": ["Reason 1"],
        },
    }
    assert validate_real_val_metrics_schema(valid_data) is True


def test_validate_real_val_metrics_schema_failure():
    """Verify schema validation fails if required keys are missing."""
    invalid_data = {
        "model_metadata": {"weights_file": "models/best.pt"},
        # missing other top level keys
    }
    assert validate_real_val_metrics_schema(invalid_data) is False


def test_validate_detection_experiments_schema_success():
    """Verify schema validation succeeds on valid detection experiments structure."""
    valid_data = {
        "timestamp": "2026-11-06T00:00:00Z",
        "baseline_model": "models/best.pt",
        "baseline_metrics": {"map50": 0.562},
        "inference_sweeps": [{"experiment_id": "test", "parameters": {}}],
        "fine_tuning_experiments": [],
        "best_inference_configuration": {"name": "test"},
        "final_decision": {"decision": "KEEP_BASELINE"},
    }
    assert validate_detection_experiments_schema(valid_data) is True


def test_validate_detection_experiments_schema_empty_sweeps():
    """Verify schema validation fails if sweeps list is empty."""
    invalid_data = {
        "timestamp": "2026-11-06T00:00:00Z",
        "baseline_model": "models/best.pt",
        "baseline_metrics": {},
        "inference_sweeps": [],  # empty sweeps
        "fine_tuning_experiments": [],
        "best_inference_configuration": {},
        "final_decision": {},
    }
    assert validate_detection_experiments_schema(invalid_data) is False


def test_generated_real_val_metrics_json():
    """Verify models/real_val_metrics.json on disk if present."""
    metrics_path = Path("models/real_val_metrics.json")
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert validate_real_val_metrics_schema(data) is True
        assert data["held_out_evaluation"]["image_count"] == 200
        assert data["held_out_evaluation"]["synthetic_leakage"] is False
        assert "helmet" in data["held_out_evaluation"]["per_class_metrics"]
        assert "no_helmet" in data["held_out_evaluation"]["per_class_metrics"]
        assert "no_vest" in data["held_out_evaluation"]["per_class_metrics"]


def test_generated_detection_experiments_json():
    """Verify models/detection_experiments.json on disk if present."""
    exp_path = Path("models/detection_experiments.json")
    if exp_path.exists():
        data = json.loads(exp_path.read_text(encoding="utf-8"))
        assert validate_detection_experiments_schema(data) is True
        assert len(data["inference_sweeps"]) >= 1
        assert "final_decision" in data


def test_box_iou_perfect_and_disjoint():
    """Verify box_iou calculation on exact overlaps and completely disjoint boxes."""
    b1 = [0.1, 0.1, 0.5, 0.5]
    assert box_iou(b1, b1) == pytest.approx(1.0)

    b2 = [0.6, 0.6, 0.9, 0.9]
    assert box_iou(b1, b2) == pytest.approx(0.0)

    # Partial overlap: [0, 0, 2, 2] area 4, [1, 0, 3, 2] area 4, inter [1, 0, 2, 2] area 2, union 6 -> 2/6 = 0.3333
    b3 = [0.0, 0.0, 2.0, 2.0]
    b4 = [1.0, 0.0, 3.0, 2.0]
    assert box_iou(b3, b4) == pytest.approx(2.0 / 6.0)


def test_box_overlap_fraction():
    """Verify box_overlap_fraction on contained and disjoint boxes."""
    inner = [0.2, 0.2, 0.4, 0.4]
    outer = [0.1, 0.1, 0.5, 0.5]
    assert box_overlap_fraction(inner, outer) == pytest.approx(1.0)

    disjoint = [0.6, 0.6, 0.8, 0.8]
    assert box_overlap_fraction(inner, disjoint) == pytest.approx(0.0)


def test_is_in_head_region():
    """Verify head region detection (upper 35% of person bounding box)."""
    # Person box: y from 0.0 to 1.0 (head is 0.0 to 0.35)
    person = [0.2, 0.0, 0.6, 1.0]

    # Helmet located at top of person (y 0.02 to 0.20) -> must be in head region
    helmet_on_head = [0.25, 0.02, 0.55, 0.18]
    assert is_in_head_region(helmet_on_head, person) is True

    # Helmet located at feet (y 0.80 to 0.95) -> must NOT be in head region
    helmet_at_feet = [0.25, 0.80, 0.55, 0.95]
    assert is_in_head_region(helmet_at_feet, person) is False

