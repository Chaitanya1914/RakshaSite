import logging
from typing import Any, Dict, List
from src.agents.state import SafetyState
from src.detection.utils import compute_box_area

logger = logging.getLogger(__name__)

VALID_VIOLATION_CLASSES = {
    "no_helmet": "no_helmet",
    "NO_HELMET": "no_helmet",
    "No Helmet": "no_helmet",
    "no_vest": "no_vest",
    "NO_VEST": "no_vest",
    "No Vest": "no_vest",
    "no_harness": "no_harness",
    "NO_HARNESS": "no_harness",
    "No Harness": "no_harness",
    "no_gloves": "no_gloves",
    "NO_GLOVES": "no_gloves",
    "No Gloves": "no_gloves",
    "no_boots": "no_boots",
    "NO_BOOTS": "no_boots",
    "No Boots": "no_boots",
}


def detection_agent(state: SafetyState) -> SafetyState:
    """
    Agent 1 — Detection & Validation Agent:
    1. Filters out low-confidence CV detections (< 0.70 threshold).
    2. Filters out tiny bounding-box hallucinations (< 25 px area when bbox is provided).
    3. Enforces consecutive-frame temporal confirmation (default >= 5 frames when
       `consecutive_frames` is tracked in stream mode).
    4. Normalizes both raw YOLO detections and DeepSORT TrackedWorker violation dicts.
    """
    raw_detections: List[Dict[str, Any]] = state.get("raw_detections", [])
    logger.info(f"Running detection agent on {len(raw_detections)} raw detections")

    validated_violations: List[Dict[str, Any]] = []

    for det in raw_detections:
        confidence = float(det.get("confidence", 0.0))

        # Check bounding box minimum area if a 4-element bbox is present
        bbox = list(det.get("bbox", []))
        if len(bbox) == 4:
            area = compute_box_area((bbox[0], bbox[1], bbox[2], bbox[3]))
            if 0.0 < area < 25.0:
                logger.debug(f"Filtered out tiny bounding box hallucination (area={area:.1f})")
                continue

        # Consecutive frame confirmation gate (5 frames when streaming)
        consecutive_frames = det.get("consecutive_frames", None)
        min_required_frames = int(det.get("min_consecutive_frames", 5))
        if consecutive_frames is not None and int(consecutive_frames) < min_required_frames:
            logger.debug(
                f"Filtered out unconfirmed violation ({consecutive_frames}/{min_required_frames} consecutive frames)"
            )
            continue

        # Case A: TrackedWorker record with a list of violations (e.g. ["NO_HELMET", "NO_VEST"])
        if "violations" in det and isinstance(det["violations"], list):
            if confidence < 0.70:
                continue
            for v_raw in det["violations"]:
                norm_v = VALID_VIOLATION_CLASSES.get(v_raw)
                if norm_v:
                    validated_violations.append(
                        {
                            "worker_id": str(det.get("worker_id", det.get("track_id", "unknown"))),
                            "violation_type": norm_v,
                            "confidence": round(confidence, 4),
                            "bbox": bbox,
                            "zone": det.get("zone", "ground_level"),
                            "consecutive_frames": int(consecutive_frames) if consecutive_frames is not None else 5,
                            "proximity": float(det.get("proximity", 0.2)),
                        }
                    )
            continue

        # Case B: Direct detection dictionary with `class_name` or `violation_type`
        raw_cls = det.get("class_name") or det.get("violation_type") or ""
        norm_cls = VALID_VIOLATION_CLASSES.get(raw_cls)

        if confidence >= 0.70 and norm_cls is not None:
            logger.info(f"Validated violation found: {norm_cls} (conf={confidence:.2f})")
            validated_violations.append(
                {
                    "worker_id": str(det.get("worker_id", det.get("track_id", "unknown"))),
                    "violation_type": norm_cls,
                    "confidence": round(confidence, 4),
                    "bbox": bbox,
                    "zone": det.get("zone", "general"),
                    "consecutive_frames": int(consecutive_frames) if consecutive_frames is not None else 5,
                    "proximity": float(det.get("proximity", 0.2)),
                }
            )

    return {**state, "validated_violations": validated_violations}
