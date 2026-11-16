"""
RakshaSite — End-to-End Live Demo & Video Compliance Auditor (`demo_runner.py`).

Runs the complete RakshaSite pipeline on real construction site video (`data/raw/hardhat_demo.mp4`),
sample images (`data/raw/samples/construction-safety.jpg`), or live webcam (`--source 0`):
  1. YOLOv8 Real-World PPE Detector (`models/best.pt`, 100-epoch Roboflow CSS checkpoint)
  2. DeepSORT Multi-Worker Tracker (`WorkerTracker`) with consecutive-frame streak confirmation
  3. Spatial Containment Analysis: PPE overlap verification (identifies workers lacking helmets/vests)
  4. 4-Agent LangGraph Pipeline:
     - Detection Agent -> Risk Agent (XGBoost) -> Compliance Agent (ChromaDB BOCW RAG) -> Report Agent
  5. Renders a crisp executive HUD overlay onto video/images and generates:
     - `reports/demo_annotated_output.mp4` (browser-playable H.264 video)
     - `reports/demo_snapshot_violation.jpg` (annotated forensic violation snapshot)
     - `reports/safety_audit.pdf` (formal BOCW Form IV PDF report)
     - `reports/demo_audit_log.json` (machine-readable audit log)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.agents.pipeline import run_pipeline
from src.agents.report_agent import generate_pdf_report
from src.detection.tracker import WorkerTracker
from src.detection.utils import infer_worker_zone
from src.detection.yolo_detector import YOLODetector


def create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """
    Create a browser-compatible video writer with tiered fallback:
      Tier 1: Windows Media Foundation (MSMF) with native H.264 encoder
      Tier 2: Windows Media Foundation with AVC1
      Tier 3: OpenCV default backend with AVC1
      Tier 4: OpenCV default backend with mp4v
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codecs: List[Tuple[int, str]] = [
        (cv2.CAP_MSMF, "H264"),
        (cv2.CAP_MSMF, "avc1"),
        (cv2.CAP_ANY, "avc1"),
        (cv2.CAP_ANY, "mp4v"),
    ]

    for api_pref, fourcc_str in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            vw = cv2.VideoWriter(str(output_path), api_pref, fourcc, fps, (width, height))
            if vw.isOpened():
                return vw
        except Exception:
            continue

    # Clean universal fallback
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def draw_executive_hud(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    latest_state: Dict[str, Any],
    frame_idx: int,
    active_workers: int,
    violating_workers: int,
    tracked_workers: Optional[List[Any]] = None,
) -> np.ndarray:
    """
    Draw bounding boxes and a high-contrast executive HUD banner showing live
    4-Agent LangGraph reasoning, compliance score, risk tier, and BOCW Act citations.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Draw raw detection boxes (PPE items like helmet, vest, machinery)
    for d in detections:
        c_name = d.get("class_name", "")
        if c_name in ("person",):
            continue  # Workers will be drawn with rich badges below
        bbox = d.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in bbox]
        conf = float(d.get("confidence", 0.0))

        if c_name in ("helmet", "vest"):
            color = (0, 230, 118) if c_name == "helmet" else (0, 215, 255)
        elif c_name in ("no_helmet", "no_vest"):
            color = (0, 0, 255)
        else:
            color = (255, 140, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{c_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255) if color != (0, 215, 255) else (10, 10, 10),
            1,
            cv2.LINE_AA,
        )

    # 2. Draw Tracked Workers with Risk-Tier Color Bounding Boxes and Status Badges
    if tracked_workers:
        for tw in tracked_workers:
            bbox = tw.get("bbox", (0, 0, 0, 0)) if isinstance(tw, dict) else tw.bbox
            x1, y1, x2, y2 = [int(v) for v in bbox]
            w_id = tw.get("worker_id", tw.get("track_id", "w-1")) if isinstance(tw, dict) else tw.track_id
            violations = tw.get("violations", []) if isinstance(tw, dict) else getattr(tw, "violations", [])
            is_comp = tw.get("is_compliant", True) if isinstance(tw, dict) else getattr(tw, "is_compliant", True)

            if is_comp:
                box_color = (0, 220, 0)
                badge_text = f"Worker #{w_id} [SAFE]"
            else:
                box_color = (0, 0, 255)
                v_str = ",".join(violations) if violations else "VIOLATION"
                badge_text = f"Worker #{w_id} [ALERT: {v_str}]"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            (tw_w, tw_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - tw_h - 6)), (x1 + tw_w + 6, y1), box_color, -1)
            cv2.putText(
                annotated,
                badge_text,
                (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # 3. Top Translucent HUD Bar
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (20, 25, 35), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    comp_score = float(latest_state.get("compliance_score", 100.0) or 100.0)
    if comp_score >= 80:
        score_color = (0, 230, 118)
        risk_label = "SAFE"
    elif comp_score >= 60:
        score_color = (0, 190, 255)
        risk_label = "MODERATE"
    else:
        score_color = (60, 60, 255)
        risk_label = "CRITICAL"

    cv2.putText(
        annotated,
        f"RAKSHASITE AI SAFETY AUDITOR | Frame #{frame_idx:03d} | Workers: {active_workers} | Violators: {violating_workers}",
        (14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (240, 245, 250),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Site Compliance: {comp_score:.1f}% [{risk_label}] | 100% Offline BOCW Compliance Engine",
        (14, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        score_color,
        2,
        cv2.LINE_AA,
    )

    # 4. Bottom Translucent Statutory Ticker if violations exist
    regs = latest_state.get("regulations", [])
    if regs:
        top_reg = regs[0]
        reg_info = top_reg.get("regulation", {})
        v_type = top_reg.get("violation_type", "no_helmet").upper()
        r_label = top_reg.get("risk_label", "HIGH")
        r_score = float(top_reg.get("risk_score", 70.0))
        sec_id = reg_info.get("section_id", "Section 38(1)(a) & Rule 46")

        bot_overlay = annotated.copy()
        cv2.rectangle(bot_overlay, (0, h - 52), (w, h), (18, 18, 48), -1)
        cv2.addWeighted(bot_overlay, 0.88, annotated, 0.12, 0, annotated)

        cv2.putText(
            annotated,
            f"[AGENT ALERT] {v_type} | Risk: {r_label} ({r_score:.0f}/100) | Law: {sec_id}",
            (14, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (80, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Statutory Penalty: {reg_info.get('penalty', 'Section 47: Fine up to Rs. 2,000 / 3 mo imprisonment')}",
            (14, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (210, 215, 225),
            1,
            cv2.LINE_AA,
        )

    return annotated


def run_demo(
    video_source: str = "data/raw/hardhat_demo.mp4",
    sample_image: str = "data/raw/samples/construction-safety.jpg",
    max_frames: int = 90,
    start_frame: int = 120,
    conf_threshold: float = 0.25,
    output_dir: str = "reports",
    device: str = "auto",
    offline: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete RakshaSite demo pipeline on real construction footage (still image + video).
    Generates 4 output artifacts in output_dir:
      1. demo_annotated_output.mp4
      2. demo_snapshot_violation.jpg
      3. safety_audit.pdf
      4. demo_audit_log.json
    """
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    reports_dir = Path(output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    target_device = None if device == "auto" else device
    detector = YOLODetector(
        model_path="models/best.pt",
        conf_threshold=conf_threshold,
        min_area_ratio=0.0005,
        device=target_device,
    )
    tracker = WorkerTracker(max_age=30, n_init=1)

    # -------------------------------------------------------------
    # 1. Process Still Image with Spatial Containment Analysis
    # -------------------------------------------------------------
    img_path = Path(sample_image)
    snapshot_out = reports_dir / "demo_snapshot_violation.jpg"
    snapshot_state: Dict[str, Any] = {}
    snapshot_annotated: Optional[np.ndarray] = None

    if img_path.exists():
        img = cv2.imread(str(img_path))
        if img is not None:
            dets = detector.detect(img)
            machinery_boxes = [d["bbox"] for d in dets if d["class_name"] in ("machinery", "vehicle")]
            person_boxes = [d for d in dets if d["class_name"] == "person"]
            ppe_dets = [d for d in dets if d["class_name"] != "person"]

            # Perform spatial containment analysis via WorkerTracker
            candidate_workers = [
                {"worker_id": f"w-{idx + 1:02d}", "bbox": d["bbox"]}
                for idx, d in enumerate(person_boxes)
            ]
            violation_records = tracker.get_worker_violations(
                candidate_workers, ppe_detections=ppe_dets, only_violators=False
            )

            agent_inputs = []
            for rec in violation_records:
                for v_code in rec.get("violations", []):
                    cls_name = "no_helmet" if v_code == "NO_HELMET" else "no_vest"
                    zone, prox = infer_worker_zone(rec["bbox"], img.shape[:2], machinery_boxes)
                    agent_inputs.append(
                        {
                            "worker_id": rec["worker_id"],
                            "class_name": cls_name,
                            "confidence": 0.88,
                            "bbox": list(rec["bbox"]),
                            "zone": zone,
                            "proximity": prox,
                            "consecutive_frames": 5,
                        }
                    )

            # Include any explicit raw violation detections not already captured
            for d in dets:
                if d["class_name"] in ("no_helmet", "no_vest"):
                    zone, prox = infer_worker_zone(d["bbox"], img.shape[:2], machinery_boxes)
                    agent_inputs.append(
                        {
                            "worker_id": f"worker-raw-{len(agent_inputs) + 1:02d}",
                            "class_name": d["class_name"],
                            "confidence": max(0.75, float(d["confidence"])),
                            "bbox": list(d["bbox"]),
                            "zone": zone,
                            "proximity": prox,
                            "consecutive_frames": 5,
                        }
                    )

            if agent_inputs:
                snapshot_state = run_pipeline(
                    agent_inputs,
                    camera_id="CAM-01-SNAPSHOT",
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                snapshot_state = {"compliance_score": 100.0, "regulations": []}

            snapshot_annotated = draw_executive_hud(
                img,
                dets,
                snapshot_state,
                frame_idx=1,
                active_workers=len(person_boxes),
                violating_workers=len(agent_inputs),
                tracked_workers=violation_records,
            )
            cv2.imwrite(str(snapshot_out), snapshot_annotated)

    # -------------------------------------------------------------
    # 2. Process Real Video Stream with VideoWriter and HUD Overlay
    # -------------------------------------------------------------
    video_out_path = reports_dir / "demo_annotated_output.mp4"
    cap = cv2.VideoCapture(0 if video_source == "0" else video_source)
    writer: Optional[cv2.VideoWriter] = None
    frame_idx = 0
    total_violation_events = 0
    best_state: Dict[str, Any] = snapshot_state
    peak_violation_frame: Optional[np.ndarray] = None
    min_compliance_seen = float(snapshot_state.get("compliance_score", 100.0) or 100.0)

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        total_stream_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Handle start_frame offset: seek straight to worker activity in hardhat_demo.mp4
        actual_start_frame = start_frame
        if video_source == "0" or (total_stream_frames > 0 and actual_start_frame >= total_stream_frames):
            actual_start_frame = 0

        if actual_start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start_frame)

        writer = create_video_writer(video_out_path, fps, width, height)

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            dets = detector.detect(frame)
            tracked_workers = tracker.update(dets, frame)

            machinery_boxes = [d["bbox"] for d in dets if d["class_name"] in ("machinery", "vehicle")]
            frame_violations: List[Dict[str, Any]] = []

            # 1. Check violations on confirmed tracked workers (spatial containment)
            for tw in tracked_workers:
                for v_code in tw.violations:
                    cls_name = "no_helmet" if v_code == "NO_HELMET" else "no_vest"
                    zone, prox = infer_worker_zone(tw.bbox, frame.shape[:2], machinery_boxes)
                    frame_violations.append(
                        {
                            "worker_id": f"w-{tw.track_id}",
                            "class_name": cls_name,
                            "confidence": 0.85,
                            "bbox": [round(float(x), 1) for x in tw.bbox],
                            "zone": zone,
                            "proximity": prox,
                            "consecutive_frames": getattr(tw, "consecutive_frames", 5),
                        }
                    )

            # 2. Check explicit raw detections from detector
            for d in dets:
                if d["class_name"] in ("no_helmet", "no_vest"):
                    # Check if already covered by tracked worker
                    already_covered = any(
                        fv["class_name"] == d["class_name"]
                        for fv in frame_violations
                    )
                    if not already_covered:
                        zone, prox = infer_worker_zone(d["bbox"], frame.shape[:2], machinery_boxes)
                        frame_violations.append(
                            {
                                "worker_id": f"worker-det-{len(frame_violations) + 1:02d}",
                                "class_name": d["class_name"],
                                "confidence": max(0.75, float(d["confidence"])),
                                "bbox": [round(float(x), 1) for x in d["bbox"]],
                                "zone": zone,
                                "proximity": prox,
                                "consecutive_frames": 5,
                            }
                        )

            if frame_violations:
                total_violation_events += len(frame_violations)
                latest_state = run_pipeline(
                    frame_violations,
                    camera_id="CAM-01-LIVE",
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                curr_score = float(latest_state.get("compliance_score", 100.0) or 100.0)
                if curr_score < min_compliance_seen or len(latest_state.get("regulations", [])) >= len(best_state.get("regulations", [])):
                    min_compliance_seen = curr_score
                    best_state = latest_state
            else:
                latest_state = best_state if best_state else {"compliance_score": 100.0, "regulations": []}

            persons_cnt = max(len(tracked_workers), sum(1 for d in dets if d["class_name"] == "person"))
            annotated_frame = draw_executive_hud(
                frame,
                dets,
                latest_state,
                frame_idx=frame_idx,
                active_workers=persons_cnt,
                violating_workers=len(frame_violations),
                tracked_workers=tracked_workers,
            )

            if frame_violations and peak_violation_frame is None:
                peak_violation_frame = annotated_frame.copy()

            if writer is not None:
                writer.write(annotated_frame)

        cap.release()
        if writer is not None:
            writer.release()

    # If the still image snapshot was not captured or had no violations, use peak video frame
    if peak_violation_frame is not None and (not snapshot_out.exists() or len(snapshot_state.get("regulations", [])) == 0):
        cv2.imwrite(str(snapshot_out), peak_violation_frame)

    # Fallback placeholder if neither video nor still image produced a snapshot
    if not snapshot_out.exists():
        blank_canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        hud_blank = draw_executive_hud(blank_canvas, [], best_state, 1, 0, 0)
        cv2.imwrite(str(snapshot_out), hud_blank)

    # -------------------------------------------------------------
    # 3. Generate Formal BOCW PDF Audit Report & JSON Audit Log
    # -------------------------------------------------------------
    pdf_out = reports_dir / "safety_audit.pdf"
    pdf_path = generate_pdf_report(best_state, str(pdf_out))

    summary: Dict[str, Any] = {
        "video_source": video_source,
        "frames_processed": frame_idx,
        "total_violation_detections": total_violation_events or len(best_state.get("regulations", [])),
        "annotated_video_output": str(video_out_path),
        "annotated_snapshot_output": str(snapshot_out),
        "pdf_audit_report": pdf_path,
        "sample_agent_output": {
            "compliance_score": best_state.get("compliance_score", 100.0),
            "alert_message": best_state.get("alert_message", "All inspected personnel compliant."),
            "regulations_cited": [
                {
                    "worker_id": r.get("worker_id", "w-01"),
                    "violation_type": r.get("violation_type", "unknown"),
                    "zone": r.get("zone", "ground_level"),
                    "risk_score": r.get("risk_score", 0.0),
                    "risk_label": r.get("risk_label", "LOW"),
                    "bocw_section": r.get("regulation", {}).get("section_id", "Section 38(1)(a)"),
                    "penalty": r.get("regulation", {}).get("penalty", "Section 47: Fine up to Rs. 2,000"),
                }
                for r in best_state.get("regulations", [])
            ],
        },
    }

    log_file = reports_dir / "demo_audit_log.json"
    log_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="RakshaSite Live Demo & Video Compliance Auditor")
    parser.add_argument("--video", "--source", dest="video", default="data/raw/hardhat_demo.mp4", help="Video file path or '0' for webcam")
    parser.add_argument("--image", default="data/raw/samples/construction-safety.jpg", help="Path to high-resolution sample still image")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold (default: 0.25)")
    parser.add_argument("--output-dir", default="reports", help="Directory where artifacts will be saved")
    parser.add_argument("--start-frame", type=int, default=120, help="Starting frame for video processing")
    parser.add_argument("--max-frames", type=int, default=90, help="Max video frames to process")
    parser.add_argument("--offline", action="store_true", default=True, help="Run 100% offline without remote LLM calls")
    parser.add_argument("--device", default="auto", help="Inference device: 'auto', 'cpu', or 'cuda'")
    args = parser.parse_args()

    result = run_demo(
        video_source=args.video,
        sample_image=args.image,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        conf_threshold=args.conf,
        output_dir=args.output_dir,
        device=args.device,
        offline=args.offline,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
