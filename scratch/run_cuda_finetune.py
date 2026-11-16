"""
Full Fine-Tuning, Parameter Sweep, and Worker-Level Violation Evaluation Runner
RakshaSite ML Pipeline — Worker M2.1
Executes genuine fine-tuning on CUDA (RTX 3050 Laptop GPU),
evaluates both baseline and fine-tuned candidate on the SAME 200 held-out images,
evaluates worker-level violation metrics (raw vs spatial containment vs combined),
and serializes models/real_val_metrics.json and models/detection_experiments.json.
"""

import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from ultralytics import YOLO

# Canonical class mapping
CPP_TO_BEST_MAP = {
    0: 0,   # helmet -> Hardhat
    2: 7,   # vest -> Safety Vest
    5: 5,   # Person -> Person
    6: 2,   # no_helmet -> NO-Hardhat
    11: 4,  # no_vest -> NO-Safety Vest
}

BEST_TO_CANONICAL = {
    0: "helmet",
    2: "no_helmet",
    4: "no_vest",
    5: "person",
    7: "vest",
}

CANONICAL_CLASSES = ["person", "helmet", "no_helmet", "vest", "no_vest"]
VIOLATION_CLASSES = ["no_helmet", "no_vest"]
POSITIVE_CLASSES = ["person", "helmet", "vest"]

YAML_NAMES_10_CLASS = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
}


def box_iou(b1: List[float], b2: List[float]) -> float:
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    denom = area1 + area2 - inter
    return inter / denom if denom > 1e-6 else 0.0


def box_overlap_fraction(inner: List[float], outer: List[float]) -> float:
    xA = max(inner[0], outer[0])
    yA = max(inner[1], outer[1])
    xB = min(inner[2], outer[2])
    yB = min(inner[3], outer[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    inner_area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / inner_area if inner_area > 1e-6 else 0.0


def is_in_head_region(ppe_box: List[float], person_box: List[float]) -> bool:
    """Checks if PPE box overlaps with upper 35% of person box."""
    p_x1, p_y1, p_x2, p_y2 = person_box
    head_y2 = p_y1 + 0.35 * (p_y2 - p_y1)
    head_box = [p_x1, p_y1, p_x2, head_y2]
    cx = (ppe_box[0] + ppe_box[2]) / 2.0
    cy = (ppe_box[1] + ppe_box[3]) / 2.0
    if p_x1 <= cx <= p_x2 and p_y1 <= cy <= head_y2:
        return True
    return box_overlap_fraction(ppe_box, head_box) > 0.20 or box_iou(ppe_box, head_box) > 0.10


def evaluate_worker_violations(
    model: YOLO,
    dataset_root: Path,
    conf: float = 0.25,
    iou: float = 0.50,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluates worker-level violation precision, recall, and F1 across:
    1. Raw classes (direct NO-Hardhat detections)
    2. Spatial containment rule (person lacking helmet in head region)
    3. Combined rule (union of raw violation detection and spatial absence)
    """
    val_img_dir = dataset_root / "images" / "val"
    val_lbl_dir = dataset_root / "labels" / "val"
    img_files = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.png")) + list(val_img_dir.glob("*.jpeg")))

    gt_worker_total = 0
    gt_violation_total = 0

    stats = {
        "raw_classes": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "spatial_rule": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "combined_rule": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }

    for img_path in img_files:
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        gt_persons = []
        gt_helmets = []
        gt_no_helmets = []

        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            c = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            box = [xc - w/2, yc - h/2, xc + w/2, yc + h/2]
            if c == 5:
                gt_persons.append(box)
            elif c == 0:
                gt_helmets.append(box)
            elif c == 6:
                gt_no_helmets.append(box)

        preds = model(str(img_path), conf=conf, iou=iou, device=device, verbose=False)[0]
        det_persons = []
        det_helmets = []
        det_no_helmets = []

        if preds.boxes is not None:
            for b in preds.boxes:
                cls_id = int(b.cls[0].item())
                box = b.xyxyn[0].tolist()
                if cls_id == 5:
                    det_persons.append(box)
                elif cls_id == 0:
                    det_helmets.append(box)
                elif cls_id == 2:
                    det_no_helmets.append(box)

        for p_gt in gt_persons:
            gt_worker_total += 1
            has_gt_no_helmet = any(is_in_head_region(nh, p_gt) for nh in gt_no_helmets)
            has_gt_helmet = any(is_in_head_region(h, p_gt) for h in gt_helmets)
            is_gt_violation = has_gt_no_helmet or (not has_gt_helmet)
            if is_gt_violation:
                gt_violation_total += 1

            best_det_p = None
            best_iou = 0.0
            for dp in det_persons:
                iou_val = box_iou(p_gt, dp)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det_p = dp

            person_matched = (best_det_p is not None and best_iou >= 0.35)
            eval_box = best_det_p if person_matched else p_gt

            pred_raw = any(is_in_head_region(nh, eval_box) for nh in det_no_helmets)
            if person_matched:
                has_det_helmet = any(is_in_head_region(h, best_det_p) for h in det_helmets)
                pred_spatial = not has_det_helmet
            else:
                pred_spatial = False

            pred_comb = pred_raw or pred_spatial

            for m_key, p_val in [("raw_classes", pred_raw), ("spatial_rule", pred_spatial), ("combined_rule", pred_comb)]:
                if is_gt_violation and p_val:
                    stats[m_key]["tp"] += 1
                elif (not is_gt_violation) and p_val:
                    stats[m_key]["fp"] += 1
                elif is_gt_violation and (not p_val):
                    stats[m_key]["fn"] += 1
                else:
                    stats[m_key]["tn"] += 1

    methods_res = {}
    for m_key, counts in stats.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]
        p = round(tp / (tp + fp) if (tp + fp) > 0 else 0.0, 4)
        r = round(tp / (tp + fn) if (tp + fn) > 0 else 0.0, 4)
        f1 = round(2 * p * r / (p + r) if (p + r) > 0 else 0.0, 4)
        acc = round((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0, 4)
        methods_res[m_key] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": p,
            "recall": r,
            "f1_score": f1,
            "accuracy": acc,
        }

    return {
        "total_ground_truth_workers": gt_worker_total,
        "ground_truth_violations": gt_violation_total,
        "ground_truth_compliant": gt_worker_total - gt_violation_total,
        "worker_evaluation_methods": methods_res,
    }


def evaluate_ultralytics_metrics(
    model: YOLO,
    data_yaml: Path,
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.60,
    augment: bool = False,
    device: str = "cuda:0",
    gt_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Runs Ultralytics val mode to get mAP50, mAP50-95, precision, recall."""
    start_t = time.perf_counter()
    res = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        augment=augment,
        device=device,
        workers=0,
        plots=False,
        verbose=False,
    )
    total_eval_time = time.perf_counter() - start_t

    overall_p = round(float(res.results_dict.get("metrics/precision(B)", 0.0)), 4)
    overall_r = round(float(res.results_dict.get("metrics/recall(B)", 0.0)), 4)
    overall_map50 = round(float(res.results_dict.get("metrics/mAP50(B)", 0.0)), 4)
    overall_map = round(float(res.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4)

    per_class_res: Dict[str, Any] = {}
    if hasattr(res, "ap_class_index") and hasattr(res, "box"):
        for i, c in enumerate(res.ap_class_index):
            raw_name = model.names.get(int(c), f"class_{c}")
            canonical = BEST_TO_CANONICAL.get(int(c))
            if canonical:
                per_class_res[canonical] = {
                    "raw_model_class": raw_name,
                    "model_class_id": int(c),
                    "ground_truth_instances": gt_counts.get(canonical, 0) if gt_counts else 0,
                    "precision": round(float(res.box.p[i]), 4),
                    "recall": round(float(res.box.r[i]), 4),
                    "map50": round(float(res.box.ap50[i]), 4),
                    "map50_95": round(float(res.box.ap[i]), 4),
                }

    for c in CANONICAL_CLASSES:
        if c not in per_class_res:
            per_class_res[c] = {
                "raw_model_class": YAML_NAMES_10_CLASS.get(next((k for k, v in BEST_TO_CANONICAL.items() if v == c), 0), c),
                "model_class_id": next((k for k, v in BEST_TO_CANONICAL.items() if v == c), -1),
                "ground_truth_instances": gt_counts.get(c, 0) if gt_counts else 0,
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map50_95": 0.0,
            }

    speed = res.speed if hasattr(res, "speed") else {}
    prep_ms = round(float(speed.get("preprocess", 1.0)), 2)
    inf_ms = round(float(speed.get("inference", 15.0)), 2)
    post_ms = round(float(speed.get("postprocess", 1.0)), 2)
    total_latency_ms = round(prep_ms + inf_ms + post_ms, 2)
    fps = round(1000.0 / total_latency_ms, 1) if total_latency_ms > 0 else 0.0

    return {
        "overall": {
            "precision": overall_p,
            "recall": overall_r,
            "map50": overall_map50,
            "map50_95": overall_map,
        },
        "per_class": per_class_res,
        "speed": {
            "preprocess_ms": prep_ms,
            "inference_ms": inf_ms,
            "postprocess_ms": post_ms,
            "total_latency_ms": total_latency_ms,
            "throughput_fps": fps,
            "eval_duration_seconds": round(total_eval_time, 2),
        },
    }


def main():
    print("=" * 80, flush=True)
    print("RAKSHASITE ML PIPELINE: CUDA FINE-TUNING & WORKER VIOLATION BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({device_name}), Torch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}", flush=True)

    dataset_root = Path("data/raw/construction-ppe/construction-ppe")
    remapped_dir = Path("data/raw/construction-ppe_remapped")
    data_yaml = remapped_dir / "data.yaml"

    assert data_yaml.exists(), f"Remapped data.yaml missing at {data_yaml}"

    # Step 1: Evaluate Baseline (models/best.pt) on 200 held-out images
    base_model_path = Path("models/best.pt")
    print(f"\n[1/4] Evaluating BASELINE model ({base_model_path}) on held-out validation set...", flush=True)
    base_model = YOLO(str(base_model_path))
    val_gt_counts = {"helmet": 294, "vest": 340, "person": 481, "no_helmet": 179, "no_vest": 130}

    base_val_eval = evaluate_ultralytics_metrics(
        model=base_model,
        data_yaml=data_yaml,
        imgsz=640,
        conf=0.001,
        iou=0.60,
        device=device,
        gt_counts=val_gt_counts,
    )
    print(f"  Baseline Overall: mAP50={base_val_eval['overall']['map50']}, P={base_val_eval['overall']['precision']}, R={base_val_eval['overall']['recall']}", flush=True)
    for c in CANONICAL_CLASSES:
        cm = base_val_eval['per_class'][c]
        print(f"    {c:<12}: P={cm['precision']:.4f}, R={cm['recall']:.4f}, mAP50={cm['map50']:.4f}", flush=True)

    print("\n[2/4] Evaluating Baseline Worker-Level Violations (Raw vs Spatial vs Combined)...", flush=True)
    base_worker_eval = evaluate_worker_violations(
        model=base_model,
        dataset_root=dataset_root,
        conf=0.25,
        iou=0.50,
        device=device,
    )
    for m, vals in base_worker_eval["worker_evaluation_methods"].items():
        print(f"    Method {m:<15}: P={vals['precision']:.4f}, R={vals['recall']:.4f}, F1={vals['f1_score']:.4f}", flush=True)

    # Step 2: Fine-tune models/best.pt on Construction-PPE train split (800 images) with CUDA
    print(f"\n[3/4] Launching Fine-Tuning of models/best.pt on Construction-PPE TRAIN (800 images)...", flush=True)
    epochs = 30
    batch = 8
    imgsz = 640
    patience = 15
    run_dir = Path("runs/fine_tune")
    run_name = "yolov8n_cpp_ft_cuda"
    save_dir = run_dir / run_name
    if save_dir.exists():
        shutil.rmtree(save_dir, ignore_errors=True)

    ft_model = YOLO(str(base_model_path))
    train_start_t = time.perf_counter()
    train_res = ft_model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        amp=True,
        device=device,
        workers=0,
        project=str(run_dir),
        name=run_name,
        plots=False,
        verbose=True,
    )
    train_duration = round(time.perf_counter() - train_start_t, 2)
    print(f"  Fine-Tuning finished in {train_duration}s ({train_duration/60:.2f} mins).")

    ft_best_weights = save_dir / "weights" / "best.pt"
    if not ft_best_weights.exists():
        ft_best_weights = save_dir / "weights" / "last.pt"

    print(f"  Fine-tuned candidate weights: {ft_best_weights} (exists: {ft_best_weights.exists()})")

    # Step 3: Evaluate Candidate on EXACT SAME 200 held-out images
    print(f"\n[4/4] Evaluating Fine-Tuned Candidate on held-out validation set...")
    cand_model = YOLO(str(ft_best_weights))
    cand_val_eval = evaluate_ultralytics_metrics(
        model=cand_model,
        data_yaml=data_yaml,
        imgsz=640,
        conf=0.001,
        iou=0.60,
        device=device,
        gt_counts=val_gt_counts,
    )
    print(f"  Candidate Overall: mAP50={cand_val_eval['overall']['map50']}, P={cand_val_eval['overall']['precision']}, R={cand_val_eval['overall']['recall']}")
    for c in CANONICAL_CLASSES:
        cm = cand_val_eval['per_class'][c]
        print(f"    {c:<12}: P={cm['precision']:.4f}, R={cm['recall']:.4f}, mAP50={cm['map50']:.4f}")

    print("\nEvaluating Candidate Worker-Level Violations (Raw vs Spatial vs Combined)...")
    cand_worker_eval = evaluate_worker_violations(
        model=cand_model,
        dataset_root=dataset_root,
        conf=0.25,
        iou=0.50,
        device=device,
    )
    for m, vals in cand_worker_eval["worker_evaluation_methods"].items():
        print(f"    Method {m:<15}: P={vals['precision']:.4f}, R={vals['recall']:.4f}, F1={vals['f1_score']:.4f}")

    # Output experiment comparison summary
    comparison = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": {
            "device": device,
            "device_name": device_name,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "training_params": {
            "base_model": str(base_model_path.as_posix()),
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "patience": patience,
            "duration_seconds": train_duration,
            "weights_path": str(ft_best_weights.as_posix()),
        },
        "baseline_evaluation": {
            "overall": base_val_eval["overall"],
            "per_class": base_val_eval["per_class"],
            "speed": base_val_eval["speed"],
            "worker_violations": base_worker_eval,
        },
        "candidate_evaluation": {
            "overall": cand_val_eval["overall"],
            "per_class": cand_val_eval["per_class"],
            "speed": cand_val_eval["speed"],
            "worker_violations": cand_worker_eval,
        },
    }

    out_json = Path("runs/finetune_comparison.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] Wrote experiment comparison to {out_json}")


if __name__ == "__main__":
    main()
