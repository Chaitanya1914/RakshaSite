"""
Detection Utility Functions for RakshaSite.

Provides helper functions for:
- Bounding box geometry (area, IoU, containment, center)
- Spatial zone classification (ground_level, scaffolding_elevated, active_machinery, restricted)
- Dataset label remapping and YOLO format conversion for multi-dataset unification
"""

import glob
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Unified 5-class mapping for RakshaSite core training
UNIFIED_CLASSES = {
    0: "person",
    1: "helmet",
    2: "no_helmet",
    3: "vest",
    4: "no_vest",
}

# Dataset-specific class ID mappings -> UNIFIED_CLASSES
CLASS_MAP_HARDHAT = {0: 1, 1: 2, 2: 0}  # 0:helmet->1, 1:head->2(no_helmet), 2:person->0
CLASS_MAP_SAFETY = {0: 1, 1: 2}         # 0:helmet->1, 1:no_helmet->2
# Snehil Sanyal / Roboflow CSS 10-class dataset mapping:
# 0:Hardhat, 1:Mask, 2:NO-Hardhat, 3:NO-Mask, 4:NO-Safety Vest, 5:Person, 6:Safety Cone, 7:Safety Vest, 8:machinery, 9:vehicle
CLASS_MAP_ROBOFLOW_CSS = {
    0: 1,  # Hardhat -> helmet
    2: 2,  # NO-Hardhat -> no_helmet
    4: 4,  # NO-Safety Vest -> no_vest
    5: 0,  # Person -> person
    7: 3,  # Safety Vest -> vest
}


def compute_box_area(bbox: Tuple[float, float, float, float]) -> float:
    """Compute area of an (x1, y1, x2, y2) bounding box."""
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def compute_box_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Compute center (cx, cy) of an (x1, y1, x2, y2) bounding box."""
    x1, y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def compute_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area <= 0.0:
        return 0.0

    area_a = compute_box_area(box_a)
    area_b = compute_box_area(box_b)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0.0 else 0.0


def infer_worker_zone(
    worker_bbox: Tuple[float, float, float, float],
    frame_shape: Tuple[int, int] = (720, 1280),
    machinery_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
) -> Tuple[str, float]:
    """
    Infer the site zone and proximity to hazards from spatial coordinates.

    Args:
        worker_bbox: (x1, y1, x2, y2) of the worker.
        frame_shape: (height, width) of the video frame.
        machinery_bboxes: Optional list of machinery/vehicle bounding boxes.

    Returns:
        Tuple of (zone_name, normalized_proximity_score [0.0 - 1.0]).
    """
    frame_h, frame_w = float(frame_shape[0]), float(frame_shape[1])
    cx, cy = compute_box_center(worker_bbox)

    # Compute proximity to active machinery if present
    proximity = 0.2
    if machinery_bboxes:
        min_dist = float("inf")
        diag = (frame_h ** 2 + frame_w ** 2) ** 0.5
        for m_box in machinery_bboxes:
            mx, my = compute_box_center(m_box)
            dist = ((cx - mx) ** 2 + (cy - my) ** 2) ** 0.5
            min_dist = min(min_dist, dist)
        if diag > 0:
            proximity = max(0.0, min(1.0, 1.0 - (min_dist / (0.5 * diag))))

    if proximity >= 0.65:
        return "active_machinery", round(proximity, 3)

    # Upper 35% of frame typically corresponds to scaffolding / elevated structure
    if cy < 0.35 * frame_h:
        return "scaffolding_elevated", round(max(proximity, 0.6), 3)

    return "ground_level", round(proximity, 3)


def remap_yolo_label_file(
    label_file: Path,
    class_map: Dict[int, int],
    output_file: Path,
) -> int:
    """
    Read a YOLO format label file, remap class IDs according to class_map, and write to output_file.

    Returns:
        Number of valid bounding box annotations written.
    """
    if not label_file.exists():
        return 0

    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    remapped: List[str] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            old_class = int(float(parts[0]))
        except ValueError:
            continue
        if old_class in class_map:
            new_class = class_map[old_class]
            remapped.append(f"{new_class} {' '.join(parts[1:5])}\n")

    if remapped:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(remapped)

    return len(remapped)


def unify_datasets(
    raw_dir: str = "data/raw",
    unified_dir: str = "data/unified",
    train_split: float = 0.85,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Merge multiple raw YOLO datasets into a unified 5-class dataset structure
    and generate `data/unified/data.yaml`.
    """
    random.seed(seed)
    unified_path = Path(unified_dir)
    for split in ("train", "val"):
        (unified_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (unified_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    sources = [
        ("hardhat", CLASS_MAP_HARDHAT, "hh"),
        ("safety", CLASS_MAP_SAFETY, "sh"),
        ("roboflow", CLASS_MAP_ROBOFLOW_CSS, "rf"),
        ("pictor", {0: 0, 1: 1, 2: 3}, "pv"),
    ]

    total_train = 0
    total_val = 0

    for folder_name, class_map, prefix in sources:
        src_root = Path(raw_dir) / folder_name
        if not src_root.exists():
            continue

        images = (
            list(src_root.rglob("*.jpg"))
            + list(src_root.rglob("*.jpeg"))
            + list(src_root.rglob("*.png"))
        )
        random.shuffle(images)
        split_idx = int(len(images) * train_split)

        for idx, img_path in enumerate(images):
            # Locate matching .txt label
            candidate_labels = [
                img_path.with_suffix(".txt"),
                img_path.parent.parent / "labels" / (img_path.stem + ".txt"),
                src_root / "labels" / (img_path.stem + ".txt"),
            ]
            label_path = next((p for p in candidate_labels if p.exists()), None)
            if label_path is None:
                continue

            split = "train" if idx < split_idx else "val"
            out_name = f"{prefix}_{img_path.name}"
            out_img = unified_path / "images" / split / out_name
            out_lbl = unified_path / "labels" / split / f"{Path(out_name).stem}.txt"

            count = remap_yolo_label_file(label_path, class_map, out_lbl)
            if count > 0:
                shutil.copy2(img_path, out_img)
                if split == "train":
                    total_train += 1
                else:
                    total_val += 1

    # Write data.yaml
    data_yaml_path = unified_path / "data.yaml"
    yaml_content = (
        f"path: {unified_path.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(UNIFIED_CLASSES)}\n"
        f"names:\n"
    )
    for k, v in sorted(UNIFIED_CLASSES.items()):
        yaml_content += f"  {k}: {v}\n"

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return {
        "train_images": total_train,
        "val_images": total_val,
        "data_yaml": str(data_yaml_path),
    }
