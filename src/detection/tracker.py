"""
Worker Tracking and PPE Violation Detection Module for RakshaSite.

Provides:
- Persistent multi-worker tracking using DeepSort (max_age=50)
- Spatial overlap analysis between worker bounding boxes and PPE detections (helmet, vest)
- Automatic detection of safety violations (NO_HELMET, NO_VEST)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

from deep_sort_realtime.deepsort_tracker import DeepSort

from src.utils.logger import get_logger

PERSON_CLASS = "person"
HELMET_CLASSES = {"helmet", "no_helmet"}
VEST_CLASSES = {"vest", "no_vest"}


class TrackedWorker(dict):
    """
    Structured dictionary representation of a tracked worker.
    Supports both attribute and dictionary key access.
    """

    def __init__(
        self,
        track_id: Union[int, str],
        bbox: Tuple[float, float, float, float],
        confidence: float = 1.0,
        class_name: str = PERSON_CLASS,
        associated_ppe: Optional[List[Dict[str, Any]]] = None,
        violations: Optional[List[str]] = None,
        missing_ppe: Optional[List[str]] = None,
        has_helmet: bool = False,
        has_vest: bool = False,
        is_compliant: bool = False,
        raw_track: Optional[Any] = None,
    ):
        data = {
            "track_id": str(track_id),
            "worker_id": str(track_id),
            "bbox": tuple(bbox),
            "confidence": float(confidence),
            "class_name": class_name,
            "center": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
            "associated_ppe": associated_ppe or [],
            "violations": violations or [],
            "missing_ppe": missing_ppe or [],
            "has_helmet": has_helmet,
            "has_vest": has_vest,
            "is_compliant": is_compliant,
        }
        super().__init__(data)
        self.raw_track = raw_track

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'TrackedWorker' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "raw_track":
            super().__setattr__(name, value)
        else:
            self[name] = value


class WorkerTracker:
    """
    Multi-object worker tracker utilizing DeepSORT with persistent track IDs
    and spatial compliance analysis for construction site PPE verification.
    """

    def __init__(
        self,
        max_age: int = 50,
        n_init: int = 3,
        nms_max_overlap: float = 1.0,
        max_cosine_distance: float = 0.2,
        nn_budget: Optional[int] = None,
        override_track_class: Optional[Any] = None,
        embedder: str = "mobilenet",
        half: bool = False,
        bgr: bool = True,
        embedder_gpu: bool = False,
        overlap_threshold: float = 0.25,
    ):
        """
        Initialize the WorkerTracker with DeepSort.

        Args:
            max_age: Maximum number of missed frames before track is terminated (default: 50).
            n_init: Number of consecutive detections needed to confirm a track.
            nms_max_overlap: Non-maximum suppression overlap threshold.
            max_cosine_distance: Cosine distance threshold for appearance feature matching.
            nn_budget: Maximum size of the appearance descriptors gallery.
            override_track_class: Custom track class override for DeepSort.
            embedder: Appearance feature embedder model ('mobilenet', 'torchreid', etc.).
            half: Use FP16 half precision for feature extractor.
            bgr: True if frames are BGR (OpenCV format), False for RGB.
            embedder_gpu: Whether to run appearance embedder on GPU.
            overlap_threshold: Minimum spatial containment ratio for PPE matching.
        """
        self.logger = get_logger("YOLO")
        self.max_age = max_age
        self.n_init = n_init
        self.nms_max_overlap = nms_max_overlap
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.override_track_class = override_track_class
        self.embedder = embedder
        self.half = half
        self.bgr = bgr
        self.embedder_gpu = embedder_gpu
        self.overlap_threshold = overlap_threshold

        self.latest_ppe_detections: List[Dict[str, Any]] = []
        self.latest_tracked_workers: List[TrackedWorker] = []

        self.logger.info(
            f"Initializing DeepSORT tracker: max_age={self.max_age}, n_init={self.n_init}, "
            f"embedder={self.embedder}, embedder_gpu={self.embedder_gpu}"
        )

        try:
            self.tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                nms_max_overlap=self.nms_max_overlap,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=self.nn_budget,
                override_track_class=self.override_track_class,
                embedder=self.embedder,
                half=self.half,
                bgr=self.bgr,
                embedder_gpu=self.embedder_gpu,
            )
        except Exception as e:
            self.logger.warning(
                f"DeepSort initialization with preferred settings failed ({e}). "
                "Retrying with CPU-only fallback settings..."
            )
            self.tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                nms_max_overlap=self.nms_max_overlap,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=self.nn_budget,
                override_track_class=self.override_track_class,
                embedder=self.embedder,
                half=False,
                bgr=self.bgr,
                embedder_gpu=False,
            )

    @classmethod
    def _load_config(cls, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load configuration dictionary from YAML file if available."""
        if config_path is not None:
            cfg_file = Path(config_path)
            if cfg_file.exists() and yaml is not None:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            return {}

        default_cfg = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
        if default_cfg.exists() and yaml is not None:
            try:
                with open(default_cfg, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return {}
        return {}

    @classmethod
    def from_config(cls, config_path: Optional[Union[str, Path]] = None, **kwargs: Any) -> "WorkerTracker":
        """
        Factory method to instantiate WorkerTracker from config YAML.

        Args:
            config_path: Path to config YAML file.
            **kwargs: Direct parameter overrides.

        Returns:
            WorkerTracker instance.
        """
        config = cls._load_config(config_path)
        tracking_cfg = config.get("model", {}).get("tracking", {})

        tracker_params: Dict[str, Any] = {
            "max_age": tracking_cfg.get("max_age", 50),
            "n_init": tracking_cfg.get("n_init", 3),
            "nn_budget": tracking_cfg.get("nn_budget", None),
            "max_cosine_distance": tracking_cfg.get("max_cosine_distance", 0.2),
        }
        # Apply caller overrides
        tracker_params.update(kwargs)
        return cls(**tracker_params)

    @staticmethod
    def _compute_containment_or_iou(
        ppe_bbox: Tuple[float, float, float, float],
        person_bbox: Tuple[float, float, float, float],
    ) -> float:
        """
        Compute containment ratio (intersection area / PPE area) of PPE inside person bbox.
        Also gives credit if PPE center falls within person bbox.

        Args:
            ppe_bbox: (x1, y1, x2, y2) of the PPE detection.
            person_bbox: (x1, y1, x2, y2) of the person detection.

        Returns:
            Containment score between 0.0 and 1.0.
        """
        px1, py1, px2, py2 = person_bbox
        hx1, hy1, hx2, hy2 = ppe_bbox

        inter_x1 = max(px1, hx1)
        inter_y1 = max(py1, hy1)
        inter_x2 = min(px2, hx2)
        inter_y2 = min(py2, hy2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        ppe_area = max(0.0, hx2 - hx1) * max(0.0, hy2 - hy1)
        if ppe_area <= 0:
            return 0.0

        containment = inter_area / ppe_area

        # If center of PPE is strictly inside person, guarantee a high base score
        center_x = (hx1 + hx2) / 2.0
        center_y = (hy1 + hy2) / 2.0
        if px1 <= center_x <= px2 and py1 <= center_y <= py2:
            return max(containment, 0.6)

        return containment

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: np.ndarray,
    ) -> List[TrackedWorker]:
        """
        Update tracker with new detections from the current frame and assign persistent IDs.

        Args:
            detections: List of detection dictionaries from YOLODetector (class_name, confidence, bbox, center).
            frame: Current video frame (NumPy BGR array).

        Returns:
            List of TrackedWorker objects representing active, confirmed tracked workers.
        """
        if detections is None:
            detections = []

        # Separate person detections from PPE detections
        person_detections: List[Dict[str, Any]] = []
        ppe_detections: List[Dict[str, Any]] = []

        for det in detections:
            cls_name = det.get("class_name", "")
            if cls_name == PERSON_CLASS:
                person_detections.append(det)
            else:
                ppe_detections.append(det)

        # Store latest PPE detections for spatial analysis
        self.latest_ppe_detections = ppe_detections

        # Format person detections for DeepSort: ([left, top, w, h], confidence, detection_class)
        raw_detections = []
        for d in person_detections:
            x1, y1, x2, y2 = d["bbox"]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            conf = float(d.get("confidence", 1.0))
            raw_detections.append(([x1, y1, w, h], conf, PERSON_CLASS))

        # Update DeepSort tracks
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        # Process confirmed active tracks
        tracked_workers: List[TrackedWorker] = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            bbox = (float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3]))
            conf = track.get_det_conf()
            confidence = float(conf) if conf is not None else 1.0

            # Match PPE detections with this worker
            worker_obj = TrackedWorker(
                track_id=track_id,
                bbox=bbox,
                confidence=confidence,
                class_name=PERSON_CLASS,
                raw_track=track,
            )
            tracked_workers.append(worker_obj)

        # Associate current PPE detections with tracked workers
        self._associate_ppe_to_workers(tracked_workers, self.latest_ppe_detections)

        # Compute violations for the current frame
        self.get_worker_violations(tracked_workers, ppe_detections=self.latest_ppe_detections, only_violators=False)

        self.latest_tracked_workers = tracked_workers
        self.logger.debug(f"Tracker updated: {len(tracked_workers)} active workers tracked.")
        return tracked_workers

    def _associate_ppe_to_workers(
        self,
        workers: List[TrackedWorker],
        ppe_detections: List[Dict[str, Any]],
    ) -> None:
        """
        Associate each PPE bounding box to the most overlapping worker.

        Args:
            workers: List of TrackedWorker objects.
            ppe_detections: List of PPE detection dictionaries.
        """
        for worker in workers:
            worker["associated_ppe"] = []

        for ppe in ppe_detections:
            best_worker = None
            best_score = 0.0

            for worker in workers:
                score = self._compute_containment_or_iou(ppe["bbox"], worker["bbox"])
                if score > best_score:
                    best_score = score
                    best_worker = worker

            # Assign to the worker if containment exceeds threshold
            if best_worker is not None and best_score >= self.overlap_threshold:
                best_worker["associated_ppe"].append(ppe)

    def get_worker_violations(
        self,
        tracked_objects: List[Union[TrackedWorker, Dict[str, Any], Any]],
        ppe_detections: Optional[List[Dict[str, Any]]] = None,
        only_violators: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Check which workers are missing PPE by spatial analysis.
        Checks if there is a 'helmet' bbox overlapping with a 'person' bbox; if not, that person has NO_HELMET.
        Checks if there is a 'vest' bbox overlapping with a 'person' bbox; if not, that person has NO_VEST.

        Args:
            tracked_objects: List of tracked worker objects or dictionaries.
            ppe_detections: Optional list of PPE detections. Defaults to self.latest_ppe_detections.
            only_violators: If True, returns only workers who have at least one safety violation.
                            If False, returns compliance records for all tracked workers.

        Returns:
            List of violation record dictionaries with keys:
            - 'worker_id' / 'track_id': Persistent identifier of the worker.
            - 'bbox': (x1, y1, x2, y2) worker bounding box.
            - 'violations': List of violation codes (e.g. ['NO_HELMET', 'NO_VEST']).
            - 'missing_ppe': List of missing gear names (e.g. ['helmet', 'vest']).
            - 'has_helmet': bool indicating helmet compliance.
            - 'has_vest': bool indicating safety vest compliance.
            - 'is_compliant': bool indicating overall safety compliance.
        """
        active_ppe = ppe_detections if ppe_detections is not None else self.latest_ppe_detections
        records: List[Dict[str, Any]] = []

        for obj in tracked_objects:
            # Normalize object extraction whether dict, TrackedWorker, or DeepSort Track
            if isinstance(obj, (dict, TrackedWorker)):
                worker_id = str(obj.get("track_id", obj.get("worker_id", "unknown")))
                bbox = tuple(obj.get("bbox", (0.0, 0.0, 0.0, 0.0)))
                associated_ppe = obj.get("associated_ppe", None)
            elif hasattr(obj, "track_id") and hasattr(obj, "to_ltrb"):
                worker_id = str(obj.track_id)
                ltrb = obj.to_ltrb()
                bbox = (float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3]))
                associated_ppe = None
            else:
                continue

            # If PPE was not previously associated, compute spatial overlap now
            if associated_ppe is None:
                assigned_ppe = []
                for ppe in active_ppe:
                    score = self._compute_containment_or_iou(ppe["bbox"], bbox)
                    if score >= self.overlap_threshold:
                        assigned_ppe.append(ppe)
            else:
                assigned_ppe = associated_ppe

            # Spatial Analysis for Helmet
            has_helmet_detection = any(p.get("class_name") == "helmet" for p in assigned_ppe)
            has_explicit_no_helmet = any(p.get("class_name") == "no_helmet" for p in assigned_ppe)
            has_helmet = has_helmet_detection and not has_explicit_no_helmet

            # Spatial Analysis for Vest
            has_vest_detection = any(p.get("class_name") == "vest" for p in assigned_ppe)
            has_explicit_no_vest = any(p.get("class_name") == "no_vest" for p in assigned_ppe)
            has_vest = has_vest_detection and not has_explicit_no_vest

            # Assemble violations
            violations: List[str] = []
            missing_ppe: List[str] = []

            if not has_helmet:
                violations.append("NO_HELMET")
                missing_ppe.append("helmet")

            if not has_vest:
                violations.append("NO_VEST")
                missing_ppe.append("vest")

            is_compliant = len(violations) == 0

            # Update object in place if it's a dict or TrackedWorker
            if isinstance(obj, (dict, TrackedWorker)):
                obj["has_helmet"] = has_helmet
                obj["has_vest"] = has_vest
                obj["violations"] = violations
                obj["missing_ppe"] = missing_ppe
                obj["is_compliant"] = is_compliant

            record = {
                "worker_id": worker_id,
                "track_id": worker_id,
                "bbox": bbox,
                "violations": violations,
                "missing_ppe": missing_ppe,
                "has_helmet": has_helmet,
                "has_vest": has_vest,
                "is_compliant": is_compliant,
            }

            if only_violators:
                if not is_compliant:
                    records.append(record)
            else:
                records.append(record)

        if records and only_violators:
            self.logger.warning(
                f"Detected PPE violations on {len(records)} workers: "
                f"{[(r['worker_id'], r['violations']) for r in records]}"
            )

        return records

    @staticmethod
    def draw_tracks(
        frame: np.ndarray,
        tracked_workers: List[TrackedWorker],
        show_ppe_boxes: bool = True,
    ) -> np.ndarray:
        """
        Visualize tracked workers, persistent track IDs, and compliance status.

        Args:
            frame: Video frame (BGR).
            tracked_workers: List of TrackedWorker objects.
            show_ppe_boxes: Whether to also render individual PPE bounding boxes.

        Returns:
            Annotated video frame.
        """
        for worker in tracked_workers:
            x1, y1, x2, y2 = [int(v) for v in worker["bbox"]]
            worker_id = worker.get("track_id", "N/A")
            is_compliant = worker.get("is_compliant", False)
            violations = worker.get("violations", [])

            # Green for compliant, Red for safety violation
            worker_color = (0, 200, 0) if is_compliant else (0, 0, 255)

            # Draw worker bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), worker_color, 2)

            # Label text: Worker #ID + status
            status_text = "SAFE" if is_compliant else ",".join(violations)
            label = f"Worker #{worker_id} [{status_text}]"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            cv2.rectangle(
                frame,
                (x1, max(0, y1 - text_h - baseline - 6)),
                (x1 + text_w + 4, max(0, y1)),
                worker_color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, max(text_h + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Optionally draw associated PPE boxes
            if show_ppe_boxes:
                for ppe in worker.get("associated_ppe", []):
                    px1, py1, px2, py2 = [int(v) for v in ppe["bbox"]]
                    p_name = ppe.get("class_name", "")
                    p_color = (0, 255, 0) if "no_" not in p_name else (0, 0, 255)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), p_color, 1)

        return frame
