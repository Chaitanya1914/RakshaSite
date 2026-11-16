"""
YOLOv8 Detection Module for RakshaSite.

Provides object detection for construction site safety monitoring:
- Person detection
- PPE detection (helmet, vest, no_helmet, no_vest)
- Hallucination filtering (area thresholds, spatial person containment, confidence filtering)
- Frame and video stream inference
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

from ultralytics import YOLO

from src.utils.logger import get_logger

# Standard PPE classes monitored by RakshaSite
DEFAULT_CLASSES = ["person", "helmet", "no_helmet", "vest", "no_vest"]
PPE_CLASSES = {"helmet", "no_helmet", "vest", "no_vest"}
PERSON_CLASS = "person"

# Canonical normalization map for Roboflow CSS, SH17, and Ultralytics Construction-PPE models
CANONICAL_CLASS_MAP = {
    "person": "person",
    "Person": "person",
    "helmet": "helmet",
    "Hardhat": "helmet",
    "hardhat": "helmet",
    "no_helmet": "no_helmet",
    "NO-Hardhat": "no_helmet",
    "no-hardhat": "no_helmet",
    "head": "no_helmet",
    "vest": "vest",
    "Safety Vest": "vest",
    "safety-vest": "vest",
    "safety_vest": "vest",
    "no_vest": "no_vest",
    "NO-Safety Vest": "no_vest",
    "no-safety vest": "no_vest",
    "no_safety_vest": "no_vest",
    "gloves": "gloves",
    "Gloves": "gloves",
    "no_gloves": "no_gloves",
    "boots": "boots",
    "no_boots": "no_boots",
    "harness": "harness",
    "no_harness": "no_harness",
    "machinery": "machinery",
    "Excavator": "machinery",
    "vehicle": "vehicle",
    "Safety Cone": "safety_cone",
    "Mask": "mask",
    "NO-Mask": "no_mask",
}


class YOLODetector:
    """
    YOLOv8-based object detector with built-in hallucination filtering
    tailored for construction site safety monitoring.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        min_area_ratio: float = 0.01,
        device: Optional[Union[str, int]] = None,
        class_names: Optional[List[str]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the YOLODetector.

        Args:
            model_path: Path to YOLOv8 weights (.pt file) or model name. If None, loaded from config.
            conf_threshold: Minimum confidence score for valid detections (0.0 to 1.0). If None, loaded from config.
            iou_threshold: IoU threshold for Non-Maximum Suppression (NMS). If None, loaded from config.
            min_area_ratio: Minimum ratio of bounding box area to total frame area (default: 0.01 = 1%).
            device: Computing device ('cpu', 'cuda', 0, etc.). None auto-selects.
            class_names: Optional custom class name list. If None, uses model names or default PPE classes.
            config_path: Optional path to config YAML file.
        """
        self.logger = get_logger("YOLO")

        # Attempt to load configuration from config file if available
        config_data = self._load_config(config_path)
        model_cfg = config_data.get("model", {})

        # Resolve confidence threshold from config or default to 0.25
        if conf_threshold is not None:
            self.conf_threshold = float(conf_threshold)
        elif "confidence_threshold" in model_cfg:
            self.conf_threshold = float(model_cfg["confidence_threshold"])
        else:
            self.conf_threshold = 0.25

        # Resolve IoU threshold from config or default to 0.45
        if iou_threshold is not None:
            self.iou_threshold = float(iou_threshold)
        elif "iou_threshold" in model_cfg:
            self.iou_threshold = float(model_cfg["iou_threshold"])
        else:
            self.iou_threshold = 0.45

        # Resolve model path from config or default to 'yolov8n.pt'
        if model_path is not None:
            self.model_path = str(model_path)
        elif "weights_path" in model_cfg:
            self.model_path = str(model_cfg["weights_path"])
        else:
            self.model_path = "yolov8n.pt"

        # Resolve device
        if device is not None:
            self.device = device
        elif "device" in model_cfg:
            self.device = model_cfg["device"]
        else:
            self.device = None

        try:
            import torch

            if self.device in ("0", 0, "cuda") and not torch.cuda.is_available():
                self.device = "cpu"
        except Exception:
            pass

        self.min_area_ratio = float(min_area_ratio)

        self.logger.info(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(self.model_path)

        # Configure class names
        if class_names is not None:
            self.class_names = class_names
        elif hasattr(self.model, "names") and isinstance(self.model.names, dict):
            # Model has internal names dictionary
            self.class_names = [self.model.names[i] for i in sorted(self.model.names.keys())]
        else:
            self.class_names = DEFAULT_CLASSES

        self.logger.info(
            f"YOLODetector initialized with {len(self.class_names)} classes. "
            f"Conf={self.conf_threshold}, IoU={self.iou_threshold}, MinAreaRatio={self.min_area_ratio}"
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

        # Check default config location (rakshasite/configs/config.yaml)
        default_cfg = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
        if default_cfg.exists() and yaml is not None:
            try:
                with open(default_cfg, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return {}
        return {}

    @classmethod
    def from_config(cls, config_path: Union[str, Path], **kwargs: Any) -> "YOLODetector":
        """
        Factory method to instantiate YOLODetector directly from a config YAML file.

        Args:
            config_path: Path to YAML configuration file.
            **kwargs: Overrides for configuration settings.

        Returns:
            YOLODetector instance.
        """
        return cls(config_path=config_path, **kwargs)

    def get_class_name(self, class_id: int) -> str:
        """
        Map class ID to canonical RakshaSite class name.

        Args:
            class_id: Numerical class index.

        Returns:
            Canonical class name string ('person', 'helmet', 'no_helmet', 'vest', 'no_vest', etc.).
        """
        raw_name = None
        if hasattr(self.model, "names") and isinstance(self.model.names, dict):
            raw_name = self.model.names.get(class_id)
        if raw_name is None and 0 <= class_id < len(self.class_names):
            raw_name = self.class_names[class_id]
        if raw_name is None:
            return f"class_{class_id}"
        return CANONICAL_CLASS_MAP.get(raw_name, CANONICAL_CLASS_MAP.get(raw_name.lower(), raw_name))

    @staticmethod
    def _is_inside_person(
        ppe_bbox: Tuple[float, float, float, float],
        person_bbox: Tuple[float, float, float, float],
        containment_threshold: float = 0.3,
    ) -> bool:
        """
        Check if a PPE bounding box is spatially located inside or overlapping
        with a person bounding box.

        Args:
            ppe_bbox: (x1, y1, x2, y2) of the PPE detection.
            person_bbox: (x1, y1, x2, y2) of the person detection.
            containment_threshold: Minimum fraction of PPE box area that must intersect
                                   with the person box to be considered inside.

        Returns:
            True if PPE box is inside/contained in person box, False otherwise.
        """
        hx1, hy1, hx2, hy2 = ppe_bbox
        px1, py1, px2, py2 = person_bbox

        # 1. Check center point containment
        center_x = (hx1 + hx2) / 2.0
        center_y = (hy1 + hy2) / 2.0
        if px1 <= center_x <= px2 and py1 <= center_y <= py2:
            return True

        # 2. Check intersection over PPE area
        inter_x1 = max(px1, hx1)
        inter_y1 = max(py1, hy1)
        inter_x2 = min(px2, hx2)
        inter_y2 = min(py2, hy2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        ppe_area = max(0.0, hx2 - hx1) * max(0.0, hy2 - hy1)
        if ppe_area > 0 and (inter_area / ppe_area) >= containment_threshold:
            return True

        return False

    def filter_hallucinations(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, ...],
        min_area_ratio: Optional[float] = None,
        conf_threshold: Optional[float] = None,
        containment_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Filter out hallucinated or false-positive detections based on:
        1. Confidence threshold check: Discard detections below threshold.
        2. Min area filter: Ignore any bounding box with area < 1% of total frame area.
        3. Person containment check: Helmet and vest detections MUST be inside or
           overlapping a person bounding box. PPE detections floating in isolation are discarded.

        Args:
            detections: List of raw detection dictionaries.
            frame_shape: Shape of the input frame (height, width, ...).
            min_area_ratio: Minimum area ratio threshold (defaults to self.min_area_ratio = 0.01).
            conf_threshold: Minimum confidence threshold (defaults to self.conf_threshold).
            containment_threshold: Minimum intersection-over-PPE-area ratio to consider inside.

        Returns:
            Filtered list of detection dictionaries.
        """
        if not detections:
            return []

        frame_h, frame_w = frame_shape[0], frame_shape[1]
        frame_area = float(frame_h * frame_w)
        active_min_area_ratio = min_area_ratio if min_area_ratio is not None else self.min_area_ratio
        min_box_area = frame_area * active_min_area_ratio
        active_conf_thresh = conf_threshold if conf_threshold is not None else self.conf_threshold

        # Step 1 & 2: Confidence and Min Area Filtering
        valid_detections: List[Dict[str, Any]] = []
        for det in detections:
            confidence = det.get("confidence", 0.0)
            if confidence < active_conf_thresh:
                self.logger.debug(
                    f"Filtered out {det.get('class_name')} due to low confidence {confidence:.2f} < {active_conf_thresh:.2f}"
                )
                continue

            x1, y1, x2, y2 = det["bbox"]
            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if box_area < min_box_area:
                self.logger.debug(
                    f"Filtered out {det.get('class_name')} due to small area {box_area:.1f} < {min_box_area:.1f} (1% of frame)"
                )
                continue

            valid_detections.append(det)

        # Step 3: Person containment check for PPE detections
        person_detections = [d for d in valid_detections if d.get("class_name") == PERSON_CLASS]
        ppe_detections = [d for d in valid_detections if d.get("class_name") in PPE_CLASSES]
        other_detections = [
            d for d in valid_detections if d.get("class_name") not in PPE_CLASSES and d.get("class_name") != PERSON_CLASS
        ]

        valid_ppe_detections: List[Dict[str, Any]] = []
        for ppe in ppe_detections:
            # Check if PPE detection is inside at least one detected person
            is_inside = any(
                self._is_inside_person(ppe["bbox"], person["bbox"], containment_threshold=containment_threshold)
                for person in person_detections
            )
            if is_inside:
                valid_ppe_detections.append(ppe)
            else:
                self.logger.debug(
                    f"Filtered out orphaned {ppe.get('class_name')} at {ppe.get('bbox')}; not inside any person bounding box"
                )

        # Combine verified person, PPE, and other valid detections
        filtered_results = person_detections + valid_ppe_detections + other_detections
        return filtered_results

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 inference on a single frame and apply hallucination filtering.

        Args:
            frame: Input video frame as a NumPy array (BGR format).

        Returns:
            List of detection dicts with keys:
            - 'class_id': int
            - 'class_name': str
            - 'confidence': float
            - 'bbox': tuple (x1, y1, x2, y2)
            - 'center': tuple (center_x, center_y)
        """
        if frame is None or frame.size == 0:
            self.logger.warning("Empty frame passed to detect()")
            return []

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        raw_detections: List[Dict[str, Any]] = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                class_name = self.get_class_name(cls_id)

                raw_detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "bbox": (x1, y1, x2, y2),
                        "center": center,
                    }
                )

        # Post-process to filter hallucinations
        filtered_detections = self.filter_hallucinations(raw_detections, frame.shape)
        return filtered_detections

    def detect_video(
        self,
        source: Union[int, str] = 0,
        output_path: Optional[Union[str, Path]] = None,
        display: bool = False,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, List[Dict[str, Any]]], None, None]:
        """
        Process a video source (webcam index or file path) frame by frame.

        Yields:
            Tuple of (frame, detections) for each processed frame.

        Args:
            source: Video source (0 for default webcam, or path to video file).
            output_path: Optional path to save annotated output video.
            display: Whether to display frames in an OpenCV GUI window.
            max_frames: Optional maximum number of frames to process.
        """
        self.logger.info(f"Starting video processing from source: {source}")
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            error_msg = f"Failed to open video source: {source}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        writer = None
        frame_count = 0

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

            if output_path:
                out_path = Path(output_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                self.logger.info(f"Saving output video to: {out_path}")

            while cap.isOpened():
                if max_frames is not None and frame_count >= max_frames:
                    self.logger.info(f"Reached max_frames limit ({max_frames}). Halting.")
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    self.logger.info("Video stream reached end of file or interrupted.")
                    break

                frame_count += 1
                detections = self.detect(frame)

                # Optional annotation and display
                if display or writer is not None:
                    annotated_frame = self.draw_detections(frame.copy(), detections)
                    if writer is not None:
                        writer.write(annotated_frame)
                    if display:
                        cv2.imshow("RakshaSite YOLO Detector", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self.logger.info("User requested termination via 'q' key.")
                            break

                yield frame, detections

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            self.logger.info(f"Video processing finished. Processed {frame_count} frames.")

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and class labels onto a frame.

        Args:
            frame: Input image (BGR).
            detections: List of detection dictionaries.
            color_map: Optional custom color mapping (BGR).

        Returns:
            Annotated frame.
        """
        colors = color_map or {
            "person": (255, 140, 0),      # Blue-ish orange
            "helmet": (0, 255, 0),        # Green
            "no_helmet": (0, 0, 255),     # Red
            "vest": (0, 215, 255),        # Yellow
            "no_vest": (0, 0, 255),       # Red
        }
        default_color = (200, 200, 200)

        for det in detections:
            cls_name = det.get("class_name", "unknown")
            conf = det.get("confidence", 0.0)
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            color = colors.get(cls_name, default_color)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background and text
            label = f"{cls_name} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, max(0, y1 - text_h - baseline - 4)),
                (x1 + text_w, max(0, y1)),
                color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1, max(text_h + 2, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0) if sum(color) > 400 else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame
