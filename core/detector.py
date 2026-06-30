"""
core/detector.py
Objective 1 — AI Detection Module
------------------------------------
Wraps YOLOv8 (Ultralytics) to scan every video frame and return a list of
Detection objects, each carrying a unique tag, bounding-box coords, class
label, and confidence score.

Falls back to a deterministic mock engine when Ultralytics / a GPU model is
not available, so the rest of the system can be developed and tested without
the heavyweight dependency.
"""

from __future__ import annotations

import time
import uuid
import random
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

# Priority levels used by the classifier (Objective 3)
PRIORITY_AUTHORIZED   = "AUTHORIZED"
PRIORITY_MONITOR      = "MONITOR"
PRIORITY_SUSPICIOUS   = "SUSPICIOUS"
PRIORITY_BREACH       = "BREACH"

PRIORITY_COLORS = {
    PRIORITY_AUTHORIZED: (34, 197, 94),    # green
    PRIORITY_MONITOR:    (59, 130, 246),   # blue
    PRIORITY_SUSPICIOUS: (234, 179,  8),   # amber
    PRIORITY_BREACH:     (239, 68, 68),    # red
}

YOLO_CLASS_MAP = {
    0:  "Person",
    1:  "Bicycle",
    2:  "Vehicle",
    3:  "Motorcycle",
    5:  "Bus",
    7:  "Truck",
    24: "Bag",
    26: "Handbag",
    28: "Suitcase",
}


@dataclass
class Detection:
    """Single object detection result for one frame."""
    uid:        str            # unique tag per detection (Obj 1)
    label:      str            # human-readable class name
    class_id:   int
    confidence: float          # 0.0 – 1.0
    bbox:       tuple          # (x1, y1, x2, y2) in pixels
    priority:   str = PRIORITY_MONITOR
    timestamp:  float = field(default_factory=time.time)
    frame_id:   int = 0

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


# --------------------------------------------------------------------------- #
# YOLOv8 wrapper
# --------------------------------------------------------------------------- #

class YOLODetector:
    """
    Objective 1 — AI Detection Module.

    Uses Ultralytics YOLOv8 when available; falls back to MockDetector
    automatically.  Call `process_frame()` on every BGR frame from OpenCV.
    """

    CONF_THRESHOLD = 0.45
    IOU_THRESHOLD  = 0.50
    TARGET_CLASSES = list(YOLO_CLASS_MAP.keys())

    def __init__(self, model_path: str = "yolov8n.pt"):
        self._model = None
        self._use_mock = False
        self._model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO          # type: ignore
            self._model = YOLO(self._model_path)
            print(f"[Detector] YOLOv8 loaded from '{self._model_path}'")
        except Exception as exc:
            print(f"[Detector] YOLOv8 unavailable ({exc}). Using mock engine.")
            self._use_mock = True

    # ---------------------------------------------------------------------- #

    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> List[Detection]:
        """Run inference on a single BGR frame, return Detection list."""
        if self._use_mock:
            return _MockEngine.generate(frame, frame_id)

        t0 = time.perf_counter()
        results = self._model.predict(
            source=frame,
            conf=self.CONF_THRESHOLD,
            iou=self.IOU_THRESHOLD,
            classes=self.TARGET_CLASSES,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        detections: List[Detection] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = YOLO_CLASS_MAP.get(cls_id, f"Class-{cls_id}")
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                det = Detection(
                    uid=str(uuid.uuid4())[:8],
                    label=label,
                    class_id=cls_id,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    frame_id=frame_id,
                )
                detections.append(det)

        return detections

    @property
    def is_mock(self) -> bool:
        return self._use_mock


# --------------------------------------------------------------------------- #
# Mock engine — deterministic, repeatable, good for testing
# --------------------------------------------------------------------------- #

class _MockEngine:
    """
    Simulates realistic YOLOv8 output so the full pipeline can be tested
    without a GPU.  Spawns 1–3 objects per scene; scenes rotate every ~2s.
    """

    _SCENE_POOL = [
        [{"label": "Person",  "cls": 0, "conf_range": (0.82, 0.97)}],
        [{"label": "Person",  "cls": 0, "conf_range": (0.88, 0.96)},
         {"label": "Person",  "cls": 0, "conf_range": (0.75, 0.90)}],
        [{"label": "Vehicle", "cls": 2, "conf_range": (0.79, 0.95)}],
        [{"label": "Bag",     "cls": 24,"conf_range": (0.68, 0.88)}],
        [{"label": "Person",  "cls": 0, "conf_range": (0.91, 0.99)},
         {"label": "Vehicle", "cls": 2, "conf_range": (0.80, 0.94)}],
        [{"label": "Motorcycle","cls":3,"conf_range": (0.72, 0.90)}],
    ]
    _scene_idx   = 0
    _scene_start = 0.0
    _SCENE_SECS  = 2.5

    @classmethod
    def generate(cls, frame: np.ndarray, frame_id: int) -> List[Detection]:
        now = time.time()
        if now - cls._scene_start > cls._SCENE_SECS:
            cls._scene_idx  = (cls._scene_idx + 1) % len(cls._SCENE_POOL)
            cls._scene_start = now

        h, w = frame.shape[:2]
        scene = cls._SCENE_POOL[cls._scene_idx]
        detections: List[Detection] = []

        rng = random.Random(frame_id % 100)   # pseudo-stable per frame
        for obj in scene:
            lo, hi = obj["conf_range"]
            conf   = rng.uniform(lo, hi)
            bw     = rng.randint(60, 160) if obj["label"] != "Vehicle" else rng.randint(120, 240)
            bh     = rng.randint(90, 200) if obj["label"] != "Vehicle" else rng.randint(70, 120)
            x1     = rng.randint(10, max(11, w - bw - 10))
            y1     = rng.randint(10, max(11, h - bh - 10))
            det = Detection(
                uid=str(uuid.uuid4())[:8],
                label=obj["label"],
                class_id=obj["cls"],
                confidence=round(conf, 3),
                bbox=(x1, y1, x1 + bw, y1 + bh),
                frame_id=frame_id,
            )
            detections.append(det)
        return detections
