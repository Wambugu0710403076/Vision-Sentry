"""
core/video_processor.py
Video capture + frame pipeline
--------------------------------
Ties together OpenCV capture, the YOLOv8 detector, classifier, metadata
engine, and alert manager into a single processing loop.

Designed to be called from a Tkinter `after()` loop — it is NOT threaded
internally; the dashboard schedules each tick.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from core.detector    import YOLODetector, Detection, PRIORITY_COLORS
from core.classifier  import EventClassifier
from core.metadata_engine import MetadataEngine
from core.alert_manager   import AlertManager


class VideoProcessor:
    """
    Orchestrates the full detection pipeline for one camera source.

        vp = VideoProcessor(metadata_engine, alert_manager)
        vp.open("path/to/video.mp4")
        frame, detections = vp.tick()   # call ~30× per second
        vp.close()
    """

    def __init__(
        self,
        metadata: MetadataEngine,
        alerts:   AlertManager,
        model_path: str = "yolov8n.pt",
        camera_id:  str = "CAM-01",
    ):
        self._metadata  = metadata
        self._alerts    = alerts
        self._detector  = YOLODetector(model_path)
        self._classifier= EventClassifier()
        self._cap:  Optional[cv2.VideoCapture] = None
        self._frame_id  = 0
        self._fps_target= 25
        self._last_tick = 0.0
        self._running   = False
        self.camera_id  = camera_id

        # Performance counters
        self._inf_times: List[float] = []
        self._frame_times: List[float] = []

    # ---------------------------------------------------------------------- #
    # Source management
    # ---------------------------------------------------------------------- #

    def open(self, source) -> bool:
        """
        source: int (webcam index), str (file path), or RTSP URL string.
        Returns True on success.
        """
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            print(f"[VideoProcessor] Cannot open source: {source}")
            return False
        self._running  = True
        self._frame_id = 0
        self._inf_times.clear()
        self._frame_times.clear()
        print(f"[VideoProcessor] Opened: {source}")
        return True

    def close(self):
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None

    # ---------------------------------------------------------------------- #
    # Zone injection
    # ---------------------------------------------------------------------- #

    def set_zones(self, zones):
        self._classifier.set_zones(zones)

    # ---------------------------------------------------------------------- #
    # Main tick — call from Tkinter after() loop
    # ---------------------------------------------------------------------- #

    def tick(self) -> Tuple[Optional[np.ndarray], List[Detection]]:
        """
        Reads one frame, runs the full pipeline.
        Returns (annotated_frame, detections) or (None, []) if no frame.
        """
        if not self._running or self._cap is None:
            return None, []

        ret, frame = self._cap.read()
        if not ret:
            # Reached end of file — loop back
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
            if not ret:
                return None, []

        self._frame_id += 1
        t0 = time.perf_counter()

        # 1. Detect
        detections = self._detector.process_frame(frame, self._frame_id)

        # 2. Classify (mutates .priority in place)
        self._classifier.classify_batch(detections)

        # 3. Record metadata
        self._metadata.record(detections)

        # 4. Evaluate alerts
        self._alerts.evaluate(detections)

        inf_ms = (time.perf_counter() - t0) * 1000
        self._inf_times.append(inf_ms)
        if len(self._inf_times) > 60:
            self._inf_times.pop(0)

        # 5. Annotate frame
        annotated = self._annotate(frame.copy(), detections)

        now = time.perf_counter()
        if self._last_tick:
            self._frame_times.append(now - self._last_tick)
            if len(self._frame_times) > 60:
                self._frame_times.pop(0)
        self._last_tick = now

        return annotated, detections

    # ---------------------------------------------------------------------- #
    # Annotation
    # ---------------------------------------------------------------------- #

    def _annotate(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        for det in detections:
            color = PRIORITY_COLORS.get(det.priority, (200, 200, 200))
            x1, y1, x2, y2 = det.bbox

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            label_text = f"{det.label} {det.confidence*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Priority badge
            badge = det.priority[:3]
            cv2.putText(frame, badge, (x2 - 28, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # HUD overlay (top-left)
        hud_lines = [
            f"Frame: {self._frame_id}",
            f"Inf:   {self.avg_inference_ms:.1f} ms",
            f"FPS:   {self.current_fps:.1f}",
            f"Dets:  {len(detections)}",
        ]
        for i, line in enumerate(hud_lines):
            cv2.putText(frame, line, (6, 18 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        return frame

    # ---------------------------------------------------------------------- #
    # Performance properties
    # ---------------------------------------------------------------------- #

    @property
    def avg_inference_ms(self) -> float:
        return sum(self._inf_times) / len(self._inf_times) if self._inf_times else 0.0

    @property
    def current_fps(self) -> float:
        if not self._frame_times:
            return 0.0
        avg = sum(self._frame_times) / len(self._frame_times)
        return round(1.0 / avg, 1) if avg > 0 else 0.0

    @property
    def is_mock(self) -> bool:
        return self._detector.is_mock
