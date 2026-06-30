"""
core/metadata_engine.py
Objective 2 — High-Frequency Metadata Engine
----------------------------------------------
Records every Detection into a searchable JSON event log.

Each entry stores:
  uid, label, confidence, priority, bbox, timestamp (ISO-8601),
  frame_id, camera_id

The log is held in memory (deque, capped at MAX_EVENTS) and flushed to
disk as a .json file on demand or on shutdown.  The search() method
supports filtering by label, priority, time range, and free text.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional

from core.detector import Detection


MAX_EVENTS = 5_000          # rolling in-memory cap
LOG_DIR    = "data/logs"


class MetadataEngine:
    """
    Objective 2 — High-Frequency Metadata Engine.
    Thread-safe enough for single-threaded Tkinter loop.
    """

    def __init__(self, camera_id: str = "CAM-01"):
        self.camera_id = camera_id
        self._log: Deque[Dict] = deque(maxlen=MAX_EVENTS)
        self._session_start = datetime.now().isoformat(timespec="seconds")
        os.makedirs(LOG_DIR, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Ingestion
    # ---------------------------------------------------------------------- #

    def record(self, detections: List[Detection]) -> None:
        """Append all detections from one frame to the log."""
        for det in detections:
            entry = self._to_dict(det)
            self._log.appendleft(entry)

    def _to_dict(self, det: Detection) -> Dict:
        return {
            "uid":        det.uid,
            "label":      det.label,
            "class_id":   det.class_id,
            "confidence": round(det.confidence, 4),
            "priority":   det.priority,
            "bbox":       list(det.bbox),
            "center":     list(det.center),
            "frame_id":   det.frame_id,
            "camera_id":  self.camera_id,
            "timestamp":  datetime.fromtimestamp(det.timestamp).isoformat(timespec="milliseconds"),
            "epoch":      det.timestamp,
        }

    # ---------------------------------------------------------------------- #
    # Search  (Objective 4 backend)
    # ---------------------------------------------------------------------- #

    def search(
        self,
        query:     str  = "",
        priority:  Optional[str] = None,
        label:     Optional[str] = None,
        after_ts:  Optional[float] = None,
        before_ts: Optional[float] = None,
        limit:     int = 200,
    ) -> List[Dict]:
        """
        Full-text + structured search across the event log.
        Returns results newest-first.
        """
        q = query.strip().lower()
        results: List[Dict] = []

        for entry in self._log:
            # Structured filters
            if priority and entry["priority"] != priority:
                continue
            if label and label.lower() not in entry["label"].lower():
                continue
            if after_ts  and entry["epoch"] < after_ts:
                continue
            if before_ts and entry["epoch"] > before_ts:
                continue

            # Free-text match against label, priority, timestamp, uid
            if q:
                haystack = (
                    entry["label"].lower() + " " +
                    entry["priority"].lower() + " " +
                    entry["timestamp"].lower() + " " +
                    entry["uid"].lower()
                )
                if q not in haystack:
                    continue

            results.append(entry)
            if len(results) >= limit:
                break

        return results

    # ---------------------------------------------------------------------- #
    # Stats helper
    # ---------------------------------------------------------------------- #

    def summary(self) -> Dict:
        total   = len(self._log)
        by_pri  = {}
        by_lbl  = {}
        confs   = []
        for e in self._log:
            by_pri[e["priority"]] = by_pri.get(e["priority"], 0) + 1
            base_lbl = e["label"].split(" [")[0]
            by_lbl[base_lbl]  = by_lbl.get(base_lbl, 0) + 1
            confs.append(e["confidence"])
        avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0
        return {
            "total_events": total,
            "by_priority":  by_pri,
            "by_label":     by_lbl,
            "avg_confidence": avg_conf,
            "session_start":  self._session_start,
        }

    # ---------------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------------- #

    def export_json(self, path: Optional[str] = None) -> str:
        if path is None:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(LOG_DIR, f"vision_sentry_{ts}.json")

        payload = {
            "meta": {
                "system":        "Vision-Sentry",
                "author":        "Victor Dennis Wambugu",
                "student_id":    "DCF-01-0184/2025",
                "session_start": self._session_start,
                "export_time":   datetime.now().isoformat(timespec="seconds"),
                "total_entries": len(self._log),
            },
            "summary": self.summary(),
            "events":  list(self._log),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return path

    def load_json(self, path: str) -> int:
        """Load a previously exported log file into memory (appends)."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        events = data.get("events", [])
        for e in reversed(events):          # preserve newest-first order
            self._log.appendleft(e)
        return len(events)
