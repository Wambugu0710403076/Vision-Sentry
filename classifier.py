"""
core/classifier.py
Objective 3 — Intelligent Event Classification
------------------------------------------------
Assigns a priority level to every Detection based on:
  • Object class (Person, Vehicle, Bag…)
  • Whether the detection centre falls inside any active restricted zone
  • Time-of-day rules (off-hours = higher priority)
  • Confidence threshold bands

Priority ladder:
  AUTHORIZED  — expected activity, high confidence, not in a zone
  MONITOR     — low-confidence or non-critical class
  SUSPICIOUS  — unattended bag, low-light person, unusual class
  BREACH      — any object inside a restricted zone or off-hours person
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import List, Tuple

from core.detector import (
    Detection,
    PRIORITY_AUTHORIZED,
    PRIORITY_MONITOR,
    PRIORITY_SUSPICIOUS,
    PRIORITY_BREACH,
)

# Zone: (x1, y1, x2, y2) in pixel coords; name
Zone = Tuple[str, Tuple[int, int, int, int]]


class EventClassifier:
    """
    Stateless classifier — call classify_batch() on each frame's detections.
    Zones are defined by the Security Admin via the dashboard and injected here.
    """

    # Off-hours window (24-hour clock)
    OFF_HOURS_START = 20   # 20:00
    OFF_HOURS_END   =  6   # 06:00

    # Confidence bands
    CONF_HIGH   = 0.80
    CONF_MEDIUM = 0.55

    def __init__(self):
        self._zones: List[Zone] = []

    # ---------------------------------------------------------------------- #
    # Zone management (called from dashboard)
    # ---------------------------------------------------------------------- #

    def set_zones(self, zones: List[Zone]):
        self._zones = zones

    def add_zone(self, name: str, bbox: Tuple[int, int, int, int]):
        self._zones.append((name, bbox))

    def clear_zones(self):
        self._zones = []

    # ---------------------------------------------------------------------- #
    # Classification logic
    # ---------------------------------------------------------------------- #

    def classify_batch(self, detections: List[Detection]) -> List[Detection]:
        """Mutates each Detection's .priority in place; returns the list."""
        hour = datetime.now().hour
        off_hours = self._is_off_hours(hour)

        for det in detections:
            det.priority = self._classify_one(det, off_hours)
        return detections

    def _classify_one(self, det: Detection, off_hours: bool) -> str:
        in_zone, zone_name = self._check_zones(det)

        # — Restricted zone = always a breach
        if in_zone:
            det.label = f"{det.label} [ZONE:{zone_name}]"
            return PRIORITY_BREACH

        # — Unattended bag is suspicious regardless of time
        if det.label in ("Bag", "Handbag", "Suitcase"):
            return PRIORITY_SUSPICIOUS

        # — Off-hours person or vehicle triggers breach
        if off_hours and det.label in ("Person", "Motorcycle", "Bicycle"):
            return PRIORITY_BREACH

        # — Low confidence → monitor only
        if det.confidence < self.CONF_MEDIUM:
            return PRIORITY_MONITOR

        # — High confidence, normal hours, no zone → authorized
        if det.confidence >= self.CONF_HIGH:
            return PRIORITY_AUTHORIZED

        return PRIORITY_MONITOR

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _check_zones(self, det: Detection) -> Tuple[bool, str]:
        cx, cy = det.center
        for name, (zx1, zy1, zx2, zy2) in self._zones:
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                return True, name
        return False, ""

    @staticmethod
    def _is_off_hours(hour: int) -> bool:
        return hour >= EventClassifier.OFF_HOURS_START or hour < EventClassifier.OFF_HOURS_END
