"""
core/alert_manager.py
Objective 4 — Real-Time Alerting System (backend)
---------------------------------------------------
Receives classified detections, debounces duplicates, maintains an active
alert queue, and fires registered callback(s) so the dashboard can react.

Debounce: the same label+priority combo is suppressed for COOLDOWN_SEC
seconds to avoid flooding the UI with identical alerts.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

from core.detector import Detection, PRIORITY_BREACH, PRIORITY_SUSPICIOUS


COOLDOWN_SEC  = 5.0    # minimum seconds between identical alerts
MAX_QUEUE     = 50     # rolling active-alert queue size


@dataclass
class Alert:
    uid:       str
    label:     str
    priority:  str
    confidence:float
    timestamp: float = field(default_factory=time.time)
    camera_id: str   = "CAM-01"
    acknowledged: bool = False

    @property
    def age_sec(self) -> float:
        return time.time() - self.timestamp


class AlertManager:
    """
    Objective 4 — automated alert system.

    Usage:
        mgr = AlertManager()
        mgr.register_callback(my_ui_function)
        mgr.evaluate(detections)   # call every frame
    """

    ALERT_PRIORITIES = {PRIORITY_BREACH, PRIORITY_SUSPICIOUS}

    def __init__(self, camera_id: str = "CAM-01"):
        self.camera_id = camera_id
        self._callbacks: List[Callable[[Alert], None]] = []
        self._queue: Deque[Alert] = deque(maxlen=MAX_QUEUE)
        self._cooldown_map: Dict[str, float] = {}   # key → last fire time
        self._total_fired = 0
        self._mttd_start: Optional[float] = None
        self._first_breach_time: Optional[float] = None

    # ---------------------------------------------------------------------- #
    # Callback registration
    # ---------------------------------------------------------------------- #

    def register_callback(self, fn: Callable[[Alert], None]) -> None:
        self._callbacks.append(fn)

    # ---------------------------------------------------------------------- #
    # Main evaluation loop — called every frame
    # ---------------------------------------------------------------------- #

    def evaluate(self, detections: List[Detection]) -> List[Alert]:
        """Inspect detections; fire alerts for BREACH / SUSPICIOUS events."""
        fired: List[Alert] = []
        for det in detections:
            if det.priority not in self.ALERT_PRIORITIES:
                continue

            key     = f"{det.label}|{det.priority}"
            last_ts = self._cooldown_map.get(key, 0.0)
            if time.time() - last_ts < COOLDOWN_SEC:
                continue   # still in cooldown

            alert = Alert(
                uid=det.uid,
                label=det.label,
                priority=det.priority,
                confidence=det.confidence,
                camera_id=self.camera_id,
            )
            self._queue.appendleft(alert)
            self._cooldown_map[key] = time.time()
            self._total_fired += 1
            fired.append(alert)

            # Track MTTD (Mean Time to Detect) — time from monitoring start
            if self._mttd_start and self._first_breach_time is None:
                if det.priority == PRIORITY_BREACH:
                    self._first_breach_time = time.time()

            for cb in self._callbacks:
                try:
                    cb(alert)
                except Exception as exc:
                    print(f"[AlertManager] Callback error: {exc}")

        return fired

    # ---------------------------------------------------------------------- #
    # Queue access
    # ---------------------------------------------------------------------- #

    def get_active(self, limit: int = 20) -> List[Alert]:
        return [a for a in self._queue if not a.acknowledged][:limit]

    def acknowledge(self, uid: str) -> None:
        for a in self._queue:
            if a.uid == uid:
                a.acknowledged = True
                break

    def acknowledge_all(self) -> None:
        for a in self._queue:
            a.acknowledged = True

    def clear(self) -> None:
        self._queue.clear()
        self._cooldown_map.clear()

    # ---------------------------------------------------------------------- #
    # MTTD tracking
    # ---------------------------------------------------------------------- #

    def start_session(self) -> None:
        self._mttd_start = time.time()
        self._first_breach_time = None

    @property
    def mttd_seconds(self) -> Optional[float]:
        if self._mttd_start is None:
            return None
        if self._first_breach_time:
            return round(self._first_breach_time - self._mttd_start, 2)
        return round(time.time() - self._mttd_start, 2)

    @property
    def total_alerts(self) -> int:
        return self._total_fired
