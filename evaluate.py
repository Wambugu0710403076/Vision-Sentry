"""
evaluate.py
Objective 3 — System Evaluation
---------------------------------
Standalone script that runs Vision-Sentry against a video file (or webcam)
for N seconds, then prints a performance report:

  • Total frames processed
  • Average inference time (ms)
  • Average FPS
  • Detection counts by class & priority
  • Alert count + MTTD
  • Average confidence per class

Usage:
    python evaluate.py                        # uses mock engine, 30-second run
    python evaluate.py --source sample.mp4
    python evaluate.py --source 0 --duration 60
"""

import argparse
import time
import sys
from collections import defaultdict
from datetime import datetime

import cv2

from core.detector        import YOLODetector, PRIORITY_COLORS
from core.classifier      import EventClassifier
from core.metadata_engine import MetadataEngine
from core.alert_manager   import AlertManager
from core.video_processor import VideoProcessor


def run_evaluation(source, duration_sec: int = 30):
    print("=" * 60)
    print("  VISION-SENTRY  |  System Evaluation")
    print(f"  Source   : {source}")
    print(f"  Duration : {duration_sec}s")
    print(f"  Started  : {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60)

    metadata = MetadataEngine()
    alerts   = AlertManager()
    vp       = VideoProcessor(metadata, alerts)

    if not vp.open(source):
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    alerts.start_session()

    inf_times     = []
    fps_samples   = []
    frame_count   = 0
    det_by_label  = defaultdict(int)
    det_by_pri    = defaultdict(int)
    conf_by_label = defaultdict(list)

    start = time.time()
    last  = start

  


    # ── Report ────────────────────────────────────────────────────────────
    avg_inf  = sum(inf_times) / len(inf_times) if inf_times else 0
    avg_fps  = sum(fps_samples) / len(fps_samples) if fps_samples else 0
    mttd     = alerts.mttd_seconds

    print()
    print("=" * 60)
    print("  PERFORMANCE REPORT")
    print("=" * 60)
    print(f"  Frames processed  : {frame_count}")
    print(f"  Avg inference     : {avg_inf:.2f} ms")
    print(f"  Avg FPS           : {avg_fps:.2f}")
    print(f"  Total detections  : {sum(det_by_label.values())}")
    print(f"  Total alerts      : {alerts.total_alerts}")
    print(f"  MTTD              : {mttd:.2f}s" if mttd else "  MTTD              : N/A (no breach)")
    print(f"  Engine mode       : {'Mock (no YOLOv8)' if vp.is_mock else 'YOLOv8'}")
    print()
    print("  Detections by class:")
    for label, count in sorted(det_by_label.items(), key=lambda x: -x[1]):
        avg_conf = sum(conf_by_label[label]) / len(conf_by_label[label])
        print(f"    {label:<20} {count:>5}  (avg conf: {avg_conf*100:.1f}%)")
    print()
    print("  Detections by priority:")
    for pri, count in sorted(det_by_pri.items(), key=lambda x: -x[1]):
        print(f"    {pri:<15} {count:>5}")
    print()
    print("  Summary JSON:")
    import json
    print(json.dumps(metadata.summary(), indent=4))
    print("=" * 60)

    # Export log
    path = metadata.export_json()
    print(f"  Log exported to: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision-Sentry Evaluator")
    parser.add_argument("--source",   default=0,  help="Video file path or webcam index")
    parser.add_argument("--duration", default=30, type=int, help="Evaluation duration (seconds)")
    args = parser.parse_args()

    src = args.source
    try:
        src = int(src)   # webcam index
    except ValueError:
        pass             # file path string

    run_evaluation(src, args.duration)
