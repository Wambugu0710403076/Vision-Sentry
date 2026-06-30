# Vision-Sentry
### Automated Event Search Methods for Security Footage to Improve Incident Response

**Author:** Victor Dennis Wambugu | DCF-01-0184/2025  
**Institution:** Zetech University — Diploma in ICT  
**Submitted:** April 2026

---

## Project Overview

Vision-Sentry is a Python-based intelligent surveillance framework that transforms
passive CCTV recording into a proactive, AI-driven security system.

### Objectives Implemented

| # | Objective | Module |
|---|-----------|--------|
| 1 | AI Detection Module (YOLOv8) | `core/detector.py` |
| 2 | High-Frequency Metadata Engine | `core/metadata_engine.py` |
| 3 | Intelligent Event Classification | `core/classifier.py` |
| 4 | Real-Time Alerting & Search Dashboard | `ui/dashboard.py` |

---

## Quick Start

### 1. Install dependencies
```bash
pip install ultralytics opencv-python Pillow
```

### 2. Run the dashboard
```bash
python main.py
```

### 3. Run the evaluation script
```bash
# With a video file:
python evaluate.py --source your_video.mp4 --duration 60

# With webcam:
python evaluate.py --source 0 --duration 30

# Mock mode (no camera/GPU needed):
python evaluate.py
```

---

## Project Structure

```
vision_sentry/
├── main.py                   Entry point
├── evaluate.py               Performance evaluation script
├── requirements.txt
├── core/
│   ├── detector.py           Obj 1 — YOLOv8 AI detection module
│   ├── classifier.py         Obj 3 — Event priority classification
│   ├── metadata_engine.py    Obj 2 — JSON metadata indexing engine
│   ├── alert_manager.py      Obj 4 — Automated alert system
│   └── video_processor.py    Pipeline orchestrator (OpenCV loop)
├── ui/
│   └── dashboard.py          Obj 4 — Tkinter security dashboard
└── data/
    └── logs/                 Auto-saved JSON event logs
```

---

## Priority Levels (Objective 3)

| Level | Colour | Trigger |
|-------|--------|---------|
| AUTHORIZED | Green | High-confidence, known class, normal hours, no zone |
| MONITOR | Blue | Low confidence or non-critical detection |
| SUSPICIOUS | Amber | Unattended bag or unknown class |
| BREACH | Red | Object inside restricted zone OR person detected off-hours |

---

## Mock Engine

If YOLOv8 / a GPU is unavailable, Vision-Sentry automatically switches to a
deterministic **mock engine** that generates realistic detection data. All other
modules (metadata, classifier, alerting, dashboard) run identically — making the
system fully testable without hardware.

---

## Notes on Problem Statement

> *"Traditional surveillance systems lack automated indexing and intelligent event
> detection mechanisms, creating a 'needle in a haystack' challenge."*

Vision-Sentry addresses this through:
- **Automated JSON indexing** (every detection timestamped & stored)
- **Full-text forensic search** (search by label, priority, time, or free text)
- **Priority classification** (BREACH events surface instantly)
- **MTTD tracking** (Mean Time to Detect — a key KPI for this project)
