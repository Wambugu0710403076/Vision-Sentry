"""
ui/dashboard.py
Objective 4 — Real-Time Alerting and Search Dashboard
-------------------------------------------------------
Full Tkinter GUI:
  • Live annotated video feed panel
  • Detection stats sidebar
  • Scrollable live event stream
  • Forensic search panel
  • Performance metrics bar
  • Alert popup system

Colour palette (dark security aesthetic):
  Background:   #0D0D14  (near-black navy)
  Surface:      #13131F
  Card:         #1A1A2E
  Accent-blue:  #3B82F6
  Accent-green: #22C55E
  Accent-amber: #F59E0B
  Accent-red:   #EF4444
  Text-primary: #F1F5F9
  Text-muted:   #64748B
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
from typing import Optional
import threading

import cv2
from PIL import Image, ImageTk

from core.video_processor  import VideoProcessor
from core.metadata_engine  import MetadataEngine
from core.alert_manager    import AlertManager, Alert
from core.detector         import (
    PRIORITY_AUTHORIZED, PRIORITY_MONITOR,
    PRIORITY_SUSPICIOUS, PRIORITY_BREACH,
)

# ── Palette ────────────────────────────────────────────────────────────────
BG      = "#0D0D14"
SURFACE = "#13131F"
CARD    = "#1A1A2E"
BORDER  = "#252540"
BLUE    = "#3B82F6"
GREEN   = "#22C55E"
AMBER   = "#F59E0B"
RED     = "#EF4444"
TEXT    = "#F1F5F9"
MUTED   = "#64748B"

PRI_COLOR = {
    PRIORITY_AUTHORIZED: GREEN,
    PRIORITY_MONITOR:    BLUE,
    PRIORITY_SUSPICIOUS: AMBER,
    PRIORITY_BREACH:     RED,
}

TICK_MS   = 33   # ~30 fps UI refresh
FEED_W    = 640
FEED_H    = 400


class VisionSentryApp:
    """Root application class — builds and owns the full UI."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self._running   = False
        self._source    = None
        self._alert_popup: Optional[tk.Toplevel] = None

        # Core engine instances
        self._metadata = MetadataEngine()
        self._alerts   = AlertManager()
        self._vp       = VideoProcessor(self._metadata, self._alerts)

        self._alerts.register_callback(self._on_alert)

        # Counters
        self._cnt = {p: 0 for p in [PRIORITY_AUTHORIZED, PRIORITY_MONITOR,
                                     PRIORITY_SUSPICIOUS, PRIORITY_BREACH]}
        self._uptime_start: Optional[float] = None

        self._build_ui()
        self._tick_clock()

    # ====================================================================== #
    # UI construction
    # ====================================================================== #

    def _build_ui(self):
        self.root.configure(bg=BG)

        # ── Top bar ──────────────────────────────────────────────────────
        topbar = tk.Frame(self.root, bg=SURFACE, height=48)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="⬡ VISION-SENTRY", bg=SURFACE, fg=TEXT,
                 font=("Courier", 13, "bold")).pack(side="left", padx=16, pady=12)
        tk.Label(topbar, text="YOLOv8 · Security Intelligence System",
                 bg=SURFACE, fg=MUTED, font=("Courier", 9)).pack(side="left")

        self._lbl_status = tk.Label(topbar, text="● IDLE", bg=SURFACE, fg=MUTED,
                                    font=("Courier", 9, "bold"))
        self._lbl_status.pack(side="right", padx=16)

        self._lbl_clock = tk.Label(topbar, text="--:--:--", bg=SURFACE, fg=MUTED,
                                   font=("Courier", 9))
        self._lbl_clock.pack(side="right", padx=8)

        # ── Alert banner ─────────────────────────────────────────────────
        self._alert_frame = tk.Frame(self.root, bg=RED, height=0)
        self._alert_frame.pack(fill="x")
        self._alert_frame.pack_propagate(False)
        self._lbl_alert = tk.Label(self._alert_frame, text="", bg=RED, fg="white",
                                   font=("Courier", 10, "bold"))
        self._lbl_alert.pack(side="left", padx=12)
        tk.Button(self._alert_frame, text="✕ Dismiss", bg=RED, fg="white",
                  relief="flat", cursor="hand2", font=("Courier", 9),
                  command=self._dismiss_alert).pack(side="right", padx=8)

        # ── Body ─────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Left: video + controls
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        self._build_video_panel(left)
        self._build_controls(left)
        self._build_search_panel(left)

        # Right: stats + event stream + performance
        right = tk.Frame(body, bg=BG, width=280)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        self._build_stats_panel(right)
        self._build_event_stream(right)
        self._build_perf_panel(right)

    # ── Video panel ──────────────────────────────────────────────────────

    def _build_video_panel(self, parent):
        frame = tk.Frame(parent, bg=CARD, bd=0, highlightthickness=1,
                         highlightbackground=BORDER)
        frame.pack(fill="x")

        hdr = tk.Frame(frame, bg=CARD)
        hdr.pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(hdr, text="▶ Camera 1 — Main Entrance", bg=CARD, fg=MUTED,
                 font=("Courier", 9)).pack(side="left")
        self._lbl_live = tk.Label(hdr, text="IDLE", bg=CARD, fg=MUTED,
                                  font=("Courier", 8, "bold"))
        self._lbl_live.pack(side="right")

        self._canvas = tk.Canvas(frame, width=FEED_W, height=FEED_H,
                                 bg="#050508", bd=0, highlightthickness=0)
        self._canvas.pack(padx=2, pady=(4, 2))

        # Placeholder text
        self._canvas.create_text(FEED_W//2, FEED_H//2,
                                 text="Load a video file or start webcam to begin",
                                 fill=MUTED, font=("Courier", 11), tags="placeholder")

    # ── Controls bar ─────────────────────────────────────────────────────

    def _build_controls(self, parent):
        bar = tk.Frame(parent, bg=SURFACE, pady=6)
        bar.pack(fill="x", pady=(4, 0))

        def btn(text, cmd, color=BORDER, fg=TEXT):
            b = tk.Button(bar, text=text, command=cmd, bg=color, fg=fg,
                          relief="flat", cursor="hand2", font=("Courier", 9),
                          padx=10, pady=5, activebackground=CARD, activeforeground=TEXT)
            b.pack(side="left", padx=5)
            return b

        self._btn_start = btn("▶  Start Monitoring", self._toggle_monitoring, BLUE, "white")
        btn("📁  Load Video File", self._load_file)
        btn("🎥  Webcam (index 0)", self._open_webcam)
        btn("💾  Export Log", self._export_log)
        btn("🗑  Clear Alerts", self._clear_alerts, RED, "white")

    # ── Search panel ─────────────────────────────────────────────────────

    def _build_search_panel(self, parent):
        card = tk.LabelFrame(parent, text=" Forensic Event Search ",
                             bg=CARD, fg=MUTED, font=("Courier", 9),
                             bd=1, relief="flat", highlightbackground=BORDER)
        card.pack(fill="x", pady=(8, 0))

        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", padx=8, pady=6)

        self._search_var = tk.StringVar()
        entry = tk.Entry(row, textvariable=self._search_var, bg=SURFACE, fg=TEXT,
                         insertbackground=TEXT, relief="flat", font=("Courier", 10),
                         bd=4)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda e: self._do_search())

        tk.Button(row, text="Search", command=self._do_search, bg=BLUE, fg="white",
                  relief="flat", cursor="hand2", font=("Courier", 9),
                  padx=10).pack(side="left", padx=(6, 0))

        # Filter row
        frow = tk.Frame(card, bg=CARD)
        frow.pack(fill="x", padx=8, pady=(0, 4))
        tk.Label(frow, text="Filter:", bg=CARD, fg=MUTED,
                 font=("Courier", 8)).pack(side="left")
        self._filter_var = tk.StringVar(value="ALL")
        for p in ["ALL", "BREACH", "SUSPICIOUS", "MONITOR", "AUTHORIZED"]:
            color = PRI_COLOR.get(p, MUTED)
            rb = tk.Radiobutton(frow, text=p, variable=self._filter_var, value=p,
                                bg=CARD, fg=color, selectcolor=CARD,
                                font=("Courier", 8), command=self._do_search)
            rb.pack(side="left", padx=4)

        # Results list
        res_frame = tk.Frame(card, bg=CARD)
        res_frame.pack(fill="x", padx=8, pady=(0, 8))

        self._search_list = tk.Listbox(res_frame, bg=SURFACE, fg=TEXT,
                                       selectbackground=BLUE, relief="flat",
                                       font=("Courier", 9), height=6,
                                       activestyle="none")
        scroll = tk.Scrollbar(res_frame, orient="vertical",
                              command=self._search_list.yview)
        self._search_list.configure(yscrollcommand=scroll.set)
        self._search_list.pack(side="left", fill="x", expand=True)
        scroll.pack(side="right", fill="y")

    # ── Stats panel ──────────────────────────────────────────────────────

    def _build_stats_panel(self, parent):
        card = tk.Frame(parent, bg=CARD, bd=0)
        card.pack(fill="x", pady=(0, 6))

        tk.Label(card, text="DETECTION STATS", bg=CARD, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=10, pady=(8, 4))

        grid = tk.Frame(card, bg=CARD)
        grid.pack(fill="x", padx=10, pady=(0, 8))

        self._stat_labels = {}
        items = [
            ("AUTH",  PRIORITY_AUTHORIZED, GREEN),
            ("MON",   PRIORITY_MONITOR,    BLUE),
            ("SUSP",  PRIORITY_SUSPICIOUS, AMBER),
            ("BREACH",PRIORITY_BREACH,     RED),
        ]
        for col, (short, key, color) in enumerate(items):
            f = tk.Frame(grid, bg=SURFACE, padx=8, pady=6)
            f.grid(row=0, column=col, padx=3, sticky="nsew")
            grid.columnconfigure(col, weight=1)
            lv = tk.Label(f, text="0", bg=SURFACE, fg=color, font=("Courier", 18, "bold"))
            lv.pack()
            tk.Label(f, text=short, bg=SURFACE, fg=MUTED, font=("Courier", 7)).pack()
            self._stat_labels[key] = lv

    # ── Event stream ─────────────────────────────────────────────────────

    def _build_event_stream(self, parent):
        card = tk.Frame(parent, bg=CARD)
        card.pack(fill="both", expand=True, pady=(0, 6))

        tk.Label(card, text="LIVE EVENT STREAM", bg=CARD, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=10, pady=(8, 4))

        inner = tk.Frame(card, bg=CARD)
        inner.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._event_list = tk.Listbox(inner, bg=SURFACE, fg=TEXT,
                                      selectbackground=BLUE, relief="flat",
                                      font=("Courier", 8), activestyle="none")
        sb = tk.Scrollbar(inner, orient="vertical", command=self._event_list.yview)
        self._event_list.configure(yscrollcommand=sb.set)
        self._event_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # ── Performance panel ────────────────────────────────────────────────

    def _build_perf_panel(self, parent):
        card = tk.Frame(parent, bg=CARD)
        card.pack(fill="x")

        tk.Label(card, text="PERFORMANCE METRICS", bg=CARD, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=10, pady=(8, 4))

        self._perf_labels = {}
        metrics = [
            ("inf_ms",   "Inference"),
            ("fps",      "Frame rate"),
            ("mttd",     "MTTD"),
            ("total",    "Total events"),
            ("uptime",   "Uptime"),
            ("mode",     "Engine mode"),
        ]
        for key, label in metrics:
            row = tk.Frame(card, bg=CARD)
            row.pack(fill="x", padx=10, pady=1)
            tk.Label(row, text=label, bg=CARD, fg=MUTED,
                     font=("Courier", 8), width=12, anchor="w").pack(side="left")
            lv = tk.Label(row, text="--", bg=CARD, fg=TEXT, font=("Courier", 8, "bold"))
            lv.pack(side="right")
            self._perf_labels[key] = lv

        tk.Frame(card, bg=CARD, height=8).pack()   # bottom padding

    # ====================================================================== #
    # Processing loop
    # ====================================================================== #

    def _tick_clock(self):
        """Master scheduler — updates clock and processes video frame."""
        now = datetime.now().strftime("%H:%M:%S")
        self._lbl_clock.configure(text=now)

        if self._running:
            frame, detections = self._vp.tick()
            if frame is not None:
                self._render_frame(frame)
            self._update_stats(detections)
            self._update_perf()

        self.root.after(TICK_MS, self._tick_clock)

    def _render_frame(self, frame):
        """Convert OpenCV BGR frame → Tkinter PhotoImage and blit to canvas."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (FEED_W, FEED_H))
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self._canvas.delete("placeholder")
        self._canvas.create_image(0, 0, anchor="nw", image=img, tags="feed")
        self._canvas._photo = img   # prevent GC

    # ====================================================================== #
    # Stats / metrics updates
    # ====================================================================== #

    def _update_stats(self, detections):
        if not detections:
            return
        for det in detections:
            self._cnt[det.priority] = self._cnt.get(det.priority, 0) + 1

        for key, lbl in self._stat_labels.items():
            lbl.configure(text=str(self._cnt[key]))

        # Update live event stream (newest detections)
        for det in detections:
            ts    = datetime.now().strftime("%H:%M:%S")
            color = PRI_COLOR.get(det.priority, MUTED)
            line  = f"{ts}  {det.priority:<12} {det.label:<18} {det.confidence*100:.0f}%"
            self._event_list.insert(0, line)
            self._event_list.itemconfigure(0, fg=color)
            if self._event_list.size() > 200:
                self._event_list.delete(200, tk.END)

    def _update_perf(self):
        self._perf_labels["inf_ms"].configure(
            text=f"{self._vp.avg_inference_ms:.1f} ms")
        self._perf_labels["fps"].configure(
            text=f"{self._vp.current_fps:.1f} fps")
        mttd = self._alerts.mttd_seconds
        self._perf_labels["mttd"].configure(
            text=f"{mttd:.1f}s" if mttd else "--")
        self._perf_labels["total"].configure(
            text=str(len(self._metadata.search(limit=9999))))
        self._perf_labels["mode"].configure(
            text="Mock" if self._vp.is_mock else "YOLOv8")

        if self._uptime_start:
            import time
            secs = int(time.time() - self._uptime_start)
            m, s = divmod(secs, 60)
            h, m = divmod(m, 60)
            self._perf_labels["uptime"].configure(text=f"{h:02d}:{m:02d}:{s:02d}")

    # ====================================================================== #
    # Alert handling  (Objective 4)
    # ====================================================================== #

    def _on_alert(self, alert: Alert):
        """Callback fired by AlertManager — runs in main thread via after()."""
        self.root.after(0, lambda: self._show_alert(alert))

    def _show_alert(self, alert: Alert):
        color = RED if alert.priority == PRIORITY_BREACH else AMBER
        self._alert_frame.configure(height=32)
        self._lbl_alert.configure(
            text=f"⚠  {alert.priority}: {alert.label}  |  "
                 f"Conf: {alert.confidence*100:.0f}%  |  {datetime.now().strftime('%H:%M:%S')}",
            bg=color)
        self._alert_frame.configure(bg=color)

        self._lbl_status.configure(text="● ALERT", fg=color)

    def _dismiss_alert(self):
        self._alert_frame.configure(height=0)
        self._alerts.acknowledge_all()
        if self._running:
            self._lbl_status.configure(text="● LIVE", fg=GREEN)

    def _clear_alerts(self):
        self._dismiss_alert()
        self._alerts.clear()
        self._cnt[PRIORITY_BREACH] = 0
        self._stat_labels[PRIORITY_BREACH].configure(text="0")

    # ====================================================================== #
    # Control callbacks
    # ====================================================================== #

    def _toggle_monitoring(self):
        import time
        if not self._running:
            if self._source is None:
                # Default to webcam 0
                if not self._vp.open(0):
                    messagebox.showerror("Source Error",
                        "Cannot open webcam.\nPlease load a video file instead.")
                    return
                self._source = 0
            self._running = True
            self._uptime_start = time.time()
            self._alerts.start_session()
            self._btn_start.configure(text="⏸  Pause Monitoring", bg=AMBER, fg="black")
            self._lbl_live.configure(text="● LIVE", fg=GREEN)
            self._lbl_status.configure(text="● LIVE", fg=GREEN)
        else:
            self._running = False
            self._btn_start.configure(text="▶  Start Monitoring", bg=BLUE, fg="white")
            self._lbl_live.configure(text="PAUSED", fg=AMBER)
            self._lbl_status.configure(text="● PAUSED", fg=AMBER)

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All files", "*.*")])
        if not path:
            return
        self._vp.close()
        if self._vp.open(path):
            self._source = path
            self._running = True
            import time
            self._uptime_start = time.time()
            self._alerts.start_session()
            self._btn_start.configure(text="⏸  Pause Monitoring", bg=AMBER, fg="black")
            self._lbl_live.configure(text="● LIVE", fg=GREEN)
            self._lbl_status.configure(text="● LIVE", fg=GREEN)
        else:
            messagebox.showerror("File Error", f"Cannot open:\n{path}")

    def _open_webcam(self):
        self._vp.close()
        idx = 0
        if self._vp.open(idx):
            self._source = idx
            self._running = True
            import time
            self._uptime_start = time.time()
            self._alerts.start_session()
            self._btn_start.configure(text="⏸  Pause Monitoring", bg=AMBER, fg="black")
            self._lbl_live.configure(text="● LIVE", fg=GREEN)
            self._lbl_status.configure(text="● LIVE", fg=GREEN)
        else:
            messagebox.showerror("Webcam Error", "Cannot open webcam (index 0).")

    def _export_log(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="vision_sentry_export.json")
        if not path:
            return
        saved = self._metadata.export_json(path)
        messagebox.showinfo("Export Complete", f"Log exported to:\n{saved}")

    # ── Forensic search ──────────────────────────────────────────────────

    def _do_search(self):
        self._search_list.delete(0, tk.END)
        q = self._search_var.get().strip()
        priority_filter = self._filter_var.get()
        pri = None if priority_filter == "ALL" else priority_filter

        results = self._metadata.search(query=q, priority=pri, limit=100)

        if not results:
            self._search_list.insert(tk.END, "  No events matched your query.")
            self._search_list.itemconfigure(0, fg=MUTED)
            return

        for r in results:
            ts    = r["timestamp"].split("T")[1][:8]
            line  = (f"  {ts}  [{r['priority']:<12}]  "
                     f"{r['label']:<20}  {r['confidence']*100:.0f}%  "
                     f"uid:{r['uid']}")
            self._search_list.insert(tk.END, line)
            color = PRI_COLOR.get(r["priority"], MUTED)
            self._search_list.itemconfigure(tk.END, fg=color)

    # ====================================================================== #
    # Lifecycle
    # ====================================================================== #

    def on_close(self):
        self._running = False
        self._vp.close()
        # Auto-save log on exit
        try:
            self._metadata.export_json()
        except Exception:
            pass
        self.root.destroy()
