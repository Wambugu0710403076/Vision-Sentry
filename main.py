"""
Vision-Sentry: Automated Event Search Methods for Security Footage
Author : Victor Dennis Wambugu | DCF-01-0184/2025
College: Zetech University — Diploma in ICT, April 2026

Entry point — launches the dashboard.
"""

import tkinter as tk
from ui.dashboard import VisionSentryApp


def main():
    root = tk.Tk()
    root.title("Vision-Sentry — Security Intelligence System")
    root.geometry("1280x800")
    root.minsize(1100, 700)
    root.configure(bg="#0D0D14")
    try:
        root.iconbitmap(default="")
    except Exception:
        pass

    app = VisionSentryApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
