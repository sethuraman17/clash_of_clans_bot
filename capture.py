"""
One-window workflow:
  1. Starts scrcpy.exe (if not already open)
  2. Captures THAT window for CV (no second game mirror)
  3. Draws a transparent overlay ON TOP of scrcpy
  4. Click overlay → ADB tap on the phone

Controls (focus overlay / keep it on top):
  Left click  → tap phone at that spot
  A           → full attack: corners → Attack → Single Player → Attack → drop troops
  D           → run CV detect (templates + pink elixir blobs)
  S           → save current frame to captures/
  R           → refresh window bounds + device size
  H           → hide / show overlay drawings
  Q / Esc     → quit (scrcpy stays open)
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path

import cv2
import mss
import numpy as np

from actions import run_attack_flow
from adb_control import AdbDevice
from config import PROJECT_DIR, WINDOW_TITLE
from detect import detect_all, draw_detections
from scrcpy_manager import get_scrcpy_rect, grab_window_bgr, start_scrcpy

CAPTURE_DIR = PROJECT_DIR / "captures"
CAPTURE_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = PROJECT_DIR / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

# Color key for Windows transparent overlay (drawn as "invisible")
CLEAR = "#010203"

# Overlay colors matching detect.py resources
RESOURCE_COLORS = {
    "town_hall": "#00ff50",
    "elixir_storage": "#ff00dc",
    "elixir_collector": "#ff78ff",
    "gold_storage": "#00b4ff",
    "gold_mine": "#00dcff",
    "template": "#32ff32",
}


def map_to_device(
    x: int,
    y: int,
    frame_w: int,
    frame_h: int,
    device_w: int,
    device_h: int,
) -> tuple[int, int]:
    dx = int(x * device_w / max(frame_w, 1))
    dy = int(y * device_h / max(frame_h, 1))
    return max(0, min(device_w - 1, dx)), max(0, min(device_h - 1, dy))


def match_template(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.85,
) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of best match top-left box, or None."""
    if template is None or frame is None:
        return None
    th, tw = template.shape[:2]
    if th > frame.shape[0] or tw > frame.shape[1]:
        return None
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val < threshold:
        return None
    return max_loc[0], max_loc[1], tw, th


def load_templates() -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for path in sorted(TEMPLATES_DIR.glob("*.png")):
        img = cv2.imread(str(path))
        if img is not None:
            out.append((path.stem, img))
    return out


class OverlayApp:
    def __init__(self) -> None:
        self.adb = AdbDevice()
        if not self.adb.is_connected():
            raise RuntimeError("ADB device not connected")

        start_scrcpy(WINDOW_TITLE)
        self.rect = get_scrcpy_rect(WINDOW_TITLE)
        self.device_w, self.device_h = self.adb.get_size()
        self.templates = load_templates()
        # (label, x, y, w, h, resource, score)
        self.detections: list[tuple[str, int, int, int, int, str, float]] = []
        self.last_tap: tuple[int, int, int, int] | None = None
        self.show_drawings = True
        self.save_i = 0
        self.sct = mss.mss()  # fallback if PrintWindow hwnd missing
        self.running = True

        self.root = tk.Tk()
        self.root.title("CoC CV Overlay")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        # Make CLEAR pixels fully transparent on Windows
        self.root.attributes("-transparentcolor", CLEAR)
        self.root.configure(bg=CLEAR)

        self.canvas = tk.Canvas(
            self.root,
            bg=CLEAR,
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)

        self._place_over_scrcpy()
        self._bind()

        print("scrcpy + CV overlay ready (no second mirror window)")
        print(f"  scrcpy client: {self.rect.width}x{self.rect.height}")
        print(f"  device:        {self.device_w}x{self.device_h}")
        print(f"  templates:     {len(self.templates)} in ./templates")
        print(
            "Click = ADB tap | A = full attack flow | D = detect | "
            "S save | R refresh | H hide | Q quit"
        )
        self._action_busy = False
        self.live_detect = True  # keep scanning each tick when templates/color on

    def _place_over_scrcpy(self) -> None:
        self.rect = get_scrcpy_rect(WINDOW_TITLE)
        self.root.geometry(
            f"{self.rect.width}x{self.rect.height}+{self.rect.left}+{self.rect.top}"
        )

    def _bind(self) -> None:
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<KeyPress-q>", lambda _e: self.quit())
        self.root.bind("<KeyPress-Q>", lambda _e: self.quit())
        self.root.bind("<Escape>", lambda _e: self.quit())
        self.root.bind("<KeyPress-s>", lambda _e: self.save_frame())
        self.root.bind("<KeyPress-S>", lambda _e: self.save_frame())
        self.root.bind("<KeyPress-r>", lambda _e: self.refresh())
        self.root.bind("<KeyPress-R>", lambda _e: self.refresh())
        self.root.bind("<KeyPress-h>", lambda _e: self.toggle_drawings())
        self.root.bind("<KeyPress-H>", lambda _e: self.toggle_drawings())
        self.root.bind("<KeyPress-a>", lambda _e: self.run_attack_sequence())
        self.root.bind("<KeyPress-A>", lambda _e: self.run_attack_sequence())
        self.root.bind("<KeyPress-d>", lambda _e: self.scan_now())
        self.root.bind("<KeyPress-D>", lambda _e: self.scan_now())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def grab(self) -> np.ndarray:
        # Prefer PrintWindow on scrcpy hwnd so our overlay is NOT in the frame
        if self.rect.hwnd:
            return grab_window_bgr(self.rect.hwnd, self.rect.width, self.rect.height)
        shot = self.sct.grab(self.rect.monitor)
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    def on_click(self, event: tk.Event) -> None:
        dx, dy = map_to_device(
            event.x,
            event.y,
            self.rect.width,
            self.rect.height,
            self.device_w,
            self.device_h,
        )
        print(f"tap overlay=({event.x},{event.y}) → device=({dx},{dy})")
        self.adb.tap(dx, dy)
        self.last_tap = (event.x, event.y, dx, dy)

    def save_frame(self) -> None:
        frame = self.grab()
        path = CAPTURE_DIR / f"frame_{self.save_i:04d}.png"
        cv2.imwrite(str(path), frame)
        print(f"saved {path}")
        self.save_i += 1

    def refresh(self) -> None:
        self._place_over_scrcpy()
        self.device_w, self.device_h = self.adb.get_size()
        self.templates = load_templates()
        print(
            f"refreshed → {self.rect.width}x{self.rect.height}, "
            f"device {self.device_w}x{self.device_h}, "
            f"templates {len(self.templates)}"
        )

    def toggle_drawings(self) -> None:
        self.show_drawings = not self.show_drawings
        print("drawings", "ON" if self.show_drawings else "OFF")

    def run_attack_sequence(self) -> None:
        """Full attack flow (corners → menus → detect → troop drops)."""
        if self._action_busy:
            print("action already running...")
            return

        def worker() -> None:
            self._action_busy = True
            try:
                run_attack_flow(self.adb)
            except Exception as exc:
                print("action failed:", exc)
            finally:
                self._action_busy = False

        threading.Thread(target=worker, daemon=True).start()

    def scan_now(self) -> None:
        """One-shot detect, save neat annotated image, refresh overlay boxes."""
        try:
            frame = self.grab()
            dets, masks = detect_all(frame)
            self.detections = [
                (d.name, d.x, d.y, d.w, d.h, d.resource, d.score) for d in dets
            ]
            annotated = draw_detections(frame, dets, masks)
            out = CAPTURE_DIR / "detect_preview.png"
            cv2.imwrite(str(out), annotated)
            print(f"Detect: {len(dets)} hits → saved {out}")
            for i, d in enumerate(dets[:20], start=1):
                cx, cy = d.center
                print(
                    f"  #{i:02d} {d.resource:12s} ({cx},{cy}) "
                    f"{d.w}x{d.h} score={d.score:.2f}"
                )
            if not dets:
                print("No resource blobs found — check lighting / HSV ranges.")
            self.redraw()
        except Exception as exc:
            print("detect failed:", exc)

    def quit(self) -> None:
        self.running = False
        try:
            self.sct.close()
        except Exception:
            pass
        self.root.destroy()

    def redraw(self) -> None:
        self.canvas.delete("all")
        # Keep a full CLEAR fill so transparentcolor works every frame
        self.canvas.create_rectangle(
            0,
            0,
            self.rect.width,
            self.rect.height,
            fill=CLEAR,
            outline=CLEAR,
        )

        if not self.show_drawings:
            return

        # Status chip (top-left)
        self.canvas.create_rectangle(8, 8, 520, 40, fill="#111111", outline="#ffffff", width=2)
        self.canvas.create_text(
            16,
            24,
            anchor="w",
            fill="#ffffff",
            font=("Segoe UI", 11, "bold"),
            text=(
                f"CV | {self.device_w}x{self.device_h} | "
                f"A=attack D=detect hits={len(self.detections)} | q quit"
            ),
        )

        for i, item in enumerate(self.detections, start=1):
            name, x, y, w, h, resource, score = item
            color = RESOURCE_COLORS.get(resource, "#32ff32")
            x2, y2 = x + w, y + h
            # Thick box + outer dark edge for contrast
            self.canvas.create_rectangle(x - 1, y - 1, x2 + 1, y2 + 1, outline="#000000", width=5)
            self.canvas.create_rectangle(x, y, x2, y2, outline=color, width=3)

            # Corner ticks
            tick = max(10, min(w, h) // 4)
            for (cx, cy, dx, dy) in (
                (x, y, 1, 1),
                (x2, y, -1, 1),
                (x, y2, 1, -1),
                (x2, y2, -1, -1),
            ):
                self.canvas.create_line(cx, cy, cx + dx * tick, cy, fill=color, width=4)
                self.canvas.create_line(cx, cy, cx, cy + dy * tick, fill=color, width=4)

            label = f"#{i} {resource.upper()} {score:.2f}"
            ly = max(22, y - 8)
            # Strong label chip
            self.canvas.create_rectangle(
                x, ly - 18, x + 11 * len(label), ly + 4,
                fill=color, outline="#ffffff", width=2,
            )
            self.canvas.create_text(
                x + 6,
                ly - 7,
                anchor="w",
                fill="#ffffff",
                font=("Segoe UI", 10, "bold"),
                text=label,
            )

        if self.last_tap:
            x, y, dx, dy = self.last_tap
            r = 10
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r, outline="#ff3333", width=2
            )
            self.canvas.create_text(
                x + 14,
                y - 10,
                anchor="w",
                fill="#ff3333",
                font=("Segoe UI", 9, "bold"),
                text=f"({dx},{dy})",
            )

    def tick(self) -> None:
        if not self.running:
            return
        try:
            # Stay glued to scrcpy if user moves/resizes it
            new_rect = get_scrcpy_rect(WINDOW_TITLE)
            if (
                new_rect.left != self.rect.left
                or new_rect.top != self.rect.top
                or new_rect.width != self.rect.width
                or new_rect.height != self.rect.height
            ):
                self.rect = new_rect
                self.root.geometry(
                    f"{self.rect.width}x{self.rect.height}+{self.rect.left}+{self.rect.top}"
                )

            frame = self.grab()
            if self.live_detect:
                found, _masks = detect_all(frame)
                self.detections = [
                    (d.name, d.x, d.y, d.w, d.h, d.resource, d.score) for d in found
                ]
            else:
                dets_fb: list[tuple[str, int, int, int, int, str, float]] = []
                for name, tpl in self.templates:
                    hit = match_template(frame, tpl)
                    if hit:
                        dets_fb.append((name, hit[0], hit[1], hit[2], hit[3], "template", 1.0))
                self.detections = dets_fb

            self.redraw()
        except Exception as exc:
            print("tick error:", exc)

        self.root.after(80, self.tick)  # ~12 FPS CV / overlay refresh

    def run(self) -> None:
        self.tick()
        self.root.mainloop()


def main() -> None:
    try:
        OverlayApp().run()
    except Exception as exc:
        print(exc)


if __name__ == "__main__":
    main()
