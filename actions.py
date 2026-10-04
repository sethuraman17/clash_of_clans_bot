"""
ADB game actions for Clash of Clans.

Attack flow:
  1) Swipe map 4 corners
  2) Tap Attack! (home)
  3) Tap Single Player
  4) Tap Attack! again (start)
  5) Detect buildings → drop troops on lines near those detections

Run:
  python actions.py
  python bot.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from adb_control import AdbDevice
from config import PROJECT_DIR
from detect import Detection, detect_all

TEMPLATES_DIR = PROJECT_DIR / "templates"

# Locked absolute taps from your successful run @ 2392x1080
# (from bot.py log — Attack / Single Player / Attack start / troop slots)
LOCKED_SIZE = (2392, 1080)
LOCKED_XY = {
    "attack_home": (155, 993),
    "single_player": (669, 594),
    "attack_start": (1196, 842),
    "troop_slot_0": (334, 1004),
    "troop_slot_1": (526, 1004),
}

# Fallback ratios if resolution differs from LOCKED_SIZE
UI = {
    "attack_home": (0.065, 0.92),
    "single_player": (0.28, 0.55),
    "attack_start": (0.50, 0.78),
    "troop_slot_0": (0.14, 0.93),
    "troop_slot_1": (0.22, 0.93),
}


def coc_input_size(adb: AdbDevice) -> tuple[int, int]:
    w, h = adb.get_size()
    if w < h:
        w, h = h, w
    return w, h


def landscape_frame(adb: AdbDevice) -> np.ndarray | None:
    frame = adb.screencap()
    if frame is None:
        return None
    if frame.shape[0] > frame.shape[1]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def match_template_center(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.78,
) -> tuple[int, int] | None:
    th, tw = template.shape[:2]
    if th > frame.shape[0] or tw > frame.shape[1]:
        return None
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val < threshold:
        return None
    return max_loc[0] + tw // 2, max_loc[1] + th // 2


def locked_xy(name: str, w: int, h: int) -> tuple[int, int] | None:
    """Scale locked 2392x1080 taps if the current size differs."""
    if name not in LOCKED_XY:
        return None
    x0, y0 = LOCKED_XY[name]
    lw, lh = LOCKED_SIZE
    if (w, h) == (lw, lh):
        return x0, y0
    return int(x0 * w / lw), int(y0 * h / lh)


def tap_xy(adb: AdbDevice, x: int, y: int, label: str = "", source: str = "") -> None:
    src = f" via {source}" if source else ""
    print(f"Tap {label} @ ({x},{y}){src}".strip())
    adb.tap(x, y)


def tap_ratio(adb: AdbDevice, w: int, h: int, rx: float, ry: float, label: str = "") -> None:
    x, y = int(w * rx), int(h * ry)
    tap_xy(adb, x, y, label=label, source=f"ratio=({rx:.3f},{ry:.3f})")


def tap_locked_or_template(
    adb: AdbDevice,
    w: int,
    h: int,
    key: str,
    template_name: str,
    label: str,
    threshold: float = 0.75,
) -> None:
    """Prefer locked XY from your clicks, then template, then ratio."""
    xy = locked_xy(key, w, h)
    if xy is not None:
        tap_xy(adb, xy[0], xy[1], label=label, source="locked")
        return

    path = TEMPLATES_DIR / template_name
    frame = landscape_frame(adb)
    if path.exists() and frame is not None:
        tpl = cv2.imread(str(path))
        hit = match_template_center(frame, tpl, threshold=threshold) if tpl is not None else None
        if hit is not None:
            tap_xy(adb, hit[0], hit[1], label=label, source=f"template {template_name}")
            return

    rx, ry = UI[key]
    tap_ratio(adb, w, h, rx, ry, label=f"{label} (fallback)")


def corner_points(w: int, h: int, margin: int = 90) -> list[tuple[int, int]]:
    return [
        (margin, margin),
        (w - margin, margin),
        (w - margin, h - margin),
        (margin, h - margin),
    ]


def swipe_four_corners(
    adb: AdbDevice,
    w: int,
    h: int,
    duration_ms: int = 450,
    pause_s: float = 0.35,
) -> None:
    cx, cy = w // 2, h // 2
    names = ["top-left", "top-right", "bottom-right", "bottom-left"]
    print(f"Swiping 4 corners @ {w}x{h}")
    for name, (x, y) in zip(names, corner_points(w, h)):
        print(f"  swipe center->{name} ({cx},{cy})->({x},{y})")
        adb.swipe(cx, cy, x, y, duration_ms)
        time.sleep(pause_s)


def tap_attack_home(adb: AdbDevice, w: int, h: int) -> None:
    tap_locked_or_template(adb, w, h, "attack_home", "attack_button.png", "Attack! (home)")


def tap_single_player(adb: AdbDevice, w: int, h: int) -> None:
    tap_locked_or_template(adb, w, h, "single_player", "single_player.png", "Single Player")


def tap_attack_start(adb: AdbDevice, w: int, h: int) -> None:
    tap_locked_or_template(adb, w, h, "attack_start", "attack_start.png", "Attack! (start)")


def wait_for_battle(adb: AdbDevice, timeout_s: float = 45.0) -> bool:
    """
    Wait until battle UI looks ready (troop bar region has content / Attack home gone).
    Simple heuristic: bottom strip is darker/orange troop cards, and we're landscape.
    """
    print(f"Waiting for battle screen (up to {timeout_s:.0f}s)...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        frame = landscape_frame(adb)
        if frame is None:
            time.sleep(0.8)
            continue
        h, w = frame.shape[:2]
        bar = frame[int(h * 0.88) : h, int(w * 0.05) : int(w * 0.70)]
        # troop cards tend to be colorful; mean saturation bump
        hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        sat = float(np.mean(hsv[:, :, 1]))
        # also check a known home Attack orange spot is NOT present strongly
        if sat > 55:
            print(f"Battle UI likely ready (bar sat={sat:.1f})")
            return True
        time.sleep(0.8)
    print("Timed out waiting for battle screen")
    return False


def select_troop(adb: AdbDevice, w: int, h: int, slot: int = 0) -> None:
    key = f"troop_slot_{slot}" if f"troop_slot_{slot}" in LOCKED_XY else "troop_slot_0"
    xy = locked_xy(key, w, h)
    if xy is not None:
        tap_xy(adb, xy[0], xy[1], label=f"troop slot {slot}", source="locked")
        return
    rx, ry = UI.get(key, UI["troop_slot_0"])
    tap_ratio(adb, w, h, rx, ry, label=f"troop slot {slot}")


def drop_line_near_detections(
    dets: list[Detection],
    w: int,
    h: int,
    offset: int = 90,
    points_per_target: int = 4,
    line_span: int = 70,
) -> list[tuple[int, int]]:
    """
    For each detected building, build a short drop LINE just outside it
    (away from the village center) — where troops can usually be placed.
    """
    if not dets:
        # Fallback diamond outside typical base
        return [
            (int(w * 0.18), int(h * 0.50)),
            (int(w * 0.50), int(h * 0.18)),
            (int(w * 0.82), int(h * 0.50)),
            (int(w * 0.50), int(h * 0.82)),
        ]

    # Prefer important targets
    priority = {
        "town_hall": 0,
        "elixir_storage": 1,
        "gold_storage": 1,
        "elixir_collector": 2,
        "gold_mine": 2,
    }
    ordered = sorted(dets, key=lambda d: priority.get(d.resource, 9))

    cx = sum(d.center[0] for d in ordered) / len(ordered)
    cy = sum(d.center[1] for d in ordered) / len(ordered)

    points: list[tuple[int, int]] = []
    margin = 40
    for d in ordered[:6]:
        dx, dy = d.center
        vx, vy = dx - cx, dy - cy
        norm = math.hypot(vx, vy) or 1.0
        ux, uy = vx / norm, vy / norm
        # outward from village center, past the building
        ox = dx + ux * (max(d.w, d.h) * 0.35 + offset)
        oy = dy + uy * (max(d.w, d.h) * 0.35 + offset)
        # perpendicular for a short drop line
        px, py = -uy, ux
        for i in range(points_per_target):
            t = (i / max(points_per_target - 1, 1)) - 0.5
            x = int(ox + px * line_span * t)
            y = int(oy + py * line_span * t)
            x = max(margin, min(w - margin, x))
            y = max(int(h * 0.12), min(int(h * 0.82), y))  # keep off HUD / troop bar
            points.append((x, y))
    return points


def deploy_troops_near_detections(
    adb: AdbDevice,
    w: int,
    h: int,
    taps_per_point: int = 2,
    pause_s: float = 0.12,
) -> None:
    frame = landscape_frame(adb)
    if frame is None:
        print("No frame for detection — using fallback drop points")
        dets: list[Detection] = []
    else:
        dets, _masks = detect_all(frame)
        print(f"Battle detections: {len(dets)}")
        for d in dets[:10]:
            print(f"  {d.resource:18s} @ {d.center} score={d.score:.2f}")

    points = drop_line_near_detections(dets, w, h)
    print(f"Deploying along {len(points)} drop points near detections")

    select_troop(adb, w, h, slot=0)
    time.sleep(0.25)

    for i, (x, y) in enumerate(points):
        for _ in range(taps_per_point):
            adb.tap(x, y)
            time.sleep(pause_s)
        if i % 4 == 0:
            print(f"  dropped near point #{i+1} ({x},{y})")

    # second troop type if available
    select_troop(adb, w, h, slot=1)
    time.sleep(0.2)
    for x, y in points[::2]:
        adb.tap(x, y)
        time.sleep(pause_s)


def run_attack_flow(adb: AdbDevice | None = None) -> None:
    """
    Full path you asked for:
      corners → Attack → Single Player → Attack → detect → drop troops
    """
    adb = adb or AdbDevice()
    if not adb.is_connected():
        raise RuntimeError(adb.connection_help())

    w, h = coc_input_size(adb)
    print(f"=== CoC attack flow @ {w}x{h} ===")

    # 1) look around base
    swipe_four_corners(adb, w, h)
    time.sleep(0.5)

    # 2) Attack from home
    tap_attack_home(adb, w, h)
    time.sleep(1.4)

    # 3) Single Player
    tap_single_player(adb, w, h)
    time.sleep(1.2)

    # 4) Attack again (start fight / search)
    tap_attack_start(adb, w, h)
    time.sleep(1.0)

    # 5) Wait for battle map, then drop near detections
    if not wait_for_battle(adb, timeout_s=50):
        print("Continuing anyway — try deploy")
    time.sleep(0.8)

    deploy_troops_near_detections(adb, w, h)
    print("=== Attack flow done ===")


def run_corners_then_attack(adb: AdbDevice | None = None) -> None:
    """Legacy: corners + home Attack only."""
    adb = adb or AdbDevice()
    if not adb.is_connected():
        raise RuntimeError(adb.connection_help())
    w, h = coc_input_size(adb)
    swipe_four_corners(adb, w, h)
    time.sleep(0.4)
    tap_attack_home(adb, w, h)
    print("Done: 4 corners + Attack")


if __name__ == "__main__":
    run_attack_flow()
