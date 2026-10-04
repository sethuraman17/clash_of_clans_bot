"""
Detect:
  - town_hall          (high-accuracy template match — primary goal)
  - elixir_storage / elixir_collector
  - gold_storage / gold_mine

Town Hall uses dedicated multi-scale template matching (strict score).
Resources still use color rules + templates.

Run:
  python detect.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from adb_control import AdbDevice
from config import PROJECT_DIR

TEMPLATES_DIR = PROJECT_DIR / "templates"
OUT_DIR = PROJECT_DIR / "captures"
OUT_DIR.mkdir(exist_ok=True)

STYLE = {
    "town_hall": {"bgr": (0, 255, 80), "fill": (0, 180, 40), "label": "TOWN HALL"},
    "elixir_storage": {"bgr": (255, 0, 220), "fill": (200, 0, 180), "label": "ELIXIR STORAGE"},
    "elixir_collector": {"bgr": (255, 120, 255), "fill": (200, 80, 200), "label": "ELIXIR MINE"},
    "gold_storage": {"bgr": (0, 180, 255), "fill": (0, 140, 220), "label": "GOLD STORAGE"},
    "gold_mine": {"bgr": (0, 220, 255), "fill": (0, 180, 220), "label": "GOLD MINE"},
    "template": {"bgr": (50, 255, 50), "fill": (30, 180, 30), "label": "TEMPLATE"},
}

# Town Hall: template confirm + orange-roof geometry (single best hit)
TOWN_HALL_MIN_SCORE = 0.75
TOWN_HALL_SCALES = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35)

# Keep original good pink/gold thresholds (slightly tightened)
RESOURCE_HSV = {
    "elixir": [
        (np.array([142, 100, 110], dtype=np.uint8), np.array([168, 255, 255], dtype=np.uint8)),
    ],
    "gold": [
        (np.array([16, 150, 150], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8)),
    ],
}

IOU_MERGE = 0.30


@dataclass
class Detection:
    name: str
    x: int
    y: int
    w: int
    h: int
    score: float
    method: str
    resource: str
    mask: np.ndarray | None = None

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def style_key(self) -> str:
        return self.resource if self.resource in STYLE else "template"


def _hud_exclude_mask(h: int, w: int) -> np.ndarray:
    """Ignore HUD / buttons / side icons (Attack, Shop, Chat, etc.)."""
    keep = np.ones((h, w), dtype=np.uint8) * 255
    keep[0 : int(h * 0.14), :] = 0                      # top bars
    keep[int(h * 0.78) : h, :] = 0                      # Attack / Shop / troop bar
    keep[:, 0 : int(w * 0.10)] = 0                      # left icons (chat/settings)
    keep[:, int(w * 0.92) : w] = 0                      # right edge UI
    # Extra kill-zones for big orange buttons
    keep[int(h * 0.72) : h, 0 : int(w * 0.22)] = 0      # Attack corner
    keep[int(h * 0.72) : h, int(w * 0.78) : w] = 0      # Shop corner
    return keep


def _build_mask(hsv: np.ndarray, resource: str) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in RESOURCE_HSV[resource]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
    # Stronger open kills tiny archer-head speckles before we even contour
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)
    return mask


def _circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def _iou(a: Detection, b: Detection) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / float(a.w * a.h + b.w * b.h - inter)


def _nms(dets: list[Detection], iou_thresh: float = IOU_MERGE) -> list[Detection]:
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for d in dets:
        if any(_iou(d, k) >= iou_thresh for k in kept):
            continue
        kept.append(d)
    return kept


def _classify_building(
    kind: str,  # elixir | gold
    area: float,
    side: float,
    circ: float,
    compactness: float,
    min_dim: int,
) -> str | None:
    """
    Map a blob to one of the 4 building classes, or reject (None).

    Critical: Town Hall yellow roof is a BIG gold blob → reject oversized gold.
    """
    # Tiny = troops / archer hair / UI crumbs
    # Collectors' glass vials can be smaller than storages — allow a lower floor
    min_side = max(36, int(0.032 * min_dim)) if kind == "elixir" else max(48, int(0.045 * min_dim))
    min_area = max(900, (0.030 * min_dim) ** 2) if kind == "elixir" else max(1400, (0.045 * min_dim) ** 2)
    if side < min_side:
        return None
    if area < min_area:
        return None
    if compactness < 0.28:
        return None

    # Town Hall / builder hut roofs: too large to be a mine/storage gold patch
    th_side = max(125, int(0.115 * min_dim))
    th_area = max(7000, (0.10 * min_dim) ** 2)
    if kind == "gold" and (side >= th_side or area >= th_area):
        return None

    storage_side = max(78, int(0.072 * min_dim))
    storage_area = max(3500, (0.065 * min_dim) ** 2)

    if kind == "elixir":
        # Sphere storages: larger + rounder
        if side >= storage_side and area >= storage_area and circ >= 0.28:
            return "elixir_storage"
        # Collectors / smaller tanks
        if side <= max(140, int(0.14 * min_dim)) and circ >= 0.20:
            return "elixir_collector"
        return None

    # gold
    # Storage: medium box of coins (not TH-huge)
    if (
        storage_side * 0.7 <= side < th_side
        and area >= storage_area * 0.55
        and compactness >= 0.35
    ):
        return "gold_storage"

    # Mine: smaller gold patch / collect bubble
    mine_side_max = max(110, int(0.10 * min_dim))
    if side <= mine_side_max and circ >= 0.20:
        return "gold_mine"

    return None


def find_resource_buildings(frame: np.ndarray) -> tuple[list[Detection], dict[str, np.ndarray]]:
    """Color + geometry rules → only the 4 building types."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    hud = _hud_exclude_mask(h, w)

    masks: dict[str, np.ndarray] = {}
    dets: list[Detection] = []

    for kind in ("elixir", "gold"):
        mask = _build_mask(hsv, kind)
        mask = cv2.bitwise_and(mask, hud)
        masks[kind] = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            x, y, bw, bh = cv2.boundingRect(cnt)
            side = float(max(bw, bh))
            aspect = bw / max(bh, 1)
            if aspect > 2.4 or aspect < 0.42:
                continue  # buildings are roughly square/round, not strips

            compactness = area / float(max(bw * bh, 1))
            circ = _circularity(cnt)
            cls = _classify_building(kind, area, side, circ, compactness, min_dim)
            if cls is None:
                continue

            # Score: size + how "building-like" (compact + circular for storages)
            size_score = min(1.0, area / 6000.0)
            shape_score = 0.5 * compactness + 0.5 * min(circ, 1.0)
            score = float(0.55 * size_score + 0.45 * shape_score)

            dets.append(
                Detection(
                    name=cls,
                    x=x,
                    y=y,
                    w=bw,
                    h=bh,
                    score=score,
                    method="color+rules",
                    resource=cls,
                )
            )

    dets = [d for d in dets if d.score >= 0.35]
    dets = _nms(dets)
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets, masks


def _find_town_hall_by_roof(frame: np.ndarray) -> Detection | None:
    """
    Geometry: the Town Hall has the largest solid orange/yellow wooden roof
    in the village (not UI buttons, not gold-mine coins).
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Warm orange-yellow roof planks (TH5–TH7 style)
    mask = cv2.inRange(
        hsv,
        np.array([8, 90, 120], dtype=np.uint8),
        np.array([26, 255, 255], dtype=np.uint8),
    )
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)

    # Village band only — kill HUD / Attack / Shop
    mask[0 : int(h * 0.14), :] = 0
    mask[int(h * 0.72) : h, :] = 0
    mask[:, 0 : int(w * 0.12)] = 0
    mask[:, int(w * 0.90) : w] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    best_area = 0.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 5000:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        side = max(bw, bh)
        # TH roof is large but not screen-wide
        if side < max(100, int(0.09 * min(h, w))):
            continue
        if side > int(0.35 * min(h, w)):
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.7 or aspect > 2.2:
            continue
        compactness = area / float(max(bw * bh, 1))
        if compactness < 0.25:
            continue
        if area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is None:
        return None

    x, y, bw, bh = cv2.boundingRect(best_cnt)
    # Expand downward to cover the full Town Hall body under the roof
    pad_x = int(bw * 0.12)
    pad_top = int(bh * 0.10)
    pad_bot = int(bh * 0.55)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_top)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_bot)
    score = min(0.99, 0.70 + best_area / 40000.0)
    return Detection(
        name="town_hall",
        x=x0,
        y=y0,
        w=x1 - x0,
        h=y1 - y0,
        score=float(score),
        method="th_roof",
        resource="town_hall",
    )


def find_town_hall(frame: np.ndarray) -> Detection | None:
    """
    High-accuracy Town Hall:
      1) Multi-scale template (town_hall.png / town_hall_roof.png)
      2) Fallback: largest orange wooden roof in the village band
    Always returns at most ONE detection.
    """
    if frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    paths = [
        TEMPLATES_DIR / "town_hall.png",
        TEMPLATES_DIR / "town_hall_roof.png",
    ]
    best: Detection | None = None

    for path in paths:
        if not path.exists():
            continue
        tpl = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if tpl is None:
            continue
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        min_score = TOWN_HALL_MIN_SCORE if path.stem == "town_hall" else 0.72

        for scale in TOWN_HALL_SCALES:
            th = max(16, int(tpl_gray.shape[0] * scale))
            tw = max(16, int(tpl_gray.shape[1] * scale))
            if th >= gray.shape[0] or tw >= gray.shape[1]:
                continue
            scaled = cv2.resize(tpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
            _mn, max_val, _ml, max_loc = cv2.minMaxLoc(result)
            if max_val < min_score:
                continue
            cand = Detection(
                name="town_hall",
                x=int(max_loc[0]),
                y=int(max_loc[1]),
                w=tw,
                h=th,
                score=float(max_val),
                method="template_th",
                resource="town_hall",
            )
            if best is None or cand.score > best.score:
                best = cand

    roof = _find_town_hall_by_roof(frame)
    if best is None:
        return roof
    if roof is None:
        return best

    # If both agree (overlap), boost confidence and prefer larger box (full building)
    if _iou(best, roof) >= 0.15:
        x0 = min(best.x, roof.x)
        y0 = min(best.y, roof.y)
        x1 = max(best.x + best.w, roof.x + roof.w)
        y1 = max(best.y + best.h, roof.y + roof.h)
        return Detection(
            name="town_hall",
            x=x0,
            y=y0,
            w=x1 - x0,
            h=y1 - y0,
            score=min(0.99, max(best.score, roof.score) + 0.05),
            method="th_fused",
            resource="town_hall",
        )

    # Disagree: trust roof geometry for this TH level (template may be zoom-shifted)
    return roof if roof.score >= 0.75 else best


def find_templates(
    frame: np.ndarray,
    threshold: float = 0.70,
    max_per_template: int = 8,
    scales: tuple[float, ...] = (0.80, 0.90, 1.0, 1.10, 1.20),
) -> list[Detection]:
    """
    Multi-scale template match for building crops in templates/.

    Uses BUILDING bodies (elixir_collector / gold_mine), not Full! icons,
    so collectors also match on opponent bases where collect bubbles are absent.
    Extra variants: elixir_collector_b.png, gold_mine_b.png
    """
    if frame is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets: list[Detection] = []
    wanted = {
        "elixir_storage",
        "elixir_collector",
        "gold_storage",
        "gold_mine",
    }
    # Building-body templates — collectors need a bit looser threshold
    thr_for = {
        "elixir_storage": 0.72,
        "elixir_collector": 0.65,
        "gold_storage": 0.72,
        "gold_mine": 0.65,
    }

    for path in sorted(TEMPLATES_DIR.glob("*.png")):
        key = path.stem.lower()
        if key.startswith("town_hall"):
            continue  # handled by find_town_hall
        resource = None
        # Map elixir_collector_b -> elixir_collector, gold_mine_b -> gold_mine
        for name in sorted(wanted, key=len, reverse=True):
            if key == name or key.startswith(name):
                resource = name
                break
        if resource is None:
            continue

        tpl = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if tpl is None:
            continue
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        thr = thr_for[resource]

        for scale in scales:
            th = max(8, int(tpl_gray.shape[0] * scale))
            tw = max(8, int(tpl_gray.shape[1] * scale))
            if th >= gray.shape[0] or tw >= gray.shape[1]:
                continue
            scaled = cv2.resize(tpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
            work = result.copy()
            found = 0
            while found < max_per_template:
                _mn, max_val, _ml, max_loc = cv2.minMaxLoc(work)
                if max_val < thr:
                    break
                x, y = max_loc
                dets.append(
                    Detection(
                        name=resource,
                        x=x,
                        y=y,
                        w=tw,
                        h=th,
                        score=float(max_val),
                        method="template",
                        resource=resource,
                    )
                )
                work[
                    max(0, y - th // 2) : min(work.shape[0], y + th // 2),
                    max(0, x - tw // 2) : min(work.shape[1], x + tw // 2),
                ] = 0
                found += 1
    return dets


def detect_all(
    frame: np.ndarray,
    use_resource_color: bool = True,
    use_templates: bool = True,
) -> tuple[list[Detection], dict[str, np.ndarray]]:
    """Town Hall first (strict), then resource templates/color."""
    dets: list[Detection] = []
    masks: dict[str, np.ndarray] = {}

    th = find_town_hall(frame)
    if th is not None:
        dets.append(th)

    if use_templates:
        dets.extend(find_templates(frame))

    if use_resource_color:
        color_dets, masks = find_resource_buildings(frame)
        templated = {d.resource for d in dets if d.method.startswith("template")}
        for d in color_dets:
            if d.resource in templated and d.score < 0.70:
                continue
            # Never re-label a gold blob on top of the Town Hall box
            if th is not None and d.resource.startswith("gold") and _iou(d, th) > 0.15:
                continue
            dets.append(d)

    dets.sort(key=lambda d: d.score, reverse=True)
    dets = _nms(dets)
    return dets, masks


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float = 0.68,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 8, 6
    y2 = max(th + pad_y * 2 + 2, y)
    x1 = max(0, x)
    x2 = min(img.shape[1] - 1, x1 + tw + pad_x * 2)
    y0 = max(0, y2 - th - pad_y * 2)
    cv2.rectangle(img, (x1 + 2, y0 + 2), (x2 + 2, y2 + 2), (0, 0, 0), -1)
    cv2.rectangle(img, (x1, y0), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y0), (x2, y2), (255, 255, 255), 2)
    tx, ty = x1 + pad_x, y2 - pad_y - baseline + 1
    cv2.putText(img, text, (tx, ty), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_segmentation_overlay(
    frame: np.ndarray,
    masks: dict[str, np.ndarray],
    alpha: float = 0.34,
) -> np.ndarray:
    out = frame.copy().astype(np.float32)
    color_map = {
        "elixir": STYLE["elixir_collector"]["fill"],
        "gold": STYLE["gold_mine"]["fill"],
    }
    for key, mask in masks.items():
        if mask is None or key not in color_map:
            continue
        color = np.array(color_map[key], dtype=np.float32)
        layer = out.copy()
        layer[mask > 0] = color
        blend = cv2.addWeighted(out, 1.0 - alpha, layer, alpha, 0)
        m = (mask > 0)[:, :, None]
        out = np.where(m, blend, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_detections(
    frame: np.ndarray,
    dets: list[Detection],
    masks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    out = draw_segmentation_overlay(frame, masks) if masks else frame.copy()
    used: list[tuple[int, int, int]] = []

    for i, d in enumerate(dets, start=1):
        st = STYLE[d.style_key]
        color = st["bgr"]
        x1, y1, x2, y2 = d.x, d.y, d.x + d.w, d.y + d.h
        cv2.rectangle(out, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 0), 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 4)
        tick = max(12, min(d.w, d.h) // 3)
        for (cx, cy, dx, dy) in (
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ):
            cv2.line(out, (cx, cy), (cx + dx * tick, cy), color, 5, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy + dy * tick), color, 5, cv2.LINE_AA)

        label = f"#{i} {st['label']}"
        lx, ly = x1, max(28, y1 - 6)
        for _ in range(8):
            if any(abs(ly - uy) < 26 and abs(lx - ux) < max(uw, 140) for ux, uy, uw in used):
                ly += 26
            else:
                break
        _draw_label(out, label, lx, ly, color, scale=0.68)
        used.append((lx, ly, 160))

        cx, cy = d.center
        cv2.circle(out, (cx, cy), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

    _draw_legend(out, dets)
    return out


def _draw_legend(img: np.ndarray, dets: list[Detection]) -> None:
    keys = [
        "town_hall",
        "elixir_storage",
        "elixir_collector",
        "gold_storage",
        "gold_mine",
    ]
    counts = {k: 0 for k in keys}
    for d in dets:
        if d.resource in counts:
            counts[d.resource] += 1

    lines = [(STYLE[k]["label"], k, counts[k]) for k in keys]
    panel_w, row_h = 280, 34
    panel_h = 18 + row_h * len(lines)
    x0, y0 = 16, 16
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), 2)
    for i, (title, key, n) in enumerate(lines):
        y = y0 + 28 + i * row_h
        color = STYLE[key]["bgr"]
        cv2.rectangle(img, (x0 + 12, y - 14), (x0 + 36, y + 6), color, -1)
        cv2.rectangle(img, (x0 + 12, y - 14), (x0 + 36, y + 6), (255, 255, 255), 1)
        cv2.putText(
            img,
            f"{title}: {n}",
            (x0 + 48, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_mask_panel(frame: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    panel = np.zeros_like(frame)
    color_map = {
        "elixir": STYLE["elixir_collector"]["bgr"],
        "gold": STYLE["gold_mine"]["bgr"],
    }
    for key, mask in masks.items():
        if key not in color_map:
            continue
        panel[mask > 0] = color_map[key]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, (255, 255, 255), 1)
    return cv2.addWeighted(frame, 0.35, panel, 0.65, 0)


def main() -> None:
    adb = AdbDevice()
    if not adb.is_connected():
        print(adb.connection_help())
        # Fallback: restyle last capture if present
        fallback = OUT_DIR / "detect_preview.png"
        if not fallback.exists():
            return
        print(f"Using fallback image: {fallback}")
        frame = cv2.imread(str(fallback))
    else:
        frame = adb.screencap()
        if frame is None:
            print("screencap failed")
            return
        if frame.shape[0] > frame.shape[1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    dets, masks = detect_all(frame)
    print(f"Found {len(dets)} buildings (mines/storages only):")
    for i, d in enumerate(dets, start=1):
        cx, cy = d.center
        print(
            f"  #{i:02d} {d.resource:18s} center=({cx:4d},{cy:4d}) "
            f"{d.w}x{d.h} score={d.score:.2f}"
        )

    annotated = draw_detections(frame, dets, masks)
    mask_view = draw_mask_panel(frame, masks)
    cv2.imwrite(str(OUT_DIR / "detect_preview.png"), annotated)
    cv2.imwrite(str(OUT_DIR / "detect_masks.png"), mask_view)
    print(f"Saved {OUT_DIR / 'detect_preview.png'}")
    print(f"Saved {OUT_DIR / 'detect_masks.png'}")
    print(
        "\nFor even stricter accuracy, crop these into templates/:\n"
        "  elixir_storage.png  elixir_collector.png  gold_storage.png  gold_mine.png"
    )


if __name__ == "__main__":
    main()
