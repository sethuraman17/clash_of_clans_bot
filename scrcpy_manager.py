"""Start scrcpy and locate its window (client area) on Windows."""

from __future__ import annotations

import ctypes
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pygetwindow as gw

from config import DEVICE_SERIAL, SCRCPY_PATH, WINDOW_TITLE

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


@dataclass
class WindowRect:
    left: int
    top: int
    width: int
    height: int
    hwnd: int

    @property
    def monitor(self) -> dict:
        return {
            "left": max(0, self.left),
            "top": max(0, self.top),
            "width": max(1, self.width),
            "height": max(1, self.height),
        }


def find_hwnd_by_title(title: str) -> Optional[int]:
    hwnd = user32.FindWindowW(None, title)
    return int(hwnd) if hwnd else None


def get_client_rect(hwnd: int) -> WindowRect:
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("GetClientRect failed")

    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        raise RuntimeError("ClientToScreen failed")

    width = rect.right - rect.left
    height = rect.bottom - rect.top
    return WindowRect(left=pt.x, top=pt.y, width=width, height=height, hwnd=hwnd)


def scrcpy_window_open(title: str = WINDOW_TITLE) -> bool:
    if find_hwnd_by_title(title):
        return True
    return bool(gw.getWindowsWithTitle(title))


def start_scrcpy(
    title: str = WINDOW_TITLE,
    serial: str = DEVICE_SERIAL,
    wait_seconds: float = 20.0,
) -> subprocess.Popen:
    """
    Launch scrcpy if needed. Returns the process (or a dummy-ended one if
    the window was already open).
    """
    if scrcpy_window_open(title):
        print(f'scrcpy already open ("{title}")')
        return subprocess.Popen(["cmd", "/c", "exit", "0"])  # no-op handle

    if not SCRCPY_PATH.exists():
        raise FileNotFoundError(f"scrcpy not found: {SCRCPY_PATH}")

    cmd = [
        str(SCRCPY_PATH),
        "-s",
        serial,
        "--window-title",
        title,
        "--stay-awake",
    ]
    print("Starting scrcpy...")
    print(" ", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(SCRCPY_PATH.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if scrcpy_window_open(title):
            time.sleep(0.4)  # let first frame settle
            print(f'scrcpy ready ("{title}")')
            return proc
        if proc.poll() is not None:
            raise RuntimeError("scrcpy exited early — check USB / adb / phone unlock")
        time.sleep(0.25)

    raise TimeoutError(f'scrcpy window "{title}" did not appear in {wait_seconds}s')


def get_scrcpy_rect(title: str = WINDOW_TITLE) -> WindowRect:
    hwnd = find_hwnd_by_title(title)
    if not hwnd:
        matches = gw.getWindowsWithTitle(title)
        if not matches:
            raise RuntimeError(f'scrcpy window "{title}" not found')
        # Fallback: outer bounds if FindWindow missed
        w = matches[0]
        return WindowRect(
            left=w.left,
            top=w.top,
            width=max(1, w.width),
            height=max(1, w.height),
            hwnd=0,
        )
    return get_client_rect(hwnd)


def grab_window_bgr(hwnd: int, width: int, height: int) -> np.ndarray:
    """
    Capture a window's client pixels via PrintWindow (ignores our overlay on top).
    Returns BGR uint8 array shaped (height, width, 3).
    """
    if not hwnd or width <= 0 or height <= 0:
        raise RuntimeError("invalid hwnd/size for grab_window_bgr")

    hwnd_dc = user32.GetDC(hwnd)
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
    old = gdi32.SelectObject(mem_dc, bmp)

    # PW_CLIENTONLY | PW_RENDERFULLCONTENT
    PW_CLIENTONLY = 0x1
    PW_RENDERFULLCONTENT = 0x2
    ok = user32.PrintWindow(hwnd, mem_dc, PW_CLIENTONLY | PW_RENDERFULLCONTENT)
    if not ok:
        # Fallback BitBlt from window DC
        gdi32.BitBlt(mem_dc, 0, 0, width, height, hwnd_dc, 0, 0, 0x00CC0020)

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    bmi.bmiHeader.biHeight = -height  # top-down
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0  # BI_RGB

    buf_len = width * height * 4
    buf = (ctypes.c_char * buf_len)()
    bits = gdi32.GetDIBits(mem_dc, bmp, 0, height, buf, ctypes.byref(bmi), 0)
    if bits == 0:
        gdi32.SelectObject(mem_dc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)
        raise RuntimeError("GetDIBits failed")

    bgra = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
    bgr = bgra[:, :, :3].copy()

    gdi32.SelectObject(mem_dc, old)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, hwnd_dc)
    return bgr
