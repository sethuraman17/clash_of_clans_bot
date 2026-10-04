"""ADB helpers: taps, swipes, keys, and device screenshots."""

from __future__ import annotations

import subprocess
from typing import Optional, Tuple

import cv2
import numpy as np

from config import ADB_PATH, DEVICE_SERIAL


class AdbDevice:
    def __init__(self, serial: str = DEVICE_SERIAL, adb_path: str = str(ADB_PATH)):
        self.serial = serial
        self.adb_path = adb_path

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        cmd = [self.adb_path, "-s", self.serial, *args]
        return subprocess.run(
            cmd,
            capture_output=True,
            check=check,
        )

    def shell(self, *args: str) -> str:
        result = self._run("shell", *args)
        return result.stdout.decode("utf-8", errors="ignore").strip()

    def get_state(self) -> str:
        """Return adb state: device | unauthorized | offline | none."""
        # Prefer `adb devices` so we can see unauthorized clearly
        result = subprocess.run(
            [self.adb_path, "devices"],
            capture_output=True,
            check=False,
        )
        text = result.stdout.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            if self.serial in line:
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
        result = self._run("get-state", check=False)
        out = result.stdout.decode("utf-8", errors="ignore").strip()
        return out or "none"

    def is_connected(self) -> bool:
        return self.get_state() == "device"

    def connection_help(self) -> str:
        state = self.get_state()
        if state == "device":
            return "ADB connected."
        if state == "unauthorized":
            return (
                "ADB unauthorized: unlock phone and tap Allow USB debugging, "
                "then run: adb kill-server && adb devices"
            )
        if state == "offline":
            return "ADB offline: unplug/replug USB, unlock phone, retry."
        return f"ADB not connected (state={state}). Enable USB debugging and plug in phone."

    def get_size(self) -> Tuple[int, int]:
        """
        Size used for input / vision.

        Prefer screencap dimensions (matches game orientation). Fall back to wm size.
        """
        frame = self.screencap()
        if frame is not None:
            h, w = frame.shape[:2]
            return w, h

        out = self.shell("wm", "size")
        line = out.splitlines()[-1]
        wh = line.split(":")[-1].strip()
        w, h = wh.split("x")
        return int(w), int(h)

    def tap(self, x: int, y: int) -> None:
        self.shell("input", "tap", str(int(x)), str(int(y)))

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> None:
        self.shell(
            "input",
            "swipe",
            str(int(x1)),
            str(int(y1)),
            str(int(x2)),
            str(int(y2)),
            str(int(duration_ms)),
        )

    def keyevent(self, keycode: str | int) -> None:
        self.shell("input", "keyevent", str(keycode))

    def screencap(self) -> Optional[np.ndarray]:
        """Capture device screen at native resolution (BGR)."""
        cmd = [
            self.adb_path,
            "-s",
            self.serial,
            "exec-out",
            "screencap",
            "-p",
        ]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0 or not result.stdout:
            return None
        arr = np.frombuffer(result.stdout, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame


if __name__ == "__main__":
    adb = AdbDevice()
    print("Connected:", adb.is_connected())
    print("Size:", adb.get_size())
