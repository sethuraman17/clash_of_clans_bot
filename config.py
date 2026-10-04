"""Shared settings for Clash of Clans automation."""

from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
SCRCPY_DIR = Path(r"C:\Users\sethu\Downloads\scrcpy-win64-v4.1")
ADB_PATH = SCRCPY_DIR / "adb.exe"
SCRCPY_PATH = SCRCPY_DIR / "scrcpy.exe"

# Device (from `adb devices`)
DEVICE_SERIAL = "10BF4X1YZ2001GF"

# scrcpy window title — usually the device model (I2407)
WINDOW_TITLE = "I2407"

# Native portrait size from `adb shell wm size`
DEVICE_WIDTH = 1080
DEVICE_HEIGHT = 2392
