"""
Clash of Clans attack bot entrypoint.

Flow:
  1. Swipe all 4 corners
  2. Tap Attack! (home)
  3. Tap Single Player
  4. Tap Attack! again
  5. Detect buildings on the battle map
  6. Drop troops on short lines just outside those detections

Usage:
  python bot.py

Optional templates (better than ratio fallbacks):
  templates/attack_button.png
  templates/single_player.png
  templates/attack_start.png

If Single Player / Attack start lands wrong, tell me the printed device=(x,y)
from a manual overlay click and we lock the ratios.
"""

from __future__ import annotations

from actions import run_attack_flow
from adb_control import AdbDevice


def main() -> None:
    adb = AdbDevice()
    if not adb.is_connected():
        print(adb.connection_help())
        return
    try:
        run_attack_flow(adb)
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
