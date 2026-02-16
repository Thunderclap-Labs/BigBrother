"""
process_monitor.py — Detect distracting apps and browser tabs.

Polls active window titles (via ctypes on Windows) and running process
names (via psutil) to catch social-media browsing, streaming, and
gaming without needing a camera.

Detected sources
────────────────
  Web  — Instagram, TikTok, YouTube Shorts, Reels, Netflix, Twitch,
          Reddit, Twitter/X, Facebook, Snapchat, Hinge, Tinder, Roblox
  Apps — Steam, Epic Games Launcher, Discord, Minecraft, Roblox,
          numerous named game executables
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)

# ── Distraction tables ────────────────────────────────────────────────────────

# Browser window-title substrings → human-readable label
# Keys are lowercased; matching is case-insensitive.
DISTRACTING_WEB: Dict[str, str] = {
    "instagram":         "Instagram",
    "tiktok":            "TikTok",
    "youtube shorts":    "YouTube Shorts",
    "reels":             "Instagram Reels",
    "netflix":           "Netflix",
    "twitch":            "Twitch",
    "reddit":            "Reddit",
    "twitter":           "Twitter/X",
    " x.com":            "Twitter/X",
    "facebook":          "Facebook",
    "snapchat":          "Snapchat",
    "hinge":             "Hinge",
    "tinder":            "Tinder",
    "feeld":             "Feeld",
    "bumble":            "Bumble",
    "discord":           "Discord (web)",
    "roblox":            "Roblox (web)",
    "primevideo":        "Prime Video",
    "prime video":       "Prime Video",
    "disneyplus":        "Disney+",
    "disney+":           "Disney+",
    "hbo max":           "HBO Max",
    "max.com":           "HBO Max",
    "youtube - youtube": "YouTube",   # catch "... - YouTube" tab titles
}

# Browser process names (so we only match web titles against browser windows)
BROWSER_PROCESSES = {
    "chrome.exe", "firefox.exe", "msedge.exe", "opera.exe",
    "brave.exe", "vivaldi.exe", "iexplore.exe", "safari.exe",
}

# Process executable names → human-readable label (case-insensitive)
DISTRACTING_APPS: Dict[str, str] = {
    "steam.exe":                   "Steam",
    "steamwebhelper.exe":          "Steam",
    "epicgameslauncher.exe":       "Epic Games",
    "discord.exe":                 "Discord",
    "discordptb.exe":              "Discord PTB",
    "discordcanary.exe":           "Discord Canary",
    "minecraft.exe":               "Minecraft",
    "minecraftlauncher.exe":       "Minecraft Launcher",
    "javaw.exe":                   "Minecraft (Java)",   # common Minecraft process
    "robloxplayerbeta.exe":        "Roblox",
    "robloxplayerlauncher.exe":    "Roblox Launcher",
    "leagueclient.exe":            "League of Legends",
    "leagueclientux.exe":          "League of Legends",
    "league of legends.exe":       "League of Legends",
    "valorant.exe":                "Valorant",
    "valorant-win64-shipping.exe": "Valorant",
    "fortnite.exe":                "Fortnite",
    "fortniteclient-win64-shipping.exe": "Fortnite",
    "csgo.exe":                    "CS:GO",
    "cs2.exe":                     "CS2",
    "gta5.exe":                    "GTA V",
    "gtavcockpit.exe":             "GTA V",
    "destiny2.exe":                "Destiny 2",
    "overwatch.exe":               "Overwatch",
    "overwatch2.exe":              "Overwatch 2",
    "apexlegends.exe":             "Apex Legends",
    "geforce experience.exe":      "GeForce Experience",
    "origin.exe":                  "Origin/EA",
    "eadesktop.exe":               "EA Desktop",
    "twitch.exe":                  "Twitch App",
    "spotify.exe":                 "Spotify",            # optional — remove if you want music
    "netflix.exe":                 "Netflix (app)",
}

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DistractionResult:
    detected:        bool  = False
    label:           str   = ""           # e.g. "Instagram", "Steam"
    source:          str   = ""           # "web" | "app"
    raw_title:       str   = ""           # window title that triggered the match
    timestamp:       float = field(default_factory=time.time)
    detection_start: float = 0.0

    @property
    def detection_duration(self) -> float:
        if self.detected and self.detection_start:
            return time.time() - self.detection_start
        return 0.0


# ── Windows helpers ───────────────────────────────────────────────────────────

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    _user32 = ctypes.windll.user32
    _user32.GetWindowTextW.restype  = ctypes.c_int
    _user32.GetWindowTextW.argtypes = [
        ctypes.wintypes.HWND,
        ctypes.wintypes.LPWSTR,
        ctypes.c_int,
    ]
    _user32.IsWindowVisible.restype = ctypes.wintypes.BOOL
    _EnumWindowsProc = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL,
        ctypes.wintypes.HWND,
        ctypes.wintypes.LPARAM,
    )


def _get_all_window_titles() -> List[str]:
    """Return titles of all currently visible top-level windows."""
    if not _IS_WINDOWS:
        return []
    titles: List[str] = []

    def _cb(hwnd, _param):
        if _user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(512)
            _user32.GetWindowTextW(hwnd, buf, 512)
            if buf.value:
                titles.append(buf.value)
        return True

    _user32.EnumWindows(_EnumWindowsProc(_cb), 0)
    return titles


def _running_process_names() -> List[str]:
    """Return lowercase exe names of every running process."""
    names: List[str] = []
    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info["name"] or "").lower()
            if name:
                names.append(name)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return names


# ── Main class ───────────────────────────────────────────────────────────────

class ProcessMonitor:
    """
    Background poller that detects distracting apps and browser tabs.

    Usage::

        monitor = ProcessMonitor(enabled=True, poll_interval=2.0)
        monitor.start()
        ...
        result = monitor.get_latest()   # DistractionResult
        monitor.stop()
    """

    def __init__(
        self,
        enabled: bool = True,
        poll_interval: float = 2.0,
        check_web: bool = True,
        check_apps: bool = True,
        extra_web_keywords: Optional[List[str]] = None,
        extra_app_names: Optional[List[str]] = None,
    ):
        self.enabled        = enabled
        self.poll_interval  = poll_interval
        self.check_web      = check_web
        self.check_apps     = check_apps

        # Build mutable copies of the tables so callers can extend them
        self._web_patterns: Dict[str, str] = dict(DISTRACTING_WEB)
        if extra_web_keywords:
            for kw in extra_web_keywords:
                self._web_patterns[kw.lower()] = kw

        self._app_patterns: Dict[str, str] = dict(DISTRACTING_APPS)
        if extra_app_names:
            for name in extra_app_names:
                self._app_patterns[name.lower()] = name

        self._result: DistractionResult = DistractionResult()
        self._lock   = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        logger.info(
            "ProcessMonitor ready — enabled=%s  poll=%.1fs  web=%s  apps=%s",
            enabled, poll_interval, check_web, check_apps,
        )

    # ── Public interface ──────────────────────────────────────────────

    def start(self):
        if not self.enabled:
            logger.info("ProcessMonitor disabled — skipping start")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="process-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("ProcessMonitor started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("ProcessMonitor stopped")

    def get_latest(self) -> DistractionResult:
        with self._lock:
            return self._result

    @property
    def is_distracted(self) -> bool:
        return self.get_latest().detected

    def update_settings(
        self,
        enabled: Optional[bool] = None,
        poll_interval: Optional[float] = None,
        check_web: Optional[bool] = None,
        check_apps: Optional[bool] = None,
    ):
        if enabled is not None:
            self.enabled = enabled
        if poll_interval is not None:
            self.poll_interval = poll_interval
        if check_web is not None:
            self.check_web = check_web
        if check_apps is not None:
            self.check_apps = check_apps

    # ── Background loop ───────────────────────────────────────────────

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                result = self._poll()
                with self._lock:
                    if result.detected and not self._result.detected:
                        # rising edge: record start time
                        result.detection_start = time.time()
                    elif result.detected and self._result.detected:
                        # keep the original start time
                        result.detection_start = self._result.detection_start
                    self._result = result
            except Exception as exc:
                logger.debug("ProcessMonitor poll error: %s", exc)
            self._stop_event.wait(self.poll_interval)

    def _poll(self) -> DistractionResult:
        """One scan pass — returns a fresh DistractionResult."""

        # ── 1. Browser-tab check via window titles ─────────────────
        if self.check_web:
            titles = _get_all_window_titles()
            for title in titles:
                title_lower = title.lower()
                for pattern, label in self._web_patterns.items():
                    if pattern in title_lower:
                        logger.debug("Web distraction: '%s' matched '%s'", title, pattern)
                        return DistractionResult(
                            detected=True,
                            label=label,
                            source="web",
                            raw_title=title,
                            timestamp=time.time(),
                        )

        # ── 2. Process name check ──────────────────────────────────
        if self.check_apps:
            names = _running_process_names()
            for proc_name in names:
                if proc_name in self._app_patterns:
                    label = self._app_patterns[proc_name]
                    logger.debug("App distraction: process '%s' → '%s'", proc_name, label)
                    return DistractionResult(
                        detected=True,
                        label=label,
                        source="app",
                        raw_title=proc_name,
                        timestamp=time.time(),
                    )

        return DistractionResult(detected=False, timestamp=time.time())
