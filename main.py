#!/usr/bin/env python3
"""
main.py — BigBrother entry point and orchestrator.

Loads configuration, initialises all core modules, runs the detection
loop in a background thread, and starts the Flask web dashboard.
"""

import argparse
import os
import sys
import time
import signal
import logging
import threading
import webbrowser
from pathlib import Path

import yaml

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bigbrother")


# ── Configuration ────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "detection": {
        "model": "yolov8n.pt",
        "confidence_threshold": 0.5,
        "min_duration": 2.0,
        # -1 = auto-select first available camera
        "camera_index": 0,
        "roi": None,
    },
    "camera_calibration": {
        "enabled": True,
        "width": None,
        "height": None,
        "fps": None,
        "brightness": None,
        "contrast": None,
        "exposure": None,
    },
    "scare": {
        "enabled": True,
        "volume": 1.0,
        "cooldown_seconds": 15,
        "escalation_enabled": True,
        "escalation_delay": 10,
        "sound_folder": "sounds/",
    },
    "ai_coach": {
        "enabled": True,
        "ollama_model": "llama3",
        "ollama_url": "http://localhost:11434",
        "tts_enabled": True,
        "tts_engine": "pyttsx3",
        "fallback_roasts": "config/roasts.yaml",
    },
    "hardware": {
        "serial_enabled": False,
        "serial_port": "COM3",
        "baud_rate": 9600,
        "wifi_enabled": False,
        "arduino_url": "http://192.168.1.100",
    },
    "calendar": {
        "enabled": True,
        "calendar_id": "primary",
        "refresh_interval": 300,
    },
    "web": {
        "host": "0.0.0.0",
        "port": 5000,
    },
    "process_monitor": {
        "enabled": True,
        "poll_interval": 2.0,
        "check_web": True,
        "check_apps": True,
        "min_duration": 3.0,   # seconds of continuous distraction before triggering
    },
}


def load_config(path: str = "config/settings.yaml") -> dict:
    """Load settings from YAML, merging with defaults.

    Load order (later layers win):
      1. DEFAULT_CONFIG hardcoded defaults
      2. config/settings.yaml  (tracked — shared defaults)
      3. config/local.yaml     (gitignored — personal overrides, never committed)
    """
    config = DEFAULT_CONFIG.copy()

    def _merge(base: dict, override: dict):
        for section, values in override.items():
            if section in base and isinstance(values, dict):
                base[section].update(values)
            else:
                base[section] = values

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
            _merge(config, user_config)
            logger.info("Configuration loaded from %s", path)
        except Exception as exc:
            logger.warning("Could not load config from %s: %s — using defaults", path, exc)
    else:
        logger.info("No config file found at %s — using defaults", path)

    # Personal override layer — gitignored, never committed
    local_path = "config/local.yaml"
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                local_config = yaml.safe_load(f) or {}
            _merge(config, local_config)
            logger.info("Local overrides loaded from %s", local_path)
        except Exception as exc:
            logger.warning("Could not load local config from %s: %s", local_path, exc)

    return config


# ── Module initialisation ────────────────────────────────────────────

def init_modules(config: dict):
    """Initialise all core modules and return them as a dict."""
    from core.stats_tracker import StatsTracker
    from core.calendar_sync import CalendarSync
    from core.serial_gun import SerialGun
    from core.ai_coach import AICoach
    from core.scare_system import ScareSystem
    from core.phone_detector import PhoneDetector
    from core.process_monitor import ProcessMonitor

    # Stats
    stats = StatsTracker(stats_path="data/stats.json")

    # Calendar
    cal_cfg = config["calendar"]
    calendar = CalendarSync(
        credentials_path="config/credentials.json",
        token_path="config/token.json",
        calendar_id=cal_cfg["calendar_id"],
        refresh_interval=cal_cfg["refresh_interval"],
    )
    if cal_cfg["enabled"]:
        try:
            calendar.authenticate()
        except Exception as exc:
            logger.warning("Calendar auth failed: %s — continuing without calendar", exc)

    # Serial / WiFi gun (optional)
    hw_cfg = config["hardware"]
    serial_gun = SerialGun(
        port=hw_cfg["serial_port"],
        baud_rate=hw_cfg["baud_rate"],
        enabled=hw_cfg["serial_enabled"],
        wifi_enabled=hw_cfg.get("wifi_enabled", False),
        arduino_url=hw_cfg.get("arduino_url", "http://192.168.1.100"),
    )

    # AI Coach
    ai_cfg = config["ai_coach"]
    coach = AICoach(
        calendar_sync=calendar,
        stats_tracker=stats,
        ollama_model=ai_cfg["ollama_model"],
        ollama_url=ai_cfg["ollama_url"],
        tts_enabled=ai_cfg["tts_enabled"],
        tts_engine=ai_cfg["tts_engine"],
        fallback_roasts_path=ai_cfg["fallback_roasts"],
    )

    # Scare system
    sc_cfg = config["scare"]
    scare = ScareSystem(
        ai_coach=coach,
        stats_tracker=stats,
        sound_folder=sc_cfg["sound_folder"],
        cooldown=sc_cfg["cooldown_seconds"],
        volume=sc_cfg["volume"],
        serial_gun=serial_gun if (hw_cfg["serial_enabled"] or hw_cfg.get("wifi_enabled", False)) else None,
        escalation_enabled=sc_cfg["escalation_enabled"],
        escalation_delay=sc_cfg["escalation_delay"],
    )

    # Phone detector
    det_cfg = config["detection"]
    roi = None
    if det_cfg["roi"]:
        roi = tuple(det_cfg["roi"])

    # Build calibration dict — only pass keys that are explicitly set
    cal_section = config.get("camera_calibration", {})
    calibration = None
    if cal_section.get("enabled", True):
        calibration = {k: v for k, v in cal_section.items() if k != "enabled" and v is not None}
        if calibration:
            logger.info("Camera calibration settings: %s", calibration)

    detector = PhoneDetector(
        model_path=det_cfg["model"],
        confidence=det_cfg["confidence_threshold"],
        camera_index=det_cfg["camera_index"],
        roi=roi,
        calibration=calibration,
    )

    return {
        "detector": detector,
        "scare": scare,
        "coach": coach,
        "calendar": calendar,
        "stats": stats,
        "serial_gun": serial_gun,
        "process_monitor": ProcessMonitor(
            enabled=config.get("process_monitor", {}).get("enabled", True),
            poll_interval=config.get("process_monitor", {}).get("poll_interval", 2.0),
            check_web=config.get("process_monitor", {}).get("check_web", True),
            check_apps=config.get("process_monitor", {}).get("check_apps", True),
        ),
    }


# ── Detection loop ──────────────────────────────────────────────────

_running = True


def detection_loop(detector, scare, config, process_monitor=None):
    """
    Main detection loop — runs in a background thread.

    Captures frames, runs YOLOv8 inference, and triggers the scare
    system when a phone is detected for long enough.
    Also checks ProcessMonitor for web/app distractions.
    """
    det_cfg = config["detection"]
    pm_cfg  = config.get("process_monitor", {})
    min_duration    = det_cfg["min_duration"]
    pm_min_duration = pm_cfg.get("min_duration", 3.0)

    if not detector.open_camera():
        logger.error("Failed to open camera — detection loop aborting")
        return

    logger.info("Detection loop started (min_duration=%.1fs)", min_duration)
    triggered   = False
    pm_triggered = False

    try:
        while _running:
            frame = detector.grab_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            result = detector.process_frame(frame)

            # ── Camera phone detection ──────────────────────────────────
            if detector.is_phone_persistent(min_duration):
                if not triggered:
                    scare.trigger(result.detection_duration)
                    triggered = True
                elif scare.should_escalate(result.detection_duration):
                    scare.escalate()
            else:
                if triggered:
                    scare.stop_escalation()
                    triggered = False

            # ── Process / browser-tab distraction detection ─────────────
            if process_monitor and process_monitor.enabled:
                pm_result = process_monitor.get_latest()
                if pm_result.detected and pm_result.detection_duration >= pm_min_duration:
                    if not pm_triggered:
                        logger.info(
                            "Process distraction: %s (%s) — triggering scare",
                            pm_result.label, pm_result.source,
                        )
                        scare.trigger(pm_result.detection_duration)
                        pm_triggered = True
                else:
                    if pm_triggered:
                        scare.stop_escalation()
                        pm_triggered = False

            # ~15 FPS
            time.sleep(0.066)

    except Exception as exc:
        logger.error("Detection loop error: %s", exc, exc_info=True)
    finally:
        detector.release_camera()
        logger.info("Detection loop stopped")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    global _running

    # ── Parse CLI arguments ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="BigBrother — phone-detection system")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Scan for available cameras, print results, and exit.",
    )
    parser.add_argument(
        "--select-camera",
        action="store_true",
        help="Interactively choose a camera index before starting.",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings YAML (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the dashboard in a browser on startup.",
    )
    args = parser.parse_args()

    # ── --list-cameras ───────────────────────────────────────────────────
    if args.list_cameras:
        from core.phone_detector import PhoneDetector
        print("\nScanning for available cameras...")
        cameras = PhoneDetector.list_cameras()
        if cameras:
            print(f"\nFound {len(cameras)} camera(s):")
            for cam in cameras:
                print(f"  [{cam['index']}]  {cam['width']}x{cam['height']}  @  {cam['fps']:.0f} fps")
        else:
            print("  No cameras found.")
        print()
        sys.exit(0)

    print(r"""
    ____  _       ____             _   _
   | __ )(_) __ _| __ ) _ __ ___ | |_| |__   ___ _ __
   |  _ \| |/ _` |  _ \| '__/ _ \| __| '_ \ / _ \ '__|
   | |_) | | (_| | |_) | | | (_) | |_| | | |  __/ |
   |____/|_|\__, |____/|_|  \___/ \__|_| |_|\___|_|
            |___/
                    Always Watching. v0.1
    """)

    # Load config
    config = load_config(args.config)

    # ── --select-camera / auto-select (camera_index == -1) ───────────────
    if args.select_camera or config["detection"]["camera_index"] == -1:
        from core.phone_detector import PhoneDetector
        print("\nScanning for available cameras...")
        cameras = PhoneDetector.list_cameras()
        if not cameras:
            logger.error("No cameras found. Please connect a camera and retry.")
            sys.exit(1)

        print(f"\nAvailable cameras ({len(cameras)} found):")
        for cam in cameras:
            print(f"  [{cam['index']}]  {cam['width']}x{cam['height']}  @  {cam['fps']:.0f} fps")

        if args.select_camera:
            valid_indices = [str(c["index"]) for c in cameras]
            while True:
                choice = input(f"\nSelect camera index [{'/'.join(valid_indices)}]: ").strip()
                if choice in valid_indices:
                    config["detection"]["camera_index"] = int(choice)
                    break
                print(f"  Invalid choice. Enter one of: {', '.join(valid_indices)}")
        else:
            # camera_index == -1: auto-select first available
            config["detection"]["camera_index"] = cameras[0]["index"]
            logger.info("Auto-selected camera index %d", cameras[0]["index"])

    # ── Init modules ───────────────────────────────────────────────────
    logger.info("Initialising modules...")
    modules = init_modules(config)
    detector = modules["detector"]
    scare    = modules["scare"]
    coach    = modules["coach"]
    calendar = modules["calendar"]
    stats    = modules["stats"]
    process_monitor = modules["process_monitor"]

    # Start process monitor background thread
    process_monitor.start()

    # Inject into Flask
    from web.server import init_app, run_server
    init_app(detector, scare, coach, calendar, stats, config, process_monitor)

    # Start detection loop in background thread
    det_thread = threading.Thread(
        target=detection_loop,
        args=(detector, scare, config, process_monitor),
        daemon=True,
    )
    det_thread.start()

    # Handle shutdown
    def shutdown(sig, frame):
        global _running
        logger.info("Shutting down...")
        _running = False
        process_monitor.stop()
        detector.release_camera()
        if modules["serial_gun"].is_connected:
            modules["serial_gun"].disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start web server (blocks)
    web_cfg = config["web"]
    url = f"http://localhost:{web_cfg['port']}"
    logger.info("Dashboard: %s", url)

    # Auto-open browser after a short delay so Flask has time to bind
    if not args.no_browser:
        def _open_browser():
            time.sleep(2.0)
            webbrowser.open(url)
        threading.Thread(target=_open_browser, daemon=True).start()

    run_server(host=web_cfg["host"], port=web_cfg["port"])


if __name__ == "__main__":
    main()
