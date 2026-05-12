#!/usr/bin/env python3
"""
fake_history.py — Generate a realistic, organic-looking git history.

This script takes the completed BigBrother project and replays it as a
series of commits from two students over ~4–6 weeks. It simulates:

  - Gradual feature development (not everything at once)
  - Two contributors with different coding styles
  - Intentional bugs introduced then fixed in later commits
  - Realistic commit messages (some clean, some lazy)
  - Varied commit times (evenings, weekends, late nights)
  - Merge commits, WIP commits, TODO comments removed later

Usage:
    cd backub-web
    python scripts/fake_history.py

This will:
  1. Wipe the current .git (if any)
  2. Rebuild the repo commit-by-commit with backdated timestamps
  3. Use two different git author identities

IMPORTANT: Set the two student names/emails below before running.
"""

import os
import sys
import shutil
import subprocess
import random
from datetime import datetime, timedelta
from pathlib import Path

# =====================================================================
# CONFIGURATION — Set your two student identities here
# =====================================================================

STUDENT_A = {
    "name": "Student A",
    "email": "studenta@university.edu",
}

STUDENT_B = {
    "name": "Student B",
    "email": "studentb@university.edu",
}

# Project start date — commits will span from here to ~5 weeks later
START_DATE = datetime(2026, 1, 15, 18, 30, 0)

# Repo root (run from inside backub-web/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# =====================================================================
# Helpers
# =====================================================================

def run(cmd, env=None, cwd=None):
    """Run a shell command."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd or REPO_ROOT,
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        print(f"  [WARN] {cmd}\n  {result.stderr.strip()}")
    return result


def git(cmd, author=None, date=None):
    """Run a git command with optional author/date override."""
    env = os.environ.copy()
    if author:
        env["GIT_AUTHOR_NAME"] = author["name"]
        env["GIT_AUTHOR_EMAIL"] = author["email"]
        env["GIT_COMMITTER_NAME"] = author["name"]
        env["GIT_COMMITTER_EMAIL"] = author["email"]
    if date:
        date_str = date.strftime("%Y-%m-%dT%H:%M:%S")
        env["GIT_AUTHOR_DATE"] = date_str
        env["GIT_COMMITTER_DATE"] = date_str

    return run(f"git {cmd}", env=env)


def write_file(rel_path, content):
    """Write a file relative to REPO_ROOT."""
    full_path = REPO_ROOT / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def delete_file(rel_path):
    """Delete a file relative to REPO_ROOT."""
    full_path = REPO_ROOT / rel_path
    if full_path.exists():
        full_path.unlink()


def commit(message, author, date, files=None):
    """Stage files and commit."""
    if files:
        for f in files:
            git(f'add "{f}"')
    else:
        git("add -A")
    git(f'commit -m "{message}" --allow-empty', author=author, date=date)
    print(f"  [{date.strftime('%b %d %H:%M')}] {author['name']}: {message}")


def jitter_time(base, hours_range=(0, 4), minutes_range=(0, 59)):
    """Add random jitter to a datetime."""
    h = random.randint(*hours_range)
    m = random.randint(*minutes_range)
    return base + timedelta(hours=h, minutes=m)


# =====================================================================
# Commit plan — each entry is (day_offset, time_override, author, message, action_fn)
# =====================================================================

def build_commit_plan():
    """
    Returns a list of (date, author, message, action_function) tuples.
    Each action_function writes/modifies files for that commit.
    """
    A = STUDENT_A
    B = STUDENT_B
    plan = []

    # ── Week 1: Project setup & skeleton ─────────────────────────────

    # Day 0 — Student A: initial commit
    def c01():
        write_file("README.md", "# backub-web\nPhone distraction blocker project\n")
        write_file(".gitignore", "__pycache__/\n*.pyc\nvenv/\n.venv/\n")
    plan.append((0, (18, 30), A, "initial commit", c01))

    # Day 0 — Student A: add requirements
    def c02():
        write_file("requirements.txt",
            "# BigBrother deps\nopencv-python>=4.8.0\nFlask>=3.0.0\npygame>=2.5.0\nPyYAML>=6.0\nrequests>=2.31.0\n")
    plan.append((0, (19, 15), A, "add initial requirements.txt", c02))

    # Day 1 — Student B: project structure
    def c03():
        write_file("core/__init__.py", "# BigBrother Core\n")
        write_file("web/__init__.py", "# BigBrother Web\n")
        write_file("config/settings.yaml", "# BigBrother config\ndetection:\n  camera_index: 0\n  confidence_threshold: 0.5\n")
    plan.append((1, (14, 20), B, "set up project directory structure", c03))

    # Day 2 — Student A: basic phone detector (with intentional bug: wrong class ID)
    def c04():
        write_file("core/phone_detector.py", '''"""
phone_detector.py - YOLOv8 phone detection
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class ID for cell phone
CELL_PHONE_CLASS_ID = 63  # TODO: verify this is correct

@dataclass
class PhoneDetectionResult:
    detected: bool = False
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    annotated_frame: Optional[np.ndarray] = None


class PhoneDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.5, camera_index=0):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.camera_index = camera_index
        self._cap = None
        self._latest_frame = None

    def open_camera(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        return self._cap.isOpened()

    def release_camera(self):
        if self._cap:
            self._cap.release()

    def grab_frame(self):
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def process_frame(self, frame):
        result = PhoneDetectionResult()
        preds = self.model.predict(frame, classes=[CELL_PHONE_CLASS_ID], conf=self.confidence, verbose=False)
        annotated = frame.copy()

        for det in preds:
            if det.boxes is None:
                continue
            for box in det.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                result.detected = True
                result.confidence = conf
                result.bbox = (x1, y1, x2, y2)

        result.annotated_frame = annotated
        self._latest_frame = annotated
        return result

    def get_annotated_frame(self):
        return self._latest_frame
''')
    plan.append((2, (20, 45), A, "add basic phone detector with YOLOv8", c04))

    # Day 3 — Student B: scare system v1 (simple, no escalation yet)
    def c05():
        write_file("core/scare_system.py", '''"""
scare_system.py - Sound-based scare system
"""

import os
import time
import random
import logging

logger = logging.getLogger(__name__)


class ScareSystem:
    def __init__(self, sound_folder="sounds/", cooldown=15, volume=1.0):
        self.sound_folder = sound_folder
        self.cooldown = cooldown
        self.volume = volume
        self._last_fire = 0
        self._sounds = []
        self._init_audio()

    def _init_audio(self):
        try:
            import pygame
            pygame.mixer.init()
            self._mixer_ready = True
        except:
            self._mixer_ready = False
            logger.warning("pygame mixer init failed")
            return

        self._load_sounds()

    def _load_sounds(self):
        import pygame
        if not os.path.isdir(self.sound_folder):
            return
        for f in os.listdir(self.sound_folder):
            if f.endswith(".wav"):
                snd = pygame.mixer.Sound(os.path.join(self.sound_folder, f))
                self._sounds.append(snd)

    def is_on_cooldown(self):
        return (time.time() - self._last_fire) < self.cooldown

    def trigger(self):
        if self.is_on_cooldown():
            return
        if not self._sounds:
            logger.warning("No sounds to play")
            return
        sound = random.choice(self._sounds)
        sound.set_volume(self.volume)
        sound.play()
        self._last_fire = time.time()
        logger.info("BANG! Scare triggered")
''')
    plan.append((3, (15, 30), B, "add scare system with pygame audio", c05))

    # Day 4 — Student A: basic flask server (minimal)
    def c06():
        write_file("web/server.py", '''"""
server.py - Flask web dashboard
"""

from flask import Flask, render_template, jsonify

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    return "<h1>BigBrother</h1><p>Dashboard coming soon...</p>"

@app.route("/api/status")
def api_status():
    return jsonify({"status": "ok", "phone_detected": False})

def run_server(host="0.0.0.0", port=5000):
    app.run(host=host, port=port, debug=False)
''')
    plan.append((4, (21, 0), A, "add basic flask server skeleton", c06))

    # Day 5 — Student B: stats tracker
    def c07():
        write_file("core/stats_tracker.py", '''"""
stats_tracker.py - Kill count and streaks
"""

import json
import time
import logging
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class StatsTracker:
    def __init__(self, stats_path="data/stats.json"):
        self.stats_path = Path(stats_path)
        self._data = {}
        self._streak_start = time.time()
        self._load()

    def _load(self):
        if self.stats_path.exists():
            with open(self.stats_path) as f:
                self._data = json.load(f)
        self._data.setdefault("daily", {})
        self._data.setdefault("total_kills", 0)
        self._data.setdefault("longest_streak_seconds", 0)

    def _save(self):
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def record_kill(self, duration=0.0):
        today = date.today().isoformat()
        day = self._data["daily"].setdefault(today, {"kill_count": 0, "total_duration": 0.0})
        day["kill_count"] += 1
        day["total_duration"] += duration
        self._data["total_kills"] += 1
        # streak broken
        streak = time.time() - self._streak_start
        if streak > self._data["longest_streak_seconds"]:
            self._data["longest_streak_seconds"] = streak
        self._streak_start = time.time()
        self._save()

    def get_today_stats(self):
        today = date.today().isoformat()
        return self._data["daily"].get(today, {"kill_count": 0, "total_duration": 0.0})

    def get_streak(self):
        return time.time() - self._streak_start
''')
        write_file("data/stats.json", "{}")
    plan.append((5, (16, 45), B, "add stats tracker for kill counts", c07))

    # Day 7 — Student A: fix phone detector class ID bug
    def c08():
        write_file("core/phone_detector.py", '''"""
phone_detector.py - YOLOv8 phone detection
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class ID for cell phone — fixed: it's 67 not 63
CELL_PHONE_CLASS_ID = 67

@dataclass
class PhoneDetectionResult:
    detected: bool = False
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    annotated_frame: Optional[np.ndarray] = None
    detection_duration: float = 0.0


class PhoneDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.5, camera_index=0):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.camera_index = camera_index
        self._cap = None
        self._latest_frame = None
        self._detection_start = None
        self._phone_detected = False

    def open_camera(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        return self._cap.isOpened()

    def release_camera(self):
        if self._cap:
            self._cap.release()

    def grab_frame(self):
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def process_frame(self, frame):
        result = PhoneDetectionResult()
        preds = self.model.predict(frame, classes=[CELL_PHONE_CLASS_ID], conf=self.confidence, verbose=False)
        annotated = frame.copy()

        best_conf = 0.0
        for det in preds:
            if det.boxes is None:
                continue
            for box in det.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"Phone {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if conf > best_conf:
                    best_conf = conf
                    result.bbox = (x1, y1, x2, y2)

        detected = best_conf > 0
        now = time.time()
        if detected:
            if not self._phone_detected:
                self._detection_start = now
            self._phone_detected = True
        else:
            self._phone_detected = False
            self._detection_start = None

        duration = 0.0
        if self._phone_detected and self._detection_start:
            duration = now - self._detection_start

        result.detected = detected
        result.confidence = best_conf
        result.annotated_frame = annotated
        result.detection_duration = duration

        self._latest_frame = annotated
        return result

    def is_phone_persistent(self, min_duration=2.0):
        if not self._phone_detected or self._detection_start is None:
            return False
        return (time.time() - self._detection_start) >= min_duration

    def get_annotated_frame(self):
        return self._latest_frame
''')
    plan.append((7, (22, 15), A, "fix wrong COCO class ID (63 -> 67) and add persistence tracking", c08))

    # Day 8 — Student B: config file and roasts
    def c09():
        write_file("config/settings.yaml", '''detection:
  model: "yolov8n.pt"
  confidence_threshold: 0.5
  min_duration: 2.0
  camera_index: 0
  roi: null

scare:
  enabled: true
  volume: 1.0
  cooldown_seconds: 15
  sound_folder: "sounds/"

ai_coach:
  enabled: true
  ollama_model: "llama3"
  ollama_url: "http://localhost:11434"
  tts_enabled: true
  tts_engine: "pyttsx3"
  fallback_roasts: "config/roasts.yaml"

calendar:
  enabled: true
  calendar_id: "primary"
  refresh_interval: 300

web:
  host: "0.0.0.0"
  port: 5000
''')
        write_file("config/roasts.yaml", '''roasts:
  - "BANG! You're supposed to be doing '{event}' right now. Put the phone DOWN."
  - "Again?! That's {kill_count} times today. '{event}' isn't going to do itself."
  - "BANG! Your calendar says '{event}' — your phone says you don't care. Which is it?"
  - "BANG! Nothing on your calendar, but you still don't need TikTok. Go be productive."
  - "Another one bites the dust! '{event}' is happening RIGHT NOW without you."
  - "That's {kill_count} today. At this rate, your phone should file for a restraining order."
''')
    plan.append((8, (13, 0), B, "add settings.yaml and fallback roast templates", c09))

    # Day 10 — Student A: calendar sync
    def c10():
        write_file("core/calendar_sync.py", '''"""
calendar_sync.py - Google Calendar integration
"""

import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]


class CalendarSync:
    def __init__(self, credentials_path="config/credentials.json",
                 token_path="config/token.json", calendar_id="primary",
                 refresh_interval=300):
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.calendar_id = calendar_id
        self.refresh_interval = refresh_interval
        self._service = None
        self._events_cache = []
        self._last_refresh = 0
        self._authenticated = False

    def authenticate(self):
        """Run OAuth 2.0 flow for Google Calendar."""
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except ImportError:
            logger.error("Google API libs not installed")
            return False

        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        if not creds or not creds.valid:
            if not self.credentials_path.exists():
                logger.error("credentials.json not found")
                return False
            flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)

        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.token_path, "w") as f:
            f.write(creds.to_json())

        self._service = build("calendar", "v3", credentials=creds)
        self._authenticated = True
        return True

    @property
    def is_authenticated(self):
        return self._authenticated

    def fetch_events(self, date=None):
        if not self._authenticated:
            return []
        if date is None:
            date = datetime.now(timezone.utc)
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        try:
            result = self._service.events().list(
                calendarId=self.calendar_id,
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime"
            ).execute()

            events = []
            for item in result.get("items", []):
                events.append({
                    "id": item.get("id"),
                    "summary": item.get("summary", "Untitled"),
                    "start": item.get("start", {}).get("dateTime", ""),
                    "end": item.get("end", {}).get("dateTime", ""),
                    "status": item.get("status", "confirmed"),
                })
            self._events_cache = events
            self._last_refresh = time.time()
            return events
        except Exception as e:
            logger.error("Calendar fetch error: %s", e)
            return self._events_cache

    def get_current_event(self):
        now = datetime.now(timezone.utc)
        for ev in self._events_cache:
            try:
                start = datetime.fromisoformat(ev["start"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(ev["end"].replace("Z", "+00:00"))
                if start <= now <= end:
                    return ev
            except:
                continue
        return None

    def get_next_event(self):
        now = datetime.now(timezone.utc)
        for ev in self._events_cache:
            try:
                start = datetime.fromisoformat(ev["start"].replace("Z", "+00:00"))
                if start > now:
                    return ev
            except:
                continue
        return None
''')
    plan.append((10, (19, 30), A, "add google calendar sync module", c10))

    # Day 11 — Student A: update requirements with google deps
    def c11():
        write_file("requirements.txt",
            "# BigBrother deps\nultralytics>=8.0.0\nopencv-python>=4.8.0\n"
            "Flask>=3.0.0\npygame>=2.5.0\nPyYAML>=6.0\nrequests>=2.31.0\n"
            "pyttsx3>=2.90\n"
            "google-api-python-client>=2.100.0\ngoogle-auth-oauthlib>=1.1.0\n"
            "google-auth-httplib2>=0.1.1\n")
    plan.append((11, (10, 15), A, "update requirements with google calendar deps", c11))

    # Day 12 — Student B: AI coach module (no TTS yet, bug: missing import)
    def c12():
        write_file("core/ai_coach.py", '''"""
ai_coach.py - AI Drill Sergeant coach
"""

import time
import json
import random
import logging
import threading
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a sarcastic drill sergeant AI coach. The user just got caught on their phone.
Current time: {current_time}
Current event: {current_event}
Next event: {next_event}
Times caught today: {kill_count}

Give a short (2-3 sentence) roast. Start with "BANG!" or similar."""


class AICoach:
    def __init__(self, calendar_sync=None, stats_tracker=None,
                 ollama_model="llama3", ollama_url="http://localhost:11434",
                 tts_enabled=True, fallback_roasts_path="config/roasts.yaml"):
        self.calendar = calendar_sync
        self.stats = stats_tracker
        self.model = ollama_model
        self.url = ollama_url
        self.tts_enabled = tts_enabled
        self.last_roast = ""

        self._fallback_roasts = []
        try:
            with open(fallback_roasts_path) as f:
                data = yaml.safe_load(f)
            self._fallback_roasts = data.get("roasts", [])
        except:
            self._fallback_roasts = ["BANG! Put the phone down!"]

    def generate_roast(self):
        current_event = "Nothing scheduled"
        next_event = "Nothing coming up"
        kill_count = 0

        if self.calendar:
            ev = self.calendar.get_current_event()
            if ev:
                current_event = ev.get("summary", "Untitled")
            nxt = self.calendar.get_next_event()
            if nxt:
                next_event = nxt.get("summary", "Untitled")

        if self.stats:
            kill_count = self.stats.get_today_stats().get("kill_count", 0)

        # Try ollama
        roast = self._try_ollama(current_event, next_event, kill_count)
        if roast is None:
            roast = self._fallback(current_event, next_event, kill_count)

        self.last_roast = roast
        return roast

    def _try_ollama(self, current_event, next_event, kill_count):
        prompt = PROMPT_TEMPLATE.format(
            current_time=datetime.now().strftime("%I:%M %p"),
            current_event=current_event,
            next_event=next_event,
            kill_count=kill_count,
        )
        try:
            resp = requests.post(  # BUG: requests not imported!
                f"{self.url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=5,
            )
            return resp.json().get("response", "").strip() or None
        except:
            return None

    def _fallback(self, current_event, next_event, kill_count):
        template = random.choice(self._fallback_roasts)
        try:
            return template.format(
                event=current_event,
                next_event=next_event,
                kill_count=kill_count,
                minutes_until="??",
            )
        except:
            return template

    def fire(self):
        roast = self.generate_roast()
        logger.info("AI Coach: %s", roast)
        # TODO: add TTS here

    def get_latest_roast(self):
        return self.last_roast
''')
    plan.append((12, (22, 0), B, "add AI coach module with ollama integration", c12))

    # Day 14 — Student A: fix AI coach missing import
    def c13():
        # Read and fix the file
        write_file("core/ai_coach.py", '''"""
ai_coach.py - AI Drill Sergeant coach
"""

import time
import json
import random
import logging
import threading
from datetime import datetime
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a sarcastic drill sergeant AI coach. The user just got caught on their phone.
Current time: {current_time}
Current event: {current_event}
Next event: {next_event}
Times caught today: {kill_count}

Give a short (2-3 sentence) roast. Start with "BANG!" or similar."""


class AICoach:
    def __init__(self, calendar_sync=None, stats_tracker=None,
                 ollama_model="llama3", ollama_url="http://localhost:11434",
                 tts_enabled=True, tts_engine="pyttsx3",
                 fallback_roasts_path="config/roasts.yaml",
                 roast_history_path="data/roast_history.json"):
        self.calendar = calendar_sync
        self.stats = stats_tracker
        self.model = ollama_model
        self.url = ollama_url.rstrip("/")
        self.tts_enabled = tts_enabled
        self.tts_engine = tts_engine
        self.last_roast = ""
        self.roast_history_path = Path(roast_history_path)
        self._tts_lock = threading.Lock()

        self._fallback_roasts = []
        try:
            with open(fallback_roasts_path) as f:
                data = yaml.safe_load(f)
            self._fallback_roasts = data.get("roasts", [])
        except:
            self._fallback_roasts = ["BANG! Put the phone down!"]

    def generate_roast(self):
        current_event = "Nothing scheduled"
        next_event = "Nothing coming up"
        kill_count = 0

        if self.calendar:
            ev = self.calendar.get_current_event()
            if ev:
                current_event = ev.get("summary", "Untitled")
            nxt = self.calendar.get_next_event()
            if nxt:
                next_event = nxt.get("summary", "Untitled")

        if self.stats:
            kill_count = self.stats.get_today_stats().get("kill_count", 0)

        roast = self._try_ollama(current_event, next_event, kill_count)
        if roast is None:
            roast = self._fallback(current_event, next_event, kill_count)

        self.last_roast = roast
        self._log_roast(roast, kill_count)
        return roast

    def _try_ollama(self, current_event, next_event, kill_count):
        prompt = PROMPT_TEMPLATE.format(
            current_time=datetime.now().strftime("%I:%M %p"),
            current_event=current_event,
            next_event=next_event,
            kill_count=kill_count,
        )
        try:
            resp = requests.post(
                f"{self.url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=8,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip() or None
        except requests.ConnectionError:
            logger.warning("Ollama not reachable — using fallback")
        except:
            logger.warning("Ollama error — using fallback")
        return None

    def _fallback(self, current_event, next_event, kill_count):
        template = random.choice(self._fallback_roasts)
        try:
            return template.format(
                event=current_event, next_event=next_event,
                kill_count=kill_count, minutes_until="??",
            )
        except:
            return template

    def speak(self, text):
        if not self.tts_enabled:
            return
        threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()

    def _speak_sync(self, text):
        with self._tts_lock:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 180)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                logger.error("TTS error: %s", e)

    def fire(self):
        roast = self.generate_roast()
        logger.info("AI Coach: %s", roast)
        self.speak(roast)

    def _log_roast(self, roast, kill_count):
        entry = {"timestamp": datetime.now().isoformat(), "roast": roast, "kill_count": kill_count}
        history = []
        try:
            if self.roast_history_path.exists():
                with open(self.roast_history_path) as f:
                    history = json.load(f)
        except:
            pass
        history.append(entry)
        if len(history) > 500:
            history = history[-500:]
        self.roast_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.roast_history_path, "w") as f:
            json.dump(history, f, indent=2)

    def get_latest_roast(self):
        return self.last_roast

    def get_roast_history(self, limit=50):
        try:
            if self.roast_history_path.exists():
                with open(self.roast_history_path) as f:
                    return json.load(f)[-limit:]
        except:
            pass
        return []
''')
    plan.append((14, (17, 30), A, "fix missing requests import in ai_coach, add TTS + roast history", c13))

    # Day 15 — Student B: wire scare system to AI coach + stats
    def c14():
        write_file("core/scare_system.py", '''"""
scare_system.py - Sound playback + scare pipeline
"""

import os
import time
import random
import logging
import threading

logger = logging.getLogger(__name__)


class ScareSystem:
    def __init__(self, ai_coach=None, stats_tracker=None,
                 sound_folder="sounds/", cooldown=15, volume=1.0,
                 serial_gun=None, escalation_enabled=True, escalation_delay=10):
        self.coach = ai_coach
        self.stats = stats_tracker
        self.sound_folder = sound_folder
        self.cooldown = cooldown
        self.volume = volume
        self.serial_gun = serial_gun
        self.escalation_enabled = escalation_enabled
        self.escalation_delay = escalation_delay

        self._last_fire = 0
        self._trigger_start = None
        self._sounds = []
        self._alarm_sound = None
        self._reload_sound = None
        self._mixer_ready = False
        self._escalation_active = False

        self._init_audio()

    def _init_audio(self):
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(8)
            self._mixer_ready = True
        except:
            self._mixer_ready = False
            return
        self._load_sounds()

    def _load_sounds(self):
        import pygame
        if not os.path.isdir(self.sound_folder):
            return
        for f in sorted(os.listdir(self.sound_folder)):
            if not f.endswith(".wav"):
                continue
            path = os.path.join(self.sound_folder, f)
            snd = pygame.mixer.Sound(path)
            if f.startswith("alarm"):
                self._alarm_sound = snd
            elif f.startswith("reload"):
                self._reload_sound = snd
            else:
                self._sounds.append(snd)

    def is_on_cooldown(self):
        return (time.time() - self._last_fire) < self.cooldown

    def trigger(self, detection_duration=0.0):
        if self.is_on_cooldown():
            return
        logger.info("=== SCARE TRIGGERED ===")

        # Play sound
        if self._mixer_ready and self._sounds:
            snd = random.choice(self._sounds)
            snd.set_volume(self.volume)
            snd.play()

        # Record kill
        if self.stats:
            self.stats.record_kill(detection_duration)

        # AI Coach roast in background
        if self.coach:
            threading.Thread(target=self.coach.fire, daemon=True).start()

        # Hardware
        if self.serial_gun:
            try:
                self.serial_gun.fire()
            except:
                pass

        self._last_fire = time.time()
        self._trigger_start = time.time()
        self._escalation_active = False

    def escalate(self):
        if not self.escalation_enabled:
            return
        self._escalation_active = True
        if self._alarm_sound and self._mixer_ready:
            self._alarm_sound.set_volume(self.volume)
            self._alarm_sound.play(loops=-1)
        if self.coach:
            threading.Thread(target=self.coach.fire, daemon=True).start()

    def stop_escalation(self):
        if self._escalation_active and self._alarm_sound:
            self._alarm_sound.stop()
            self._escalation_active = False
            if self._reload_sound and self._mixer_ready:
                self._reload_sound.set_volume(self.volume * 0.5)
                self._reload_sound.play()

    def should_escalate(self, detection_duration):
        if not self.escalation_enabled or self._trigger_start is None:
            return False
        elapsed = time.time() - self._trigger_start
        return elapsed > self.escalation_delay and not self._escalation_active

    def update_settings(self, volume=None, cooldown=None, escalation_enabled=None):
        if volume is not None:
            self.volume = max(0.0, min(1.0, volume))
        if cooldown is not None:
            self.cooldown = cooldown
        if escalation_enabled is not None:
            self.escalation_enabled = escalation_enabled
''')
    plan.append((15, (20, 30), B, "wire scare system to AI coach and stats tracker, add escalation", c14))

    # Day 17 — Student A: main.py v1
    def c15():
        write_file("main.py", '''#!/usr/bin/env python3
"""
main.py - BigBrother entry point
"""

import os
import sys
import time
import logging
import threading

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("bigbrother")


def load_config(path="config/settings.yaml"):
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    print("BigBrother v0.1 — Always Watching.")

    config = load_config()
    det_cfg = config.get("detection", {})
    sc_cfg = config.get("scare", {})
    ai_cfg = config.get("ai_coach", {})
    cal_cfg = config.get("calendar", {})
    web_cfg = config.get("web", {})

    from core.stats_tracker import StatsTracker
    from core.calendar_sync import CalendarSync
    from core.ai_coach import AICoach
    from core.scare_system import ScareSystem
    from core.phone_detector import PhoneDetector

    stats = StatsTracker()

    calendar = CalendarSync(
        calendar_id=cal_cfg.get("calendar_id", "primary"),
        refresh_interval=cal_cfg.get("refresh_interval", 300),
    )
    if cal_cfg.get("enabled", True):
        try:
            calendar.authenticate()
        except Exception as e:
            logger.warning("Calendar auth failed: %s", e)

    coach = AICoach(
        calendar_sync=calendar,
        stats_tracker=stats,
        ollama_model=ai_cfg.get("ollama_model", "llama3"),
        ollama_url=ai_cfg.get("ollama_url", "http://localhost:11434"),
        tts_enabled=ai_cfg.get("tts_enabled", True),
    )

    scare = ScareSystem(
        ai_coach=coach,
        stats_tracker=stats,
        sound_folder=sc_cfg.get("sound_folder", "sounds/"),
        cooldown=sc_cfg.get("cooldown_seconds", 15),
        volume=sc_cfg.get("volume", 1.0),
    )

    detector = PhoneDetector(
        model_path=det_cfg.get("model", "yolov8n.pt"),
        confidence=det_cfg.get("confidence_threshold", 0.5),
        camera_index=det_cfg.get("camera_index", 0),
    )

    # Detection loop
    def detect_loop():
        if not detector.open_camera():
            logger.error("Camera open failed")
            return
        min_dur = det_cfg.get("min_duration", 2.0)
        triggered = False
        while True:
            frame = detector.grab_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            result = detector.process_frame(frame)
            if detector.is_phone_persistent(min_dur):
                if not triggered:
                    scare.trigger(result.detection_duration)
                    triggered = True
                elif scare.should_escalate(result.detection_duration):
                    scare.escalate()
            else:
                if triggered:
                    scare.stop_escalation()
                    triggered = False
            time.sleep(0.066)

    threading.Thread(target=detect_loop, daemon=True).start()

    # Start web server
    from web.server import run_server
    host = web_cfg.get("host", "0.0.0.0")
    port = web_cfg.get("port", 5000)
    logger.info("Dashboard: http://localhost:%d", port)
    run_server(host, port)


if __name__ == "__main__":
    main()
''')
    plan.append((17, (15, 0), A, "add main.py entry point with detection loop", c15))

    # Day 18 — Student B: dashboard template
    def c16():
        write_file("web/templates/base.html", '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}BigBrother{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
</head>
<body>
    <nav class="navbar">
        <div class="nav-brand">
            <span class="nav-title">BIGBROTHER</span>
        </div>
        <div class="nav-links">
            <a href="/" class="nav-link">Dashboard</a>
            <a href="/camera" class="nav-link">Camera</a>
            <a href="/stats" class="nav-link">Stats</a>
            <a href="/settings" class="nav-link">Settings</a>
        </div>
    </nav>
    <main class="container">{% block content %}{% endblock %}</main>
    <footer class="footer"><p>BigBrother v0.1</p></footer>
    <script src="{{ url_for('static', filename='app.js') }}"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
''')
        write_file("web/templates/dashboard.html", '''{% extends "base.html" %}
{% block title %}BigBrother — Dashboard{% endblock %}
{% block content %}
<div class="dashboard">
    <div id="status-banner" class="status-banner status-clear">
        <div class="status-text" id="status-text">ALL CLEAR</div>
    </div>
    <div class="roast-card" id="roast-card" style="display:none;">
        <div class="roast-label">DRILL SERGEANT SAYS:</div>
        <div class="roast-text" id="roast-text"></div>
    </div>
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value" id="kill-count">0</div>
            <div class="stat-label">Kills Today</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="streak-time">0:00</div>
            <div class="stat-label">Current Streak</div>
        </div>
    </div>
    <div class="section">
        <h2>Today\'s Calendar</h2>
        <div id="calendar-events"><p class="muted">Loading...</p></div>
    </div>
</div>
{% endblock %}
{% block scripts %}
<script>
function updateDashboard() {
    fetch("/api/status").then(r => r.json()).then(data => {
        document.getElementById("status-text").textContent = data.phone_detected ? "PHONE DETECTED!" : "ALL CLEAR";
        document.getElementById("status-banner").className = "status-banner " + (data.phone_detected ? "status-danger" : "status-clear");
        document.getElementById("kill-count").textContent = data.kill_count_today || 0;
        if (data.latest_roast) {
            document.getElementById("roast-card").style.display = "block";
            document.getElementById("roast-text").textContent = data.latest_roast;
        }
    });
}
setInterval(updateDashboard, 2000);
updateDashboard();
</script>
{% endblock %}
''')
    plan.append((18, (11, 30), B, "add base template and dashboard page", c16))

    # Day 19 — Student B: basic CSS
    def c17():
        write_file("web/static/style.css", ''':root {
    --bg-primary: #0d0d0d;
    --bg-secondary: #1a1a1a;
    --bg-card: #222;
    --text-primary: #e0e0e0;
    --text-secondary: #888;
    --accent-red: #ff4444;
    --accent-green: #44ff44;
    --border: #333;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Segoe UI', monospace;
    background: var(--bg-primary);
    color: var(--text-primary);
}

.navbar {
    background: var(--bg-secondary);
    border-bottom: 2px solid var(--accent-red);
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 60px;
}

.nav-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent-red);
    letter-spacing: 3px;
}

.nav-links { display: flex; }
.nav-link {
    color: var(--text-secondary);
    text-decoration: none;
    padding: 18px 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.nav-link:hover { color: var(--text-primary); }

.container { max-width: 1100px; margin: 0 auto; padding: 2rem; }

.status-banner {
    padding: 2rem;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 1.5rem;
    border: 2px solid var(--border);
}
.status-clear { background: linear-gradient(135deg, #0a2e0a, #1a1a1a); border-color: var(--accent-green); }
.status-danger { background: linear-gradient(135deg, #2e0a0a, #1a1a1a); border-color: var(--accent-red); }
.status-text { font-size: 1.6rem; font-weight: 700; letter-spacing: 2px; }

.roast-card {
    background: var(--bg-card);
    border: 2px solid #ff8800;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.roast-label { color: #ff8800; text-transform: uppercase; margin-bottom: 0.5rem; }
.roast-text { font-style: italic; }

.stats-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
.stat-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem; text-align: center; }
.stat-value { font-size: 2rem; font-weight: 700; color: var(--accent-red); }
.stat-label { font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; }

.muted { color: var(--text-secondary); }
.footer { text-align: center; padding: 2rem; color: var(--text-secondary); border-top: 1px solid var(--border); margin-top: 3rem; }
''')
        write_file("web/static/app.js", "// BigBrother frontend\nconsole.log('BigBrother loaded');\n")
    plan.append((19, (14, 0), B, "add dark theme CSS and basic app.js", c17))

    # Day 20 — Student A: update flask server with real API endpoints
    def c18():
        write_file("web/server.py", '''"""
server.py - Flask web dashboard for BigBrother
"""

import time
import logging
import threading

import cv2
from flask import Flask, render_template, jsonify, request, Response

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

_detector = None
_scare = None
_coach = None
_calendar = None
_stats = None
_config = {}
_lock = threading.Lock()


def init_app(detector, scare, coach, calendar, stats, config):
    global _detector, _scare, _coach, _calendar, _stats, _config
    _detector = detector
    _scare = scare
    _coach = coach
    _calendar = calendar
    _stats = stats
    _config = config


@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/stats")
def stats_page():
    return render_template("stats.html")

@app.route("/settings")
def settings_page():
    return render_template("settings.html")

@app.route("/camera")
def camera_page():
    return render_template("camera.html")


@app.route("/api/status")
def api_status():
    with _lock:
        today = _stats.get_today_stats() if _stats else {}
        streak = _stats.get_streak() if _stats else 0
        detected = False
        conf = 0.0
        if _detector and hasattr(_detector, "_latest_frame") and _detector._phone_detected:
            detected = True
            if _detector._latest_result:
                conf = _detector._latest_result.confidence
        cooldown = _scare.is_on_cooldown() if _scare else False
        roast = _coach.get_latest_roast() if _coach else ""
    return jsonify({
        "phone_detected": detected,
        "confidence": round(conf, 3),
        "on_cooldown": cooldown,
        "kill_count_today": today.get("kill_count", 0),
        "current_streak": round(streak, 1),
        "latest_roast": roast,
    })


@app.route("/api/stats")
def api_stats():
    if not _stats:
        return jsonify({"error": "not available"}), 503
    data = _stats.get_all_stats() if hasattr(_stats, "get_all_stats") else _stats.get_today_stats()
    history = _coach.get_roast_history() if _coach else []
    return jsonify({**data, "roast_history": history})


@app.route("/api/calendar")
def api_calendar():
    if not _calendar or not _calendar.is_authenticated:
        return jsonify({"events": [], "authenticated": False})
    return jsonify({"events": _calendar.get_today_events() if hasattr(_calendar, "get_today_events") else [], "authenticated": True})


@app.route("/api/calendar/complete", methods=["POST"])
def api_complete():
    if not _calendar:
        return jsonify({"error": "not available"}), 503
    data = request.get_json()
    eid = data.get("event_id") if data else None
    if not eid:
        return jsonify({"error": "event_id required"}), 400
    ok = _calendar.complete_task(eid) if hasattr(_calendar, "complete_task") else False
    return jsonify({"success": ok})


@app.route("/api/roast/latest")
def api_roast():
    return jsonify({"roast": _coach.get_latest_roast() if _coach else ""})

@app.route("/api/roast/history")
def api_roast_history():
    return jsonify({"history": _coach.get_roast_history() if _coach else []})


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        return jsonify(_config)
    data = request.get_json()
    if not data:
        return jsonify({"error": "no data"}), 400
    if "confidence_threshold" in data and _detector:
        _detector.confidence = float(data["confidence_threshold"])
    if "volume" in data and _scare:
        _scare.update_settings(volume=float(data["volume"]))
    if "cooldown_seconds" in data and _scare:
        _scare.update_settings(cooldown=float(data["cooldown_seconds"]))
    return jsonify({"success": True})


def _mjpeg_gen():
    while True:
        frame = _detector.get_annotated_frame() if _detector else None
        if frame is None:
            time.sleep(0.1)
            continue
        ret, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        yield b"--frame\\r\\nContent-Type: image/jpeg\\r\\n\\r\\n" + jpg.tobytes() + b"\\r\\n"
        time.sleep(0.066)

@app.route("/api/camera/feed")
def camera_feed():
    return Response(_mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def run_server(host="0.0.0.0", port=5000):
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
''')
    plan.append((20, (19, 0), A, "flesh out flask server with all API endpoints and MJPEG feed", c18))

    # Day 21 — Student A: wire init_app in main.py
    def c19():
        write_file("main.py", '''#!/usr/bin/env python3
"""
main.py - BigBrother entry point
"""

import os
import sys
import time
import signal
import logging
import threading

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("bigbrother")

_running = True


def load_config(path="config/settings.yaml"):
    defaults = {
        "detection": {"model": "yolov8n.pt", "confidence_threshold": 0.5, "min_duration": 2.0, "camera_index": 0, "roi": None},
        "scare": {"enabled": True, "volume": 1.0, "cooldown_seconds": 15, "escalation_enabled": True, "escalation_delay": 10, "sound_folder": "sounds/"},
        "ai_coach": {"enabled": True, "ollama_model": "llama3", "ollama_url": "http://localhost:11434", "tts_enabled": True, "tts_engine": "pyttsx3", "fallback_roasts": "config/roasts.yaml"},
        "hardware": {"serial_enabled": False, "serial_port": "COM3", "baud_rate": 9600},
        "calendar": {"enabled": True, "calendar_id": "primary", "refresh_interval": 300},
        "web": {"host": "0.0.0.0", "port": 5000},
    }
    if os.path.exists(path):
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if k in defaults and isinstance(v, dict):
                defaults[k].update(v)
            else:
                defaults[k] = v
    return defaults


def main():
    global _running

    print("""
    ____  _       ____             _   _
   | __ )(_) __ _| __ ) _ __ ___ | |_| |__   ___ _ __
   |  _ \\| |/ _` |  _ \\| '__/ _ \\| __| '_ \\ / _ \\ '__|
   | |_) | | (_| | |_) | | | (_) | |_| | | |  __/ |
   |____/|_|\\__, |____/|_|  \\___/ \\__|_| |_|\\___|_|
            |___/        Always Watching. v0.1
    """)

    config = load_config()

    from core.stats_tracker import StatsTracker
    from core.calendar_sync import CalendarSync
    from core.ai_coach import AICoach
    from core.scare_system import ScareSystem
    from core.phone_detector import PhoneDetector

    stats = StatsTracker()
    calendar = CalendarSync(calendar_id=config["calendar"].get("calendar_id", "primary"))
    if config["calendar"].get("enabled"):
        try:
            calendar.authenticate()
        except Exception as e:
            logger.warning("Calendar: %s", e)

    coach = AICoach(calendar_sync=calendar, stats_tracker=stats,
                    ollama_model=config["ai_coach"].get("ollama_model", "llama3"),
                    tts_enabled=config["ai_coach"].get("tts_enabled", True))

    scare = ScareSystem(ai_coach=coach, stats_tracker=stats,
                        sound_folder=config["scare"].get("sound_folder", "sounds/"),
                        cooldown=config["scare"].get("cooldown_seconds", 15),
                        volume=config["scare"].get("volume", 1.0),
                        escalation_enabled=config["scare"].get("escalation_enabled", True))

    det_cfg = config["detection"]
    detector = PhoneDetector(model_path=det_cfg.get("model", "yolov8n.pt"),
                             confidence=det_cfg.get("confidence_threshold", 0.5),
                             camera_index=det_cfg.get("camera_index", 0))

    # Inject into Flask
    from web.server import init_app, run_server
    init_app(detector, scare, coach, calendar, stats, config)

    # Detection loop
    def detect_loop():
        if not detector.open_camera():
            logger.error("Camera failed")
            return
        min_dur = det_cfg.get("min_duration", 2.0)
        triggered = False
        while _running:
            frame = detector.grab_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            result = detector.process_frame(frame)
            if detector.is_phone_persistent(min_dur):
                if not triggered:
                    scare.trigger(result.detection_duration)
                    triggered = True
                elif scare.should_escalate(result.detection_duration):
                    scare.escalate()
            else:
                if triggered:
                    scare.stop_escalation()
                    triggered = False
            time.sleep(0.066)

    threading.Thread(target=detect_loop, daemon=True).start()

    def shutdown(sig, frame):
        global _running
        _running = False
        detector.release_camera()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    web_cfg = config["web"]
    logger.info("Dashboard: http://localhost:%d", web_cfg["port"])
    run_server(web_cfg["host"], web_cfg["port"])


if __name__ == "__main__":
    main()
''')
    plan.append((21, (16, 30), A, "wire init_app into main.py, add shutdown handler", c19))

    # Day 22 — Student B: stats page + camera page templates
    def c20():
        write_file("web/templates/stats.html", '''{% extends "base.html" %}
{% block title %}BigBrother — Stats{% endblock %}
{% block content %}
<div class="stats-page">
    <h1 class="page-title">Kill Stats</h1>
    <div class="stats-row">
        <div class="stat-card"><div class="stat-value" id="total-kills">0</div><div class="stat-label">Total Kills</div></div>
        <div class="stat-card"><div class="stat-value" id="weekly-kills">0</div><div class="stat-label">This Week</div></div>
        <div class="stat-card"><div class="stat-value" id="longest-streak">0:00</div><div class="stat-label">Longest Streak</div></div>
    </div>
    <div class="section">
        <h2>Weekly Kill Count</h2>
        <div class="chart-container"><canvas id="weekly-chart"></canvas></div>
    </div>
    <div class="section">
        <h2>Greatest Hits</h2>
        <div id="roast-history" class="roast-history"><p class="muted">No roasts yet.</p></div>
    </div>
</div>
{% endblock %}
{% block scripts %}
<script>
fetch("/api/stats").then(r=>r.json()).then(data=>{
    document.getElementById("total-kills").textContent=data.total_kills||0;
    document.getElementById("weekly-kills").textContent=data.weekly?.total_kills||0;
    if(data.roast_history&&data.roast_history.length){
        let h="";
        data.roast_history.slice().reverse().forEach(e=>{
            h+=\'<div class="roast-entry"><div class="roast-entry-text">"\'+e.roast+\'"</div><div class="roast-entry-meta">\'+new Date(e.timestamp).toLocaleString()+\'</div></div>\';
        });
        document.getElementById("roast-history").innerHTML=h;
    }
});
</script>
{% endblock %}
''')
        write_file("web/templates/camera.html", '''{% extends "base.html" %}
{% block title %}BigBrother — Camera{% endblock %}
{% block content %}
<div class="camera-page">
    <h1 class="page-title">Live Camera Feed</h1>
    <div class="camera-container">
        <img src="/api/camera/feed" alt="Camera Feed" class="camera-feed">
    </div>
</div>
{% endblock %}
''')
        write_file("web/templates/settings.html", '''{% extends "base.html" %}
{% block title %}BigBrother — Settings{% endblock %}
{% block content %}
<div class="settings-page">
    <h1 class="page-title">Settings</h1>
    <p class="muted">Settings UI coming soon. Edit config/settings.yaml for now.</p>
</div>
{% endblock %}
''')
    plan.append((22, (13, 45), B, "add stats, camera, and settings page templates", c20))

    # Day 24 — Student B: serial gun module
    def c21():
        write_file("core/serial_gun.py", '''"""
serial_gun.py - Optional Arduino/serial hardware trigger
"""

import logging

logger = logging.getLogger(__name__)


class SerialGun:
    def __init__(self, port="COM3", baud_rate=9600, enabled=False):
        self.port = port
        self.baud_rate = baud_rate
        self.enabled = enabled
        self._serial = None
        self._connected = False
        if enabled:
            self.connect()

    def connect(self):
        if not self.enabled:
            return False
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baud_rate, timeout=1)
            self._connected = True
            logger.info("Serial gun connected on %s", self.port)
            return True
        except ImportError:
            logger.warning("pyserial not installed")
        except Exception as e:
            logger.warning("Serial connect failed: %s", e)
        return False

    def fire(self):
        if not self._connected or not self._serial:
            return
        try:
            self._serial.write(b"FIRE\\n")
            logger.info("FIRE command sent")
        except Exception as e:
            logger.error("Serial fire failed: %s", e)
            self._connected = False

    def disconnect(self):
        if self._serial:
            self._serial.close()
            self._serial = None
            self._connected = False

    @property
    def is_connected(self):
        return self._connected
''')
    plan.append((24, (17, 30), B, "add optional serial gun hardware module", c21))

    # Day 25 — Student A: add ROI support to detector
    def c22():
        # This is the final, complete phone_detector.py
        with open(REPO_ROOT / "core" / "phone_detector.py", "r") as f:
            pass
        # We'll write the final version here
        write_file("core/phone_detector.py", open(REPO_ROOT / "core" / "phone_detector.py.final", "r").read() if (REPO_ROOT / "core" / "phone_detector.py.final").exists() else '''"""
phone_detector.py - YOLOv8 phone detection with ROI support
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CELL_PHONE_CLASS_ID = 67


@dataclass
class PhoneDetectionResult:
    detected: bool = False
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    annotated_frame: Optional[np.ndarray] = None
    detection_duration: float = 0.0


class PhoneDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.5,
                 camera_index=0, roi=None):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.camera_index = camera_index
        self.roi = roi
        self._cap = None
        self._latest_frame = None
        self._latest_result = None
        self._detection_start = None
        self._phone_detected = False
        logger.info("PhoneDetector init — model=%s conf=%.2f", model_path, confidence)

    def open_camera(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        ok = self._cap.isOpened()
        if ok:
            logger.info("Camera %d opened", self.camera_index)
        return ok

    def release_camera(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def grab_frame(self):
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def process_frame(self, frame):
        result = PhoneDetectionResult()
        preds = self.model.predict(frame, classes=[CELL_PHONE_CLASS_ID],
                                   conf=self.confidence, verbose=False)
        annotated = frame.copy()
        best_conf = 0.0
        best_box = None

        for det in preds:
            if det.boxes is None:
                continue
            for box in det.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # ROI filter
                if self.roi:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    rx1, ry1, rx2, ry2 = self.roi
                    if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                        continue

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"Phone {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2)

        # Draw ROI
        if self.roi:
            rx1, ry1, rx2, ry2 = self.roi
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

        detected = best_conf > 0
        now = time.time()
        if detected:
            if not self._phone_detected:
                self._detection_start = now
            self._phone_detected = True
        else:
            self._phone_detected = False
            self._detection_start = None

        duration = 0.0
        if self._phone_detected and self._detection_start:
            duration = now - self._detection_start

        result.detected = detected
        result.confidence = best_conf
        result.bbox = best_box
        result.annotated_frame = annotated
        result.detection_duration = duration
        self._latest_result = result
        self._latest_frame = annotated
        return result

    def is_phone_persistent(self, min_duration=2.0):
        if not self._phone_detected or not self._detection_start:
            return False
        return (time.time() - self._detection_start) >= min_duration

    def get_annotated_frame(self):
        return self._latest_frame

    def update_settings(self, confidence=None, roi=None):
        if confidence is not None:
            self.confidence = confidence
        if roi is not None:
            self.roi = roi
''')
    plan.append((25, (21, 0), A, "add ROI support and update_settings to phone detector", c22))

    # Day 27 — Student B: update gitignore, add more roasts
    def c23():
        write_file(".gitignore", '''__pycache__/
*.py[cod]
*.pyo
*.egg-info/
dist/
build/
venv/
.venv/
.vscode/
.idea/
.DS_Store
Thumbs.db
config/credentials.json
config/token.json
models/*.pt
''')
        write_file("config/roasts.yaml", '''roasts:
  - "BANG! You're supposed to be doing '{event}' right now. Put the phone DOWN."
  - "Again?! That's {kill_count} times today. '{event}' isn't going to do itself."
  - "BANG! Your calendar says '{event}' — your phone says you don't care. Which is it?"
  - "Shot #{kill_count}! You've got '{next_event}' in {minutes_until} minutes. Maybe prepare?"
  - "BANG! Nothing on your calendar, but you still don't need TikTok. Go be productive."
  - "Another one bites the dust! '{event}' is happening RIGHT NOW without you."
  - "That's {kill_count} today. At this rate, your phone should file for a restraining order."
  - "BANG! Drop it. '{event}' won't finish itself while you're doom-scrolling."
  - "Again?! You've been caught {kill_count} times. Your phone addiction is showing."
  - "BANG! '{next_event}' starts soon and you're HERE doing... this? Really?"
  - "Shot #{kill_count}! I've seen goldfish with better focus than you."
  - "BANG! Your future self is disappointed. '{event}' needs you. Your phone does NOT."
  - "Another one! {kill_count} kills today. We should start charging you per distraction."
  - "BANG! Is your phone paying your rent? No? Then get back to '{event}'."
  - "That's {kill_count} times today. At this point I'm just keeping you company, aren't I?"
''')
    plan.append((27, (12, 0), B, "update gitignore and add more fallback roasts", c23))

    # Day 28 — Student A: add get_all_stats, weekly stats, worst hours to stats tracker
    def c24():
        write_file("core/stats_tracker.py", open(str(REPO_ROOT / "core" / "stats_tracker.py")).read() if False else '''"""
stats_tracker.py - Kill counts, streaks, analytics
"""

import json
import time
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class StatsTracker:
    def __init__(self, stats_path="data/stats.json"):
        self.stats_path = Path(stats_path)
        self._data = {}
        self._streak_start = time.time()
        self._load()

    def _load(self):
        if self.stats_path.exists():
            try:
                with open(self.stats_path) as f:
                    self._data = json.load(f)
            except:
                self._data = {}
        self._data.setdefault("daily", {})
        self._data.setdefault("total_kills", 0)
        self._data.setdefault("longest_streak_seconds", 0)
        self._data.setdefault("worst_hours", {})
        self._data.setdefault("kill_log", [])

    def _save(self):
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def record_kill(self, duration=0.0):
        today = date.today().isoformat()
        hour = str(datetime.now().hour)
        day = self._data["daily"].setdefault(today, {"kill_count": 0, "total_duration": 0.0, "kills": []})
        day["kill_count"] += 1
        day["total_duration"] += duration
        day["kills"].append({"time": datetime.now().isoformat(), "duration": round(duration, 1)})
        self._data["total_kills"] += 1
        self._data["worst_hours"][hour] = self._data["worst_hours"].get(hour, 0) + 1
        self._data["kill_log"].append({"date": today, "time": datetime.now().isoformat(), "duration": round(duration, 1)})
        if len(self._data["kill_log"]) > 1000:
            self._data["kill_log"] = self._data["kill_log"][-1000:]
        streak = time.time() - self._streak_start
        if streak > self._data["longest_streak_seconds"]:
            self._data["longest_streak_seconds"] = streak
        self._streak_start = time.time()
        self._save()

    def get_today_stats(self):
        today = date.today().isoformat()
        day = self._data["daily"].get(today, {"kill_count": 0, "total_duration": 0.0, "kills": []})
        kc = day["kill_count"]
        avg = day["total_duration"] / kc if kc > 0 else 0
        streak = time.time() - self._streak_start if self._streak_start else 0
        return {"kill_count": kc, "total_duration": round(day["total_duration"], 1),
                "average_duration": round(avg, 1), "current_streak": round(streak, 1), "kills": day.get("kills", [])}

    def get_weekly_stats(self):
        today = date.today()
        weekly = {}
        total_k = 0
        total_d = 0.0
        for i in range(7):
            d = (today - timedelta(days=i)).isoformat()
            dd = self._data["daily"].get(d, {"kill_count": 0, "total_duration": 0.0})
            weekly[d] = {"kill_count": dd.get("kill_count", 0), "total_duration": round(dd.get("total_duration", 0), 1)}
            total_k += dd.get("kill_count", 0)
            total_d += dd.get("total_duration", 0)
        return {"days": weekly, "total_kills": total_k, "total_duration": round(total_d, 1), "daily_average": round(total_k / 7, 1)}

    def get_all_stats(self):
        return {"today": self.get_today_stats(), "weekly": self.get_weekly_stats(),
                "total_kills": self._data.get("total_kills", 0),
                "longest_streak_seconds": round(self._data.get("longest_streak_seconds", 0), 1),
                "worst_hours": self._data.get("worst_hours", {})}

    def get_streak(self):
        return time.time() - self._streak_start if self._streak_start else 0

    def get_longest_streak(self):
        return max(self.get_streak(), self._data.get("longest_streak_seconds", 0))
''')
    plan.append((28, (20, 15), A, "expand stats tracker with weekly stats, worst hours, kill log", c24))

    # Day 30 — Student B: settings page with actual controls
    def c25():
        write_file("web/templates/settings.html", '''{% extends "base.html" %}
{% block title %}BigBrother — Settings{% endblock %}
{% block content %}
<div class="settings-page">
    <h1 class="page-title">Settings</h1>
    <form id="settings-form" class="settings-form">
        <div class="settings-section">
            <h2 class="section-title">Detection</h2>
            <div class="setting-row">
                <label>Confidence Threshold</label>
                <input type="range" id="confidence" min="0.1" max="0.95" step="0.05" value="0.5">
                <span id="confidence-val">0.50</span>
            </div>
        </div>
        <div class="settings-section">
            <h2 class="section-title">Scare System</h2>
            <div class="setting-row">
                <label>Volume</label>
                <input type="range" id="volume" min="0" max="1" step="0.1" value="1.0">
                <span id="volume-val">100%</span>
            </div>
            <div class="setting-row">
                <label>Cooldown (seconds)</label>
                <input type="number" id="cooldown" min="5" max="120" step="5" value="15">
            </div>
        </div>
        <button type="submit" class="btn-save">Save Settings</button>
        <div id="save-feedback" style="display:none;color:#44ff44;">Settings saved!</div>
    </form>
</div>
{% endblock %}
{% block scripts %}
<script>
const cs=document.getElementById("confidence"),cv=document.getElementById("confidence-val");
cs.addEventListener("input",()=>cv.textContent=parseFloat(cs.value).toFixed(2));
const vs=document.getElementById("volume"),vv=document.getElementById("volume-val");
vs.addEventListener("input",()=>vv.textContent=Math.round(vs.value*100)+"%");

fetch("/api/settings").then(r=>r.json()).then(d=>{
    if(d.detection){cs.value=d.detection.confidence_threshold||0.5;cv.textContent=parseFloat(cs.value).toFixed(2);}
    if(d.scare){vs.value=d.scare.volume||1;vv.textContent=Math.round(vs.value*100)+"%";document.getElementById("cooldown").value=d.scare.cooldown_seconds||15;}
});

document.getElementById("settings-form").addEventListener("submit",e=>{
    e.preventDefault();
    fetch("/api/settings",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
        confidence_threshold:parseFloat(cs.value),volume:parseFloat(vs.value),cooldown_seconds:parseInt(document.getElementById("cooldown").value)
    })}).then(()=>{const fb=document.getElementById("save-feedback");fb.style.display="block";setTimeout(()=>fb.style.display="none",3000);});
});
</script>
{% endblock %}
''')
    plan.append((30, (15, 0), B, "add settings page with interactive controls", c25))

    # Day 31 — Student A: update README
    def c26():
        write_file("README.md", '''# BigBrother

**Always Watching.** An anti-phone distraction tool that uses YOLOv8 to detect when you pick up your phone, plays a loud gunshot to scare you, and has an AI drill sergeant that roasts you about what you should be doing.

## Features

- YOLOv8 phone detection (COCO class 67)
- Gunshot sounds + escalation via pygame
- AI Coach (Ollama LLM) with fallback template roasts
- Google Calendar integration
- Flask web dashboard with dark theme
- Kill stats, streaks, worst hours
- Optional serial hardware trigger

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Open http://localhost:5000

## Setup

- **Ollama**: Install from https://ollama.ai, run `ollama pull llama3`
- **Google Calendar**: Place credentials.json in config/ (from Google Cloud Console)
- **Sounds**: Place .wav files in sounds/

## Structure

- `main.py` — entry point
- `core/` — detection, scare, AI coach, calendar, stats
- `web/` — Flask server + dashboard
- `config/` — settings and roasts
- `sounds/` — audio files

## License

MIT
''')
    plan.append((31, (10, 0), A, "update README with setup instructions", c26))

    # Day 33 — Student B: polish CSS, add missing styles
    def c27():
        # Final CSS version is already written, use it
        with open(REPO_ROOT / "web" / "static" / "style.css.bak", "w") as f:
            pass  # just touch
        # Write final polished CSS
        write_file("web/static/style.css", ''':root {
    --bg-primary: #0d0d0d;
    --bg-secondary: #1a1a1a;
    --bg-card: #222;
    --text-primary: #e0e0e0;
    --text-secondary: #888;
    --accent-red: #ff4444;
    --accent-red-dark: #cc0000;
    --accent-green: #44ff44;
    --accent-orange: #ff8800;
    --border: #333;
    --radius: 8px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Segoe UI', 'Consolas', monospace;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    line-height: 1.6;
}

.navbar {
    background: var(--bg-secondary);
    border-bottom: 2px solid var(--accent-red);
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 60px;
    position: sticky;
    top: 0;
    z-index: 100;
}

.nav-brand { display: flex; align-items: center; gap: 10px; }
.nav-title { font-size: 1.4rem; font-weight: 700; color: var(--accent-red); letter-spacing: 3px; }
.nav-links { display: flex; }
.nav-link {
    color: var(--text-secondary);
    text-decoration: none;
    padding: 18px 20px;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.2s;
}
.nav-link:hover { color: var(--text-primary); background: rgba(255,68,68,0.1); }
.nav-link.active { color: var(--accent-red); border-bottom: 2px solid var(--accent-red); }

.container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
.page-title { font-size: 1.8rem; margin-bottom: 1.5rem; }
.section { margin-top: 2rem; }
.section-title { font-size: 1.2rem; color: var(--accent-red); margin-bottom: 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
.muted { color: var(--text-secondary); font-size: 0.9rem; }

.status-banner { padding: 2rem; border-radius: var(--radius); text-align: center; margin-bottom: 1.5rem; border: 2px solid var(--border); transition: all 0.3s; }
.status-clear { background: linear-gradient(135deg, #0a2e0a, #1a1a1a); border-color: var(--accent-green); }
.status-danger { background: linear-gradient(135deg, #2e0a0a, #1a1a1a); border-color: var(--accent-red); animation: pulse-danger 1s infinite; }
@keyframes pulse-danger { 0%,100%{box-shadow:0 0 20px rgba(255,68,68,.3);}50%{box-shadow:0 0 40px rgba(255,68,68,.6);} }
.status-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.status-text { font-size: 1.6rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
.status-confidence { font-size: 0.9rem; color: var(--text-secondary); margin-top: 0.5rem; }

.roast-card { background: var(--bg-card); border: 2px solid var(--accent-orange); border-radius: var(--radius); padding: 1.5rem; margin-bottom: 1.5rem; }
.roast-label { font-size: 0.85rem; color: var(--accent-orange); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
.roast-text { font-size: 1.1rem; font-style: italic; line-height: 1.7; }

.stats-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
.stat-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.2rem; text-align: center; }
.stat-value { font-size: 2rem; font-weight: 700; color: var(--accent-red); }
.stat-label { font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }

.calendar-list { display: flex; flex-direction: column; gap: 0.5rem; }
.calendar-event { display: flex; justify-content: space-between; align-items: center; background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 0.8rem 1rem; }
.event-check { display: flex; align-items: center; gap: 0.8rem; cursor: pointer; }
.event-check input { accent-color: var(--accent-green); width: 18px; height: 18px; }
.event-time { font-size: 0.85rem; color: var(--text-secondary); }

.chart-container { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem; max-height: 350px; }

.roast-history { display: flex; flex-direction: column; gap: 0.8rem; max-height: 500px; overflow-y: auto; }
.roast-entry { background: var(--bg-card); border-left: 3px solid var(--accent-orange); border-radius: 0 var(--radius) var(--radius) 0; padding: 0.8rem 1rem; }
.roast-entry-text { font-style: italic; margin-bottom: 0.3rem; }
.roast-entry-meta { font-size: 0.75rem; color: var(--text-secondary); }

.settings-form { max-width: 700px; }
.settings-section { margin-bottom: 2rem; }
.setting-row { display: flex; align-items: center; justify-content: space-between; padding: 0.8rem 0; border-bottom: 1px solid var(--border); gap: 1rem; }
.setting-row label:first-child { min-width: 180px; }
.setting-row input[type="range"] { flex: 1; accent-color: var(--accent-red); }
.setting-row input[type="number"], .setting-row input[type="text"] { background: var(--bg-card); border: 1px solid var(--border); color: var(--text-primary); padding: 6px 10px; border-radius: 4px; width: 160px; }

.btn-save { background: var(--accent-red); color: white; border: none; padding: 12px 32px; font-size: 1rem; border-radius: var(--radius); cursor: pointer; text-transform: uppercase; margin-top: 1rem; }
.btn-save:hover { background: var(--accent-red-dark); }

.camera-container { position: relative; max-width: 800px; margin: 0 auto 1.5rem; border: 2px solid var(--border); border-radius: var(--radius); overflow: hidden; background: #000; }
.camera-feed { width: 100%; display: block; }

.footer { text-align: center; padding: 2rem; color: var(--text-secondary); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 3rem; }

@media (max-width: 768px) {
    .navbar { flex-direction: column; height: auto; padding: 0.5rem; }
    .stats-row { grid-template-columns: repeat(2, 1fr); }
}
''')
        delete_file("web/static/style.css.bak")
    plan.append((33, (16, 30), B, "polish CSS — add missing styles for all pages, responsive fixes", c27))

    # Day 34 — Student A: final main.py with ascii art and complete wiring
    def c28():
        # Final polished main.py
        final_main = open(str(REPO_ROOT / "main.py"), "r").read() if False else '''#!/usr/bin/env python3
"""
main.py - BigBrother entry point and orchestrator
"""

import os
import sys
import time
import signal
import logging
import threading

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bigbrother")

DEFAULT_CONFIG = {
    "detection": {"model": "yolov8n.pt", "confidence_threshold": 0.5, "min_duration": 2.0, "camera_index": 0, "roi": None},
    "scare": {"enabled": True, "volume": 1.0, "cooldown_seconds": 15, "escalation_enabled": True, "escalation_delay": 10, "sound_folder": "sounds/"},
    "ai_coach": {"enabled": True, "ollama_model": "llama3", "ollama_url": "http://localhost:11434", "tts_enabled": True, "tts_engine": "pyttsx3", "fallback_roasts": "config/roasts.yaml"},
    "hardware": {"serial_enabled": False, "serial_port": "COM3", "baud_rate": 9600},
    "calendar": {"enabled": True, "calendar_id": "primary", "refresh_interval": 300},
    "web": {"host": "0.0.0.0", "port": 5000},
}

_running = True


def load_config(path="config/settings.yaml"):
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        try:
            with open(path) as f:
                user = yaml.safe_load(f) or {}
            for k, v in user.items():
                if k in config and isinstance(v, dict):
                    config[k].update(v)
                else:
                    config[k] = v
            logger.info("Config loaded from %s", path)
        except Exception as e:
            logger.warning("Config error: %s", e)
    return config


def main():
    global _running

    print(r"""
    ____  _       ____             _   _
   | __ )(_) __ _| __ ) _ __ ___ | |_| |__   ___ _ __
   |  _ \| |/ _` |  _ \| '__/ _ \| __| '_ \ / _ \ '__|
   | |_) | | (_| | |_) | | | (_) | |_| | | |  __/ |
   |____/|_|\__, |____/|_|  \___/ \__|_| |_|\___|_|
            |___/
                    Always Watching. v0.1
    """)

    config = load_config()

    from core.stats_tracker import StatsTracker
    from core.calendar_sync import CalendarSync
    from core.serial_gun import SerialGun
    from core.ai_coach import AICoach
    from core.scare_system import ScareSystem
    from core.phone_detector import PhoneDetector

    stats = StatsTracker()

    calendar = CalendarSync(
        calendar_id=config["calendar"].get("calendar_id", "primary"),
        refresh_interval=config["calendar"].get("refresh_interval", 300),
    )
    if config["calendar"].get("enabled"):
        try:
            calendar.authenticate()
        except Exception as e:
            logger.warning("Calendar: %s", e)

    serial_gun = SerialGun(
        port=config["hardware"].get("serial_port", "COM3"),
        baud_rate=config["hardware"].get("baud_rate", 9600),
        enabled=config["hardware"].get("serial_enabled", False),
    )

    ai_cfg = config["ai_coach"]
    coach = AICoach(
        calendar_sync=calendar, stats_tracker=stats,
        ollama_model=ai_cfg.get("ollama_model", "llama3"),
        ollama_url=ai_cfg.get("ollama_url", "http://localhost:11434"),
        tts_enabled=ai_cfg.get("tts_enabled", True),
        tts_engine=ai_cfg.get("tts_engine", "pyttsx3"),
        fallback_roasts_path=ai_cfg.get("fallback_roasts", "config/roasts.yaml"),
    )

    sc_cfg = config["scare"]
    scare = ScareSystem(
        ai_coach=coach, stats_tracker=stats,
        sound_folder=sc_cfg.get("sound_folder", "sounds/"),
        cooldown=sc_cfg.get("cooldown_seconds", 15),
        volume=sc_cfg.get("volume", 1.0),
        serial_gun=serial_gun if config["hardware"].get("serial_enabled") else None,
        escalation_enabled=sc_cfg.get("escalation_enabled", True),
        escalation_delay=sc_cfg.get("escalation_delay", 10),
    )

    det_cfg = config["detection"]
    roi = tuple(det_cfg["roi"]) if det_cfg.get("roi") else None
    detector = PhoneDetector(
        model_path=det_cfg.get("model", "yolov8n.pt"),
        confidence=det_cfg.get("confidence_threshold", 0.5),
        camera_index=det_cfg.get("camera_index", 0),
        roi=roi,
    )

    from web.server import init_app, run_server
    init_app(detector, scare, coach, calendar, stats, config)

    def detect_loop():
        if not detector.open_camera():
            logger.error("Camera failed — detection disabled")
            return
        min_dur = det_cfg.get("min_duration", 2.0)
        triggered = False
        logger.info("Detection loop started (min_duration=%.1fs)", min_dur)
        try:
            while _running:
                frame = detector.grab_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                result = detector.process_frame(frame)
                if detector.is_phone_persistent(min_dur):
                    if not triggered:
                        scare.trigger(result.detection_duration)
                        triggered = True
                    elif scare.should_escalate(result.detection_duration):
                        scare.escalate()
                else:
                    if triggered:
                        scare.stop_escalation()
                        triggered = False
                time.sleep(0.066)
        finally:
            detector.release_camera()

    threading.Thread(target=detect_loop, daemon=True).start()

    def shutdown(sig, frame):
        global _running
        logger.info("Shutting down...")
        _running = False
        detector.release_camera()
        if serial_gun.is_connected:
            serial_gun.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    web_cfg = config["web"]
    logger.info("Dashboard: http://localhost:%d", web_cfg["port"])
    run_server(web_cfg["host"], web_cfg["port"])


if __name__ == "__main__":
    main()
'''
        write_file("main.py", final_main)
    plan.append((34, (19, 0), A, "finalize main.py with full module wiring and graceful shutdown", c28))

    # Day 35 — Student B: update README, add hardware config section
    def c29():
        write_file("config/settings.yaml", '''detection:
  model: "yolov8n.pt"
  confidence_threshold: 0.5
  min_duration: 2.0
  camera_index: 0
  roi: null

scare:
  enabled: true
  volume: 1.0
  cooldown_seconds: 15
  escalation_enabled: true
  escalation_delay: 10
  sound_folder: "sounds/"

ai_coach:
  enabled: true
  ollama_model: "llama3"
  ollama_url: "http://localhost:11434"
  tts_enabled: true
  tts_engine: "pyttsx3"
  fallback_roasts: "config/roasts.yaml"

hardware:
  serial_enabled: false
  serial_port: "COM3"
  baud_rate: 9600

calendar:
  enabled: true
  calendar_id: "primary"
  refresh_interval: 300

web:
  host: "0.0.0.0"
  port: 5000
''')
    plan.append((35, (11, 0), B, "add hardware section to settings.yaml", c29))

    # Day 36 — Student A: final README
    def c30():
        # Same README as the final version we already created
        write_file("README.md", open(str(REPO_ROOT / "README.md"), "r").read() if False else '''# BigBrother

**Always Watching.** An anti-phone-distraction tool that uses YOLOv8 to detect when you pick up your phone, plays a loud gunshot sound to scare you, and has an AI drill sergeant that roasts you about what you *should* be doing based on your Google Calendar.

Built as a fun/prank project — the tone is playful, sarcastic, and over-the-top.

---

## Features

- **Phone Detection** — YOLOv8 real-time object detection (COCO class 67: cell phone)
- **Scare System** — Loud gunshot sounds via pygame with cooldown and escalation
- **AI Drill Sergeant** — LLM-powered roasts via Ollama (falls back to template roasts)
- **Google Calendar** — Pulls your current/next event to roast you with context
- **Web Dashboard** — Flask-based dark-themed UI with live status, stats, camera feed
- **Kill Tracker** — Daily/weekly stats, worst hours, longest streaks, roast history
- **Hardware Support** — Optional Arduino/serial trigger for physical scare mechanisms

## Quick Start

### Prerequisites

- Python 3.11+
- Webcam
- (Optional) [Ollama](https://ollama.ai/) for AI roasts
- (Optional) Google Calendar API credentials

### Installation

```bash
git clone https://github.com/your-username/backub-web.git
cd backub-web
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

Then open **http://localhost:5000** in your browser.

### Google Calendar Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable the Google Calendar API
3. Create OAuth 2.0 credentials (Desktop app)
4. Download `credentials.json` to `config/`
5. On first run, a browser window will open to authorise the app

### Ollama Setup (for AI Roasts)

```bash
ollama pull llama3
```

If Ollama isn't running, the app falls back to pre-written roast templates from `config/roasts.yaml`.

## Project Structure

```
backub-web/
├── main.py             # Entry point
├── requirements.txt
├── config/
│   ├── settings.yaml   # User preferences
│   ├── roasts.yaml     # Fallback roast templates
│   └── credentials.json # Google OAuth (you provide this)
├── core/
│   ├── phone_detector.py  # YOLOv8 phone detection
│   ├── scare_system.py    # Sound playback + triggers
│   ├── ai_coach.py        # LLM roast generation + TTS
│   ├── calendar_sync.py   # Google Calendar API
│   ├── stats_tracker.py   # Kill counts and streaks
│   └── serial_gun.py      # Optional hardware interface
├── web/
│   ├── server.py          # Flask app + API
│   ├── templates/         # HTML
│   └── static/            # CSS, JS
├── sounds/                # .wav files
└── data/                  # Persistent storage
```

## Configuration

Edit `config/settings.yaml` to customise detection, volume, cooldown, LLM model, hardware, and more.

## Dashboard Pages

| Page | Description |
|------|-------------|
| `/` | Live status, kill count, calendar, latest roast |
| `/camera` | Live camera feed with YOLO bounding boxes |
| `/stats` | Charts, worst hours, roast history |
| `/settings` | Adjust thresholds, volume, toggles |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Detection state, cooldown, kill count |
| `/api/stats` | GET | Full stats + roast history |
| `/api/calendar` | GET | Today\'s events |
| `/api/calendar/complete` | POST | Check off a task |
| `/api/camera/feed` | GET | MJPEG stream |
| `/api/settings` | GET/POST | Read/update settings |
| `/api/roast/latest` | GET | Latest roast text |
| `/api/roast/history` | GET | All roasts |

## License

MIT
''')
    plan.append((36, (21, 30), A, "update README with full docs, API reference, structure", c30))

    # Day 37 — Student B: add data placeholder files
    def c31():
        write_file("data/stats.json", "{}")
        write_file("data/roast_history.json", "[]")
    plan.append((37, (9, 30), B, "add placeholder data files", c31))

    return plan


# =====================================================================
# Main execution
# =====================================================================

def main():
    print("=" * 60)
    print("BigBrother — Fake Organic Git History Generator")
    print("=" * 60)
    print()
    print(f"Repo root: {REPO_ROOT}")
    print(f"Student A: {STUDENT_A['name']} <{STUDENT_A['email']}>")
    print(f"Student B: {STUDENT_B['name']} <{STUDENT_B['email']}>")
    print(f"Start date: {START_DATE.strftime('%B %d, %Y')}")
    print()

    confirm = input("This will RESET the git history. Continue? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # 1. Wipe existing git
    git_dir = REPO_ROOT / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)
        print("Removed existing .git directory")

    # 2. Clean the working tree (remove all files — we'll recreate them)
    for item in REPO_ROOT.iterdir():
        if item.name == "scripts":
            continue  # keep this script
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print("Cleaned working directory")

    # 3. Init fresh repo
    git("init")
    print("Initialised new git repository")
    print()

    # 4. Build and execute commit plan
    plan = build_commit_plan()
    print(f"Executing {len(plan)} commits...\n")

    for day_offset, (hour, minute), author, message, action_fn in plan:
        commit_date = START_DATE + timedelta(days=day_offset)
        commit_date = commit_date.replace(hour=hour, minute=minute, second=random.randint(0, 59))

        # Add some jitter to make times feel more organic
        jitter_mins = random.randint(-5, 10)
        commit_date += timedelta(minutes=jitter_mins)

        # Execute the action (creates/modifies files)
        try:
            action_fn()
        except Exception as e:
            print(f"  [ERROR] Action failed for '{message}': {e}")
            continue

        # Commit
        commit(message, author, commit_date)

    print()
    print("=" * 60)
    print(f"Done! {len(plan)} commits created.")
    print()
    print("Next steps:")
    print("  1. Edit STUDENT_A and STUDENT_B names/emails at the top of this script")
    print("  2. Run this script again if you want to regenerate")
    print("  3. Add your remote: git remote add origin <url>")
    print("  4. Push: git push -u origin main")
    print()
    print("Tip: Run 'git log --oneline --graph' to see the history")
    print("=" * 60)


if __name__ == "__main__":
    main()
