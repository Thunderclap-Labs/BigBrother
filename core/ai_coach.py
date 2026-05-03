"""
ai_coach.py — The Drill Sergeant AI Coach for BigBrother.

Generates sarcastic, drill-sergeant-style roasts when the user is caught
picking up their phone. Uses a local LLM via Ollama for dynamic roasts,
falls back to pre-written templates from config/roasts.yaml.
"""

import time
import json
import random
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
import yaml

logger = logging.getLogger(__name__)

DRILL_SERGEANT_PROMPT = """You are a sarcastic, over-the-top drill sergeant AI coach in an anti-phone-distraction app called BigBrother.
The user just got caught picking up their phone. A loud gunshot sound was played to scare them.
Now deliver a short (2-3 sentences max), funny, biting roast telling them what they should be doing instead.

Context:
- Current time: {current_time}
- Day of week: {day_of_week}
- Current calendar event: {current_event}
- Next upcoming event: {next_event}
- Pending tasks this week: {pending_tasks}
- Times caught today: {kill_count}
- Time since last caught: {time_since_last}

Rules:
- Be funny and sarcastic but not mean-spirited
- If there is a current calendar event, reference it by name
- If there are pending tasks, mention one of them by name (and when it's due) to shame the user
- If NO event is currently scheduled (event says "Nothing scheduled"), make an EDUCATED GUESS about what they should be doing based on the time and day — e.g. if it's evening on a weekday guess they have assignments/studying, if it's morning guess they have work/classes, if it's late guess they should sleep. Be specific and creative with the guess.
- If they've been caught many times today, roast them harder
- Keep it under 3 sentences
- Start with a reaction to the gunshot (e.g., "BANG!", "Another one!", "Again?!")
"""

GUESS_PROMPT = """You are a brutally honest but funny productivity coach.
The user has no calendar events scheduled right now.
Based on the context below, make a SHORT (1-2 sentences) educated guess about what they SHOULD be doing right now instead of checking their phone.

Context:
- Current time: {current_time}
- Day of week: {day_of_week}
- Time of day: {time_of_day}
- Next upcoming event: {next_event}
- Times caught today: {kill_count}

Make a realistic, specific guess. Examples:
- Sunday evening → "probably have a week's worth of assignments due tomorrow"
- Monday morning → "should be preparing for the week / morning standup"
- Late night → "should be sleeping, not doomscrolling"
- Weekday afternoon → "should be working on whatever deadline is creeping up"

Be direct, funny, and specific. Just output the guess, no preamble.
"""


class AICoach:
    """
    Generates sarcastic drill-sergeant roasts about the user's phone usage.

    Tries Ollama first; falls back to template roasts from roasts.yaml.
    Speaks the roast via TTS (non-blocking) and logs to roast_history.json.
    """

    def __init__(
        self,
        calendar_sync,
        stats_tracker,
        ollama_model: str = "llama3",
        ollama_url: str = "http://localhost:11434",
        tts_enabled: bool = True,
        tts_engine: str = "pyttsx3",
        fallback_roasts_path: str = "config/roasts.yaml",
        roast_history_path: str = "data/roast_history.json",
    ):
        self.calendar = calendar_sync
        self.stats = stats_tracker
        self.model = ollama_model
        self.url = ollama_url.rstrip("/")
        self.tts_enabled = tts_enabled
        self.tts_engine = tts_engine
        self.last_roast: str = ""
        self.last_guess: str = ""
        self.roast_history_path = Path(roast_history_path)

        self._fallback_roasts: List[str] = []
        self._tts_lock = threading.Lock()
        self._last_roast_time: float = 0

        self._load_fallback_roasts(fallback_roasts_path)

    # ------------------------------------------------------------------
    # Fallback roast loading
    # ------------------------------------------------------------------

    def _load_fallback_roasts(self, path: str):
        """Load pre-written roast templates from YAML."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._fallback_roasts = data.get("roasts", [])
            logger.info("Loaded %d fallback roasts from %s", len(self._fallback_roasts), path)
        except FileNotFoundError:
            logger.warning("Fallback roasts file not found: %s", path)
            self._fallback_roasts = [
                "BANG! Put the phone down. You have things to do.",
                "Again?! That's {kill_count} times today. Get back to work.",
                "BANG! Your calendar is crying right now.",
            ]
        except Exception as exc:
            logger.error("Error loading fallback roasts: %s", exc)
            self._fallback_roasts = ["BANG! Put the phone down!"]

    # ------------------------------------------------------------------
    # Roast generation
    # ------------------------------------------------------------------

    def generate_roast(self) -> str:
        """
        Generate a sarcastic roast based on current calendar and stats.
        Tries Ollama first, falls back to template roasts.
        """
        # Gather context
        now = datetime.now()
        current_event = self._get_current_event_text()
        next_event = self._get_next_event_text()
        pending_tasks = self._get_pending_tasks_text()
        kill_count = self.stats.get_today_stats().get("kill_count", 0)
        time_since = self._format_time_since_last()
        current_time = now.strftime("%I:%M %p")
        day_of_week = now.strftime("%A")

        # Try Ollama
        roast = self._try_ollama(current_time, day_of_week, current_event, next_event, pending_tasks, kill_count, time_since)

        if roast is None:
            roast = self._fallback_roast(current_event, next_event, kill_count, pending_tasks)

        self.last_roast = roast
        self._last_roast_time = time.time()

        # Log to history
        self._log_roast(roast, current_event, next_event, kill_count)

        return roast

    def _try_ollama(
        self, current_time: str, day_of_week: str, current_event: str, next_event: str,
        pending_tasks: str, kill_count: int, time_since: str,
    ) -> Optional[str]:
        """Attempt to generate a roast via Ollama. Returns None on failure."""
        prompt = DRILL_SERGEANT_PROMPT.format(
            current_time=current_time,
            day_of_week=day_of_week,
            current_event=current_event,
            next_event=next_event,
            pending_tasks=pending_tasks,
            kill_count=kill_count,
            time_since_last=time_since,
        )

        try:
            resp = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            roast = data.get("response", "").strip()
            if roast:
                logger.info("Ollama roast generated successfully")
                return roast
        except requests.ConnectionError:
            logger.warning("Ollama not reachable at %s — using fallback", self.url)
        except requests.Timeout:
            logger.warning("Ollama timed out — using fallback")
        except Exception as exc:
            logger.warning("Ollama error: %s — using fallback", exc)

        return None

    def _fallback_roast(self, current_event: str, next_event: str, kill_count: int, pending_tasks: str = "") -> str:
        """Pick a random template roast and fill in placeholders."""
        # When nothing is currently scheduled, substitute a task name or
        # a time-aware activity so the roast doesn't read "Nothing scheduled
        # right now' needs you" verbatim.
        event_substitute = current_event
        if current_event == "Nothing scheduled right now":
            if pending_tasks and pending_tasks not in ("No pending tasks this week", "No tasks"):
                # e.g. "Finish report (due today), Review PR (due tomorrow)"
                first_task = pending_tasks.split(",")[0].strip()
                if " (due " in first_task:
                    first_task = first_task[:first_task.index(" (due ")]
                event_substitute = first_task
            else:
                hour = datetime.now().hour
                if hour < 9:
                    event_substitute = "morning prep"
                elif hour < 12:
                    event_substitute = "your morning work"
                elif hour < 14:
                    event_substitute = "your afternoon grind"
                elif hour < 17:
                    event_substitute = "whatever deadline you're dodging"
                elif hour < 20:
                    event_substitute = "your evening to-do list"
                else:
                    event_substitute = "sleep (yes, sleep)"

        template = random.choice(self._fallback_roasts)

        # Compute minutes until next event (rough)
        minutes_until = "??"
        if self.calendar is not None:
            nxt = self.calendar.get_next_event()
            if nxt and nxt.get("start"):
                try:
                    from datetime import datetime as dt
                    start = dt.fromisoformat(nxt["start"].replace("Z", "+00:00"))
                    diff = (start - datetime.now(start.tzinfo)).total_seconds() / 60
                    minutes_until = str(max(1, int(diff)))
                except Exception:
                    pass

        return template.format(
            event=event_substitute,
            next_event=next_event,
            kill_count=kill_count,
            minutes_until=minutes_until,
        )

    # ------------------------------------------------------------------
    # Educated guess (no calendar event)
    # ------------------------------------------------------------------

    def generate_guess(self) -> str:
        """
        When no calendar event is active, generate an educated guess about
        what the user should be doing based on the time and day of week.
        Uses Ollama if available, otherwise returns a hardcoded time-aware fallback.
        """
        now = datetime.now()
        hour = now.hour
        current_time = now.strftime("%I:%M %p")
        day_of_week = now.strftime("%A")
        next_event = self._get_next_event_text()
        kill_count = self.stats.get_today_stats().get("kill_count", 0)

        if hour < 6:
            time_of_day = "middle of the night"
        elif hour < 9:
            time_of_day = "early morning"
        elif hour < 12:
            time_of_day = "morning"
        elif hour < 14:
            time_of_day = "midday"
        elif hour < 17:
            time_of_day = "afternoon"
        elif hour < 20:
            time_of_day = "evening"
        else:
            time_of_day = "late night"

        # Try Ollama
        try:
            prompt = GUESS_PROMPT.format(
                current_time=current_time,
                day_of_week=day_of_week,
                time_of_day=time_of_day,
                next_event=next_event,
                kill_count=kill_count,
            )
            resp = requests.post(
                f"{self.url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=8,
            )
            resp.raise_for_status()
            guess = resp.json().get("response", "").strip()
            if guess:
                self.last_guess = guess
                return guess
        except Exception:
            pass

        # Hardcoded time-aware fallbacks
        is_weekend = day_of_week in ("Saturday", "Sunday")
        if hour < 6:
            guess = "It's the dead of night. Put the phone down and sleep, you absolute goblin."
        elif hour < 9:
            guess = f"It's {current_time} on a {day_of_week} morning. Prep for the day instead of rotting in bed."
        elif 9 <= hour < 12 and not is_weekend:
            guess = f"It's {current_time} — a weekday morning. You have work or classes, don't you?"
        elif 12 <= hour < 14:
            guess = "Lunch break is for eating, not doomscrolling. Chew your food."
        elif 14 <= hour < 17 and not is_weekend:
            guess = "Mid-afternoon on a weekday — you've got deadlines lurking. Get back to it."
        elif 17 <= hour < 20 and not is_weekend:
            guess = f"{day_of_week} evening. You probably have assignments due tomorrow you haven't started."
        elif hour >= 20:
            guess = "It's late. Either finish your work or go to sleep. There's no third option."
        elif is_weekend and hour < 12:
            guess = "Weekend morning — great time to get ahead on the week. You won't though, will you."
        else:
            guess = f"You could be doing literally anything productive right now. It's {current_time} on {day_of_week}."

        self.last_guess = guess
        return guess

    def get_guess(self) -> str:
        """Return the last generated guess, or generate a fresh one."""
        if not self.last_guess:
            return self.generate_guess()
        return self.last_guess

    # ------------------------------------------------------------------
    # Calendar helpers
    # ------------------------------------------------------------------

    def _get_current_event_text(self) -> str:
        if self.calendar is None:
            return "Nothing scheduled right now"
        ev = self.calendar.get_current_event()
        if ev:
            return ev.get("summary", "Untitled event")
        return "Nothing scheduled right now"

    def _get_next_event_text(self) -> str:
        if self.calendar is None:
            return "Nothing coming up"
        ev = self.calendar.get_next_event()
        if ev:
            label = ev.get("summary", "Untitled event")
            if ev.get("day") == "tomorrow":
                label += " (tomorrow)"
            return label
        return "Nothing coming up"

    def _get_pending_tasks_text(self) -> str:
        """Return a short summary of pending Google Tasks for this week."""
        if self.calendar is None:
            return "No tasks"
        try:
            tasks = self.calendar.get_tasks()
            if not tasks:
                return "No pending tasks this week"
            # Format up to 3 tasks with due-day hint
            parts = []
            now_date = datetime.now().date()
            for t in tasks[:3]:
                name = t.get("title", "Untitled task")
                due_str = t.get("due", "")
                if due_str:
                    try:
                        # Google Tasks due dates are RFC 3339 (UTC midnight)
                        due_dt = datetime.fromisoformat(due_str.replace("Z", "+00:00"))
                        diff = (due_dt.date() - now_date).days
                        if diff < 0:
                            day_label = "overdue"
                        elif diff == 0:
                            day_label = "today"
                        elif diff == 1:
                            day_label = "tomorrow"
                        else:
                            day_label = due_dt.strftime("%A")  # Monday, Tuesday…
                        parts.append(f"{name} (due {day_label})")
                    except Exception:
                        parts.append(name)
                else:
                    parts.append(name)
            summary = ", ".join(parts)
            if len(tasks) > 3:
                summary += f" (+{len(tasks) - 3} more)"
            return summary
        except Exception:
            return "No tasks"

    def _format_time_since_last(self) -> str:
        if self._last_roast_time == 0:
            return "First time today — let's not make it a habit"
        delta = time.time() - self._last_roast_time
        if delta < 60:
            return f"{int(delta)} seconds ago"
        elif delta < 3600:
            return f"{int(delta // 60)} minutes ago"
        else:
            return f"{int(delta // 3600)} hours ago"

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def speak(self, text: str):
        """Speak the roast out loud via TTS (non-blocking, in a thread)."""
        if not self.tts_enabled:
            return
        threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()

    def _speak_sync(self, text: str):
        """Synchronous TTS — runs in background thread."""
        with self._tts_lock:
            try:
                if self.tts_engine == "pyttsx3":
                    self._speak_pyttsx3(text)
                elif self.tts_engine == "edge-tts":
                    self._speak_edge_tts(text)
                else:
                    logger.warning("Unknown TTS engine: %s", self.tts_engine)
            except Exception as exc:
                logger.error("TTS error: %s", exc)

    def _speak_pyttsx3(self, text: str):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def _speak_edge_tts(self, text: str):
        import asyncio
        import edge_tts
        import tempfile
        import pygame

        async def _generate():
            communicate = edge_tts.Communicate(text, "en-US-GuyNeural")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            await communicate.save(tmp_path)
            return tmp_path

        loop = asyncio.new_event_loop()
        tmp_path = loop.run_until_complete(_generate())
        loop.close()

        # Play with pygame
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def fire(self):
        """Full pipeline: generate roast → display → speak."""
        roast = self.generate_roast()
        logger.info("AI Coach says: %s", roast)
        self.speak(roast)

    # ------------------------------------------------------------------
    # History / logging
    # ------------------------------------------------------------------

    def _log_roast(self, roast: str, current_event: str, next_event: str, kill_count: int):
        """Append roast to history JSON."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "roast": roast,
            "current_event": current_event,
            "next_event": next_event,
            "kill_count": kill_count,
        }

        history = self._load_history()
        history.append(entry)

        # Keep only last 500 roasts
        if len(history) > 500:
            history = history[-500:]

        try:
            self.roast_history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.roast_history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save roast history: %s", exc)

    def _load_history(self) -> list:
        """Load existing roast history."""
        try:
            if self.roast_history_path.exists():
                with open(self.roast_history_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def get_roast_history(self, limit: int = 50) -> list:
        """Return the most recent roasts."""
        history = self._load_history()
        return history[-limit:]

    def get_latest_roast(self) -> str:
        """Return the last generated roast."""
        return self.last_roast
