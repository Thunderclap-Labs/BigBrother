"""
stats_tracker.py — Kill count, streaks, and history for BigBrother.

Tracks how many times the user has been caught picking up their phone,
detection durations, hourly breakdowns, and streak data. Persists
everything to data/stats.json.
"""

import time
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class StatsTracker:
    """
    Persistent statistics tracker for phone detection events.

    Stores per-day kill counts, detection durations, hourly breakdowns,
    longest phone-free streaks, and more.
    """

    def __init__(self, stats_path: str = "data/stats.json"):
        self.stats_path = Path(stats_path)
        self._data: Dict[str, Any] = {}
        self._streak_start: Optional[float] = None
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        """Load stats from disk."""
        if self.stats_path.exists():
            try:
                with open(self.stats_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info("Stats loaded from %s", self.stats_path)
            except Exception as exc:
                logger.error("Failed to load stats: %s", exc)
                self._data = {}
        else:
            self._data = {}

        # Ensure top-level keys exist
        self._data.setdefault("daily", {})
        self._data.setdefault("total_kills", 0)
        self._data.setdefault("longest_streak_seconds", 0)
        self._data.setdefault("worst_hours", {})
        self._data.setdefault("kill_log", [])

        self._streak_start = time.time()

    def _save(self):
        """Save stats to disk."""
        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save stats: %s", exc)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_kill(self, detection_duration: float = 0.0):
        """
        Record a phone detection event (a 'kill').

        Parameters
        ----------
        detection_duration : float
            How long the phone was visible before the scare fired (seconds).
        """
        today = date.today().isoformat()
        hour = str(datetime.now().hour)
        now = time.time()

        # Daily stats
        day_data = self._data["daily"].setdefault(today, {
            "kill_count": 0,
            "total_duration": 0.0,
            "kills": [],
        })
        day_data["kill_count"] += 1
        day_data["total_duration"] += detection_duration
        day_data["kills"].append({
            "time": datetime.now().isoformat(),
            "duration": round(detection_duration, 1),
        })

        # Total
        self._data["total_kills"] += 1

        # Worst hours
        hour_count = self._data["worst_hours"].get(hour, 0)
        self._data["worst_hours"][hour] = hour_count + 1

        # Kill log (global, keep last 1000)
        self._data["kill_log"].append({
            "date": today,
            "time": datetime.now().isoformat(),
            "duration": round(detection_duration, 1),
        })
        if len(self._data["kill_log"]) > 1000:
            self._data["kill_log"] = self._data["kill_log"][-1000:]

        # Update streak — current streak is broken
        if self._streak_start is not None:
            streak = now - self._streak_start
            if streak > self._data["longest_streak_seconds"]:
                self._data["longest_streak_seconds"] = streak

        # Reset streak timer
        self._streak_start = now

        self._save()
        logger.info(
            "Kill recorded — today=%d  total=%d  duration=%.1fs",
            day_data["kill_count"], self._data["total_kills"], detection_duration,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_today_stats(self) -> Dict[str, Any]:
        """Get today's kill count and stats."""
        today = date.today().isoformat()
        day_data = self._data["daily"].get(today, {
            "kill_count": 0,
            "total_duration": 0.0,
            "kills": [],
        })

        kill_count = day_data["kill_count"]
        avg_duration = 0.0
        if kill_count > 0:
            avg_duration = day_data["total_duration"] / kill_count

        # Current streak
        current_streak = 0.0
        if self._streak_start is not None:
            current_streak = time.time() - self._streak_start

        return {
            "kill_count": kill_count,
            "total_duration": round(day_data["total_duration"], 1),
            "average_duration": round(avg_duration, 1),
            "current_streak": round(current_streak, 1),
            "kills": day_data.get("kills", []),
        }

    def get_weekly_stats(self) -> Dict[str, Any]:
        """Get stats for the last 7 days."""
        today = date.today()
        weekly = {}
        total_kills = 0
        total_duration = 0.0

        for i in range(7):
            d = (today - __import__("datetime").timedelta(days=i)).isoformat()
            day_data = self._data["daily"].get(d, {"kill_count": 0, "total_duration": 0.0})
            weekly[d] = {
                "kill_count": day_data.get("kill_count", 0),
                "total_duration": round(day_data.get("total_duration", 0.0), 1),
            }
            total_kills += day_data.get("kill_count", 0)
            total_duration += day_data.get("total_duration", 0.0)

        return {
            "days": weekly,
            "total_kills": total_kills,
            "total_duration": round(total_duration, 1),
            "daily_average": round(total_kills / 7, 1),
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get complete stats overview."""
        today_stats = self.get_today_stats()
        weekly_stats = self.get_weekly_stats()

        return {
            "today": today_stats,
            "weekly": weekly_stats,
            "total_kills": self._data.get("total_kills", 0),
            "longest_streak_seconds": round(self._data.get("longest_streak_seconds", 0), 1),
            "worst_hours": self._data.get("worst_hours", {}),
        }

    def get_streak(self) -> float:
        """Get the current phone-free streak in seconds."""
        if self._streak_start is None:
            return 0.0
        return time.time() - self._streak_start

    def get_longest_streak(self) -> float:
        """Get the longest phone-free streak ever (seconds)."""
        current = self.get_streak()
        historical = self._data.get("longest_streak_seconds", 0)
        return max(current, historical)
