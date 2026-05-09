"""
calendar_sync.py — Google Calendar integration for BigBrother.

Handles OAuth 2.0 authentication, event fetching, and task completion
for the Google Calendar API. Provides current/next event data to the
AI coach for contextual roasts.
"""

import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Google API scopes — calendar read/write + tasks read
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks.readonly",
]


class CalendarSync:
    """
    Synchronises with Google Calendar via the official API.

    Caches events locally and refreshes at a configurable interval.
    """

    def __init__(
        self,
        credentials_path: str = "config/credentials.json",
        token_path: str = "config/token.json",
        calendar_id: str = "primary",
        refresh_interval: int = 300,
    ):
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.calendar_id = calendar_id
        self.refresh_interval = refresh_interval

        self._service = None
        self._tasks_service = None
        self._events_cache: List[Dict[str, Any]] = []      # today
        self._tomorrow_cache: List[Dict[str, Any]] = []    # tomorrow
        self._tasks_cache: List[Dict[str, Any]] = []       # Google Tasks
        self._last_refresh: float = 0
        self._authenticated = False

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        """
        Run the OAuth 2.0 flow. Opens a browser window on first run
        to authorise the app. Stores the token in config/token.json.
        """
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except ImportError:
            logger.error(
                "Google API libraries not installed. Run: "
                "pip install google-api-python-client google-auth-oauthlib google-auth-httplib2"
            )
            return False

        creds = None

        # Try to load existing token
        if self.token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
            except Exception as exc:
                logger.warning("Could not load token: %s", exc)

        # Refresh or re-authenticate
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("Calendar token refreshed")
            except Exception:
                creds = None

        # Force re-auth if stored token is missing required scopes (e.g. tasks.readonly added later)
        if creds and creds.valid and creds.scopes:
            missing = set(SCOPES) - set(creds.scopes)
            if missing:
                logger.info(
                    "Stored token missing scopes %s — re-authenticating to include them", missing
                )
                creds = None

        if not creds or not creds.valid:
            if not self.credentials_path.exists():
                logger.error(
                    "credentials.json not found at %s. "
                    "Download it from Google Cloud Console.",
                    self.credentials_path,
                )
                return False

            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.credentials_path), SCOPES
            )
            creds = flow.run_local_server(port=0)
            logger.info("Calendar authentication completed")

        # Save token
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.token_path, "w") as f:
            f.write(creds.to_json())

        self._service = build("calendar", "v3", credentials=creds)
        try:
            self._tasks_service = build("tasks", "v1", credentials=creds)
            logger.info("Google Tasks service ready")
        except Exception as exc:
            logger.warning("Could not build Tasks service: %s", exc)
            self._tasks_service = None
        self._authenticated = True
        logger.info("Google Calendar service ready")
        return True

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    # ------------------------------------------------------------------
    # Event fetching
    # ------------------------------------------------------------------

    def fetch_events(self, date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetch events for a given day (defaults to today).
        Returns a list of event dicts with keys: id, summary, start, end, status.
        """
        if not self._authenticated or self._service is None:
            logger.warning("Calendar not authenticated — returning empty events")
            return []

        if date is None:
            date = datetime.now(timezone.utc)

        # Start / end of the day in UTC
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        try:
            result = self._service.events().list(
                calendarId=self.calendar_id,
                timeMin=start_of_day.isoformat(),
                timeMax=end_of_day.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            ).execute()

            events = []
            for item in result.get("items", []):
                events.append({
                    "id": item.get("id"),
                    "summary": item.get("summary", "Untitled"),
                    "start": item.get("start", {}).get("dateTime", item.get("start", {}).get("date", "")),
                    "end": item.get("end", {}).get("dateTime", item.get("end", {}).get("date", "")),
                    "status": item.get("status", "confirmed"),
                    "description": item.get("description", ""),
                })

            self._events_cache = events
            self._last_refresh = time.time()
            logger.info("Fetched %d calendar events", len(events))
            return events

        except Exception as exc:
            logger.error("Failed to fetch calendar events: %s", exc)
            return self._events_cache  # return stale cache on error

    def fetch_tomorrow_events(self) -> List[Dict[str, Any]]:
        """Fetch events for tomorrow directly into _tomorrow_cache."""
        if not self._authenticated or self._service is None:
            return []
        tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
        start = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        end   = start + timedelta(days=1)
        try:
            result = self._service.events().list(
                calendarId=self.calendar_id,
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            ).execute()
            self._tomorrow_cache = [
                {
                    "id": e.get("id"),
                    "summary": e.get("summary", "Untitled"),
                    "start": e.get("start", {}).get("dateTime", e.get("start", {}).get("date", "")),
                    "end":   e.get("end",   {}).get("dateTime", e.get("end",   {}).get("date", "")),
                    "status": e.get("status", "confirmed"),
                    "description": e.get("description", ""),
                    "day": "tomorrow",
                }
                for e in result.get("items", [])
            ]
            logger.info("Fetched %d tomorrow events", len(self._tomorrow_cache))
            return self._tomorrow_cache
        except Exception as exc:
            logger.error("Failed to fetch tomorrow events: %s", exc)
            return self._tomorrow_cache

    def fetch_tasks(self) -> List[Dict[str, Any]]:
        """
        Fetch incomplete Google Tasks due by end of the current week (Sunday).
        If fewer than 3 days remain in the week, extends to the following Sunday.
        Returns a list of task dicts with keys: id, title, due, status, tasklist.
        """
        if not self._authenticated or self._tasks_service is None:
            return []
        try:
            now = datetime.now(timezone.utc)

            # Advance to the end of this Sunday; if <3 days remain, go to next Sunday
            days_until_sunday = (6 - now.weekday()) % 7  # Monday=0 … Sunday=6
            if days_until_sunday < 3:
                days_until_sunday += 7  # extend to next Sunday
            week_end = (now + timedelta(days=days_until_sunday + 1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

            task_lists = self._tasks_service.tasklists().list(maxResults=10).execute()
            tasks: List[Dict[str, Any]] = []
            for tl in task_lists.get("items", []):
                tl_id = tl["id"]
                result = self._tasks_service.tasks().list(
                    tasklist=tl_id,
                    showCompleted=False,
                    dueMax=week_end.isoformat(),
                    maxResults=30,
                ).execute()
                for item in result.get("items", []):
                    if item.get("status") == "completed":
                        continue
                    tasks.append({
                        "id": item.get("id"),
                        "title": item.get("title", "Untitled task"),
                        "due": item.get("due", ""),
                        "notes": item.get("notes", ""),
                        "status": item.get("status", "needsAction"),
                        "tasklist": tl.get("title", ""),
                    })
            # Sort by due date (tasks with no due date go last)
            tasks.sort(key=lambda t: t["due"] or "9999")
            self._tasks_cache = tasks
            logger.info("Fetched %d pending tasks for the week", len(tasks))
            return tasks
        except Exception as exc:
            logger.error("Failed to fetch tasks: %s", exc)
            return self._tasks_cache

    def _maybe_refresh(self):
        """Refresh events (today + tomorrow) and tasks if the cache is stale."""
        if time.time() - self._last_refresh > self.refresh_interval:
            self.fetch_events()          # populates _events_cache (today)
            self.fetch_tomorrow_events() # populates _tomorrow_cache
            self.fetch_tasks()           # populates _tasks_cache

    def get_today_events(self) -> List[Dict[str, Any]]:
        """Get today's events (cached, auto-refreshing)."""
        self._maybe_refresh()
        return self._events_cache

    def get_tomorrow_events(self) -> List[Dict[str, Any]]:
        """Get tomorrow's events (cached, auto-refreshing)."""
        self._maybe_refresh()
        return self._tomorrow_cache

    def get_tasks(self) -> List[Dict[str, Any]]:
        """Get pending Google Tasks due today or tomorrow (cached, auto-refreshing)."""
        self._maybe_refresh()
        return self._tasks_cache

    def get_week_plan(self) -> List[Dict[str, Any]]:
        """
        Return a day-by-day plan for the rest of the week (today → Sunday).
        Each entry: { date, label, events, tasks }
        """
        self._maybe_refresh()

        now = datetime.now(timezone.utc)
        today = now.date()

        # How many days until Sunday (0 = today is Sunday → show current week only)
        days_until_sunday = (6 - today.weekday()) % 7
        if days_until_sunday == 0:
            days_until_sunday = 6  # Sunday → show the next 6 days

        days: List[Dict[str, Any]] = []

        for offset in range(days_until_sunday + 1):
            target_date = today + timedelta(days=offset)
            date_str = target_date.isoformat()  # "2026-02-24"

            # Label
            if offset == 0:
                label = "Today"
            elif offset == 1:
                label = "Tomorrow"
            else:
                label = target_date.strftime("%A")  # Monday, Tuesday…

            # Events for this day
            if offset == 0:
                day_events = [
                    {"summary": e.get("summary", ""), "start": e.get("start", ""),
                     "end": e.get("end", ""), "id": e.get("id", "")}
                    for e in self._events_cache
                ]
            elif offset == 1:
                day_events = [
                    {"summary": e.get("summary", ""), "start": e.get("start", ""),
                     "end": e.get("end", ""), "id": e.get("id", "")}
                    for e in self._tomorrow_cache
                ]
            else:
                # Fetch on demand (will be cached by fetch_events → but we don't
                # persist further-out days — do a live fetch for this week view)
                day_events = []
                if self._authenticated and self._service is not None:
                    try:
                        target_dt = datetime(target_date.year, target_date.month,
                                             target_date.day, tzinfo=timezone.utc)
                        start = target_dt
                        end   = target_dt + timedelta(days=1)
                        result = self._service.events().list(
                            calendarId=self.calendar_id,
                            timeMin=start.isoformat(),
                            timeMax=end.isoformat(),
                            singleEvents=True,
                            orderBy="startTime",
                        ).execute()
                        day_events = [
                            {"summary": e.get("summary", ""),
                             "start": e.get("start", {}).get("dateTime", e.get("start", {}).get("date", "")),
                             "end":   e.get("end",   {}).get("dateTime", e.get("end",   {}).get("date", "")),
                             "id":    e.get("id", "")}
                            for e in result.get("items", [])
                        ]
                    except Exception as exc:
                        logger.warning("Could not fetch events for %s: %s", date_str, exc)

            # Tasks due on this day
            day_tasks = [
                {"title": t.get("title", ""), "id": t.get("id", ""),
                 "due": t.get("due", ""), "tasklist": t.get("tasklist", "")}
                for t in self._tasks_cache
                if t.get("due", "")[:10] == date_str
            ]
            # Also include tasks with no due date on the "Today" bucket
            if offset == 0:
                day_tasks += [
                    {"title": t.get("title", ""), "id": t.get("id", ""),
                     "due": "", "tasklist": t.get("tasklist", "")}
                    for t in self._tasks_cache
                    if not t.get("due")
                ]

            days.append({
                "date":   date_str,
                "label":  label,
                "events": day_events,
                "tasks":  day_tasks,
            })

        return days

    def get_current_event(self) -> Optional[Dict[str, Any]]:
        """Get the event happening right now, if any."""
        self._maybe_refresh()
        now = datetime.now(timezone.utc)

        for event in self._events_cache:
            try:
                start = datetime.fromisoformat(event["start"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(event["end"].replace("Z", "+00:00"))
                if start <= now <= end:
                    return event
            except (ValueError, KeyError):
                continue
        return None

    def get_next_event(self) -> Optional[Dict[str, Any]]:
        """Get the next upcoming event, searching today then tomorrow."""
        self._maybe_refresh()
        now = datetime.now(timezone.utc)

        for event in self._events_cache + self._tomorrow_cache:
            try:
                start = datetime.fromisoformat(event["start"].replace("Z", "+00:00"))
                if start > now:
                    return event
            except (ValueError, KeyError):
                continue
        return None

    # ------------------------------------------------------------------
    # Task completion
    # ------------------------------------------------------------------

    def complete_task(self, event_id: str) -> bool:
        """
        Mark a calendar event as completed by changing its status.
        Uses colorId to visually mark it as done (green = 11).
        """
        if not self._authenticated or self._service is None:
            return False

        try:
            event = self._service.events().get(
                calendarId=self.calendar_id, eventId=event_id
            ).execute()

            event["colorId"] = "11"  # Green in Google Calendar
            event["summary"] = "✅ " + event.get("summary", "")

            self._service.events().update(
                calendarId=self.calendar_id, eventId=event_id, body=event
            ).execute()

            logger.info("Marked event %s as complete", event_id)

            # Refresh cache
            self.fetch_events()
            return True

        except Exception as exc:
            logger.error("Failed to complete task %s: %s", event_id, exc)
            return False
