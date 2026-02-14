# BigBrother

**Always Watching.** An anti-phone-distraction tool that uses YOLOv8 to detect when you pick up your phone, plays a loud gunshot sound to scare you, and has an AI `drill sergeant` that roasts you about what you *should* be doing based on your Google Calendar.

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
# Install Ollama from https://ollama.ai
ollama pull llama3
# Ollama runs on localhost:11434 by default
```

If Ollama isn't running, the app falls back to pre-written roast templates from `config/roasts.yaml`.

## Project Structure

```
backub-web/
├── main.py                  # Entry point
├── requirements.txt
├── config/
│   ├── settings.yaml        # User preferences
│   ├── roasts.yaml          # Fallback roast templates
│   └── credentials.json     # Google OAuth (you provide this)
├── core/
│   ├── phone_detector.py    # YOLOv8 phone detection
│   ├── scare_system.py      # Sound playback + triggers
│   ├── ai_coach.py          # LLM roast generation + TTS
│   ├── calendar_sync.py     # Google Calendar API
│   ├── stats_tracker.py     # Kill counts and streaks
│   └── serial_gun.py        # Optional hardware interface
├── web/
│   ├── server.py            # Flask app + API endpoints
│   ├── templates/           # HTML templates
│   └── static/              # CSS, JS
├── sounds/                  # .wav sound effects
└── data/                    # Persistent stats + roast history
```

## Configuration

Edit `config/settings.yaml` to customise detection thresholds, volume, cooldown, LLM model, and more. See the file for all available options.

## Sound Effects

Place `.wav` files in the `sounds/` folder:
- `gunshot*.wav` — random gunshot sounds played on trigger
- `alarm.wav` — continuous alarm for escalation
- `reload.wav` — played when cooldown ends

## Dashboard

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
| `/api/calendar` | GET | Today's events |
| `/api/calendar/complete` | POST | Check off a task |
| `/api/camera/feed` | GET | MJPEG stream |
| `/api/settings` | GET/POST | Read/update settings |
| `/api/roast/latest` | GET | Latest roast text |
| `/api/roast/history` | GET | All roasts |

## Contributing

This is a student project built for fun. PRs welcome!

## License

MIT

- Completed SCRUM-17: Add face-visibility tracking to phone detector
