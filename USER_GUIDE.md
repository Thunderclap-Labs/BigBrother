# User Onboarding and Setup Guide

## Requirements

- Python 3.10+
- Webcam
- (Optional) ESP32-based scare gun connected via USB serial
- (Optional) Local [Ollama](https://ollama.com/) instance for AI coaching

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd backub-web
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure settings**

   Copy the example config and edit it for your environment:
   ```bash
   cp config/local.yaml.example config/local.yaml
   ```

   Key settings in `config/settings.yaml`:
   | Setting | Description |
   |---|---|
   | `camera_index` | Webcam device index (default `0`) |
   | `serial_port` | COM port for ESP32 (e.g. `COM3` / `/dev/ttyUSB0`) |
   | `work_apps` | List of process names classified as work |
   | `scare_cooldown_seconds` | Minimum time between scare triggers |

4. **Google Calendar (optional)**

   Follow the [Google OAuth2 setup guide](https://developers.google.com/identity/protocols/oauth2) to generate credentials. Place the token file at `config/token.json`.

## Running the App

### Windows
Double-click `start.bat`, or run:
```bash
python main.py
```

### All platforms
```bash
python main.py
```

The web dashboard will open at **http://localhost:5000**.

## Dashboard Overview

| Page | URL | Description |
|---|---|---|
| Camera | `/camera` | Live webcam feed with YOLO bounding boxes |
| Dashboard | `/` | Real-time productivity stats |
| Stats | `/stats` | Historical usage graphs |
| Settings | `/settings` | Adjust thresholds and app lists |

## Troubleshooting

**Webcam not detected** — Check `camera_index` in settings. Try `0`, `1`, or `2`.

**Serial gun not firing** — Verify `serial_port` matches your device manager. Ensure the ESP32 firmware is flashed.

**AI coach not responding** — Confirm Ollama is running locally (`ollama serve`) and the model is pulled (`ollama pull llama3`).
