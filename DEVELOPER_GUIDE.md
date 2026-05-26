# Developer Documentation for Core Systems

## Overview

This document provides technical documentation for the core subsystems of the WorkSimulation productivity monitor.

## Project Structure

```
core/              # Core detection and monitoring modules
  phone_detector.py    # YOLOv8-based phone detection
  process_monitor.py   # Active process / app-category monitoring
  scare_system.py      # Scare trigger orchestration
  serial_gun.py        # Serial communication with ESP32 hardware
  ai_coach.py          # Ollama LLM coaching integration
  calendar_sync.py     # Google Calendar OAuth sync
  stats_tracker.py     # Thread-safe stats and config I/O
web/               # Flask web server and UI
config/            # YAML configuration files
data/              # Persistent JSON data storage
scripts/           # Utility and simulation scripts
```

## Core Modules

### `core/phone_detector.py`
Runs YOLOv8 inference in a background thread. Detects phone usage and face visibility from webcam frames. Implements debounce logic to suppress transient false positives.

**Key classes:** `PhoneDetector`
**Dependencies:** `ultralytics`, `opencv-python`

### `core/process_monitor.py`
Polls the OS process list and classifies active applications as work or personal using a configurable allow-list in `config/settings.yaml`.

**Key classes:** `ProcessMonitor`

### `core/scare_system.py`
Orchestrates the scare response pipeline: selects roast text, triggers the serial gun, and falls back to audio if hardware is unavailable.

**Key classes:** `ScareSystem`

### `core/serial_gun.py`
Manages async serial communication with the ESP32 microcontroller. Sends trigger commands and reads acknowledgement responses.

**Key classes:** `SerialGun`

### `core/ai_coach.py`
Sends prompt requests to a local Ollama instance with strict latency timeouts. Generates personalised productivity coaching messages.

**Key classes:** `AICoach`

### `core/calendar_sync.py`
Handles Google Calendar OAuth2 flow and token refresh. Fetches upcoming events to provide context-aware coaching.

**Key classes:** `CalendarSync`

### `core/stats_tracker.py`
Thread-safe read/write of `data/stats.json` and `config/settings.yaml`. Provides atomic update helpers to prevent file corruption under concurrent access.

**Key classes:** `StatsTracker`

## Configuration

All runtime configuration lives in `config/settings.yaml`. Copy `config/local.yaml.example` to `config/local.yaml` for local overrides (excluded from version control).

## Running Locally

```bash
pip install -r requirements.txt
python main.py
```

The web dashboard will be available at `http://localhost:5000`.
