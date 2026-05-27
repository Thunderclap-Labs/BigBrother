# Hardware Interface Specification

## Overview

The scare system uses an ESP32 microcontroller connected to a modified Nerf-style gun mechanism. The ESP32 receives serial commands from the host PC and actuates a servo or relay to pull the trigger.

## Components

| Component | Description |
| --- | --- |
| ESP32 DevKit v1 | Main microcontroller, USB serial interface |
| Servo motor (SG90) | Trigger actuation mechanism |
| 5V relay module (optional) | Alternative to servo for solenoid-based triggers |
| USB-A to Micro-USB cable | Host PC to ESP32 serial connection |

## Serial Protocol

- **Baud rate:** 115200
- **Data bits:** 8
- **Parity:** None
- **Stop bits:** 1
- **Line ending:** `\n`

### Commands

| Command | Description | Response |
| --- | --- | --- |
| `FIRE\n` | Trigger the scare mechanism once | `OK\n` |
| `PING\n` | Health check | `PONG\n` |
| `STATUS\n` | Request device status | `READY\n` or `BUSY\n` |

### Example Interaction

```
Host  → PING\n
ESP32 ← PONG\n

Host  → FIRE\n
ESP32 ← OK\n
```

## ESP32 Firmware

The firmware is not included in this repository. Flash the ESP32 with the companion Arduino sketch (`esp32_trigger/esp32_trigger.ino`). Key parameters to configure in the sketch:

```cpp
#define SERVO_PIN     18      // GPIO pin connected to servo signal
#define TRIGGER_ANGLE 60      // Degrees to rotate for trigger pull
#define REST_ANGLE    0       // Resting position angle
#define TRIGGER_MS    300     // Duration of trigger hold in ms
```

## Wiring Diagram

```
ESP32 GPIO18 ──────── Servo Signal (Orange)
ESP32 5V     ──────── Servo VCC   (Red)
ESP32 GND    ──────── Servo GND   (Brown)
```

## Serial Port Configuration

Set the correct port in `config/settings.yaml`:

```yaml
serial_port: "COM3"       # Windows example
# serial_port: "/dev/ttyUSB0"  # Linux/macOS example
serial_baud: 115200
serial_timeout: 2.0
```

## Fallback Behaviour

If the serial port is unavailable or the ESP32 does not respond within `serial_timeout` seconds, `ScareSystem` falls back to a software audio roast. No exception is raised to the caller.
