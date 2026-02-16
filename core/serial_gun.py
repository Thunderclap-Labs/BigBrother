"""
serial_gun.py — Optional Arduino hardware interface for BigBrother.

Supports two transport modes:
  • WiFi  — sends GET /trigger to the Arduino's HTTP server (preferred).
  • Serial — sends "FIRE\\n" over a USB/serial connection (legacy fallback).

Gracefully falls back if no hardware is connected.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class SerialGun:
    """
    Optional hardware interface for physical scare hardware.

    WiFi mode (recommended): sends an HTTP GET request to the Arduino's
    /trigger endpoint.  Configure with wifi_enabled=True and arduino_url.

    Serial mode (legacy): sends ASCII commands over a serial connection.
    """

    def __init__(
        self,
        port: str = "COM3",
        baud_rate: int = 9600,
        enabled: bool = False,
        wifi_enabled: bool = False,
        arduino_url: str = "http://192.168.1.100",
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.enabled = enabled
        self.wifi_enabled = wifi_enabled
        # Ensure the URL ends with /trigger
        base = arduino_url.rstrip("/")
        self.trigger_url = base + "/trigger"

        self._serial = None
        self._connected = False

        if self.wifi_enabled:
            # No persistent connection needed for HTTP — mark as ready
            self._connected = True
            logger.info("Arduino WiFi trigger ready → %s", self.trigger_url)
        elif self.enabled:
            self.connect()

    # ------------------------------------------------------------------
    # Serial (legacy)
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Attempt to connect to the serial port."""
        if not self.enabled:
            return False

        try:
            import serial
            self._serial = serial.Serial(
                self.port,
                self.baud_rate,
                timeout=1,
            )
            self._connected = True
            logger.info("Serial gun connected on %s @ %d baud", self.port, self.baud_rate)
            return True
        except ImportError:
            logger.warning("pyserial not installed — serial gun disabled")
        except Exception as exc:
            logger.warning("Could not connect to serial port %s: %s", self.port, exc)

        self._connected = False
        return False

    # ------------------------------------------------------------------
    # Fire
    # ------------------------------------------------------------------

    def fire(self):
        """Send fire command to the hardware (non-blocking)."""
        if self.wifi_enabled:
            threading.Thread(target=self._wifi_fire, daemon=True).start()
        elif self._connected and self._serial is not None:
            self._serial_fire()
        else:
            logger.debug("Gun not connected — skipping fire")

    def _wifi_fire(self):
        """Send HTTP GET /trigger to the Arduino (runs in a thread)."""
        try:
            import urllib.request
            with urllib.request.urlopen(self.trigger_url, timeout=3) as resp:
                status = resp.status
            logger.info("Arduino WiFi trigger sent → %s (HTTP %d)", self.trigger_url, status)
        except Exception as exc:
            logger.warning("Arduino WiFi trigger failed: %s", exc)

    def _serial_fire(self):
        """Send FIRE command over serial."""
        try:
            self._serial.write(b"FIRE\n")
            logger.info("Serial gun FIRE command sent")
        except Exception as exc:
            logger.error("Serial gun fire failed: %s", exc)
            self._connected = False

    # ------------------------------------------------------------------
    # Disconnect
    # ------------------------------------------------------------------

    def disconnect(self):
        """Close the serial connection (no-op for WiFi mode)."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
            self._connected = False
            logger.info("Serial gun disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected
