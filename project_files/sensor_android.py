"""
sensor_android.py — Android sensor provider via Termux:API
Tested on Vivo V27 (Android 13, Funtouch OS 13)

Hardware reality on Vivo V27:
  GPS          → real   (termux-location)
  Altitude     → GPS-derived (from NMEA alt)
  Barometer    → NOT PRESENT → Open-Meteo API fallback
  Temperature  → NOT PRESENT → battery temp proxy (°C from termux-battery-status)
  Battery      → real   (termux-battery-status)

Install requirements on Android (Termux):
  pkg install termux-api python
  pip install requests

System requirements:
  • Termux:API app installed from F-Droid  (NOT Play Store)
  • Location permission granted to Termux:API app
  • Run: termux-setup-storage
"""

from __future__ import annotations
import json
import logging
import subprocess
import time
from typing import Optional, Tuple

import requests

from sensor_provider import SensorProvider, SensorState
import config

logger = logging.getLogger(__name__)


class TermuxAPI:
    """
    Thin wrapper around Termux:API CLI commands.
    All methods return parsed dicts or raise on failure.
    """

    @staticmethod
    def _run(cmd: list[str], timeout: int = 30) -> dict:
        try:
            out = subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL,
                text=True, timeout=timeout
            )
            return json.loads(out.strip())
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Termux command timed out: {cmd}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Bad JSON from {cmd}: {exc}")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Termux command failed {cmd}: {exc}")

    @staticmethod
    def location(provider: str = "gps",
                 timeout: int = config.TERMUX_LOC_TIMEOUT_S) -> dict:
        """
        termux-location returns:
          { "latitude": ..., "longitude": ..., "altitude": ...,
            "accuracy": ..., "bearing": ..., "speed": ... }
        provider: "gps" | "network" | "passive"
        """
        return TermuxAPI._run(
            ["termux-location", "-p", provider, "-r", "once"],
            timeout=timeout
        )

    @staticmethod
    def battery_status() -> dict:
        """
        termux-battery-status returns:
          { "health": "GOOD", "percentage": 72, "plugged": "UNPLUGGED",
            "status": "DISCHARGING", "temperature": 28.5 }
        """
        return TermuxAPI._run(["termux-battery-status"])

    @staticmethod
    def sensor_once(sensor_type: str, timeout: int = 5) -> Optional[dict]:
        """
        Try to read a named hardware sensor once.
        Returns None if sensor not available (e.g. no barometer).
        """
        try:
            return TermuxAPI._run(
                ["termux-sensor", "-s", sensor_type, "-n", "1"],
                timeout=timeout
            )
        except Exception:
            return None


class OpenMeteoClient:
    """
    Open-Meteo free weather API (no API key required).
    Used as barometric pressure fallback on devices without a baro sensor.
    Returns surface pressure (hPa) and 2 m temperature (°C).
    Caches last result for TTL seconds to avoid hammering the API.
    """

    _CACHE_TTL_S = 60   # refresh at most once per minute

    def __init__(self):
        self._last_result: Optional[Tuple[float, float]] = None
        self._last_fetch:  float = 0.0

    def fetch(self, lat: float, lon: float) -> Tuple[float, float]:
        """Returns (pressure_hPa, temperature_C)."""
        now = time.time()
        if self._last_result and (now - self._last_fetch) < self._CACHE_TTL_S:
            return self._last_result

        url = (
            f"{config.OPEN_METEO_BASE}"
            f"?latitude={lat:.5f}&longitude={lon:.5f}"
            f"&current=surface_pressure,temperature_2m"
            f"&wind_speed_unit=ms"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data    = resp.json()
            current = data["current"]
            pressure_hpa = float(current["surface_pressure"])
            temp_c       = float(current["temperature_2m"])
            self._last_result = (pressure_hpa, temp_c)
            self._last_fetch  = now
            logger.debug("Open-Meteo: %.2f hPa, %.1f°C", pressure_hpa, temp_c)
            return self._last_result
        except Exception as exc:
            logger.warning("Open-Meteo fetch failed: %s", exc)
            if self._last_result:
                logger.info("Using stale Open-Meteo cache")
                return self._last_result
            # Absolute fallback: ISA sea-level standard atmosphere
            return 1013.25, 15.0


class AndroidSensorProvider(SensorProvider):
    """
    Android sensor provider for Termux Python stack.
    Vivo V27 sensor mapping:
      GPS           → termux-location  (provider=gps, fallback=network)
      Altitude      → GPS NMEA altitude
      Pressure/hPa  → Open-Meteo API (no onboard baro)
      Temp/°C       → Battery temperature (proxy, typically 25–45°C)
      Battery %     → termux-battery-status percentage
    """

    def __init__(self):
        self._meteo    = OpenMeteoClient()
        self._last_gps = (0.0, 0.0, 0.0)   # lat, lon, alt
        self._gps_ok   = False
        logger.info("AndroidSensorProvider ready (Vivo V27 profile)")

    # ── internal reads ────────────────────────────────────────────────────────

    def _read_gps(self) -> Tuple[float, float, float]:
        """Try GPS first, fall back to network location."""
        for provider in ("gps", "network"):
            try:
                loc = TermuxAPI.location(provider=provider)
                lat = float(loc["latitude"])
                lon = float(loc["longitude"])
                alt = float(loc.get("altitude", 0.0))
                self._last_gps = (lat, lon, alt)
                self._gps_ok   = True
                logger.debug("GPS fix [%s]: %.5f, %.5f, %.1fm", provider, lat, lon, alt)
                return self._last_gps
            except Exception as exc:
                logger.warning("GPS provider '%s' failed: %s", provider, exc)

        logger.warning("All GPS providers failed — returning last known fix")
        return self._last_gps

    def _read_battery(self) -> Tuple[float, float]:
        """Returns (battery_fraction, battery_temp_c)."""
        try:
            bat = TermuxAPI.battery_status()
            pct  = float(bat.get("percentage", 50)) / 100.0
            temp = float(bat.get("temperature", 30.0))
            return pct, temp
        except Exception as exc:
            logger.warning("Battery status failed: %s", exc)
            return 0.5, 30.0

    def _try_hardware_baro(self) -> Optional[float]:
        """
        Attempt to read TYPE_PRESSURE from Android sensor HAL.
        On Vivo V27 this will return None (no barometer chip).
        Returns pressure_hPa or None.
        """
        raw = TermuxAPI.sensor_once("TYPE_PRESSURE")
        if raw is None:
            return None
        try:
            # termux-sensor returns: { "TYPE_PRESSURE": { "values": [hPa] } }
            return float(raw["TYPE_PRESSURE"]["values"][0])
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    # ── public ────────────────────────────────────────────────────────────────

    def get_state(self) -> SensorState:
        lat, lon, alt = self._read_gps()
        bat_pct, bat_temp = self._read_battery()

        # Try onboard barometer first (future-proofing)
        hw_baro = self._try_hardware_baro()

        if hw_baro is not None:
            pressure_hpa = hw_baro
            baro_source  = "hardware"
            # If hardware baro is available, it includes temperature
            temp_c       = bat_temp   # still use battery temp as env proxy
            temp_source  = "battery_proxy"
        else:
            # Vivo V27 path: Open-Meteo
            if self._gps_ok and lat != 0.0:
                api_pres, api_temp = self._meteo.fetch(lat, lon)
            else:
                api_pres, api_temp = 1013.25, 15.0   # absolute fallback

            pressure_hpa = api_pres
            baro_source  = "open_meteo_api"
            # Use battery temp as ground-truth env temp proxy
            # (battery temp ≈ ambient + 5-10°C; we subtract a rough offset)
            temp_c       = bat_temp - 7.0   # rough ambient correction
            temp_source  = "battery_proxy"

        return SensorState(
            lat          = lat,
            lon          = lon,
            altitude_m   = alt,
            pressure_hpa = pressure_hpa,
            temp_c       = temp_c,
            battery_pct  = bat_pct,
            source_flags = {
                "gps":      "real" if self._gps_ok else "stale",
                "altitude": "gps_derived",
                "pressure": baro_source,
                "temp":     temp_source,
                "battery":  "real",
            }
        )
