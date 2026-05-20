"""
sensor_provider.py — SensorState dataclass + abstract SensorProvider
Shared by RPi, Android, and Simulator implementations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class SensorState:
    """
    Multi-dimensional sensor vector S(t) — Eq. (1) in paper.
    Matches exactly:  S(t) = [sGPS, sALT, sBARO, sTEMP, sBAT]
    """
    lat:            float           # degrees
    lon:            float           # degrees
    altitude_m:     float           # metres  (GPS-derived or baro formula)
    pressure_hpa:   float           # hPa     (real sensor or API-sourced)
    temp_c:         float           # °C      (real or battery-temp proxy)
    battery_pct:    float           # 0.0 – 1.0
    timestamp:      float = field(default_factory=time.time)  # Unix epoch
    source_flags:   dict  = field(default_factory=dict)
    # source_flags keys: "gps", "altitude", "pressure", "temp", "battery"
    # source_flags values: "real" | "gps_derived" | "api" | "battery_proxy"
    #                      | "baro_formula" | "simulated"

    def copy(self) -> "SensorState":
        import copy
        return copy.deepcopy(self)

    def __str__(self) -> str:
        return (
            f"SensorState(lat={self.lat:.5f}, lon={self.lon:.5f}, "
            f"alt={self.altitude_m:.1f}m, baro={self.pressure_hpa:.2f}hPa, "
            f"temp={self.temp_c:.1f}°C, bat={self.battery_pct*100:.0f}%, "
            f"src={self.source_flags})"
        )


class SensorProvider(ABC):
    """Abstract base — implement get_state() for each platform."""

    @abstractmethod
    def get_state(self) -> SensorState:
        """Return the current physical sensor state."""
        ...

    def is_healthy(self) -> bool:
        """
        Optional liveness check. Override in hardware providers to test
        I2C bus / GPS socket / Termux-API availability before starting
        the CVW monitor.
        """
        try:
            self.get_state()
            return True
        except Exception as exc:
            logger.warning("SensorProvider health check failed: %s", exc)
            return False


def get_provider(platform: Optional[str] = None) -> SensorProvider:
    """
    Factory: returns the right SensorProvider for the running platform.
    Auto-detects when platform=None; override with PLATFORM env var
    or explicit argument.
    """
    from config import PLATFORM as CFG_PLATFORM
    p = (platform or CFG_PLATFORM).lower()

    if p == "rpi":
        from sensor_rpi import RPiSensorProvider
        logger.info("Using RPi5 hardware sensor provider")
        return RPiSensorProvider()

    elif p == "android":
        from sensor_android import AndroidSensorProvider
        logger.info("Using Android/Termux sensor provider")
        return AndroidSensorProvider()

    elif p == "sim":
        from sensor_simulator import SimulatedSensorProvider
        logger.info("Using simulated sensor provider")
        return SimulatedSensorProvider()

    else:
        raise ValueError(
            f"Unknown platform '{p}'. Set PLATFORM=rpi|android|sim"
        )
