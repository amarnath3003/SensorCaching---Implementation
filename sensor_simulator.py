"""
sensor_simulator.py — Deterministic Gaussian-drift sensor simulator
Exactly matches Section V-C of the paper.
τ_sample = 5 s,  three scenarios: static / moderate / high_vol

Usage:
    sim = SimulatedSensorProvider(scenario="high_vol", seed=42)
    state = sim.get_state()   # call every τ_sample seconds
"""

from __future__ import annotations
import math
import time
import numpy as np
import logging

from sensor_provider import SensorProvider, SensorState
import config

logger = logging.getLogger(__name__)


# Starting position: representative alpine / wilderness location (Vaud Alps area)
_START_LAT    =  46.852
_START_LON    =   7.823
_START_ALT    = 2_800.0   # metres
_START_BARO   =   720.0   # hPa  (roughly correct for 2800 m ASL)
_START_TEMP   =     8.0   # °C


class SimulatedSensorProvider(SensorProvider):
    """
    Gaussian-drift simulator as described in Section V-C.
    Each call to get_state() advances one τ_sample step.

    Drift per step (std):
      static:   GPS 0.3 m,  ALT 0.1 m,  BARO 0.05 hPa,  TEMP 0.05°C
      moderate: GPS 5.0 m,  ALT 2.0 m,  BARO 0.40 hPa,  TEMP 0.30°C
      high_vol: GPS 15 m,   ALT 8.0 m,  BARO 1.20 hPa,  TEMP 0.80°C
    """

    def __init__(
        self,
        scenario:     str   = "static",
        seed:         int   = 42,
        battery_start: float = 1.0,
        force_low_battery_at: int = -1,   # step index to drop battery to 0.05
    ):
        self._rng      = np.random.default_rng(seed)
        self._params   = config.SIM_PARAMS[scenario]
        self._scenario = scenario

        # Current physical state (cumulative)
        self._lat   = _START_LAT
        self._lon   = _START_LON
        self._alt   = _START_ALT
        self._baro  = _START_BARO
        self._temp  = _START_TEMP
        self._bat   = battery_start

        self._step                = 0
        self._force_low_bat_at    = force_low_battery_at
        self._start_wall          = time.time()

        logger.info("SimulatedSensorProvider: scenario=%s seed=%d", scenario, seed)

    def _metres_to_lat_delta(self, metres: float) -> float:
        return metres / 111_320.0

    def _metres_to_lon_delta(self, metres: float, lat: float) -> float:
        return metres / (111_320.0 * math.cos(math.radians(lat)))

    def _drift_battery(self) -> float:
        """Simple linear discharge ~4 h session."""
        return max(0.0, self._bat - (1.0 / (4 * 3600 / config.TAU_SAMPLE_S)))

    def get_state(self) -> SensorState:
        p = self._params

        # Gaussian displacement in metres → lat/lon degrees
        dx_m = self._rng.normal(0, p["gps_std"])
        dy_m = self._rng.normal(0, p["gps_std"])
        self._lat  += self._metres_to_lat_delta(dy_m)
        self._lon  += self._metres_to_lon_delta(dx_m, self._lat)

        # Altitude, baro, temp drift
        self._alt  += self._rng.normal(0, p["alt_std"])
        self._baro += self._rng.normal(0, p["baro_std"])
        self._temp += self._rng.normal(0, p["temp_std"])

        # Clamp to physical bounds
        self._baro = max(300.0, min(1100.0, self._baro))
        self._temp = max(-40.0, min(60.0, self._temp))

        # Battery
        if (self._force_low_bat_at >= 0
                and self._step >= self._force_low_bat_at):
            self._bat = 0.05   # forced critically low (C5 scenario)
        else:
            self._bat = self._drift_battery()

        self._step += 1

        return SensorState(
            lat          = self._lat,
            lon          = self._lon,
            altitude_m   = self._alt,
            pressure_hpa = self._baro,
            temp_c       = self._temp,
            battery_pct  = self._bat,
            source_flags = {
                "gps":      "simulated",
                "altitude": "simulated",
                "pressure": "simulated",
                "temp":     "simulated",
                "battery":  "simulated",
            }
        )
