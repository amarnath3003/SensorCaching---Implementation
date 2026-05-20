"""
cvw_monitor.py — Background CVW invalidation sweep
Runs in a daemon thread at τ_sample = 5 s (config.TAU_SAMPLE_S).
Proactively evicts entries whose physical drift exceeds CVW thresholds.

Paper ref: Section IV-C, Eq. (16)
"""

from __future__ import annotations
import math
import threading
import time
import logging
from typing import Callable, Optional

from sensor_provider import SensorProvider, SensorState
from cache_store import CacheStore
from semantic_lookup import haversine_m
import config

logger = logging.getLogger(__name__)


class CVWMonitor:
    """
    Daemon thread that sweeps the cache every τ_sample seconds and
    evicts entries whose per-dimension drift exceeds their CVW envelope.

    Usage:
        monitor = CVWMonitor(cache, sensor)
        monitor.start()
        # ... run eval ...
        monitor.stop()
    """

    def __init__(
        self,
        cache:           CacheStore,
        sensor:          SensorProvider,
        tau_s:           float = config.TAU_SAMPLE_S,
        on_invalidation: Optional[Callable[[str, dict], None]] = None,
    ):
        self._cache    = cache
        self._sensor   = sensor
        self._tau      = tau_s
        self._callback = on_invalidation   # hook for metrics logger

        self._stop_event      = threading.Event()
        self._thread          = threading.Thread(
            target=self._sweep_loop, daemon=True, name="CVWMonitor"
        )

        # Public counters (read from metrics_logger)
        self.invalidation_count: int = 0
        self.sweep_count:        int = 0
        self.sweep_latencies_ms: list[float] = []

    def start(self):
        logger.info("CVWMonitor starting (τ=%.1fs)", self._tau)
        self._thread.start()

    def stop(self, timeout: float = 10.0):
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        logger.info("CVWMonitor stopped (sweeps=%d, invalidations=%d)",
                    self.sweep_count, self.invalidation_count)

    # ── sweep logic ───────────────────────────────────────────────────────────

    def _sweep_loop(self):
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                self._sweep_once()
            except Exception as exc:
                logger.error("CVW sweep error: %s", exc, exc_info=True)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.sweep_latencies_ms.append(elapsed_ms)
            self.sweep_count += 1

            # Sleep, but remain responsive to stop_event
            self._stop_event.wait(timeout=self._tau)

    def _sweep_once(self):
        try:
            s_now = self._sensor.get_state()
        except Exception as exc:
            logger.warning("CVW sweep: sensor read failed (%s), skipping", exc)
            return

        to_evict = []

        for entry_id, entry in self._cache.all_entries():
            if not entry.is_valid:
                to_evict.append((entry_id, "already_invalid"))
                continue

            if self._is_cvw_violated(entry, s_now):
                to_evict.append((entry_id, "cvw_violation"))
                delta = self._compute_delta(entry, s_now)
                logger.debug(
                    "CVW violated [%s] cat=%s delta=%s",
                    entry_id[:8], entry.category, delta
                )
                if self._callback:
                    self._callback(entry_id, delta)

        for entry_id, reason in to_evict:
            self._cache.mark_invalid(entry_id)
            self._cache.evict(entry_id, reason=reason)
            if reason == "cvw_violation":
                self.invalidation_count += 1

    def _is_cvw_violated(self, entry, s_now: SensorState) -> bool:
        """
        Eq. (16): ∃d such that Δ_d(t) > V_d_i × alpha
        V_d_i already stored scaled at insertion; we compare absolute deltas.
        """
        s_ins = entry.sensor_snapshot
        v     = entry.cvw_envelope   # scaled by alpha at insertion

        delta_gps  = haversine_m(s_ins.lat, s_ins.lon, s_now.lat, s_now.lon)
        delta_alt  = abs(s_now.altitude_m   - s_ins.altitude_m)
        delta_baro = abs(s_now.pressure_hpa - s_ins.pressure_hpa)
        delta_temp = abs(s_now.temp_c       - s_ins.temp_c)

        return (
            delta_gps  > v["gps"]  or
            delta_alt  > v["alt"]  or
            delta_baro > v["baro"] or
            delta_temp > v["temp"]
        )

    @staticmethod
    def _compute_delta(entry, s_now: SensorState) -> dict:
        s_ins = entry.sensor_snapshot
        return {
            "gps_m": haversine_m(s_ins.lat, s_ins.lon, s_now.lat, s_now.lon),
            "alt_m": abs(s_now.altitude_m   - s_ins.altitude_m),
            "baro":  abs(s_now.pressure_hpa - s_ins.pressure_hpa),
            "temp":  abs(s_now.temp_c       - s_ins.temp_c),
        }

    def avg_sweep_latency_ms(self) -> float:
        if not self.sweep_latencies_ms:
            return 0.0
        return sum(self.sweep_latencies_ms) / len(self.sweep_latencies_ms)
