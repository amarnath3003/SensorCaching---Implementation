# metrics_logger.py
# CHR, VHR, SRR, TTFT, CVW-inv counter

"""
metrics_logger.py — SensorAware-Cache evaluation metrics
Tracks: CHR, VHR, SRR, TTFT-H, TTFT-M, CVW invalidation count.
Exports to JSON for table generation matching paper Table IV.

Paper ref: Section V-E, Table IV
"""

from __future__ import annotations
import json
import logging
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ─── Per-query event ──────────────────────────────────────────────────────────

@dataclass
class QueryEvent:
    """Single query record for post-hoc SRR validation."""
    query_index:    int
    query_text:     str
    category:       str
    urgency:        str
    result:         str            # "hash_hit"|"semantic_hit"|"cvw_miss"|"cache_miss"
    ttft_ms:        float
    alpha:          float
    sensor_at_query: dict          # serialisable snapshot
    sensor_at_insert: Optional[dict] = None
    cvw_delta:       Optional[dict]  = None
    response_snippet: str          = ""
    is_stale:        Optional[bool] = None    # set during manual validation


# ─── Main logger ──────────────────────────────────────────────────────────────

class MetricsLogger:
    """
    Accumulates per-query events and computes aggregate metrics.

    Usage:
        ml = MetricsLogger(config_name="C4", seed=0)
        ml.record(...)
        ml.log_cvw_invalidation()
        results = ml.summary()
        ml.export_json("results/C4_seed0.json")
    """

    def __init__(self, config_name: str = "C1", seed: int = 0):
        self.config_name = config_name
        self.seed        = seed
        self._events: list[QueryEvent] = []
        self._cvw_inv_count:  int   = 0
        self._ttft_hits:  list[float] = []
        self._ttft_misses: list[float] = []
        self._start_time: float = time.time()
        self._category_stats: dict[str, dict] = defaultdict(
            lambda: {"hits": 0, "misses": 0, "cvw_misses": 0}
        )

    # ── recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        query_index:      int,
        query_text:       str,
        category:         str,
        urgency:          str,
        result:           str,      # LookupResult.name.lower()
        ttft_ms:          float,
        alpha:            float,
        sensor_now:       "SensorState",
        sensor_insert:    Optional["SensorState"] = None,
        cvw_delta:        Optional[dict] = None,
        response_snippet: str = "",
    ):
        is_hit  = result in ("hash_hit", "semantic_hit")
        is_miss = result == "cache_miss"

        # Track TTFT
        if is_hit:
            self._ttft_hits.append(ttft_ms)
        else:
            self._ttft_misses.append(ttft_ms)

        # Category breakdown
        cs = self._category_stats[category]
        if is_hit:
            cs["hits"] += 1
        elif result == "cvw_miss":
            cs["cvw_misses"] += 1
        else:
            cs["misses"] += 1

        event = QueryEvent(
            query_index       = query_index,
            query_text        = query_text[:100],
            category          = category,
            urgency           = urgency,
            result            = result,
            ttft_ms           = ttft_ms,
            alpha             = alpha,
            sensor_at_query   = _serialise_state(sensor_now),
            sensor_at_insert  = _serialise_state(sensor_insert) if sensor_insert else None,
            cvw_delta         = cvw_delta,
            response_snippet  = response_snippet[:120],
        )
        self._events.append(event)

    def log_cvw_invalidation(self, count: int = 1):
        """Call once per CVW monitor invalidation event."""
        self._cvw_inv_count += count

    def sync_cvw_count(self, monitor_count: int):
        """Overwrite from CVWMonitor.invalidation_count at end of run."""
        self._cvw_inv_count = monitor_count

    # ── aggregates ────────────────────────────────────────────────────────────

    def total_queries(self) -> int:
        return len(self._events)

    def cache_hit_rate(self) -> float:
        """CHR — fraction of queries served from cache."""
        n = self.total_queries()
        if n == 0:
            return 0.0
        hits = sum(1 for e in self._events if e.result in ("hash_hit", "semantic_hit"))
        return hits / n

    def valid_hit_rate(self) -> float:
        """
        VHR — CVW-valid hits / total queries.
        For CVW-disabled configs, VHR = CHR (all hits assumed valid).
        """
        return self.cache_hit_rate()   # CVW monitor pre-screens invalid entries

    def stale_response_rate(self) -> Optional[float]:
        """
        SRR — requires manual is_stale labels set by adjudicators.
        Returns None if no events have been labelled yet.
        """
        labelled = [e for e in self._events if e.is_stale is not None]
        if not labelled:
            return None
        stale = sum(1 for e in labelled if e.is_stale)
        return stale / len(labelled)

    def ttft_median_hit(self) -> float:
        if not self._ttft_hits:
            return 0.0
        return statistics.median(self._ttft_hits)

    def ttft_median_miss(self) -> float:
        if not self._ttft_misses:
            return 0.0
        return statistics.median(self._ttft_misses)

    def speedup(self) -> float:
        m = self.ttft_median_miss()
        h = self.ttft_median_hit()
        return m / h if h > 0 else 0.0

    def category_hit_rates(self) -> dict[str, float]:
        rates = {}
        for cat, cs in self._category_stats.items():
            total = cs["hits"] + cs["misses"] + cs["cvw_misses"]
            rates[cat] = cs["hits"] / total if total > 0 else 0.0
        return rates

    def summary(self) -> dict:
        """Returns dict matching Table IV columns."""
        return {
            "config":         self.config_name,
            "seed":           self.seed,
            "n_queries":      self.total_queries(),
            "CHR":            round(self.cache_hit_rate() * 100, 1),
            "VHR":            round(self.valid_hit_rate() * 100, 1),
            "SRR":            self.stale_response_rate(),   # None until labelled
            "CVW_inv":        self._cvw_inv_count,
            "TTFT_H_ms":      round(self.ttft_median_hit(), 1),
            "TTFT_M_ms":      round(self.ttft_median_miss(), 1),
            "speedup":        round(self.speedup(), 0),
            "category_CHR":   self.category_hit_rates(),
            "runtime_s":      round(time.time() - self._start_time, 1),
        }

    # ── export ────────────────────────────────────────────────────────────────

    def export_json(self, path: Optional[str] = None) -> str:
        """Export full event log + summary to JSON. Returns path written."""
        if path is None:
            Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
            path = os.path.join(
                config.RESULTS_DIR,
                f"{self.config_name}_seed{self.seed}.json"
            )

        payload = {
            "summary":  self.summary(),
            "events":   [_event_to_dict(e) for e in self._events],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        logger.info("Metrics exported → %s  (CHR=%.1f%%)", path, self.cache_hit_rate()*100)
        return path

    def print_summary(self):
        s = self.summary()
        print(
            f"\n{'─'*50}\n"
            f"Config: {s['config']}  seed={s['seed']}\n"
            f"  Queries:   {s['n_queries']}\n"
            f"  CHR:       {s['CHR']:.1f}%\n"
            f"  VHR:       {s['VHR']:.1f}%\n"
            f"  SRR:       {s['SRR']}\n"
            f"  CVW-inv:   {s['CVW_inv']}\n"
            f"  TTFT-H:    {s['TTFT_H_ms']:.1f} ms\n"
            f"  TTFT-M:    {s['TTFT_M_ms']:.1f} ms\n"
            f"  Speedup:   {s['speedup']:.0f}×\n"
            f"  Runtime:   {s['runtime_s']:.0f} s\n"
            f"{'─'*50}"
        )


# ─── Aggregate across seeds ───────────────────────────────────────────────────

def aggregate_seeds(results: list[dict]) -> dict:
    """
    Takes list of summary dicts from N seeds and returns mean ± std.
    Matches paper Section V-A: "mean values across runs, std ≤ ±1.2%".
    """
    if not results:
        return {}

    import statistics as st
    keys = ["CHR", "VHR", "CVW_inv", "TTFT_H_ms", "TTFT_M_ms"]
    out  = {"config": results[0]["config"], "n_seeds": len(results)}

    for k in keys:
        vals = [r[k] for r in results if r.get(k) is not None]
        if vals:
            out[f"{k}_mean"] = round(st.mean(vals), 2)
            out[f"{k}_std"]  = round(st.stdev(vals), 3) if len(vals) > 1 else 0.0

    return out


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _serialise_state(s) -> dict:
    if s is None:
        return {}
    return {
        "lat": s.lat, "lon": s.lon,
        "alt_m": s.altitude_m, "baro_hpa": s.pressure_hpa,
        "temp_c": s.temp_c, "bat_pct": s.battery_pct,
    }

def _event_to_dict(e: QueryEvent) -> dict:
    return {
        "idx":        e.query_index,
        "category":   e.category,
        "urgency":    e.urgency,
        "result":     e.result,
        "ttft_ms":    round(e.ttft_ms, 2),
        "alpha":      round(e.alpha, 3),
        "s_now":      e.sensor_at_query,
        "s_ins":      e.sensor_at_insert,
        "cvw_delta":  e.cvw_delta,
        "is_stale":   e.is_stale,
        "snippet":    e.response_snippet,
    }
