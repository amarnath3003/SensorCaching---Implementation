# cache_store.py
# Annotated cache store with CVW envelopes

"""
cache_store.py — Annotated semantic cache with CVW envelopes
Implements Eq. (2), (12), and the eviction policy from Section IV-A.

Entry layout  (Eq. 12):
    (query_hash, embedding, response, S_insert, V_i, category, alpha, flags)

Eviction policy (Section IV-C):
    1. Physically invalid entries (CVW violated) → highest priority
    2. Remaining → LRU
"""

from __future__ import annotations
import hashlib
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sensor_provider import SensorState
import config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single annotated cache entry."""
    query_hash:      str            # SHA-256 of raw query string
    embedding:       np.ndarray     # shape (384,) from all-MiniLM-L6-v2
    response:        str            # LLM-generated response text
    sensor_snapshot: SensorState    # S_insert at time of insertion
    cvw_envelope:    dict           # {"gps":50, "alt":30, "baro":5, "temp":10}
    category:        str            # from CVW_THRESHOLDS keys
    alpha:           float  = 1.0   # UCTR scale factor applied at insertion
    pre_warm:        bool   = False  # PRE_WARM flag (Section IV-E)
    insertion_time:  float  = field(default_factory=time.time)
    last_access:     float  = field(default_factory=time.time)
    hit_count:       int    = 0
    is_valid:        bool   = True   # set False by CVWMonitor before eviction

    # Optional TTL (seconds) — for C1-T baseline comparison only
    ttl_s:           Optional[float] = None

    def touch(self):
        self.last_access = time.time()
        self.hit_count  += 1

    def is_ttl_expired(self) -> bool:
        if self.ttl_s is None:
            return False
        return (time.time() - self.insertion_time) > self.ttl_s


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


class CacheStore:
    """
    Thread-safe (GIL-level) annotated cache store.
    Capacity: config.CACHE_MAX_ENTRIES (500).

    Public API
    ──────────
    insert(query, embedding, response, sensor_state, category, alpha, pre_warm, ttl_s)
    lookup_hash(query) → Optional[CacheEntry]
    evict(entry_id, reason)
    mark_invalid(entry_id)
    all_entries()     → list[tuple[str, CacheEntry]]
    stats()           → dict
    """

    def __init__(self, max_entries: int = config.CACHE_MAX_ENTRIES):
        self._max   = max_entries
        # OrderedDict: key = query_hash, preserves insertion order for LRU
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._eviction_log: list[dict] = []
        logger.info("CacheStore initialized (max=%d)", max_entries)

    # ── insertion ─────────────────────────────────────────────────────────────

    def insert(
        self,
        query:          str,
        embedding:      np.ndarray,
        response:       str,
        sensor_state:   SensorState,
        category:       str,
        alpha:          float = 1.0,
        pre_warm:       bool  = False,
        ttl_s:          Optional[float] = None,
    ) -> str:
        """
        Insert a new cache entry.  Returns the query_hash (entry ID).
        If capacity is exceeded, evict using priority policy.
        """
        qhash = _query_hash(query)

        # Deduplicate: update hit count and response if exact hash already present
        if qhash in self._store:
            entry = self._store[qhash]
            entry.response        = response   # refresh
            entry.sensor_snapshot = sensor_state.copy()
            entry.alpha           = alpha
            entry.is_valid        = True
            self._store.move_to_end(qhash)
            return qhash

        # CVW envelope = base thresholds × alpha
        base = config.CVW_THRESHOLDS.get(category, config.CVW_THRESHOLDS["gen_knowledge"])
        cvw  = {k: v * alpha for k, v in base.items()}

        entry = CacheEntry(
            query_hash      = qhash,
            embedding       = embedding.copy(),
            response        = response,
            sensor_snapshot = sensor_state.copy(),
            cvw_envelope    = cvw,
            category        = category,
            alpha           = alpha,
            pre_warm        = pre_warm,
            ttl_s           = ttl_s,
        )

        # Capacity management before adding
        if len(self._store) >= self._max:
            self._evict_one()

        self._store[qhash] = entry
        logger.debug("Inserted [%s] cat=%s α=%.2f", qhash[:8], category, alpha)
        return qhash

    # ── lookup ────────────────────────────────────────────────────────────────

    def lookup_hash(self, query: str) -> Optional[CacheEntry]:
        """O(1) exact hash lookup — Stage 1 of three-stage pipeline."""
        qhash = _query_hash(query)
        entry = self._store.get(qhash)
        if entry is None:
            return None
        if not entry.is_valid or entry.is_ttl_expired():
            self.evict(qhash, reason="stale_on_lookup")
            return None
        entry.touch()
        self._store.move_to_end(qhash)
        return entry

    # ── eviction ──────────────────────────────────────────────────────────────

    def mark_invalid(self, entry_id: str):
        """Called by CVWMonitor when drift exceeds threshold."""
        if entry_id in self._store:
            self._store[entry_id].is_valid = False

    def evict(self, entry_id: str, reason: str = "lru"):
        """Remove entry from store, log eviction."""
        if entry_id in self._store:
            entry = self._store.pop(entry_id)
            self._eviction_log.append({
                "hash":     entry_id[:8],
                "category": entry.category,
                "reason":   reason,
                "age_s":    time.time() - entry.insertion_time,
                "hits":     entry.hit_count,
                "ts":       time.time(),
            })
            logger.debug("Evicted [%s] reason=%s", entry_id[:8], reason)

    def _evict_one(self):
        """
        Priority eviction:
        1. First invalid entry (physically violated or TTL expired)
        2. LRU (oldest last_access)
        """
        # Pass 1: invalid entries
        for qhash, entry in self._store.items():
            if not entry.is_valid or entry.is_ttl_expired():
                self.evict(qhash, reason="priority_invalid")
                return

        # Pass 2: LRU — OrderedDict head = oldest insertion
        lru_key = next(iter(self._store))
        self.evict(lru_key, reason="lru")

    # ── inspection ────────────────────────────────────────────────────────────

    def all_entries(self) -> list[tuple[str, CacheEntry]]:
        return list(self._store.items())

    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        total   = len(self._store)
        invalid = sum(1 for e in self._store.values() if not e.is_valid)
        prewarm = sum(1 for e in self._store.values() if e.pre_warm)
        return {
            "size":             total,
            "invalid_pending":  invalid,
            "pre_warm_entries": prewarm,
            "evictions_total":  len(self._eviction_log),
            "eviction_reasons": _count_reasons(self._eviction_log),
        }


def _count_reasons(log: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for e in log:
        counts[e["reason"]] = counts.get(e["reason"], 0) + 1
    return counts
