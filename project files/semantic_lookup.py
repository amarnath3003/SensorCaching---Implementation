# semantic_lookup.py
# Hash → embedding → CVW three-stage pipeline

"""
semantic_lookup.py — Three-stage semantic lookup pipeline
  Stage 1: Exact hash match          O(1)
  Stage 2: Embedding cosine sim      O(N·d)   sentence-transformers
  Stage 3: CVW validity check        O(D)

Also handles:
  • Category classification  (lightweight embedding sim vs. prototypes)
  • Haversine GPS delta
  • GPSonly-threshold baseline mode (C1-G)

Paper refs: Section IV-A (multi-tier lookup), Eq. (3), (4), (5)
"""

from __future__ import annotations
import math
import logging
import time
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from sensor_provider import SensorState
from cache_store import CacheStore, CacheEntry
import config

logger = logging.getLogger(__name__)


class LookupResult(Enum):
    HASH_HIT      = auto()   # Stage 1 exact match, CVW valid
    SEMANTIC_HIT  = auto()   # Stage 2 cosine match, CVW valid
    CVW_MISS      = auto()   # Semantic match found but CVW violated → forced miss
    CACHE_MISS    = auto()   # No match in any stage


# ─── Haversine ────────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """GPS displacement in metres — Eq. (15) of paper."""
    R = 6_371_000
    p1, p2   = math.radians(lat1), math.radians(lat2)
    dp       = math.radians(lat2 - lat1)
    dl       = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── CVW validity check ───────────────────────────────────────────────────────

def cvw_valid(entry: CacheEntry, s_now: SensorState) -> bool:
    """
    Eq. (4):  ∀d ∈ {GPS,ALT,BARO,TEMP}: |S_d(t) - S_d^insert| ≤ V_d_i
    Returns True if ALL dimensions are within tolerance.
    """
    s_ins = entry.sensor_snapshot
    v     = entry.cvw_envelope   # already scaled by alpha at insert time

    delta_gps  = haversine_m(s_ins.lat, s_ins.lon, s_now.lat, s_now.lon)
    delta_alt  = abs(s_now.altitude_m   - s_ins.altitude_m)
    delta_baro = abs(s_now.pressure_hpa - s_ins.pressure_hpa)
    delta_temp = abs(s_now.temp_c       - s_ins.temp_c)

    return (
        delta_gps  <= v["gps"]  and
        delta_alt  <= v["alt"]  and
        delta_baro <= v["baro"] and
        delta_temp <= v["temp"]
    )


def cvw_delta(entry: CacheEntry, s_now: SensorState) -> dict:
    """Return per-dimension drift deltas (for metrics logging)."""
    s_ins = entry.sensor_snapshot
    return {
        "gps_m":  haversine_m(s_ins.lat, s_ins.lon, s_now.lat, s_now.lon),
        "alt_m":  abs(s_now.altitude_m   - s_ins.altitude_m),
        "baro":   abs(s_now.pressure_hpa - s_ins.pressure_hpa),
        "temp":   abs(s_now.temp_c       - s_ins.temp_c),
    }


# ─── Category classifier ──────────────────────────────────────────────────────

class CategoryClassifier:
    """
    Maps a query string to one of the six CVW categories by computing
    cosine similarity against pre-embedded prototype queries.
    Latency < 1 ms (reuses the already-loaded embedding model).
    """

    def __init__(self, model: SentenceTransformer):
        self._model      = model
        self._prototypes = {}   # category → np.ndarray
        self._load_prototypes()

    def _load_prototypes(self):
        for cat, proto in config.CATEGORY_PROTOTYPES.items():
            self._prototypes[cat] = self._model.encode(
                proto, normalize_embeddings=True
            )
        logger.info("CategoryClassifier loaded %d prototypes", len(self._prototypes))

    def classify(self, query: str, embedding: Optional[np.ndarray] = None) -> str:
        """
        Returns the category name with highest cosine sim to the query.
        If embedding is already computed, pass it in to skip re-encoding.
        """
        if embedding is None:
            embedding = self._model.encode(query, normalize_embeddings=True)

        best_cat   = "gen_knowledge"
        best_score = -1.0
        for cat, proto_emb in self._prototypes.items():
            score = float(np.dot(embedding, proto_emb))
            if score > best_score:
                best_score = score
                best_cat   = cat

        logger.debug("Classified '%s...' → %s (%.3f)", query[:30], best_cat, best_score)
        return best_cat


# ─── Main lookup engine ───────────────────────────────────────────────────────

class SemanticLookup:
    """
    Three-stage lookup (Section IV-A):
      1. Hash  → O(1)
      2. Cosine sim with threshold σ → O(N·d)
      3. CVW check → O(D)

    Additional modes:
      gps_only=True   → Stage 3 uses GPS threshold only  (C1-G baseline)
      cvw=False       → Skip Stage 3 entirely            (C1 baseline)
    """

    def __init__(
        self,
        cache:         CacheStore,
        cvw_enabled:   bool  = True,
        gps_only:      bool  = False,
        model_name:    str   = config.EMBEDDING_MODEL,
        sim_threshold: float = config.SIMILARITY_THRESHOLD,
    ):
        self._cache         = cache
        self._cvw_enabled   = cvw_enabled
        self._gps_only      = gps_only
        self._threshold     = sim_threshold

        logger.info("Loading embedding model: %s", model_name)
        t0 = time.perf_counter()
        self._model         = SentenceTransformer(model_name)
        logger.info("Embedding model loaded in %.1f s", time.perf_counter() - t0)

        self._classifier = CategoryClassifier(self._model)

    # ── public ────────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Return normalised embedding. ~3 ms on CPU."""
        return self._model.encode(text, normalize_embeddings=True)

    def classify(self, query: str, embedding: Optional[np.ndarray] = None) -> str:
        return self._classifier.classify(query, embedding)

    def lookup(
        self,
        query:     str,
        s_now:     SensorState,
        alpha:     float = 1.0,
    ) -> Tuple[LookupResult, Optional[CacheEntry], dict]:
        """
        Full three-stage lookup.  Returns:
          (result_enum, entry_or_None, metadata_dict)

        metadata_dict contains timing, stage hit, similarity score, cvw_delta.
        """
        meta: dict = {
            "stage":        None,
            "sim_score":    None,
            "cvw_delta":    None,
            "embed_ms":     0.0,
            "lookup_ms":    0.0,
        }
        t_start = time.perf_counter()

        # ── Stage 1: exact hash ───────────────────────────────────────────────
        entry = self._cache.lookup_hash(query)
        if entry is not None:
            if self._should_accept(entry, s_now, alpha):
                meta.update(stage=1, cvw_delta=cvw_delta(entry, s_now))
                meta["lookup_ms"] = (time.perf_counter() - t_start) * 1000
                return LookupResult.HASH_HIT, entry, meta
            else:
                meta["cvw_delta"] = cvw_delta(entry, s_now)
                meta["lookup_ms"] = (time.perf_counter() - t_start) * 1000
                return LookupResult.CVW_MISS, None, meta

        # ── Stage 2: embedding similarity ────────────────────────────────────
        t_emb = time.perf_counter()
        q_emb = self.encode(query)
        meta["embed_ms"] = (time.perf_counter() - t_emb) * 1000

        best_entry: Optional[CacheEntry] = None
        best_score: float                = -1.0

        for _, entry in self._cache.all_entries():
            if not entry.is_valid:
                continue
            score = float(np.dot(q_emb, entry.embedding))
            if score > best_score:
                best_score = score
                best_entry = entry

        meta["sim_score"] = best_score

        if best_score >= self._threshold and best_entry is not None:
            if self._should_accept(best_entry, s_now, alpha):
                best_entry.touch()
                meta.update(stage=2, cvw_delta=cvw_delta(best_entry, s_now))
                meta["lookup_ms"] = (time.perf_counter() - t_start) * 1000
                return LookupResult.SEMANTIC_HIT, best_entry, meta
            else:
                meta["cvw_delta"] = cvw_delta(best_entry, s_now)
                meta["lookup_ms"] = (time.perf_counter() - t_start) * 1000
                return LookupResult.CVW_MISS, None, meta

        meta["lookup_ms"] = (time.perf_counter() - t_start) * 1000
        return LookupResult.CACHE_MISS, None, meta

    # ── private ───────────────────────────────────────────────────────────────

    def _should_accept(
        self,
        entry: CacheEntry,
        s_now: SensorState,
        alpha: float,
    ) -> bool:
        """
        Eq. (5): Hit ⟺ sem_sim ∧ CVW_valid
        If cvw_enabled=False → always accept (C1 baseline).
        If gps_only=True     → only GPS dimension checked (C1-G baseline).
        """
        if not self._cvw_enabled:
            return True

        if self._gps_only:
            s_ins      = entry.sensor_snapshot
            delta_gps  = haversine_m(s_ins.lat, s_ins.lon, s_now.lat, s_now.lon)
            threshold  = config.CVW_THRESHOLDS[entry.category]["gps"] * alpha
            return delta_gps <= threshold

        # Scale CVW envelope by current alpha (UCTR may differ from insertion-time alpha)
        # We re-compute rather than trusting entry.cvw_envelope to handle UCTR changes
        base = config.CVW_THRESHOLDS.get(entry.category, config.CVW_THRESHOLDS["gen_knowledge"])
        scaled_v = {k: v * alpha for k, v in base.items()}

        s_ins      = entry.sensor_snapshot
        delta_gps  = haversine_m(s_ins.lat, s_ins.lon, s_now.lat, s_now.lon)
        delta_alt  = abs(s_now.altitude_m   - s_ins.altitude_m)
        delta_baro = abs(s_now.pressure_hpa - s_ins.pressure_hpa)
        delta_temp = abs(s_now.temp_c       - s_ins.temp_c)

        return (
            delta_gps  <= scaled_v["gps"]  and
            delta_alt  <= scaled_v["alt"]  and
            delta_baro <= scaled_v["baro"] and
            delta_temp <= scaled_v["temp"]
        )
