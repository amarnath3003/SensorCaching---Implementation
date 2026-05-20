"""
uctr_gate.py — Urgency-Coupled Threshold Relaxation (UCTR)
Solves the per-step cost minimisation problem — Eq. (7) from paper.

J(α) = p_stale(α,Δ)·C_stale + (1 - CHR(α))·C_miss
α* = argmin_{α≥0} J(α)

Battery Conservation  (α* > 1): Bstate < 0.15  → α* ≈ 1.5
Urgency Escalation    (α* < 1): high-urgency    → α* ≈ 0.5
Normal operation                               → α* ≈ 1.0

Paper refs: Section II-C, Eq. (6)–(8), Section IV-D
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

import config

logger = logging.getLogger(__name__)

# Running hit-rate estimator window
_CHR_WINDOW = 100   # use last 100 queries for CHR(alpha) estimation


class UCTRGate:
    """
    Stateful UCTR gate.  Tracks running CHR and solves α* at each query.

    Usage:
        gate = UCTRGate()
        alpha = gate.compute_alpha(
            battery_pct=0.10,
            urgency="high",
            sensor_drift={"gps": 45, "alt": 20, "baro": 3, "temp": 5},
            category="navigation"
        )
    """

    def __init__(
        self,
        c_stale_ratio: float = config.C_STALE_RATIO,
        gamma:         float = config.GAMMA,
        alpha_bounds:  tuple = config.ALPHA_BOUNDS,
        enabled:       bool  = True,
    ):
        self._c_ratio  = c_stale_ratio
        self._gamma    = gamma
        self._bounds   = alpha_bounds
        self._enabled  = enabled

        # Running hit/miss history for CHR(alpha) calibration
        self._hit_history: list[int] = []   # 1=hit, 0=miss
        logger.info(
            "UCTRGate init (enabled=%s, C_ratio=%.1f, γ=%.1f)",
            enabled, c_stale_ratio, gamma
        )

    # ── public ────────────────────────────────────────────────────────────────

    def compute_alpha(
        self,
        battery_pct:    float,
        urgency:        str,          # "normal" | "high"
        sensor_drift:   dict,         # {"gps": Δm, "alt": Δm, "baro": ΔhPa, "temp": Δ°C}
        category:       str,
        c_stale_ratio:  Optional[float] = None,
    ) -> float:
        """
        Solve Eq. (7) and return optimal α*.
        If UCTR is disabled returns 1.0 unconditionally.
        """
        if not self._enabled:
            return 1.0

        c_ratio = c_stale_ratio if c_stale_ratio is not None else self._c_ratio

        # Get category-specific CVW thresholds
        thresholds = config.CVW_THRESHOLDS.get(
            category, config.CVW_THRESHOLDS["gen_knowledge"]
        )

        def p_stale(alpha: float) -> float:
            """Sigmoid over max normalised drift margin — Eq. (8)."""
            max_norm = max(
                sensor_drift.get(d, 0.0) / (alpha * thresholds.get(d, 1e9) + 1e-9)
                for d in ("gps", "alt", "baro", "temp")
            )
            return _sigmoid(self._gamma * (max_norm - 1.0))

        def chr_estimate(alpha: float) -> float:
            """
            Monotone CHR(α) approximation calibrated from running history.
            Base intercept 0.42 matches paper Table IV C1/C2 result.
            """
            base_chr = self._running_chr()
            return min(0.95, base_chr * alpha)

        def J(alpha: float) -> float:
            ps = p_stale(alpha)
            return ps * c_ratio + (1.0 - chr_estimate(alpha))

        result = minimize_scalar(J, bounds=self._bounds, method="bounded")
        alpha_star = float(result.x)

        # Hard override for extreme states (Section II-C explicit cases)
        if battery_pct < config.BATTERY_CRITICAL_THRESHOLD:
            # Battery conservation: expand thresholds
            alpha_star = max(alpha_star, 1.5)
            logger.debug("UCTR: low battery (%.0f%%) → α*=%.2f", battery_pct*100, alpha_star)

        if urgency.lower() == "high":
            # Urgency escalation: force fresh inference
            alpha_star = min(alpha_star, 0.5)
            logger.debug("UCTR: high urgency → α*=%.2f", alpha_star)

        return alpha_star

    def record_outcome(self, was_hit: bool):
        """Update running CHR estimator after each query."""
        self._hit_history.append(1 if was_hit else 0)
        if len(self._hit_history) > _CHR_WINDOW:
            self._hit_history.pop(0)

    def _running_chr(self) -> float:
        if not self._hit_history:
            return 0.42   # paper-calibrated prior
        return sum(self._hit_history) / len(self._hit_history)

    # ── utilities ─────────────────────────────────────────────────────────────

    def alpha_for_config(
        self,
        battery_pct: float,
        urgency:     str,
    ) -> float:
        """
        Simplified alpha computation using zero sensor drift assumption
        (when drift is not yet available at query time).
        """
        zero_drift = {"gps": 0.0, "alt": 0.0, "baro": 0.0, "temp": 0.0}
        return self.compute_alpha(battery_pct, urgency, zero_drift, "gen_knowledge")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))
