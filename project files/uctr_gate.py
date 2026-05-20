# uctr_gate.py
# Cost-minimisation α* solver (Eq. 7 from paper)

import numpy as np
from scipy.optimize import minimize_scalar

def compute_alpha_star(battery_pct: float, urgency: str,
                       sensor_drift: dict, cvw_thresholds: dict,
                       C_stale_ratio: float = 10.0, gamma: float = 5.0) -> float:
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    def p_stale(alpha):
        max_norm_drift = max(
            sensor_drift.get(d, 0) / (alpha * cvw_thresholds.get(d, 1e9) + 1e-9)
            for d in ["gps", "alt", "baro", "temp"]
        )
        return sigmoid(gamma * (max_norm_drift - 1))
    def chr_estimate(alpha):
        # monotone approximation from running hit history
        return min(0.95, 0.42 * alpha)   # calibrate from your first 100 queries
    def J(alpha):
        ps = p_stale(alpha)
        return ps * C_stale_ratio + (1 - chr_estimate(alpha))
    result = minimize_scalar(J, bounds=(0.1, 3.0), method='bounded')
    return float(result.x)