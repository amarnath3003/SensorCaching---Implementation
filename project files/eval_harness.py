# eval_harness.py
# Runs all 5 configs (C1–C5), 5 seeds

CONFIGS = {
    "C1": {"scenario": "static",    "cvw": False, "uctr": False},
    "C2": {"scenario": "static",    "cvw": True,  "uctr": False},
    "C3": {"scenario": "moderate",  "cvw": True,  "uctr": False},
    "C4": {"scenario": "high_vol",  "cvw": True,  "uctr": True},
    "C5": {"scenario": "moderate",  "cvw": True,  "uctr": True,
           "force_low_battery_at": 480},
}
N_SEEDS = 5