"""
config.py — SensorAware-Cache global configuration
All constants match Table I, Table III, and Section V of the paper.
Edit PLATFORM and MODEL_PATH before running on device.
"""

import os

# ─── Platform ────────────────────────────────────────────────────────────────
# Set via env: PLATFORM=rpi  or  PLATFORM=android
PLATFORM: str = os.environ.get("PLATFORM", "rpi")   # "rpi" | "android" | "sim"

# ─── LLM Backend ─────────────────────────────────────────────────────────────
MODEL_PATH: str = os.environ.get(
    "LLAMA_MODEL_PATH",
    "./models/gemma-3-1B-IT-Q4_K_M.gguf"
)
LLAMA_CLI_PATH: str = os.environ.get(
    "LLAMA_CLI",
    "./llama.cpp/build/bin/llama-cli"
)
LLAMA_N_THREADS: int    = int(os.environ.get("LLAMA_N_THREADS", "4"))
LLAMA_MAX_TOKENS: int   = 256

# ─── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL: str    = "all-MiniLM-L6-v2"   # 22 MB, fully offline
EMBEDDING_DIM: int      = 384
SIMILARITY_THRESHOLD: float = 0.85             # σ in paper Eq. (3)

# ─── Cache Store ─────────────────────────────────────────────────────────────
CACHE_MAX_ENTRIES: int  = 500                  # Section V-A

# ─── CVW Thresholds  — Table I ───────────────────────────────────────────────
# Units: GPS→metres, ALT→metres, BARO→hPa, TEMP→°C
# 1e9 represents ∞ (no physical constraint)
CVW_THRESHOLDS: dict = {
    "navigation":    {"gps": 50,   "alt": 30,   "baro": 5,   "temp": 10},
    "weather":       {"gps": 200,  "alt": 100,  "baro": 3,   "temp": 5},
    "first_aid":     {"gps": 1e9,  "alt": 200,  "baro": 1e9, "temp": 8},
    "resource_mgmt": {"gps": 1e9,  "alt": 1e9,  "baro": 1e9, "temp": 1e9},
    "threat_assess": {"gps": 100,  "alt": 1e9,  "baro": 1e9, "temp": 1e9},
    "gen_knowledge": {"gps": 1e9,  "alt": 1e9,  "baro": 1e9, "temp": 1e9},
}

# Prototype queries for lightweight category classification (< 1 ms)
CATEGORY_PROTOTYPES: dict = {
    "navigation":    "What is the safest route down from here?",
    "weather":       "What does the current weather mean for my safety?",
    "first_aid":     "How do I treat a wound in the wilderness?",
    "resource_mgmt": "How do I find water in the wild?",
    "threat_assess": "Are there bears in this area?",
    "gen_knowledge": "What are Leave No Trace principles?",
}

# ─── CVW Monitor ─────────────────────────────────────────────────────────────
TAU_SAMPLE_S: float = 5.0          # background sweep interval (seconds)

# ─── UCTR Gate — Section II-C ────────────────────────────────────────────────
BATTERY_CRITICAL_THRESHOLD: float  = 0.15   # Bstate < 0.15 → battery conservation
C_STALE_RATIO: float               = 10.0   # Cstale / Cmiss (wilderness domain)
GAMMA: float                       = 5.0    # sigmoid sharpness
ALPHA_BOUNDS: tuple                = (0.1, 3.0)

# ─── Evaluation Harness — Table III ──────────────────────────────────────────
EVAL_CONFIGS: dict = {
    "C1":   {"scenario": "static",    "cvw": False, "uctr": False},
    "C1_T": {"scenario": "static",    "cvw": False, "uctr": False, "ttl": 30},
    "C1_G": {"scenario": "static",    "cvw": False, "uctr": False, "gps_only": True},
    "C2":   {"scenario": "static",    "cvw": True,  "uctr": False},
    "C3":   {"scenario": "moderate",  "cvw": True,  "uctr": False},
    "C4":   {"scenario": "high_vol",  "cvw": True,  "uctr": True},
    "C5":   {"scenario": "moderate",  "cvw": True,  "uctr": True,
              "force_low_battery_at": 480},
}
N_SEEDS: int = 5

# Simulation drift parameters (Gaussian)
SIM_PARAMS: dict = {
    "static":   {"gps_std": 0.3, "alt_std": 0.1, "baro_std": 0.05, "temp_std": 0.05},
    "moderate": {"gps_std": 5.0, "alt_std": 2.0, "baro_std": 0.4,  "temp_std": 0.3},
    "high_vol": {"gps_std": 15.0,"alt_std": 8.0, "baro_std": 1.2,  "temp_std": 0.8},
}

# ─── Paths ────────────────────────────────────────────────────────────────────
CORPUS_PATH: str    = "./corpus/wilderness_1200.json"
PREWARM_PATH: str   = "./corpus/prewarm_entries.json"
RESULTS_DIR: str    = "./results"

# ─── Android / Termux ────────────────────────────────────────────────────────
OPEN_METEO_BASE: str  = "https://api.open-meteo.com/v1/forecast"
TERMUX_LOC_TIMEOUT_S: int = 30     # max seconds for termux-location

# ─── RPi ──────────────────────────────────────────────────────────────────────
BME280_I2C_ADDR: int  = 0x76
I2C_BUS: int          = 1
GPSD_HOST: str        = "localhost"
GPSD_PORT: int        = 2947
GPS_MAX_WAIT_S: int   = 10
