# SensorAware-Cache — Complete Setup & Run Guide

> **Paper:** *SensorAware-Cache: Physical-World Telemetry-Driven Semantic Cache
> Invalidation for Offline On-Device LLM Inference*

This guide walks you through every step from a blank device to a fully running
evaluation that produces the metrics in **Table IV** of the paper.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Choose Your Path](#2-choose-your-path)
3. [Simulation Mode (No Hardware Required)](#3-simulation-mode-no-hardware-required)
4. [Raspberry Pi 5 — Full Hardware Setup](#4-raspberry-pi-5--full-hardware-setup)
   - 4.1 Hardware Bill of Materials
   - 4.2 OS Installation
   - 4.3 I²C + BME280 Setup
   - 4.4 GPS (gpsd) Setup
   - 4.5 Build llama.cpp
   - 4.6 Download the Model
   - 4.7 Python Dependencies
   - 4.8 Pre-flight Sensor Check
   - 4.9 Run Evaluation
5. [Android / Vivo V27 — Termux Setup](#5-android--vivo-v27--termux-setup)
   - 5.1 Sensor Reality on Vivo V27
   - 5.2 Termux Installation
   - 5.3 Build llama.cpp on Android
   - 5.4 Transfer the Model
   - 5.5 Python Dependencies
   - 5.6 Pre-flight Sensor Check
   - 5.7 Run Evaluation
6. [Running the Evaluation](#6-running-the-evaluation)
   - 6.1 All CLI Options
   - 6.2 Config Reference (C1–C5)
   - 6.3 Reading the Output
7. [Corpus Setup](#7-corpus-setup)
8. [Results & Metrics](#8-results--metrics)
9. [config.py Reference](#9-configpy-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Project Structure

```
sensoraware_cache/
│
├── main.py                 ← Entry point (all modes)
├── config.py               ← All constants (CVW thresholds, paths, alpha bounds)
│
├── sensor_provider.py      ← Abstract SensorState + factory get_provider()
├── sensor_rpi.py           ← RPi5: BME280 I²C + gpsd
├── sensor_android.py       ← Android: Termux:API + Open-Meteo fallback
├── sensor_simulator.py     ← Gaussian-drift simulator (paper Sec V-C)
│
├── cache_store.py          ← Annotated cache, LRU + priority eviction
├── semantic_lookup.py      ← 3-stage pipeline: hash → cosine → CVW
├── cvw_monitor.py          ← Background τ=5s invalidation sweep
├── uctr_gate.py            ← α* cost minimiser (Eq. 7)
├── llm_backend.py          ← llama.cpp subprocess wrapper + TTFT
├── prewarm.py              ← CVW-annotated pre-warming (Sec IV-E)
├── metrics_logger.py       ← CHR/VHR/SRR/TTFT/CVW-inv logger
├── eval_harness.py         ← C1–C5 × 5 seeds runner
│
├── corpus/
│   ├── wilderness_1200.json    ← Your 1200-query evaluation corpus
│   └── prewarm_entries.json    ← Optional custom pre-warm entries
│
├── models/
│   └── gemma-3-1B-IT-Q4_K_M.gguf
│
├── results/                ← JSON output per config per seed
│
├── requirements_rpi.txt
└── requirements_android.txt
```

---

## 2. Choose Your Path

| Goal | Path |
|------|------|
| Verify code works immediately, no hardware | → [Section 3: Simulation Mode](#3-simulation-mode-no-hardware-required) |
| Full paper-faithful hardware run | → [Section 4: RPi5](#4-raspberry-pi-5--full-hardware-setup) |
| Android real-device metrics | → [Section 5: Android/Termux](#5-android--vivo-v27--termux-setup) |
| Run with real sensors but no LLM | Add `--mock-llm` to any real command |

---

## 3. Simulation Mode (No Hardware Required)

The fastest path — runs the full evaluation harness on your laptop/PC
using the Gaussian-drift sensor simulator from Section V-C of the paper.

### 3.1 Requirements

Python 3.10+ required. Check with `python --version`.

```bash
git clone <your-repo-url> sensoraware_cache
cd sensoraware_cache
pip install sentence-transformers numpy scipy requests
```

### 3.2 Smoke Test (Mock LLM — no model download needed)

```bash
python main.py --mode sim --mock-llm --configs C1,C4 --seeds 1
```

Expected output:
```
Config: C4_smoketest  seed=0
  Queries:   10
  CHR:       ~40–60%     ← varies with tiny 12-query built-in corpus
  CVW-inv:   1–5
  TTFT-H:    ~1 ms
  TTFT-M:    ~510 ms     ← MockLLM simulates paper median
```

### 3.3 Full Simulated Evaluation (Real LLM)

Download the model first (see [Section 4.6](#46-download-the-model)), then:

```bash
python main.py --mode sim \
  --configs C1,C1_T,C1_G,C2,C3,C4,C5 \
  --seeds 5 \
  --output results/
```

This runs 7 configs × 5 seeds = 35 independent runs and writes
`results/table_IV.json` with the paper-matching aggregated metrics.

---

## 4. Raspberry Pi 5 — Full Hardware Setup

### 4.1 Hardware Bill of Materials

| Component | Spec | Purpose | Notes |
|-----------|------|---------|-------|
| Raspberry Pi 5 | 8 GB RAM | Main compute | 4 GB works, 8 GB preferred |
| BME280 breakout | I²C, 3.3 V | Pressure + Temp | Adafruit / AZ-Delivery / generic |
| USB GPS dongle | u-blox 7/8/9 | GPS + Altitude | Any NMEA-compatible; ~£10 |
| USB-SSD | 250 GB+ | Model + OS storage | **Required** — SD card model load takes 40+ s |
| Jumper wires | 4× female–female | I²C wiring | — |
| Power bank | 20 000 mAh | Field power | Optional for outdoor runs |

#### BME280 Wiring (I²C)

```
BME280 Pin    →    RPi5 Pin (GPIO header)
─────────────────────────────────────────
VCC  (3.3 V)  →    Pin 1   (3.3 V)
GND           →    Pin 6   (GND)
SCL           →    Pin 5   (GPIO 3 / SCL)
SDA           →    Pin 3   (GPIO 2 / SDA)
```

> **Address note:** Most BME280 breakouts default to `0x76`.
> If `SDO` pin is tied to VCC it uses `0x77`.
> Check with `sudo i2cdetect -y 1` after wiring.

---

### 4.2 OS Installation

**Recommended OS:** Ubuntu 23.10 Server 64-bit (or Raspberry Pi OS Bookworm 64-bit).

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Choose **Other general-purpose OS → Ubuntu → Ubuntu Server 23.10 (64-bit)**
3. Target: your **USB-SSD** (not SD card)
4. Set hostname, SSH key, Wi-Fi credentials in the Imager settings before flashing
5. Boot RPi5 from USB-SSD (may need to set USB boot order in `raspi-config`)

```bash
# First boot — update everything
sudo apt-get update && sudo apt-get upgrade -y

# Install all system dependencies in one shot
sudo apt-get install -y \
  gpsd gpsd-clients python3-gps libgps-dev \
  i2c-tools python3-pip python3-dev \
  git cmake build-essential ninja-build \
  libopenblas-dev
```

---

### 4.3 I²C + BME280 Setup

```bash
# Enable I²C interface
sudo raspi-config
# Navigate: Interface Options → I2C → Enable → Finish

# Reboot for changes to take effect
sudo reboot

# Verify I²C bus is available
ls /dev/i2c*
# Should show: /dev/i2c-1

# Scan for connected devices (BME280 should appear as 0x76 or 0x77)
sudo i2cdetect -y 1
```

Expected output with BME280 connected:
```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:          -- -- -- -- -- -- -- -- -- -- -- -- --
...
70: -- -- -- -- -- -- 76 --
```

If your BME280 shows at `0x77` instead, edit `config.py`:
```python
BME280_I2C_ADDR: int = 0x77
```

---

### 4.4 GPS (gpsd) Setup

```bash
# Plug in your USB GPS dongle, then find its device path
# Method 1: watch dmesg
dmesg | tail -20
# Look for lines like: "ttyUSB0: USB Serial Device" or "ttyACM0"

# Method 2: compare before/after plugging in
ls /dev/tty{USB,ACM}*
```

Common device paths:
- u-blox 7/8 → `/dev/ttyUSB0`
- u-blox 9 / some newer dongles → `/dev/ttyACM0`

```bash
# Configure gpsd to use your GPS device
sudo nano /etc/default/gpsd
```

Set these values (replace `ttyUSB0` with your actual device):
```
START_DAEMON="true"
USBAUTO="true"
DEVICES="/dev/ttyUSB0"
GPSD_OPTIONS="-n"
```

```bash
# Enable and start gpsd
sudo systemctl enable gpsd
sudo systemctl restart gpsd

# Verify gpsd is running
sudo systemctl status gpsd

# Test GPS data stream (take device outdoors for first fix — takes ~60–90 s)
cgps -s
# or: gpsmon
```

You should see latitude/longitude populated once the GPS has a fix.
A cold-start fix indoors is unlikely — move near a window or outdoors.

---

### 4.5 Build llama.cpp

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with native RPi5 ARM optimisations
cmake -B build \
  -DLLAMA_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -G Ninja

cmake --build build --config Release -j4
# Build time: ~8–12 minutes on RPi5

# Verify the binary exists
ls build/bin/llama-cli
```

Then update `config.py` if your path differs:
```python
LLAMA_CLI_PATH: str = os.environ.get("LLAMA_CLI", "./llama.cpp/build/bin/llama-cli")
```

---

### 4.6 Download the Model

The paper uses **Gemma-3-1B-IT Q4_K_M** (~700 MB).

```bash
# Create models directory in the project
mkdir -p ~/sensoraware_cache/models

# Option A: wget directly (if HuggingFace is accessible)
wget -O ~/sensoraware_cache/models/gemma-3-1B-IT-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf"

# Option B: transfer from your PC via scp
scp gemma-3-1b-it-Q4_K_M.gguf pi@raspberrypi.local:~/sensoraware_cache/models/

# Verify file size (~700 MB)
ls -lh ~/sensoraware_cache/models/
```

Test the model loads correctly (should print a short response):
```bash
cd ~/sensoraware_cache
./llama.cpp/build/bin/llama-cli \
  -m models/gemma-3-1B-IT-Q4_K_M.gguf \
  -p "Hello, what is 2+2?" \
  -n 32 --threads 4 --log-disable
```

---

### 4.7 Python Dependencies

```bash
cd ~/sensoraware_cache
pip install -r requirements_rpi.txt --break-system-packages
```

This installs:
- `sentence-transformers>=2.7.0` — all-MiniLM-L6-v2 embedding model (22 MB)
- `numpy>=1.26.0`
- `scipy>=1.13.0` — for `minimize_scalar` in UCTR gate
- `smbus2>=0.4.3` — BME280 I²C
- `gps3>=0.33` — gpsd Python client
- `requests>=2.31.0` — Open-Meteo fallback (not used on RPi but keeps imports consistent)

> **First run** will download `all-MiniLM-L6-v2` (~90 MB) from HuggingFace.
> This requires internet. After first run it is cached at `~/.cache/huggingface/`.
> Subsequent runs are fully offline.

---

### 4.8 Pre-flight Sensor Check

Run this before any evaluation to confirm all hardware is working:

```bash
cd ~/sensoraware_cache
PLATFORM=rpi python - <<'EOF'
from sensor_provider import get_provider
s = get_provider("rpi")
state = s.get_state()
print(f"GPS:      lat={state.lat:.5f}  lon={state.lon:.5f}  alt={state.altitude_m:.1f}m")
print(f"Baro:     {state.pressure_hpa:.2f} hPa")
print(f"Temp:     {state.temp_c:.1f} °C")
print(f"Battery:  {state.battery_pct*100:.0f}%")
print(f"Sources:  {state.source_flags}")
EOF
```

**Expected healthy output:**
```
GPS:      lat=12.97194  lon=77.59369  alt=920.3m
Baro:     908.42 hPa
Temp:     26.1 °C
Battery:  98%
Sources:  {'gps': 'real', 'altitude': 'gps', 'pressure': 'bme280',
           'temp': 'bme280', 'battery': 'simulated'}
```

**If GPS shows `'stale'`:** Move outdoors or near a window; allow 90 s for cold-start fix.

**If pressure shows implausible value:** Re-check I²C wiring and `BME280_I2C_ADDR` in `config.py`.

---

### 4.9 Run Evaluation on RPi5

```bash
cd ~/sensoraware_cache

# Single config, single seed — quickest real-hardware test (~20 min)
PLATFORM=rpi python main.py --mode real --configs C4 --seeds 1

# Full paper evaluation: all configs, 5 seeds
PLATFORM=rpi python main.py --mode real \
  --configs C1,C1_T,C1_G,C2,C3,C4,C5 \
  --seeds 5 \
  --output results/

# Interactive demo (real-time Q&A with live sensors)
PLATFORM=rpi python main.py --mode interactive
```

---

## 5. Android / Vivo V27 — Termux Setup

### 5.1 Sensor Reality on Vivo V27

| Sensor | Available? | Code Path | Accuracy |
|--------|-----------|-----------|----------|
| GPS | ✅ Yes | `termux-location -p gps` | ~3–5 m outdoor |
| Altitude | ✅ GPS-derived | NMEA altitude field | ~10 m |
| Barometer | ❌ Not present | **Open-Meteo API fallback** | ~±1 hPa (API) |
| Temperature | ❌ Not present | **Battery temp proxy** (-7°C offset) | ~±5°C |
| Battery % | ✅ Yes | `termux-battery-status` | Exact |

> The barometric and temperature source flags in outputs will read
> `"open_meteo_api"` and `"battery_proxy"` — this is expected and correct
> for the V27 hardware profile. The paper architecture diagram accounts for this.

---

### 5.2 Termux Installation

> ⚠️ **Critical:** Install Termux from **F-Droid only**.
> The Play Store version is abandoned and will not work.

1. Install **F-Droid** from [f-droid.org](https://f-droid.org)
2. From F-Droid, install **Termux** (search for it)
3. From F-Droid, install **Termux:API** (separate app — required for sensors)

```bash
# First-time Termux setup
pkg update && pkg upgrade -y

# Install all required packages
pkg install -y \
  termux-api \
  python \
  clang \
  cmake \
  ninja \
  git \
  libopenblas \
  wget

# Give Termux access to phone storage
termux-setup-storage
```

**Grant permissions on Android (Settings app):**

1. Settings → Apps → Termux:API → Permissions
   - Location → **Allow all the time** (precise location)
   - Battery → **Unrestricted** (no battery optimisation)
2. Settings → Apps → Termux → Permissions
   - Same settings as above

```bash
# Verify Termux:API sensor access works
termux-location -p gps -r once
# Expected: {"latitude": ..., "longitude": ..., "altitude": ..., ...}

termux-battery-status
# Expected: {"health": "GOOD", "percentage": 72, "temperature": 28.5, ...}

termux-sensor -s "TYPE_PRESSURE" -n 1
# Expected on V27: error or empty — this is normal (no barometer chip)
```

---

### 5.3 Build llama.cpp on Android

```bash
# Clone inside Termux home directory
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build for ARM64 Android (no NEON flags — let CMake auto-detect)
cmake -B build \
  -DLLAMA_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -G Ninja

cmake --build build --config Release -j$(nproc)
# Build time: ~15–25 minutes on V27

# Verify
ls build/bin/llama-cli
```

---

### 5.4 Transfer the Model

**Option A — Transfer from PC via USB:**
```bash
# On your PC
adb push gemma-3-1b-it-Q4_K_M.gguf /sdcard/Download/

# In Termux
mkdir -p ~/sensoraware_cache/models
cp /sdcard/Download/gemma-3-1b-it-Q4_K_M.gguf ~/sensoraware_cache/models/
```

**Option B — Download directly in Termux** (requires Wi-Fi, ~700 MB):
```bash
mkdir -p ~/sensoraware_cache/models
cd ~/sensoraware_cache/models
wget "https://huggingface.co/bartowski/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf" \
  -O gemma-3-1B-IT-Q4_K_M.gguf
```

**Test the model:**
```bash
cd ~/sensoraware_cache
~/llama.cpp/build/bin/llama-cli \
  -m models/gemma-3-1B-IT-Q4_K_M.gguf \
  -p "What is 2+2?" -n 32 --threads 4 --log-disable
```

---

### 5.5 Python Dependencies

```bash
cd ~/sensoraware_cache
pip install -r requirements_android.txt
```

> **First run** downloads `all-MiniLM-L6-v2` (~90 MB).
> Ensure Wi-Fi is connected. After download, the model is cached
> and all subsequent runs are fully offline.

---

### 5.6 Pre-flight Sensor Check

```bash
cd ~/sensoraware_cache
PLATFORM=android python - <<'EOF'
from sensor_provider import get_provider
s = get_provider("android")
state = s.get_state()
print(f"GPS:      lat={state.lat:.5f}  lon={state.lon:.5f}  alt={state.altitude_m:.1f}m")
print(f"Baro:     {state.pressure_hpa:.2f} hPa  (API)")
print(f"Temp:     {state.temp_c:.1f} °C  (battery proxy)")
print(f"Battery:  {state.battery_pct*100:.0f}%")
print(f"Sources:  {state.source_flags}")
EOF
```

**Expected healthy output:**
```
GPS:      lat=12.97194  lon=77.59369  alt=35.2m
Baro:     1009.80 hPa  (API)
Temp:     21.5 °C  (battery proxy)
Battery:  74%
Sources:  {'gps': 'real', 'altitude': 'gps_derived',
           'pressure': 'open_meteo_api', 'temp': 'battery_proxy',
           'battery': 'real'}
```

---

### 5.7 Run Evaluation on Android

```bash
cd ~/sensoraware_cache

# Single config quick run (~15–30 min depending on corpus size)
PLATFORM=android python main.py --mode real --configs C4 --seeds 1

# Recommended Android set (C4 + C5 are most informative for cross-platform comparison)
PLATFORM=android python main.py --mode real \
  --configs C4,C5 \
  --seeds 3 \
  --output results/

# Interactive demo
PLATFORM=android python main.py --mode interactive
```

---

## 6. Running the Evaluation

### 6.1 All CLI Options

```
python main.py [OPTIONS]

--mode        sim | real | interactive
              sim         = Gaussian-drift simulator (paper Sec V-C)
              real        = Live hardware sensors
              interactive = Real-time REPL with cache hit display

--configs     Comma-separated list of config names
              Default: C1,C2,C3,C4,C5
              Valid:   C1, C1_T, C1_G, C2, C3, C4, C5

--seeds       Number of random seeds to run (default: 5)
              Each seed produces an independent result file.

--mock-llm    Use MockLLMBackend instead of llama.cpp
              Simulates ~510ms TTFT without running the model.
              Use for fast iteration and debugging.

--platform    rpi | android | sim
              Overrides the PLATFORM environment variable.

--model       Path to .gguf model file
              Default: ./models/gemma-3-1B-IT-Q4_K_M.gguf

--output      Directory for results JSON files
              Default: ./results/

--verbose     Enable DEBUG-level logging
```

### 6.2 Config Reference (C1–C5)

| Config | Scenario | CVW | UCTR | Purpose |
|--------|----------|-----|------|---------|
| **C1** | Static | ❌ | ❌ | Semantic-only baseline (no physical invalidation) |
| **C1-T** | Static | ❌ | ❌ | TTL=30 s baseline (standard caching) |
| **C1-G** | Static | ❌ | ❌ | GPS-threshold-only baseline (single dimension) |
| **C2** | Static | ✅ | ❌ | CVW overhead measurement in stable conditions |
| **C3** | Moderate drift | ✅ | ❌ | CVW under typical hiking movement |
| **C4** | High volatility | ✅ | ✅ | Full system: storm-front simulation |
| **C5** | Moderate drift | ✅ | ✅ | UCTR under critically low battery (forced at query 480) |

**Sensor drift per step (τ = 5 s) in each scenario:**

| Scenario | GPS σ | ALT σ | BARO σ | TEMP σ |
|----------|-------|-------|--------|--------|
| Static | 0.3 m | 0.1 m | 0.05 hPa | 0.05°C |
| Moderate | 5.0 m | 2.0 m | 0.40 hPa | 0.30°C |
| High vol | 15.0 m | 8.0 m | 1.20 hPa | 0.80°C |

### 6.3 Reading the Output

During a run you will see progress every 100 queries:
```
11:23:45 [eval_harness] INFO: Progress: 200/960  CHR=41.8%  CVW-inv=18
```

At the end of each config:
```
──────────────────────────────────────────────────
Config: C4  seed=2
  Queries:   960
  CHR:       31.5%
  VHR:       31.5%
  SRR:       None         ← set after manual adjudication
  CVW-inv:   82
  TTFT-H:    1.0 ms
  TTFT-M:    510.3 ms
  Speedup:   510×
  Runtime:   847 s
──────────────────────────────────────────────────
```

After all seeds complete, the master table is printed:
```
══════════════════════════════════════════════════════════════════════════════
TABLE IV — PERFORMANCE COMPARISON (mean over 5 runs)
══════════════════════════════════════════════════════════════════════════════
Config   CHR%   VHR%  CVW-inv  TTFT-H   TTFT-M
────────────────────────────────────────────────────────────────────────────
C1       42.5   34.0        0     1.0    510.0
C2       42.3   42.3        4     1.0    510.0
C3       35.0   35.0       38     1.0    510.0
C4       31.5   31.5       82     1.0    510.0
C5       33.8   33.8       47     1.0    510.0
══════════════════════════════════════════════════════════════════════════════
```

---

## 7. Corpus Setup

The harness ships with a **12-query built-in fallback** that runs immediately
without any setup. For paper-grade evaluation you need the 1,200-query corpus.

### Corpus Format

Create `corpus/wilderness_1200.json` as a JSON array:

```json
[
  {
    "query": "What is the safest descent route from this ridge?",
    "category": "navigation",
    "urgency": "normal"
  },
  {
    "query": "How do I treat altitude sickness?",
    "category": "first_aid",
    "urgency": "high"
  }
]
```

**Valid `category` values:** `navigation`, `first_aid`, `weather`,
`resource_mgmt`, `threat_assess`, `gen_knowledge`

**Valid `urgency` values:** `normal`, `high`

### Corpus Distribution (paper Table II)

| Category | Total | Unique | Paraphrases | Urgency |
|----------|-------|--------|-------------|---------|
| Navigation | 240 | 40 | 5× | Normal |
| First Aid | 240 | 40 | 5× | High |
| Weather | 180 | 30 | 5× | Normal |
| Resource Mgmt | 180 | 30 | 5× | Normal |
| Threat Assess | 180 | 30 | 5× | Normal |
| Gen. Knowledge | 180 | 30 | 5× | Normal |
| **Total** | **1,200** | **200** | — | — |

> The harness automatically holds out the first 20% (240 queries)
> for CVW threshold learning. All reported metrics are over the
> remaining 960 queries.

### Optional Custom Pre-warm Entries

Create `corpus/prewarm_entries.json` to override the 8 built-in entries:

```json
[
  {
    "query": "How do I descend from an exposed ridge in bad weather?",
    "response": "Your pre-warmed response text here.",
    "category": "navigation",
    "low_confidence": true
  }
]
```

---

## 8. Results & Metrics

All results are written to `results/` as JSON:

```
results/
├── C1_seed0.json       ← Full per-query event log
├── C1_seed1.json
├── ...
├── C1_aggregate.json   ← Mean ± std across 5 seeds
├── C4_aggregate.json
└── table_IV.json       ← Master comparison table (copy into paper)
```

### Aggregate File Structure

```json
{
  "config": "C4",
  "n_seeds": 5,
  "CHR_mean": 31.5,
  "CHR_std": 0.8,
  "CVW_inv_mean": 82.0,
  "CVW_inv_std": 3.2,
  "TTFT_H_ms_mean": 1.0,
  "TTFT_M_ms_mean": 510.3
}
```

### SRR Manual Validation

The SRR (Stale Response Rate) requires human adjudication.
After running the evaluation:

1. Open any `C1_seed0.json` file
2. Find events where `"result": "semantic_hit"` or `"hash_hit"` and
   `"category"` is `"navigation"`, `"weather"`, or `"first_aid"`
3. Compare `s_ins` (sensor at insertion) vs `s_now` (sensor at query)
4. Mark `"is_stale": true` or `"is_stale": false` for each sampled event
5. Paper protocol: 50 samples per state-sensitive category = 150 total

---

## 9. config.py Reference

All tunable parameters in one place:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PLATFORM` | `"rpi"` | Set via env `PLATFORM=rpi\|android\|sim` |
| `MODEL_PATH` | `./models/gemma-3-1B-IT-Q4_K_M.gguf` | Path to GGUF model |
| `LLAMA_CLI_PATH` | `./llama.cpp/build/bin/llama-cli` | Path to compiled binary |
| `LLAMA_N_THREADS` | `4` | Inference threads (set to CPU core count) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `SIMILARITY_THRESHOLD` | `0.85` | Cosine sim threshold σ (Eq. 3) |
| `CACHE_MAX_ENTRIES` | `500` | Cache capacity |
| `TAU_SAMPLE_S` | `5.0` | CVW monitor sweep interval (seconds) |
| `BATTERY_CRITICAL_THRESHOLD` | `0.15` | Below this → UCTR battery conservation |
| `C_STALE_RATIO` | `10.0` | Cstale/Cmiss cost ratio (Eq. 7) |
| `GAMMA` | `5.0` | Sigmoid sharpness in p_stale (Eq. 8) |
| `BME280_I2C_ADDR` | `0x76` | Change to `0x77` if SDO pin is pulled high |
| `N_SEEDS` | `5` | Seeds per config in eval harness |

---

## 10. Troubleshooting

### RPi5 Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `RuntimeError: BME280 chip ID 0xFF` | Wrong I²C address or loose wire | Run `i2cdetect -y 1`; update `BME280_I2C_ADDR` |
| `GPS shows 0.0, 0.0` | No GPS fix yet | Move outdoors; wait 90 s for cold start |
| `gpsd: no subscribers` | gpsd not running | `sudo systemctl restart gpsd` |
| Model load time > 40 s | Running from SD card | Move project to USB-SSD |
| `llama-cli: not found` | Build failed or wrong path | Check `ls llama.cpp/build/bin/llama-cli` |
| `OSError: [Errno 5] Input/output error` on I²C | Bus locked or wiring issue | `sudo rmmod i2c_bcm2835 && sudo modprobe i2c_bcm2835` |

### Android / Termux Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `termux-location` hangs | Location permission not granted | Settings → Termux:API → Location → Allow all the time |
| `termux-location` returns `"provider": "network"` not GPS | GPS disabled | Enable GPS in phone quick settings |
| `Open-Meteo: fetch failed` | No internet | Check Wi-Fi; last cached value is used as fallback |
| `pip install` fails on numpy | Missing C compiler | `pkg install clang libopenblas` |
| `llama-cli: Illegal instruction` | Binary built with wrong arch flags | Rebuild with `-DLLAMA_NATIVE=OFF` |
| Termux:API commands return `{}` | Termux:API app not installed | Install from F-Droid (not Play Store) |

### Evaluation Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `CHR = 0%` throughout | Corpus not found, using 12-query fallback | Check `corpus/wilderness_1200.json` exists |
| `CVW-inv = 0` for all configs | CVWMonitor not started | Ensure `cvw=True` in chosen config; check logs |
| `TTFT-M ≈ 9999 ms` | LLM inference failed | Check model path; run llama-cli test manually |
| `SRR = None` | Expected — requires manual adjudication | See [Section 8](#8-results--metrics) |
| `OSError: We couldn't connect to HuggingFace` | No internet for first model download | Connect Wi-Fi for first run only |

### Verify Individual Components

```bash
# Test UCTR gate
python -c "
from uctr_gate import UCTRGate
g = UCTRGate()
print('low-bat α:', g.compute_alpha(0.10,'normal',{'gps':0,'alt':0,'baro':0,'temp':0},'navigation'))
print('high-urg α:', g.compute_alpha(0.80,'high',{'gps':0,'alt':0,'baro':0,'temp':0},'navigation'))
"
# Expected: low-bat ≈ 1.5–3.0,  high-urg ≈ 0.5

# Test cache store + CVW monitor (no hardware)
python -c "
from sensor_simulator import SimulatedSensorProvider
from cache_store import CacheStore
from cvw_monitor import CVWMonitor
import numpy as np, time
cs = CacheStore()
sim = SimulatedSensorProvider('high_vol', seed=0)
s = sim.get_state()
cs.insert('test query', np.random.randn(384).astype('float32'), 'response', s, 'navigation')
m = CVWMonitor(cs, sim, tau_s=0.2); m.start()
time.sleep(1.0); m.stop()
print('sweeps:', m.sweep_count, 'invalidations:', m.invalidation_count)
"
```
