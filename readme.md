# SensorAware-Cache — Real-Device Deployment Guide

Paper: *SensorAware-Cache: Physical-World Telemetry-Driven Semantic Cache
Invalidation for Offline On-Device LLM Inference*

---

## Quick Start

```bash
# Smoke-test with mock LLM (no hardware required)
cd sensoraware_cache
pip install -r requirements_rpi.txt        # or requirements_android.txt
python main.py --mode sim --mock-llm --configs C1,C4 --seeds 1
```

---

## Phase 1: Raspberry Pi 5 (Paper-Faithful)

### 1.1 Hardware Bill of Materials

| Part | Purpose | Notes |
|------|---------|-------|
| Raspberry Pi 5 (8 GB) | Compute | Main platform |
| BME280 breakout (I2C) | Baro + temp | addr 0x76 default |
| USB GPS dongle (u-blox 7/8/9) | Location + alt | Any NMEA-compatible |
| USB-SSD (250 GB) | Model storage | SD card too slow (~40 s load) |
| Power bank 20 000 mAh | Field power | Optional for outdoor runs |

### 1.2 OS Setup

```bash
# Ubuntu 23.10 Server (64-bit) on USB-SSD
# Flash with Raspberry Pi Imager → choose Ubuntu 23.10

# After first boot:
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y gpsd gpsd-clients python3-pip python3-dev \
                        libgps-dev i2c-tools git cmake build-essential
```

### 1.3 Enable I2C

```bash
sudo raspi-config          # Interface Options → I2C → Enable
# Verify BME280 is visible:
sudo i2cdetect -y 1        # Should show 0x76 (or 0x77)
```

### 1.4 Configure gpsd

```bash
# Find GPS device (usually /dev/ttyUSB0 or /dev/ttyACM0):
ls /dev/tty*               # Plug/unplug GPS to identify

# Edit /etc/default/gpsd:
sudo nano /etc/default/gpsd
# Set:  DEVICES="/dev/ttyUSB0"
#       GPSD_OPTIONS="-n"

sudo systemctl enable gpsd
sudo systemctl restart gpsd

# Test GPS lock:
gpsmon                     # Should show satellites after ~60s outdoor
```

### 1.5 Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_NATIVE=ON
cmake --build build -j4    # ~10 min on RPi5
cd ..
```

### 1.6 Download Model

```bash
# Gemma-3-1B-IT Q4_K_M (~700 MB)
mkdir -p models
wget -O models/gemma-3-1B-IT-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf"
```

### 1.7 Install Python deps

```bash
pip install -r requirements_rpi.txt --break-system-packages
```

### 1.8 Run evaluation

```bash
# Full 5-seed evaluation (all configs)
PLATFORM=rpi python main.py --mode real --configs C1,C2,C3,C4,C5 --seeds 5

# Single interactive demo
PLATFORM=rpi python main.py --mode interactive
```

---

## Phase 2: Android — Vivo V27

### Sensor Reality Check

| Sensor | Vivo V27 | Code path |
|--------|----------|-----------|
| GPS | ✅ Real | `termux-location` |
| Altitude | ✅ GPS-derived | NMEA alt field |
| Barometer | ❌ Not present | Open-Meteo API fallback |
| Temperature | ❌ Not present | Battery temp proxy (-7°C offset) |
| Battery | ✅ Real | `termux-battery-status` |

### 2.1 Termux Setup

```bash
# 1. Install from F-Droid (NOT Play Store — Play Store version is outdated):
#    https://f-droid.org/packages/com.termux/
#    https://f-droid.org/packages/com.termux.api/

# 2. First-time setup:
pkg update && pkg upgrade
pkg install termux-api python clang libopenblas git cmake

# 3. Grant permissions (Android Settings):
#    Termux:API → Location: Allow all the time (precise)
#    Termux:API → Battery: Unrestricted

# 4. Test sensors:
termux-location -p gps -r once
termux-battery-status
```

### 2.2 Build llama.cpp on Android

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### 2.3 Download Model (use phone or transfer from PC)

```bash
# Transfer via adb:
adb push gemma-3-1B-IT-Q4_K_M.gguf /sdcard/Download/
cp /sdcard/Download/gemma-3-1B-IT-Q4_K_M.gguf ~/sensoraware_cache/models/
```

### 2.4 Install Python deps

```bash
pip install -r requirements_android.txt
```

### 2.5 Run evaluation

```bash
PLATFORM=android python main.py --mode real --configs C4,C5 --seeds 3
```

---

## Verification Checklist

After each platform setup, run the pre-flight check:

```bash
python - <<'EOF'
import config
from sensor_provider import get_provider
s = get_provider()
print("Sensor:", s.get_state())
print("Source flags:", s.get_state().source_flags)
EOF
```

Expected on RPi5:
```
source_flags = {'gps': 'real', 'altitude': 'gps', 'pressure': 'bme280',
                'temp': 'bme280', 'battery': 'simulated'}
```

Expected on Vivo V27:
```
source_flags = {'gps': 'real', 'altitude': 'gps_derived',
                'pressure': 'open_meteo_api', 'temp': 'battery_proxy',
                'battery': 'real'}
```

---

## Results Export

All metrics are exported to `./results/` as JSON files:
- `C4_seed0.json` through `C4_seed4.json` — per-seed full event logs
- `C4_aggregate.json` — mean ± std across seeds
- `table_IV.json` — master comparison table matching Table IV in paper

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `gpsd: no device` | Check `ls /dev/ttyUSB*` and update `/etc/default/gpsd` |
| BME280 not found at 0x76 | Try 0x77: `BME280_I2C_ADDR=0x77` in config.py |
| `termux-location` times out | Move outdoors; first fix takes 60–90 s |
| Open-Meteo returns 0 hPa | GPS fix required first; fallback ISA 1013.25 used |
| llama-cli not found | Rebuild: `cmake --build llama.cpp/build -j4` |
| Model load > 40 s on RPi | Use USB-SSD instead of SD card |
