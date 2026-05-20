# sensor_rpi.py
# RPi hardware: BMP280 (I2C) + gpsd

"""
sensor_rpi.py — Raspberry Pi 5 hardware sensors
  • GPS:      USB dongle (u-blox) via gpsd / gps3
  • Pressure: BME280 or BMP280 via I2C (smbus2)  ← address 0x76 or 0x77
  • Temp:     BME280 (same chip), fallback: battery-proxy via vcgencmd
  • Altitude: GPS NMEA altitude  +  ISA cross-check from baro
  • Battery:  Simulated discharge curve (RPi has no fuel-gauge by default)
              Swap in PiJuice / UPS-Hat readings if your board has one.

Install deps on RPi:
  sudo apt-get install -y gpsd gpsd-clients python3-gps
  pip install smbus2 gps3 --break-system-packages
"""

from __future__ import annotations
import math
import struct
import time
import logging
import subprocess
from typing import Optional, Tuple

from sensor_provider import SensorProvider, SensorState
import config

logger = logging.getLogger(__name__)


# ─── BME280 register map ─────────────────────────────────────────────────────
_BME_REG_ID        = 0xD0
_BME_REG_RESET     = 0xE0
_BME_REG_CTRL_HUM  = 0xF2
_BME_REG_STATUS    = 0xF3
_BME_REG_CTRL_MEAS = 0xF4
_BME_REG_CONFIG    = 0xF5
_BME_REG_PRESS_MSB = 0xF7   # 0xF7..0xF9  pressure
_BME_REG_TEMP_MSB  = 0xFA   # 0xFA..0xFC  temperature
_BME_REG_HUM_MSB   = 0xFD   # 0xFD..0xFE  humidity (BME only)
_BME_CALIB_00      = 0x88   # 24 bytes (T1..P9)
_BME_CALIB_E1      = 0xE1   # 7 bytes  (H1..H6)


def _baro_altitude(pressure_hpa: float,
                   sea_level_hpa: float = 1013.25) -> float:
    """ISA hypsometric formula → metres above sea-level."""
    return 44330.0 * (1.0 - (pressure_hpa / sea_level_hpa) ** 0.1903)


class BME280:
    """
    Minimal BME280 driver over smbus2.
    Returns (pressure_hPa, temperature_C) tuple.
    Works for BMP280 too (humidity reads are ignored).
    """

    def __init__(self, bus: "smbus2.SMBus", addr: int = config.BME280_I2C_ADDR):
        self.bus  = bus
        self.addr = addr
        self._dig = {}
        self._init()

    # ── private ──────────────────────────────────────────────────────────────

    def _read_byte(self, reg: int) -> int:
        return self.bus.read_byte_data(self.addr, reg)

    def _read_bytes(self, reg: int, length: int) -> bytes:
        return bytes(self.bus.read_i2c_block_data(self.addr, reg, length))

    def _write_byte(self, reg: int, val: int):
        self.bus.write_byte_data(self.addr, reg, val)

    def _init(self):
        chip_id = self._read_byte(_BME_REG_ID)
        if chip_id not in (0x60, 0x58):   # 0x60=BME280 0x58=BMP280
            raise RuntimeError(
                f"Unexpected BME/BMP chip ID: 0x{chip_id:02X} at addr 0x{self.addr:02X}"
            )
        logger.info("BME/BMP280 found: chip_id=0x%02X  addr=0x%02X", chip_id, self.addr)
        self._load_calibration()
        # Mode: osrs_t×2, osrs_p×16, normal mode
        self._write_byte(_BME_REG_CTRL_HUM,  0x01)
        self._write_byte(_BME_REG_CTRL_MEAS, 0x57)   # temp×2, pres×16, normal
        self._write_byte(_BME_REG_CONFIG,    0xA0)   # t_sb 1000ms, filter 16

    def _load_calibration(self):
        raw = self._read_bytes(_BME_CALIB_00, 24)
        d   = struct.unpack_from('<HhhHhhhhhhhh', raw)   # T1-T3, P1-P9 unsigned/signed mix
        # struct H=uint16, h=int16
        raw2 = self._read_bytes(_BME_CALIB_00, 26)
        self._dig['T1'] = struct.unpack_from('<H', raw2, 0)[0]
        self._dig['T2'] = struct.unpack_from('<h', raw2, 2)[0]
        self._dig['T3'] = struct.unpack_from('<h', raw2, 4)[0]
        self._dig['P1'] = struct.unpack_from('<H', raw2, 6)[0]
        for i, k in enumerate(['P2','P3','P4','P5','P6','P7','P8','P9']):
            self._dig[k] = struct.unpack_from('<h', raw2, 8 + i*2)[0]

    def _read_raw(self) -> Tuple[int, int]:
        data = self._read_bytes(_BME_REG_PRESS_MSB, 6)
        raw_press = ((data[0] << 12) | (data[1] << 4) | (data[2] >> 4))
        raw_temp  = ((data[3] << 12) | (data[4] << 4) | (data[5] >> 4))
        return raw_press, raw_temp

    def _compensate_temp(self, raw: int) -> Tuple[float, float]:
        d = self._dig
        var1 = ((raw / 16384.0) - (d['T1'] / 1024.0)) * d['T2']
        var2 = ((raw / 131072.0) - (d['T1'] / 8192.0)) ** 2 * d['T3']
        t_fine = var1 + var2
        temp_c = t_fine / 5120.0
        return temp_c, t_fine

    def _compensate_pressure(self, raw: int, t_fine: float) -> float:
        d = self._dig
        var1 = t_fine / 2.0 - 64000.0
        var2 = var1 * var1 * d['P6'] / 32768.0
        var2 = var2 + var1 * d['P5'] * 2.0
        var2 = var2 / 4.0 + d['P4'] * 65536.0
        var1 = (d['P3'] * var1 * var1 / 524288.0 + d['P2'] * var1) / 524288.0
        var1 = (1.0 + var1 / 32768.0) * d['P1']
        if var1 == 0:
            return 0.0
        pressure = 1048576.0 - raw
        pressure = ((pressure - var2 / 4096.0) * 6250.0) / var1
        var1 = d['P9'] * pressure * pressure / 2147483648.0
        var2 = pressure * d['P8'] / 32768.0
        pressure = pressure + (var1 + var2 + d['P7']) / 16.0
        return pressure / 100.0   # Pa → hPa

    # ── public ───────────────────────────────────────────────────────────────

    def read(self) -> Tuple[float, float]:
        """Returns (pressure_hPa, temperature_C)."""
        raw_press, raw_temp = self._read_raw()
        temp_c, t_fine      = self._compensate_temp(raw_temp)
        pressure_hpa        = self._compensate_pressure(raw_press, t_fine)
        return pressure_hpa, temp_c


class RPiSensorProvider(SensorProvider):
    """
    Full RPi5 sensor provider:
      GPS  →  gpsd (USB u-blox dongle)
      Baro →  BME280/BMP280 I2C bus-1
      Temp →  BME280 (same read), fallback vcgencmd
      Bat  →  Simulated 4h discharge curve (swap for PiJuice if available)
    """

    def __init__(self):
        import smbus2
        self.bus    = smbus2.SMBus(config.I2C_BUS)
        self.bme    = BME280(self.bus, config.BME280_I2C_ADDR)
        self._init_gps()
        self._start_time    = time.time()
        self._last_gps      = (0.0, 0.0, 0.0)   # lat, lon, alt
        self._gps_ok        = False
        logger.info("RPiSensorProvider ready")

    def _init_gps(self):
        """Connect to gpsd and start watching."""
        try:
            import gps as gpsd_module
            self._gpsd = gpsd_module.gps(
                host=config.GPSD_HOST,
                port=config.GPSD_PORT,
                mode=gpsd_module.WATCH_ENABLE | gpsd_module.WATCH_NEWSTYLE
            )
            self._gpsd_module = gpsd_module
            logger.info("gpsd connected at %s:%d", config.GPSD_HOST, config.GPSD_PORT)
        except Exception as exc:
            logger.warning("gpsd init failed (%s) — will use fallback zeros", exc)
            self._gpsd = None

    def _read_gps(self, timeout: float = config.GPS_MAX_WAIT_S
                  ) -> Tuple[float, float, float]:
        """
        Poll gpsd until a TPV fix is available.
        Returns (lat, lon, alt_m).  Falls back to last known if timeout.
        """
        if self._gpsd is None:
            return self._last_gps

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                report = self._gpsd.next()
                if report['class'] == 'TPV':
                    lat = getattr(report, 'lat', None)
                    lon = getattr(report, 'lon', None)
                    alt = getattr(report, 'alt', None)
                    if None not in (lat, lon, alt):
                        self._last_gps = (float(lat), float(lon), float(alt))
                        self._gps_ok   = True
                        return self._last_gps
            except StopIteration:
                break
            except Exception as exc:
                logger.debug("gpsd read error: %s", exc)
                break

        logger.debug("GPS timeout — returning last known fix")
        return self._last_gps

    def _read_vcgencmd_temp(self) -> Optional[float]:
        """CPU temperature from vcgencmd as last-resort proxy."""
        try:
            out = subprocess.check_output(
                ['vcgencmd', 'measure_temp'], text=True, timeout=2
            )
            # "temp=42.8'C"
            return float(out.strip().split('=')[1].replace("'C", ""))
        except Exception:
            return None

    def _simulate_battery(self) -> float:
        """
        Discharge curve approximation for a 3-hour session.
        Swap with PiJuice.STATUS.battery_charge / UPS-Hat API if available.
        """
        elapsed = time.time() - self._start_time
        capacity_seconds = 4 * 3600   # assume 4h runtime
        return max(0.0, 1.0 - elapsed / capacity_seconds)

    # ── PiJuice integration (uncomment if PiJuice HAT is present) ────────────
    # def _piJuice_battery(self) -> float:
    #     from pijuice import PiJuice
    #     pj = PiJuice(1, 0x14)
    #     return pj.status.GetChargeLevel()['data'] / 100.0

    def get_state(self) -> SensorState:
        lat, lon, gps_alt = self._read_gps()
        pressure_hpa, temp_c = self.bme.read()

        # Altitude: prefer GPS, cross-check with barometric ISA formula
        baro_alt = _baro_altitude(pressure_hpa)
        # Blend: GPS alt has ~5m accuracy, baro formula ~±10m
        altitude_m = gps_alt if self._gps_ok else baro_alt

        # Temperature fallback: CPU proxy if BME280 returns implausible value
        if temp_c < -40 or temp_c > 85:
            cpu_t = self._read_vcgencmd_temp()
            if cpu_t is not None:
                temp_c = cpu_t

        return SensorState(
            lat          = lat,
            lon          = lon,
            altitude_m   = altitude_m,
            pressure_hpa = pressure_hpa,
            temp_c       = temp_c,
            battery_pct  = self._simulate_battery(),
            source_flags = {
                "gps":      "real" if self._gps_ok else "stale",
                "altitude": "gps"  if self._gps_ok else "baro_formula",
                "pressure": "bme280",
                "temp":     "bme280",
                "battery":  "simulated",   # replace with "piJuice" if available
            }
        )
