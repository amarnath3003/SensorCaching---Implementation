from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class SensorState:
    lat: float          # degrees
    lon: float          # degrees
    altitude_m: float   # metres
    pressure_hpa: float # hPa — real or API-sourced
    temp_c: float       # °C — real or battery proxy
    battery_pct: float  # 0.0–1.0
    source_flags: dict  # e.g. {"pressure": "api", "temp": "battery_proxy"}

class SensorProvider(ABC):
    @abstractmethod
    def get_state(self) -> SensorState:
        pass

class RPiSensorProvider(SensorProvider):
    def __init__(self):
        import smbus2
        from gps3 import gps3
        self.bus = smbus2.SMBus(1)
        self.BME280_ADDR = 0x76
        self._init_bme280()
        self.gps_socket = gps3.GPSDSocket()
        self.gps_socket.connect()
        self.gps_socket.watch()
        self.data_stream = gps3.DataStream()

    def get_state(self) -> SensorState:
        # Read BME280 (pressure + temp)
        pressure, temp = self._read_bme280()
        # Read GPS
        lat, lon, alt = self._read_gps()
        return SensorState(
            lat=lat, lon=lon, altitude_m=alt,
            pressure_hpa=pressure, temp_c=temp,
            battery_pct=self._simulate_battery(),
            source_flags={"pressure": "bme280", "temp": "bme280",
                          "gps": "ublox", "battery": "simulated"}
        )
    # ... BME280 I2C read + GPS NMEA parse methods