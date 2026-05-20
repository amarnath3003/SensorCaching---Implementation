# cvw_monitor.py
# Background invalidation sweep (τ_sample=5s)

import threading, time, math
from sensor_provider import SensorState

CVW_THRESHOLDS = {  # Table I from paper
    "navigation":   {"gps": 50,  "alt": 30,  "baro": 5,   "temp": 10},
    "weather":      {"gps": 200, "alt": 100, "baro": 3,   "temp": 5},
    "first_aid":    {"gps": 1e9, "alt": 200, "baro": 1e9, "temp": 8},
    "resource_mgmt":{"gps": 1e9, "alt": 1e9, "baro": 1e9, "temp": 1e9},
    "threat_assess":{"gps": 100, "alt": 1e9, "baro": 1e9, "temp": 1e9},
    "gen_knowledge":{"gps": 1e9, "alt": 1e9, "baro": 1e9, "temp": 1e9},
}

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

class CVWMonitor:
    def __init__(self, cache_store, sensor_provider, tau_sample=5.0):
        self.cache = cache_store
        self.sensor = sensor_provider
        self.tau = tau_sample
        self.invalidation_count = 0
        self._thread = threading.Thread(target=self._sweep_loop, daemon=True)

    def start(self):
        self._thread.start()

    def _sweep_loop(self):
        while True:
            s_now = self.sensor.get_state()
            for entry_id, entry in list(self.cache.entries.items()):
                if self._is_invalid(entry, s_now):
                    self.cache.evict(entry_id, reason="cvw_violation")
                    self.invalidation_count += 1
            time.sleep(self.tau)

    def _is_invalid(self, entry, s_now: SensorState) -> bool:
        s_ins = entry.sensor_snapshot
        alpha = entry.alpha  # from UCTR gate
        thresholds = CVW_THRESHOLDS[entry.category]
        delta_gps  = haversine_m(s_ins.lat, s_ins.lon, s_now.lat, s_now.lon)
        delta_alt  = abs(s_now.altitude_m - s_ins.altitude_m)
        delta_baro = abs(s_now.pressure_hpa - s_ins.pressure_hpa)
        delta_temp = abs(s_now.temp_c - s_ins.temp_c)
        return (delta_gps  > alpha * thresholds["gps"]  or
                delta_alt  > alpha * thresholds["alt"]  or
                delta_baro > alpha * thresholds["baro"] or
                delta_temp > alpha * thresholds["temp"])