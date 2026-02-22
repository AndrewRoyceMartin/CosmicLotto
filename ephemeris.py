import math
import json
from datetime import datetime, timezone
from skyfield.api import load, Topos, wgs84
from skyfield.data import mpc
from skyfield.framelib import ecliptic_frame
from db import connect
from utils import from_json

PLANETS = {
    "mercury": "mercury",
    "venus": "venus",
    "mars": "mars barycenter",
    "jupiter": "jupiter barycenter",
    "saturn": "saturn barycenter",
    "uranus": "uranus barycenter",
    "neptune": "neptune barycenter",
    "moon": "moon",
    "sun": "sun",
}


class EphemerisEngine:
    def __init__(self):
        self.ts = load.timescale()
        self.eph = load("de421.bsp")
        self.earth = self.eph["earth"]

    def compute_positions(self, dt_utc, lat, lon, altitude_m=0.0):
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)

        t = self.ts.from_datetime(dt_utc)
        observer = self.earth + wgs84.latlon(lat, lon, elevation_m=altitude_m)

        positions = {}
        for name, target_name in PLANETS.items():
            target = self.eph[target_name]
            astrometric = observer.at(t).observe(target)
            ecl_pos = astrometric.apparent().frame_latlon(ecliptic_frame)
            ecl_lat_deg = ecl_pos[0].degrees
            ecl_lon_deg = ecl_pos[1].degrees % 360.0

            positions[name] = {
                "ecl_lon": round(ecl_lon_deg, 4),
                "ecl_lat": round(ecl_lat_deg, 4),
            }

        return positions

    def compute_and_store(self, draw_id, dt_utc_str, lat, lon, altitude_m, location_name, progress_callback=None):
        from dateutil import parser as dtparser
        dt_utc = dtparser.parse(dt_utc_str)
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)

        positions = self.compute_positions(dt_utc, lat, lon, altitude_m)
        positions_json = json.dumps(positions)

        conn = connect()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO planet_positions
                (draw_id, location_name, latitude, longitude, altitude_m, positions_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (draw_id, location_name, lat, lon, altitude_m, positions_json))
            conn.commit()
        finally:
            conn.close()

        return positions

    def compute_bulk(self, draw_rows, lat, lon, altitude_m, location_name, progress_callback=None):
        computed = 0
        total = len(draw_rows)
        for row in draw_rows:
            draw_id = row["draw_id"]
            dt_utc_str = row["draw_datetime_utc"]
            self.compute_and_store(draw_id, dt_utc_str, lat, lon, altitude_m, location_name)
            computed += 1
            if progress_callback:
                progress_callback(computed, total)
        return computed
