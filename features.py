import json
import math
from db import connect
from utils import from_json

DEFAULT_ASPECTS = {
    "conjunction": 0,
    "sextile": 60,
    "square": 90,
    "trine": 120,
    "opposition": 180,
}


def angular_sep(a, b):
    diff = abs(a - b) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


def longitude_bin(lon_deg, bin_size=30):
    return int(lon_deg // bin_size) % (360 // bin_size)


BIN_LABELS = [
    "Aries", "Taurus", "Gemini", "Cancer",
    "Leo", "Virgo", "Libra", "Scorpio",
    "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]


def build_features(positions, bin_size=30, aspects=None, orb_deg=6.0):
    if aspects is None:
        aspects = DEFAULT_ASPECTS

    features = {}

    planet_names = sorted(positions.keys())

    for pname in planet_names:
        ecl_lon = positions[pname]["ecl_lon"]
        b = longitude_bin(ecl_lon, bin_size)
        num_bins = 360 // bin_size
        for i in range(num_bins):
            key = f"bin_{pname}_{i}"
            features[key] = 1 if i == b else 0
        if bin_size == 30 and b < len(BIN_LABELS):
            features[f"sign_{pname}"] = BIN_LABELS[b]

    for i, p1 in enumerate(planet_names):
        for p2 in planet_names[i + 1:]:
            lon1 = positions[p1]["ecl_lon"]
            lon2 = positions[p2]["ecl_lon"]
            sep = angular_sep(lon1, lon2)

            for aspect_name, aspect_angle in aspects.items():
                diff = abs(sep - aspect_angle)
                key = f"aspect_{p1}_{p2}_{aspect_name}"
                features[key] = 1 if diff <= orb_deg else 0

    return features


def generate_and_store(draw_id, positions_json_str, bin_size=30, orb_deg=6.0):
    positions = json.loads(positions_json_str) if isinstance(positions_json_str, str) else positions_json_str
    features = build_features(positions, bin_size=bin_size, orb_deg=orb_deg)
    features_json = json.dumps(features)

    conn = connect()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO planet_features (draw_id, features_json)
            VALUES (?, ?)
        """, (draw_id, features_json))
        conn.commit()
    finally:
        conn.close()

    return features


def generate_bulk(draw_position_rows, bin_size=30, orb_deg=6.0, progress_callback=None):
    computed = 0
    total = len(draw_position_rows)
    for row in draw_position_rows:
        draw_id = row["draw_id"]
        positions_json = row["positions_json"]
        generate_and_store(draw_id, positions_json, bin_size, orb_deg)
        computed += 1
        if progress_callback:
            progress_callback(computed, total)
    return computed
