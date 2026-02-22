import pandas as pd
import json
from db import connect
from utils import ensure_tzaware_local, to_utc, safe_int, format_utc_iso, format_local_iso

REQUIRED_COLUMNS = {"draw_datetime_local"}
NUMBER_COLUMNS_7 = ["n1", "n2", "n3", "n4", "n5", "n6", "n7"]
NUMBER_COLUMNS_5 = ["n1", "n2", "n3", "n4", "n5"]
PB_COLUMN = "powerball"

MAIN_NUMBER_MIN = 1
MAIN_NUMBER_MAX_7 = 35
MAIN_NUMBER_MAX_5 = 69
PB_MIN = 1
PB_MAX_7 = 20
PB_MAX_5 = 26


def detect_format(df):
    cols = set(c.strip().lower() for c in df.columns)
    has_7 = all(c in cols for c in NUMBER_COLUMNS_7)
    has_5 = all(c in cols for c in NUMBER_COLUMNS_5) and not has_7
    has_pb = PB_COLUMN in cols
    has_dt = "draw_datetime_local" in cols

    if has_7 and has_pb and has_dt:
        return "au_7", NUMBER_COLUMNS_7, MAIN_NUMBER_MAX_7, PB_MAX_7
    elif has_5 and has_pb and has_dt:
        return "us_5", NUMBER_COLUMNS_5, MAIN_NUMBER_MAX_5, PB_MAX_5
    else:
        return None, None, None, None


def validate_csv_columns(df):
    df.columns = [c.strip().lower() for c in df.columns]
    fmt, num_cols, main_max, pb_max = detect_format(df)
    if fmt is None:
        missing = []
        if "draw_datetime_local" not in df.columns:
            missing.append("draw_datetime_local")
        if PB_COLUMN not in df.columns:
            missing.append("powerball")
        for c in NUMBER_COLUMNS_7:
            if c not in df.columns:
                missing.append(c)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}. "
                             f"Expected columns: draw_datetime_local, n1..n7 (AU) or n1..n5 (US), powerball")
    return fmt, num_cols, main_max, pb_max


def parse_draw_row(row, num_cols, main_max, pb_max, timezone_str="Australia/Sydney"):
    errors = []

    try:
        dt_local = ensure_tzaware_local(row["draw_datetime_local"], timezone_str)
        dt_utc = to_utc(dt_local)
    except Exception as e:
        errors.append(f"datetime error: {e}")
        return None, errors

    numbers = []
    for c in num_cols:
        v = safe_int(row.get(c))
        if v is None or v < MAIN_NUMBER_MIN or v > main_max:
            errors.append(f"{c}={row.get(c)} out of range [1,{main_max}]")
        else:
            numbers.append(v)

    if len(numbers) != len(num_cols):
        return None, errors

    if len(set(numbers)) != len(numbers):
        errors.append(f"Duplicate main numbers: {numbers}")
        return None, errors

    pb = safe_int(row.get(PB_COLUMN))
    if pb is None or pb < PB_MIN or pb > pb_max:
        errors.append(f"powerball={row.get(PB_COLUMN)} out of range [1,{pb_max}]")
        return None, errors

    return {
        "draw_datetime_utc": format_utc_iso(dt_utc),
        "draw_datetime_local": format_local_iso(dt_local),
        "numbers_json": json.dumps(sorted(numbers)),
        "powerball": pb,
        "game": "Powerball",
    }, errors


def upsert_draw(conn, draw_dict):
    import sqlite3
    try:
        conn.execute("""
            INSERT INTO draws (draw_datetime_utc, draw_datetime_local, game, numbers_json, powerball)
            VALUES (:draw_datetime_utc, :draw_datetime_local, :game, :numbers_json, :powerball)
        """, draw_dict)
        return "inserted"
    except sqlite3.IntegrityError:
        return "skipped"


def ingest_from_csv(file_or_path, timezone_str="Australia/Sydney"):
    df = pd.read_csv(file_or_path)
    df.columns = [c.strip().lower() for c in df.columns]
    fmt, num_cols, main_max, pb_max = validate_csv_columns(df)

    conn = connect()
    inserted = 0
    skipped = 0
    error_rows = []

    for idx, row in df.iterrows():
        parsed, errs = parse_draw_row(row, num_cols, main_max, pb_max, timezone_str)
        if parsed is None:
            error_rows.append({"row": idx + 2, "errors": errs})
            continue
        result = upsert_draw(conn, parsed)
        if result == "inserted":
            inserted += 1
        else:
            skipped += 1

    conn.commit()
    conn.close()

    return {
        "total": len(df),
        "inserted": inserted,
        "skipped": skipped,
        "errors": error_rows,
        "format": fmt,
    }
