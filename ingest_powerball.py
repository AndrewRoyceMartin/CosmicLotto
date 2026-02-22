import pandas as pd
import json
import sqlite3
import re
from datetime import datetime, time
from dateutil import tz, parser as dtparser
from db import connect
from utils import safe_int, format_utc_iso, format_local_iso

NUMBER_COLUMNS_7 = ["n1", "n2", "n3", "n4", "n5", "n6", "n7"]
NUMBER_COLUMNS_5 = ["n1", "n2", "n3", "n4", "n5"]
PB_COLUMN = "powerball"

MAIN_NUMBER_MIN = 1
MAIN_NUMBER_MAX_7 = 35
MAIN_NUMBER_MAX_5 = 69
PB_MIN = 1
PB_MAX_7 = 20
PB_MAX_5 = 26

LOCAL_TZ = tz.gettz("Australia/Sydney")
DEFAULT_DRAW_TIME = time(20, 30)

DATE_COLUMN_ALIASES = [
    "draw_datetime_local", "draw date", "date", "draw_date",
    "drawdate", "datetime", "draw datetime", "draw_datetime",
]

PB_COLUMN_ALIASES = [
    "powerball", "pb", "power ball", "power_ball",
    "bonus", "bonus ball", "bonus_ball", "supplementary",
]

BALL_COLUMN_PATTERNS_ORDERED = [
    (r"^(?:winning\s*)?(?:ball|number|num|n)\s*(\d+)$", None),
    (r"^(?:main\s*)?(?:ball|number|num)\s*(\d+)$", None),
    (r"^n(\d+)$", None),
]


def normalize_columns(df):
    df.columns = [c.strip().lower() for c in df.columns]

    col_map = {}
    original_cols = list(df.columns)

    date_col_found = False
    for alias in DATE_COLUMN_ALIASES:
        if alias in original_cols:
            col_map[alias] = "draw_datetime_local"
            date_col_found = True
            break

    if not date_col_found:
        for col in original_cols:
            if "date" in col or "time" in col or "draw" in col:
                col_map[col] = "draw_datetime_local"
                date_col_found = True
                break

    pb_col_found = False
    for alias in PB_COLUMN_ALIASES:
        if alias in original_cols:
            col_map[alias] = "powerball"
            pb_col_found = True
            break

    if not pb_col_found:
        for col in original_cols:
            if "power" in col or col == "pb":
                col_map[col] = "powerball"
                pb_col_found = True
                break

    ball_cols = {}
    mapped_cols = set(col_map.keys())
    for col in original_cols:
        if col in mapped_cols:
            continue
        for pattern, _ in BALL_COLUMN_PATTERNS_ORDERED:
            m = re.match(pattern, col)
            if m:
                idx = int(m.group(1))
                ball_cols[idx] = col
                break

    if not ball_cols:
        remaining = [c for c in original_cols if c not in mapped_cols
                     and c != "powerball" and c not in col_map.values()]
        numeric_candidates = []
        for c in remaining:
            try:
                sample = df[c].dropna().head(20)
                if sample.apply(lambda x: str(x).strip().isdigit()).all():
                    numeric_candidates.append(c)
            except Exception:
                pass

        for i, c in enumerate(numeric_candidates):
            ball_cols[i + 1] = c

    for idx in sorted(ball_cols.keys()):
        col_map[ball_cols[idx]] = f"n{idx}"

    if col_map:
        df = df.rename(columns=col_map)

    return df, col_map


def detect_format(df):
    cols = set(df.columns)
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
    fmt, num_cols, main_max, pb_max = detect_format(df)
    if fmt is None:
        missing = []
        if "draw_datetime_local" not in df.columns:
            missing.append("draw_datetime_local (or 'Draw Date', 'Date')")
        if PB_COLUMN not in df.columns:
            missing.append("powerball (or 'Powerball', 'PB')")

        detected_n = [c for c in df.columns if re.match(r"^n\d+$", c)]
        if len(detected_n) < 5:
            missing.append("ball number columns (e.g., 'Ball 1'...'Ball 7' or 'n1'...'n7')")

        actual_cols = ", ".join(df.columns.tolist()[:15])
        raise ValueError(
            f"Could not detect CSV format. Missing: {', '.join(missing)}. "
            f"Detected columns: [{actual_cols}]"
        )
    return fmt, num_cols, main_max, pb_max


def parse_draw_date(date_str, draw_time_local=None, timezone_str="Australia/Sydney"):
    date_str = str(date_str).strip()
    local_tz = tz.gettz(timezone_str)

    try:
        dt = dtparser.parse(date_str, dayfirst=True)
    except Exception:
        raise ValueError(f"Cannot parse date: {date_str}")

    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        use_time = draw_time_local if draw_time_local else DEFAULT_DRAW_TIME
        dt = datetime.combine(dt.date(), use_time)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=local_tz)

    return dt


def parse_draw_row(row, num_cols, main_max, pb_max, timezone_str="Australia/Sydney", draw_time_local=None):
    errors = []

    try:
        dt_local = parse_draw_date(row["draw_datetime_local"], draw_time_local, timezone_str)
        dt_utc = dt_local.astimezone(tz.UTC)
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
    try:
        conn.execute("""
            INSERT INTO draws (draw_datetime_utc, draw_datetime_local, game, numbers_json, powerball)
            VALUES (:draw_datetime_utc, :draw_datetime_local, :game, :numbers_json, :powerball)
        """, draw_dict)
        return "inserted"
    except sqlite3.IntegrityError:
        return "skipped"


def ingest_from_csv(file_or_path, timezone_str="Australia/Sydney", draw_time_local=None):
    df = pd.read_csv(file_or_path)
    df, col_mapping = normalize_columns(df)
    fmt, num_cols, main_max, pb_max = validate_csv_columns(df)

    conn = connect()
    inserted = 0
    skipped = 0
    error_rows = []

    for idx, row in df.iterrows():
        parsed, errs = parse_draw_row(row, num_cols, main_max, pb_max, timezone_str, draw_time_local)
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
        "column_mapping": col_mapping,
    }
