import json
from datetime import datetime, timezone
from dateutil import parser as dtparser
from dateutil import tz


def ensure_tzaware_local(dt_input, timezone_str="Australia/Sydney"):
    if isinstance(dt_input, str):
        dt_input = dtparser.parse(dt_input)
    if dt_input.tzinfo is None:
        local_tz = tz.gettz(timezone_str)
        dt_input = dt_input.replace(tzinfo=local_tz)
    return dt_input


def to_utc(dt_local):
    if dt_local.tzinfo is None:
        raise ValueError("Cannot convert naive datetime to UTC; provide timezone-aware input.")
    return dt_local.astimezone(timezone.utc)


def safe_int(x, default=None):
    try:
        return int(x)
    except (ValueError, TypeError):
        return default


def to_json(d) -> str:
    return json.dumps(d, default=str)


def from_json(s) -> dict:
    if s is None:
        return {}
    return json.loads(s)


def validate_number_range(val, min_val, max_val, label="number"):
    v = safe_int(val)
    if v is None:
        raise ValueError(f"{label} is not a valid integer: {val}")
    if v < min_val or v > max_val:
        raise ValueError(f"{label} {v} out of range [{min_val}, {max_val}]")
    return v


def format_utc_iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def format_local_iso(dt):
    return dt.isoformat()
