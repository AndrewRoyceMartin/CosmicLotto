import json
from datetime import datetime, time, timedelta
from dateutil import tz
from ephemeris import EphemerisEngine
from features import build_features


LOCAL_TZ = tz.gettz("Australia/Sydney")
DEFAULT_DRAW_TIME = time(20, 30)
DRAW_WEEKDAY = 3  # Thursday


def validate_draw_datetime_local(dt_local):
    issues = []
    if dt_local.tzinfo is None:
        issues.append("Datetime is not timezone-aware")
    if dt_local.weekday() != DRAW_WEEKDAY:
        issues.append(f"Not a Thursday (weekday={dt_local.weekday()})")
    if not (dt_local.hour == 20 and dt_local.minute == 30):
        issues.append(f"Time is not 20:30 (got {dt_local.hour:02d}:{dt_local.minute:02d})")
    return (len(issues) == 0, issues)


def get_next_draw_dates(n=10, draw_time=None, timezone_str="Australia/Sydney"):
    if draw_time is None:
        draw_time = DEFAULT_DRAW_TIME

    local_tz = tz.gettz(timezone_str)
    now_local = datetime.now(tz=local_tz)

    today = now_local.date()
    days_until_thursday = (DRAW_WEEKDAY - today.weekday()) % 7
    if days_until_thursday == 0:
        candidate = datetime.combine(today, draw_time).replace(tzinfo=local_tz)
        if candidate <= now_local:
            days_until_thursday = 7
    next_thursday = today + timedelta(days=days_until_thursday)

    draw_dates = []
    for i in range(n):
        draw_date = next_thursday + timedelta(weeks=i)
        dt_local = datetime.combine(draw_date, draw_time).replace(tzinfo=local_tz)
        draw_dates.append(dt_local)

    return draw_dates


def compute_future_positions(draw_dates, lat, lon, altitude_m=0.0, bin_size=30, orb_deg=6.0):
    engine = EphemerisEngine()
    results = []

    for dt_local in draw_dates:
        dt_utc = dt_local.astimezone(tz.UTC)
        positions = engine.compute_positions(dt_utc, lat, lon, altitude_m)
        features = build_features(positions, bin_size=bin_size, orb_deg=orb_deg)

        results.append({
            "dt_local": dt_local,
            "dt_utc": dt_utc,
            "positions": positions,
            "features": features,
        })

    return results


def score_numbers_for_draw(features, analysis_results_df, number_max=35, pb_max=20):
    if analysis_results_df is None or analysis_results_df.empty:
        return [], []

    active_features = set(k for k, v in features.items() if v == 1)

    main_scores = {}
    pb_scores = {}

    for _, row in analysis_results_df.iterrows():
        feat_name = row["feature"]
        number_col = row["number"]
        lift = row.get("lift", 0)
        q_val = row.get("q_value_bh", 1.0)

        if feat_name not in active_features:
            continue

        weight = lift * max(0, 1 - q_val)

        if number_col.startswith("ball_"):
            num = int(number_col.replace("ball_", ""))
            if 1 <= num <= number_max:
                main_scores[num] = main_scores.get(num, 0) + weight
        elif number_col.startswith("pb_"):
            num = int(number_col.replace("pb_", ""))
            if 1 <= num <= pb_max:
                pb_scores[num] = pb_scores.get(num, 0) + weight

    import pandas as pd
    scored_main = pd.DataFrame([
        {"number": k, "score": v} for k, v in main_scores.items()
    ]).sort_values("score", ascending=False).reset_index(drop=True) if main_scores else pd.DataFrame(columns=["number", "score"])

    scored_pb = pd.DataFrame([
        {"number": k, "score": v} for k, v in pb_scores.items()
    ]).sort_values("score", ascending=False).reset_index(drop=True) if pb_scores else pd.DataFrame(columns=["number", "score"])

    return scored_main, scored_pb


def forecast_next_draws(
    n_draws=10,
    lat=-33.8688,
    lon=151.2093,
    altitude_m=0.0,
    location_name="Sydney, Australia",
    timezone_str="Australia/Sydney",
    draw_time=None,
    analysis_results_df=None,
    number_max=35,
    pb_max=20,
    main_count=7,
    bin_size=30,
    orb_deg=6.0,
):
    draw_dates = get_next_draw_dates(n_draws, draw_time, timezone_str)
    future_data = compute_future_positions(draw_dates, lat, lon, altitude_m, bin_size, orb_deg)

    rows = []
    for item in future_data:
        dt_local = item["dt_local"]
        dt_utc = item["dt_utc"]
        features = item["features"]

        scored_main, scored_pb = score_numbers_for_draw(
            features, analysis_results_df, number_max, pb_max
        )

        if len(scored_main) >= main_count:
            main_nums = sorted(scored_main.head(main_count)["number"].tolist())
            main_score_sum = float(scored_main.head(main_count)["score"].sum())
        else:
            main_nums = sorted(scored_main["number"].tolist()) if len(scored_main) > 0 else []
            main_score_sum = float(scored_main["score"].sum()) if len(scored_main) > 0 else 0.0

        if len(scored_pb) > 0:
            pb = int(scored_pb.iloc[0]["number"])
            pb_score = float(scored_pb.iloc[0]["score"])
        else:
            pb = None
            pb_score = 0.0

        is_valid_dt, dt_issues = validate_draw_datetime_local(dt_local)
        utc_offset_hours = dt_local.utcoffset().total_seconds() / 3600 if dt_local.utcoffset() else None

        rows.append({
            "draw_datetime_local": dt_local.isoformat(),
            "draw_datetime_utc": dt_utc.isoformat(),
            "weekday_local": dt_local.strftime("%A"),
            "local_time": dt_local.strftime("%H:%M"),
            "utc_offset_hours": utc_offset_hours,
            "draw_time_alignment_ok": bool(is_valid_dt),
            "draw_time_alignment_issues": "; ".join(dt_issues) if dt_issues else "",
            "location_name": location_name,
            "main_numbers": main_nums,
            "powerball": pb,
            "main_score_sum": main_score_sum,
            "pb_score": pb_score,
            "positions": item["positions"],
            "active_features_count": sum(1 for v in features.values() if v == 1),
        })

    import pandas as pd
    return pd.DataFrame(rows)
