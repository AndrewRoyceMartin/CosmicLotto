import itertools
import json
import numpy as np
import pandas as pd
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


def score_numbers_from_rules(
    active_features,
    rules_df,
    baseline_probs,
    q_max=0.10,
    min_lift=0.0,
    alpha=2.0,
    target_col="number",
):
    active_set = set(k for k, v in active_features.items() if v == 1)

    scores = {}
    for num, base_p in baseline_probs.items():
        scores[int(num)] = float(base_p)

    if rules_df is not None and len(rules_df) > 0:
        for _, row in rules_df.iterrows():
            feat_name = row["feature"]
            if feat_name not in active_set:
                continue
            q_val = float(row.get("q_value_bh", 1.0))
            lift = float(row.get("lift", 0.0))
            if q_val > q_max or lift < min_lift:
                continue

            num_col = str(row.get("number", row.get(target_col, "")))
            if num_col.startswith("ball_"):
                num = int(num_col.replace("ball_", ""))
            elif num_col.startswith("pb_"):
                num = int(num_col.replace("pb_", ""))
            else:
                continue

            weight = lift * max(0, 1 - q_val) * alpha
            scores[num] = scores.get(num, 0.0) + weight

    col_name = "number" if target_col == "number" else "number"
    rows = [{"number": int(n), "score": float(s)} for n, s in scores.items()]
    out = pd.DataFrame(rows).sort_values(["score", "number"], ascending=[False, True]).reset_index(drop=True)
    return out


def score_numbers_for_draw(features, analysis_results_df, number_max=35, pb_max=20):
    if analysis_results_df is None or analysis_results_df.empty:
        return pd.DataFrame(columns=["number", "score"]), pd.DataFrame(columns=["number", "score"])

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

    scored_main = pd.DataFrame([
        {"number": k, "score": v} for k, v in main_scores.items()
    ]).sort_values("score", ascending=False).reset_index(drop=True) if main_scores else pd.DataFrame(columns=["number", "score"])

    scored_pb = pd.DataFrame([
        {"number": k, "score": v} for k, v in pb_scores.items()
    ]).sort_values("score", ascending=False).reset_index(drop=True) if pb_scores else pd.DataFrame(columns=["number", "score"])

    return scored_main, scored_pb


def confidence_label_from_rank(rank_idx, total_count):
    if total_count <= 0:
        return "Unknown"
    pct = (rank_idx + 1) / total_count
    if pct <= 0.2:
        return "Higher confidence"
    elif pct <= 0.6:
        return "Moderate confidence"
    return "Lower confidence"


def normalize_confidence_0_100(scores):
    arr = np.array(scores, dtype=float)
    if len(arr) == 0:
        return []
    smin = float(arr.min())
    smax = float(arr.max())
    if abs(smax - smin) < 1e-12:
        return [50.0 for _ in arr]
    return [float(100.0 * (s - smin) / (smax - smin)) for s in arr]


def score_main_combination(combo_numbers, score_map):
    nums = sorted(int(n) for n in combo_numbers)
    main_score = float(sum(score_map.get(n, 0.0) for n in nums))

    gaps = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
    spread_bonus = float((max(nums) - min(nums)) / 100.0)
    clustering_penalty = 0.0
    if any(g == 1 for g in gaps):
        clustering_penalty = 0.05

    combined_score = main_score + spread_bonus - clustering_penalty
    return {
        "main_numbers": nums,
        "main_combo_score": combined_score,
        "main_score_raw_sum": main_score,
    }


def generate_top_game_cards_for_draw(
    scored_main_df,
    scored_pb_df,
    main_count=7,
    top_n_games=10,
    combo_pool_main_n=14,
    pb_candidates_n=3,
):
    available = len(scored_main_df)
    if available < main_count:
        return pd.DataFrame()

    effective_pool = min(combo_pool_main_n, available)
    main_pool = scored_main_df.head(effective_pool)
    pb_pool = scored_pb_df.head(pb_candidates_n) if len(scored_pb_df) > 0 else pd.DataFrame(columns=["number", "score"])

    score_map = dict(zip(scored_main_df["number"].astype(int), scored_main_df["score"].astype(float)))
    main_nums_pool = main_pool["number"].astype(int).tolist()

    main_candidates = []
    for combo in itertools.combinations(main_nums_pool, main_count):
        main_candidates.append(score_main_combination(combo, score_map))

    if not main_candidates:
        return pd.DataFrame()

    main_candidates_df = pd.DataFrame(main_candidates).sort_values(
        "main_combo_score", ascending=False
    )

    pb_score_map = dict(zip(scored_pb_df["number"].astype(int), scored_pb_df["score"].astype(float))) if len(scored_pb_df) > 0 else {}

    rows = []
    top_main_for_pairing = main_candidates_df.head(max(200, top_n_games * 20))

    if len(pb_pool) > 0:
        for _, mrow in top_main_for_pairing.iterrows():
            for _, pbrow in pb_pool.iterrows():
                pb = int(pbrow["number"])
                pb_score = float(pb_score_map.get(pb, 0.0))
                game_score = float(mrow["main_combo_score"] + pb_score)

                rows.append({
                    "main_numbers": mrow["main_numbers"],
                    "powerball": pb,
                    "main_combo_score": float(mrow["main_combo_score"]),
                    "pb_score": pb_score,
                    "game_score": game_score,
                })
    else:
        for _, mrow in top_main_for_pairing.iterrows():
            rows.append({
                "main_numbers": mrow["main_numbers"],
                "powerball": None,
                "main_combo_score": float(mrow["main_combo_score"]),
                "pb_score": 0.0,
                "game_score": float(mrow["main_combo_score"]),
            })

    games_df = pd.DataFrame(rows)
    if len(games_df) == 0:
        return games_df

    games_df["main_key"] = games_df["main_numbers"].apply(lambda x: ",".join(map(str, x)))
    games_df["game_key"] = games_df.apply(
        lambda r: f"{r['main_key']}|{int(r['powerball']) if r['powerball'] is not None else 'none'}", axis=1
    )
    games_df = games_df.drop_duplicates(subset=["game_key"]).copy()

    games_df = games_df.sort_values(
        ["game_score", "main_combo_score", "pb_score"], ascending=False
    ).reset_index(drop=True)

    conf_0_100 = normalize_confidence_0_100(games_df["game_score"].tolist())
    games_df["confidence_score_0_100"] = conf_0_100
    games_df["confidence_label"] = [
        confidence_label_from_rank(i, len(games_df)) for i in range(len(games_df))
    ]
    games_df["rank"] = np.arange(1, len(games_df) + 1)

    return games_df.head(top_n_games).drop(columns=["main_key", "game_key"]).copy()


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
    top_n_games_per_draw=10,
    combo_pool_main_n=14,
    pb_candidates_n=3,
):
    draw_dates = get_next_draw_dates(n_draws, draw_time, timezone_str)
    future_data = compute_future_positions(draw_dates, lat, lon, altitude_m, bin_size, orb_deg)

    all_rows = []
    for item in future_data:
        dt_local = item["dt_local"]
        dt_utc = item["dt_utc"]
        features = item["features"]

        scored_main, scored_pb = score_numbers_for_draw(
            features, analysis_results_df, number_max, pb_max
        )

        games_df = generate_top_game_cards_for_draw(
            scored_main_df=scored_main,
            scored_pb_df=scored_pb,
            main_count=main_count,
            top_n_games=top_n_games_per_draw,
            combo_pool_main_n=combo_pool_main_n,
            pb_candidates_n=pb_candidates_n,
        )

        is_valid_dt, dt_issues = validate_draw_datetime_local(dt_local)
        utc_offset_hours = dt_local.utcoffset().total_seconds() / 3600 if dt_local.utcoffset() else None
        active_feat_count = sum(1 for v in features.values() if v == 1)

        if len(games_df) > 0:
            for _, grow in games_df.iterrows():
                all_rows.append({
                    "draw_datetime_local": dt_local.isoformat(),
                    "draw_datetime_utc": dt_utc.isoformat(),
                    "weekday_local": dt_local.strftime("%A"),
                    "local_time": dt_local.strftime("%H:%M"),
                    "utc_offset_hours": utc_offset_hours,
                    "draw_time_alignment_ok": bool(is_valid_dt),
                    "draw_time_alignment_issues": "; ".join(dt_issues) if dt_issues else "",
                    "location_name": location_name,
                    "game_rank_for_draw": int(grow["rank"]),
                    "confidence_score_0_100": float(grow["confidence_score_0_100"]),
                    "confidence_label": str(grow["confidence_label"]),
                    "main_numbers": grow["main_numbers"],
                    "powerball": grow["powerball"],
                    "game_score": float(grow["game_score"]),
                    "main_combo_score": float(grow["main_combo_score"]),
                    "pb_score": float(grow["pb_score"]),
                    "active_features_count": active_feat_count,
                    "insufficient_data": False,
                })

    out = pd.DataFrame(all_rows)
    if len(out):
        out = out.sort_values(
            ["draw_datetime_local", "game_rank_for_draw"],
            ascending=[True, True]
        ).reset_index(drop=True)

    return out
