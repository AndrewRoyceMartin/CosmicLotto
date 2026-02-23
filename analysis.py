import json
import math
import numpy as np
import pandas as pd
from scipy import stats


def build_feature_matrix(draws_df, feats_df):
    merged = draws_df.merge(feats_df, on="draw_id", how="inner")

    feature_dicts = []
    for _, row in merged.iterrows():
        fd = json.loads(row["features_json"])
        fd_numeric = {k: v for k, v in fd.items() if isinstance(v, (int, float))}
        feature_dicts.append(fd_numeric)

    if not feature_dicts:
        return pd.DataFrame(), merged

    X = pd.DataFrame(feature_dicts)
    X.index = merged["draw_id"].values
    return X, merged


def build_number_indicators(draws_df, number_min, number_max, pb_max=26):
    indicators = {}
    for num in range(number_min, number_max + 1):
        col_name = f"ball_{num}"
        indicators[col_name] = draws_df["numbers_json"].apply(
            lambda s: 1 if num in json.loads(s) else 0
        )

    pb_min = 1
    for num in range(pb_min, pb_max + 1):
        col_name = f"pb_{num}"
        indicators[col_name] = (draws_df["powerball"] == num).astype(int)

    Y = pd.DataFrame(indicators)
    Y.index = draws_df.index
    return Y


def two_proportion_ztest(k1, n1, k0, n0):
    if n1 == 0 or n0 == 0:
        return 0.0, 1.0
    p1 = k1 / n1
    p0 = k0 / n0
    p_pool = (k1 + k0) / (n1 + n0)
    if p_pool == 0 or p_pool == 1:
        return 0.0, 1.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p0) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


def benjamini_hochberg(pvals):
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    q_values = np.zeros(n)

    for i in range(n):
        rank = i + 1
        q_values[i] = sorted_pvals[i] * n / rank

    for i in range(n - 2, -1, -1):
        q_values[i] = min(q_values[i], q_values[i + 1])

    q_values = np.minimum(q_values, 1.0)
    result = np.zeros(n)
    result[sorted_idx] = q_values
    return result


def feature_number_scan(draws_df, feats_df, number_min=1, number_max=35, pb_max=26, min_group_size=5):
    X, merged = build_feature_matrix(draws_df, feats_df)
    if X.empty:
        return pd.DataFrame()

    merged_indexed = merged.set_index("draw_id")
    aligned_draws = merged_indexed.loc[X.index]

    Y = build_number_indicators(aligned_draws, number_min, number_max, pb_max=pb_max)

    results = []
    feature_cols = [c for c in X.columns if X[c].nunique() > 1]

    for feat in feature_cols:
        mask1 = X[feat] == 1
        mask0 = X[feat] == 0
        n1 = mask1.sum()
        n0 = mask0.sum()

        if n1 < min_group_size or n0 < min_group_size:
            continue

        for num_col in Y.columns:
            k1 = Y.loc[mask1.values, num_col].sum()
            k0 = Y.loc[mask0.values, num_col].sum()
            rate1 = k1 / n1 if n1 > 0 else 0
            rate0 = k0 / n0 if n0 > 0 else 0
            lift = (rate1 / rate0 - 1) if rate0 > 0 else 0

            z, p_val = two_proportion_ztest(k1, n1, k0, n0)

            results.append({
                "feature": feat,
                "number": num_col,
                "n1": int(n1),
                "n0": int(n0),
                "k1": int(k1),
                "k0": int(k0),
                "rate1": round(rate1, 4),
                "rate0": round(rate0, 4),
                "lift": round(lift, 4),
                "z_score": round(z, 4),
                "p_value": p_val,
            })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df["q_value_bh"] = benjamini_hochberg(results_df["p_value"].values)
    results_df = results_df.sort_values("p_value").reset_index(drop=True)
    return results_df


def feature_powerball_scan(draws_df, feats_df, pb_min=1, pb_max=20, min_group_size=5):
    X, merged = build_feature_matrix(draws_df, feats_df)
    if X.empty:
        return pd.DataFrame()

    merged_indexed = merged.set_index("draw_id")
    aligned_draws = merged_indexed.loc[X.index]

    indicators = {}
    for num in range(pb_min, pb_max + 1):
        col_name = f"pb_{num}"
        indicators[col_name] = (aligned_draws["powerball"] == num).astype(int)
    Y = pd.DataFrame(indicators)
    Y.index = aligned_draws.index

    results = []
    feature_cols = [c for c in X.columns if X[c].nunique() > 1]

    for feat in feature_cols:
        mask1 = X[feat] == 1
        mask0 = X[feat] == 0
        n1 = mask1.sum()
        n0 = mask0.sum()

        if n1 < min_group_size or n0 < min_group_size:
            continue

        for num_col in Y.columns:
            k1 = Y.loc[mask1.values, num_col].sum()
            k0 = Y.loc[mask0.values, num_col].sum()
            rate1 = k1 / n1 if n1 > 0 else 0
            rate0 = k0 / n0 if n0 > 0 else 0
            lift = (rate1 / rate0 - 1) if rate0 > 0 else 0

            z, p_val = two_proportion_ztest(k1, n1, k0, n0)

            results.append({
                "feature": feat,
                "number": num_col,
                "n1": int(n1),
                "n0": int(n0),
                "k1": int(k1),
                "k0": int(k0),
                "rate1": round(rate1, 4),
                "rate0": round(rate0, 4),
                "lift": round(lift, 4),
                "z_score": round(z, 4),
                "p_value": p_val,
            })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df["q_value_bh"] = benjamini_hochberg(results_df["p_value"].values)
    results_df = results_df.sort_values("p_value").reset_index(drop=True)
    return results_df


def humanize_feature_name(feature_name):
    if feature_name.startswith("aspect_"):
        parts = feature_name.split("_")
        if len(parts) >= 4:
            _, p1, p2, aspect = parts[0], parts[1], parts[2], parts[3]
            return f"{p1.title()}\u2013{p2.title()} {aspect.title()} aspect"
    if feature_name.startswith("bin_"):
        parts = feature_name.split("_")
        if len(parts) >= 3:
            _, planet, bin_idx = parts[0], parts[1], parts[2]
            try:
                b = int(bin_idx)
                start_deg = b * 30
                end_deg = start_deg + 30
                return f"{planet.title()} in longitude bin {start_deg}\u00b0\u2013{end_deg}\u00b0"
            except Exception:
                return f"{planet.title()} longitude bin {bin_idx}"
    return feature_name


def summarize_correlations_plain_english(
    results_df,
    top_n=10,
    q_threshold=0.05,
    min_abs_lift=0.005,
    target_label="main numbers"
):
    if results_df is None or len(results_df) == 0:
        return f"No analysis results were available for {target_label}."

    df = results_df.copy()

    if "q_value_bh" in df.columns:
        df = df[df["q_value_bh"] <= q_threshold]
    if "lift" in df.columns:
        df = df[df["lift"].abs() >= min_abs_lift]

    if len(df) == 0:
        return (
            f"No statistically strong correlations were identified for {target_label} "
            f"after multiple-comparisons correction (q \u2264 {q_threshold}). "
            "Any apparent relationships in the raw scan are likely weak or due to chance."
        )

    df = df.sort_values(
        by=["q_value_bh", "lift"],
        ascending=[True, False]
    ).copy()

    lines = []
    lines.append(
        f"The analysis identified **{len(df)}** statistically filtered correlation signals for {target_label} "
        f"(q \u2264 {q_threshold}, |lift| \u2265 {min_abs_lift:.3f})."
    )

    top = df.head(top_n)
    for _, row in top.iterrows():
        feature_raw = str(row.get("feature", "unknown feature"))
        feature = humanize_feature_name(feature_raw)

        number_col = str(row.get("number", ""))
        if number_col.startswith("ball_"):
            target_val = f"main number {number_col.replace('ball_', '')}"
        elif number_col.startswith("pb_"):
            target_val = f"Powerball {number_col.replace('pb_', '')}"
        else:
            target_val = number_col

        rate1 = float(row.get("rate1", 0.0))
        rate0 = float(row.get("rate0", 0.0))
        lift = float(row.get("lift", 0.0))
        qv = float(row.get("q_value_bh", 1.0))

        direction = "more often" if lift > 0 else "less often"
        lines.append(
            f"- When **{feature}** was present, {target_val} appeared {direction} "
            f"than baseline (feature-present rate {rate1:.3f} vs feature-absent rate {rate0:.3f}; "
            f"lift {lift:+.3f}, q={qv:.4f})."
        )

    lines.append("")
    lines.append(
        "*These are statistical associations only and do not establish causation. "
        "They should be validated with out-of-sample backtesting before using them as forecast rules.*"
    )

    return "\n".join(lines)
