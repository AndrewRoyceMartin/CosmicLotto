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
