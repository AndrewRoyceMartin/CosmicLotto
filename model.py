"""
V2 placeholder: backtesting and prediction module.
Will be implemented in a future phase.
"""


def time_split(draws_df, test_fraction=0.2):
    raise NotImplementedError("V2 feature — not yet implemented.")


def fit_baseline_model(train_df):
    raise NotImplementedError("V2 feature — not yet implemented.")


def fit_planet_model(train_df, features_df):
    raise NotImplementedError("V2 feature — not yet implemented.")


def evaluate_models(test_df, baseline_model, planet_model, features_df):
    raise NotImplementedError("V2 feature — not yet implemented.")


def forecast_for_datetime(dt_utc, lat, lon, planet_model):
    raise NotImplementedError("V2 feature — not yet implemented.")
