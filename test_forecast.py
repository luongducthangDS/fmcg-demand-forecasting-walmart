"""python -m pytest test_forecast.py  (needs data/ from Kaggle)"""
import numpy as np
import pandas as pd

import forecast as F


def test_no_future_leakage():
    # Features for the forecast window must be identical whether or not the future sales exist.
    train, feat, stores = F.load()
    train = train[train["Store"] <= 3]
    cutoff = train["Date"].max() - pd.Timedelta(weeks=F.HORIZON)
    dates = F.weeks_between(cutoff + pd.Timedelta(weeks=1), cutoff + pd.Timedelta(weeks=F.HORIZON))
    series = train[["Store", "Dept"]].drop_duplicates()

    seen = F.build_features(train[train["Date"] <= cutoff], feat, stores, dates, series)
    full = F.build_features(train, feat, stores, dates, series)
    cols = [c for c in F.FEATURES if c.startswith(("lag_", "roll"))]
    pd.testing.assert_frame_equal(seen[cols], full[cols])


def test_wmae_weights_holidays_5x():
    actual, pred = pd.Series([10.0, 10.0]), pd.Series([0.0, 10.0])
    assert F.wmae(actual, pred, pd.Series([True, False])) == 50 / 6
    assert F.wmae(actual, pred, pd.Series([False, False])) == 5


def test_psi_flags_shift_only():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, 5000))
    assert F.psi(base, pd.Series(rng.normal(0, 1, 5000))) < 0.1
    assert F.psi(base, pd.Series(rng.normal(1.5, 1, 5000))) > 0.2
