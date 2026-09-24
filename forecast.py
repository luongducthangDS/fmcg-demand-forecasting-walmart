"""
LightGBM demand forecast + rolling-origin backtest + drift check.

    python forecast.py train     # backtest vs baselines, fit on all history, log to MLflow
    python forecast.py score     # forecast the 12 weeks after the last observed week + drift report

Model is trained at Store x Dept level and aggregated to Store level, so it is scored on
exactly the same target / holdout / WMAE metric as the baselines in analysis.py.
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = "data"
OUT = "outputs"
HORIZON = 12  # weeks forecast ahead; also the minimum lag, so no feature can see the future
N_FOLDS = 3   # rolling-origin backtest folds before the final holdout

# Every lag is >= HORIZON: a row at cutoff + h (h <= 12) only reads sales from <= cutoff.
LAGS = [12, 13, 14, 26, 51, 52, 53]  # 51/53 catch holidays that shift week (Thanksgiving)
# Weather / fuel / CPI / unemployment are left out on purpose: they are not known 12 weeks ahead.
# Markdowns and IsHoliday are planned in advance, so they stay.
MARKDOWNS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
FEATURES = (["Store", "Dept", "Type", "Size", "week", "month", "IsHoliday"]
            + [f"lag_{k}" for k in LAGS] + ["roll4_lag12", "roll12_lag12"] + MARKDOWNS)
CATEGORICAL = ["Store", "Dept", "Type"]
PARAMS = {
    # ponytail: fixed params, chosen on backtest folds only (never on the holdout).
    # Optuna over the same folds if more accuracy is needed.
    # L2 beat L1 and Huber on the fold-1 validation window (store WMAE 51.0k vs 62.4k / 66.5k):
    # errors are summed from dept to store level, where means aggregate better than medians.
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
}


def load():
    train = pd.read_csv(f"{DATA}/train.csv", parse_dates=["Date"])
    feat = pd.read_csv(f"{DATA}/features.csv", parse_dates=["Date"])
    stores = pd.read_csv(f"{DATA}/stores.csv")
    return train, feat, stores


def build_features(sales, feat, stores, dates, series):
    """Feature rows for every (Store, Dept) in `series` x every week in `dates`.

    `sales` holds the only sales history the features may read from.
    """
    wide = sales.pivot_table(index="Date", columns=["Store", "Dept"], values="Weekly_Sales")
    all_weeks = pd.date_range(min(wide.index.min(), min(dates)), max(wide.index.max(), max(dates)), freq="7D")
    wide = wide.reindex(index=all_weeks, columns=pd.MultiIndex.from_frame(series))

    cols = {f"lag_{k}": wide.shift(k) for k in LAGS}
    cols["roll4_lag12"] = wide.shift(HORIZON).rolling(4, min_periods=1).mean()
    cols["roll12_lag12"] = wide.shift(HORIZON).rolling(12, min_periods=1).mean()
    stacked = [c.loc[dates].stack(["Store", "Dept"], future_stack=True).rename(name) for name, c in cols.items()]
    X = pd.concat(stacked, axis=1).reset_index().rename(columns={"level_0": "Date"})

    X = X.merge(feat[["Store", "Date", "IsHoliday"] + MARKDOWNS], on=["Store", "Date"], how="left")
    X = X.merge(stores, on="Store", how="left")
    X["week"] = X["Date"].dt.isocalendar().week.astype(int)
    X["month"] = X["Date"].dt.month
    X["IsHoliday"] = X["IsHoliday"].astype(int)
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def wmae(actual, pred, is_holiday):
    """Kaggle competition metric: holiday weeks weigh 5x."""
    w = np.where(is_holiday, 5, 1)
    return float((w * (actual - pred).abs()).sum() / w.sum())


def weeks_between(start, end):
    return list(pd.date_range(start, end, freq="7D"))


def fit(train, feat, stores, cutoff):
    """Fit on weeks <= cutoff. The 12 weeks before cutoff pick the number of trees, then refit on all."""
    series = train[["Store", "Dept"]].drop_duplicates()
    hist = train[train["Date"] <= cutoff]
    dates = weeks_between(hist["Date"].min(), cutoff)
    X = build_features(hist, feat, stores, dates, series)
    X = X.merge(hist[["Store", "Dept", "Date", "Weekly_Sales"]], on=["Store", "Dept", "Date"], how="inner")
    for c in CATEGORICAL:  # merging on a category key drops the dtype
        X[c] = X[c].astype("category")
    w =np.where(X["IsHoliday"] == 1, 5, 1)

    val_start = cutoff - pd.Timedelta(weeks=HORIZON)
    tr, va = X["Date"] <= val_start, X["Date"] > val_start
    d_tr = lgb.Dataset(X.loc[tr, FEATURES], X.loc[tr, "Weekly_Sales"], weight=w[tr])
    d_va = lgb.Dataset(X.loc[va, FEATURES], X.loc[va, "Weekly_Sales"], weight=w[va], reference=d_tr)
    probe = lgb.train(PARAMS, d_tr, 2000, valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])

    d_all = lgb.Dataset(X[FEATURES], X["Weekly_Sales"], weight=w)
    return lgb.train(PARAMS, d_all, probe.best_iteration), probe.best_iteration


def predict(model, sales, feat, stores, dates, series):
    X = build_features(sales, feat, stores, dates, series)
    X["pred"] = model.predict(X[FEATURES])
    return X


def baselines(train, cutoff):
    """Same two baselines as analysis.py, at Store level."""
    sw = train.groupby(["Store", "Date"])["Weekly_Sales"].sum().reset_index()
    hist = sw[sw["Date"] <= cutoff]
    last_year = sw.assign(Date=sw["Date"] + pd.Timedelta(weeks=52)).set_index(["Store", "Date"])["Weekly_Sales"]
    ma4 = hist[hist["Date"] > cutoff - pd.Timedelta(weeks=4)].groupby("Store")["Weekly_Sales"].mean()
    return last_year, ma4


def evaluate_fold(train, feat, stores, cutoff):
    test_dates = weeks_between(cutoff + pd.Timedelta(weeks=1), cutoff + pd.Timedelta(weeks=HORIZON))
    model, n_trees = fit(train, feat, stores, cutoff)

    test = train[train["Date"].isin(test_dates)]
    series = test[["Store", "Dept"]].drop_duplicates()
    X = predict(model, train[train["Date"] <= cutoff], feat, stores, test_dates, series)
    # score only store-dept-weeks that really exist, same rows the baselines see
    X = X.merge(test[["Store", "Dept", "Date"]], on=["Store", "Dept", "Date"], how="inner")
    X["Store"] = X["Store"].astype(int)
    lgbm = X.groupby(["Store", "Date"])["pred"].sum()

    actual = test.groupby(["Store", "Date"]).agg(y=("Weekly_Sales", "sum"), hol=("IsHoliday", "first"))
    last_year, ma4 = baselines(train, cutoff)
    actual["lgbm"] = lgbm
    actual["seasonal_naive"] = last_year.reindex(actual.index)
    actual["moving_avg"] = actual.index.get_level_values("Store").map(ma4)

    ok = actual["seasonal_naive"].notna()  # same row mask for every method
    a = actual[ok]
    scores = {m: wmae(a["y"], a[m], a["hol"]) for m in ["lgbm", "seasonal_naive", "moving_avg"]}
    return scores, n_trees, actual.reset_index(), model


def psi(expected, actual, bins=10):
    """Population Stability Index. > 0.2 is the usual 'investigate' threshold."""
    # ponytail: hand-rolled PSI on numeric features; switch to Evidently if you need per-feature HTML reports.
    expected, actual = expected.dropna(), actual.dropna()
    if expected.nunique() < 2 or actual.empty:
        return np.nan
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(((a - e) * np.log(a / e)).sum())


def cmd_train():
    import matplotlib.pyplot as plt
    import mlflow

    train, feat, stores = load()
    holdout_cutoff = train["Date"].max() - pd.Timedelta(weeks=HORIZON)
    cutoffs = [holdout_cutoff - pd.Timedelta(weeks=HORIZON * i) for i in range(N_FOLDS, 0, -1)]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("walmart-demand-forecast")
    with mlflow.start_run(run_name="lgbm-store-dept"):
        mlflow.log_params({**PARAMS, "horizon": HORIZON, "lags": LAGS, "n_features": len(FEATURES)})

        folds = []
        for i, c in enumerate(cutoffs, 1):
            s, n_trees, _, _ = evaluate_fold(train, feat, stores, c)
            folds.append({"fold": i, "cutoff": str(c.date()), "n_trees": n_trees, **s})
            mlflow.log_metrics({f"cv_{k}_wmae": v for k, v in s.items()}, step=i)
            print(f"fold {i} cutoff {c.date()}: " + "  ".join(f"{k}={v:,.0f}" for k, v in s.items()))
        cv = pd.DataFrame(folds)
        cv_mean = cv[["lgbm", "seasonal_naive", "moving_avg"]].mean()
        mlflow.log_metrics({f"cv_mean_{k}_wmae": v for k, v in cv_mean.items()})

        hold, n_trees, hold_rows, _ = evaluate_fold(train, feat, stores, holdout_cutoff)
        mlflow.log_metrics({f"holdout_{k}_wmae": v for k, v in hold.items()})
        best_baseline = min(hold["seasonal_naive"], hold["moving_avg"])
        gain = 1 - hold["lgbm"] / best_baseline
        print(f"holdout cutoff {holdout_cutoff.date()}: " + "  ".join(f"{k}={v:,.0f}" for k, v in hold.items()))
        print(f"LightGBM vs best baseline: {gain:+.1%} WMAE reduction")

        # production model: all history, tree count from the latest validation window
        final, final_trees = fit(train, feat, stores, train["Date"].max())
        final.save_model(f"{OUT}/model.txt")
        imp = pd.DataFrame({"feature": FEATURES, "gain": final.feature_importance("gain")})
        imp = imp.sort_values("gain", ascending=False)

        cv.to_csv(f"{OUT}/backtest_folds.csv", index=False)
        hold_rows.to_csv(f"{OUT}/lgbm_holdout_predictions.csv", index=False)
        imp.to_csv(f"{OUT}/feature_importance.csv", index=False)
        summary = {
            "holdout_cutoff": str(holdout_cutoff.date()),
            "holdout_wmae": {k: round(v, 1) for k, v in hold.items()},
            "holdout_reduction_vs_best_baseline": round(float(gain), 4),
            "cv_mean_wmae": {k: round(float(v), 1) for k, v in cv_mean.items()},
            "cv_folds_lgbm_wins": int((cv["lgbm"] < cv[["seasonal_naive", "moving_avg"]].min(axis=1)).sum()),
            "n_folds": N_FOLDS,
            "final_model_trees": final_trees,
        }
        with open(f"{OUT}/model_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        chain = hold_rows.groupby("Date")[["y", "lgbm", "seasonal_naive", "moving_avg"]].sum()
        fig, ax = plt.subplots(figsize=(10, 5))
        chain["y"].plot(ax=ax, color="black", lw=2, label="Actual")
        chain["lgbm"].plot(ax=ax, color="#2E8B57", label=f"LightGBM (WMAE {hold['lgbm']:,.0f})")
        chain["seasonal_naive"].plot(ax=ax, color="#2E86AB", ls="--", label=f"Seasonal-naive ({hold['seasonal_naive']:,.0f})")
        chain["moving_avg"].plot(ax=ax, color="#A23B72", ls=":", label=f"4-wk moving avg ({hold['moving_avg']:,.0f})")
        ax.set_title("Chain-wide sales, 12-week holdout (WMAE is store-level)")
        ax.set_ylabel("Weekly sales ($)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{OUT}/lgbm_holdout.png", dpi=150)
        plt.close()

        for p in ["model.txt", "backtest_folds.csv", "feature_importance.csv", "model_metrics.json", "lgbm_holdout.png"]:
            mlflow.log_artifact(f"{OUT}/{p}")
        print(json.dumps(summary, indent=2))


def cmd_score(cutoff):
    train, feat, stores = load()
    cutoff = pd.Timestamp(cutoff) if cutoff else train["Date"].max()
    model = lgb.Booster(model_file=f"{OUT}/model.txt")
    hist = train[train["Date"] <= cutoff]

    dates = weeks_between(cutoff + pd.Timedelta(weeks=1), cutoff + pd.Timedelta(weeks=HORIZON))
    if max(dates) > feat["Date"].max():
        raise SystemExit(f"features.csv ends {feat['Date'].max().date()}, cannot score up to {max(dates).date()}")
    # series still selling in the last HORIZON weeks; long-dead departments are not forecast
    recent = hist[hist["Date"] > cutoff - pd.Timedelta(weeks=HORIZON)]
    series = recent[["Store", "Dept"]].drop_duplicates()
    X = predict(model, hist, feat, stores, dates, series)
    X[["Store", "Dept", "Date", "IsHoliday", "pred"]].to_csv(f"{OUT}/forecast_next_{HORIZON}w.csv", index=False)
    by_store = X.groupby(["Store", "Date"], observed=True)["pred"].sum().reset_index()
    by_store.to_csv(f"{OUT}/forecast_next_{HORIZON}w_by_store.csv", index=False)

    # drift: scoring window vs the last 52 weeks the model was trained on
    ref_dates = weeks_between(cutoff - pd.Timedelta(weeks=51), cutoff)
    ref = build_features(hist, feat, stores, ref_dates, series)
    numeric = [f for f in FEATURES if f not in CATEGORICAL + ["week", "month", "IsHoliday"]]
    drift = pd.DataFrame({"feature": numeric, "psi": [psi(ref[f], X[f]) for f in numeric]})
    drift["status"] = np.select([drift["psi"] > 0.2, drift["psi"] > 0.1], ["DRIFT", "watch"], "ok")
    drift.to_csv(f"{OUT}/drift_report.csv", index=False)

    print(f"Forecast {dates[0].date()} .. {dates[-1].date()} for {len(series):,} store-dept series "
          f"-> {OUT}/forecast_next_{HORIZON}w.csv")
    print(drift.round(3).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    s = sub.add_parser("score")
    s.add_argument("--cutoff", help="last observed week (YYYY-MM-DD); default = last week in train.csv")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cmd_train() if a.cmd == "train" else cmd_score(a.cutoff)
