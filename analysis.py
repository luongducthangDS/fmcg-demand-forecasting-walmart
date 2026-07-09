"""
FMCG Demand Forecasting & Promotion Effectiveness
Dataset: Walmart Recruiting - Store Sales Forecasting (Kaggle)
45 stores, 81 departments, weekly sales 2010-02 to 2012-10.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, json

DATA = "data"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv", parse_dates=["Date"])
feat = pd.read_csv(f"{DATA}/features.csv", parse_dates=["Date"])
stores = pd.read_csv(f"{DATA}/stores.csv")

df = train.merge(feat, on=["Store", "Date"], how="left", suffixes=("", "_feat")).merge(stores, on="Store", how="left")
if "IsHoliday_feat" in df.columns:
    df = df.drop(columns=["IsHoliday_feat"])

print(f"Rows: {len(df):,}  Stores: {df['Store'].nunique()}  Depts: {df['Dept'].nunique()}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ---------- 1. Holiday effect ----------
holiday_avg = df.groupby("IsHoliday")["Weekly_Sales"].mean()
holiday_uplift = holiday_avg[True] / holiday_avg[False] - 1
print(f"\nHoliday week avg sales: {holiday_avg[True]:.0f} vs non-holiday: {holiday_avg[False]:.0f} ({holiday_uplift:+.1%})")

# ---------- 2. Promotion (markdown) effectiveness ----------
# Markdown data only exists from 2011-11-11 onward. Once it starts, almost every
# store-week has SOME markdown active, so a binary has/no-markdown split doesn't
# work -> bucket by markdown $ INTENSITY (store-week total) into terciles instead,
# and compare within holiday / non-holiday separately to avoid confounding.
md_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
df["markdown_total"] = df[md_cols].fillna(0).sum(axis=1)
post_md = df[df["Date"] >= "2011-11-11"].copy()

def bucket(x):
    try:
        return pd.qcut(x, 3, labels=["Low", "Medium", "High"])
    except ValueError:
        return pd.Series(["Low"] * len(x), index=x.index)

post_md["markdown_bucket"] = post_md.groupby("IsHoliday")["markdown_total"].transform(bucket)

promo_effect = post_md.groupby(["IsHoliday", "markdown_bucket"], observed=True)["Weekly_Sales"].mean().unstack("markdown_bucket")
promo_effect.to_csv(f"{OUT}/promo_effect_by_holiday.csv")
print("\n=== Avg weekly sales by markdown intensity tercile, split by holiday flag ===")
print(promo_effect.round(0))
uplift_nonholiday = promo_effect.loc[False, "High"] / promo_effect.loc[False, "Low"] - 1
uplift_holiday = promo_effect.loc[True, "High"] / promo_effect.loc[True, "Low"] - 1
print(f"High vs Low markdown uplift -- non-holiday weeks: {uplift_nonholiday:+.1%} | holiday weeks: {uplift_holiday:+.1%}")

# ---------- 3. Store type performance ----------
type_perf = df.groupby("Type").agg(
    avg_weekly_sales=("Weekly_Sales", "mean"),
    avg_store_size=("Size", "mean"),
    n_stores=("Store", "nunique"),
).round(0)
type_perf.to_csv(f"{OUT}/store_type_performance.csv")
print("\n=== Store type performance ===")
print(type_perf)

# chart: weekly sales trend (chain-wide) with holiday markers
chain_weekly = df.groupby("Date")["Weekly_Sales"].sum()
fig, ax = plt.subplots(figsize=(12, 5))
chain_weekly.plot(ax=ax, color="#2E86AB")
holidays = df[df["IsHoliday"]]["Date"].unique()
for h in holidays:
    ax.axvline(pd.Timestamp(h), color="red", alpha=0.2, linestyle="--")
ax.set_title("Chain-wide weekly sales (red dashed = holiday weeks)")
ax.set_ylabel("Total weekly sales ($)")
plt.tight_layout()
plt.savefig(f"{OUT}/weekly_sales_trend.png", dpi=150)
plt.close()

# ---------- 4. Simple forecast baseline: seasonal-naive vs 4-wk moving avg ----------
store_weekly = df.groupby(["Store", "Date"])["Weekly_Sales"].sum().reset_index()
store_weekly = store_weekly.sort_values(["Store", "Date"])

cutoff = store_weekly["Date"].max() - pd.Timedelta(weeks=12)
train_sw = store_weekly[store_weekly["Date"] <= cutoff]
test_sw = store_weekly[store_weekly["Date"] > cutoff].copy()

sn = store_weekly.copy()
sn["Date_plus_1y"] = sn["Date"] + pd.Timedelta(weeks=52)
sn_lookup = sn.set_index(["Store", "Date_plus_1y"])["Weekly_Sales"]
test_sw["pred_seasonal_naive"] = test_sw.set_index(["Store", "Date"]).index.map(sn_lookup.to_dict())

last4 = train_sw[train_sw["Date"] > cutoff - pd.Timedelta(weeks=4)].groupby("Store")["Weekly_Sales"].mean()
test_sw["pred_moving_avg"] = test_sw["Store"].map(last4)

test_sw = test_sw.merge(df[["Store", "Date", "IsHoliday"]].drop_duplicates(), on=["Store", "Date"], how="left")
test_sw["weight"] = np.where(test_sw["IsHoliday"], 5, 1)  # official Kaggle competition WMAE weighting

def wmae(actual, pred, weight):
    mask = pred.notna()
    return (weight[mask] * (actual[mask] - pred[mask]).abs()).sum() / weight[mask].sum()

wmae_sn = wmae(test_sw["Weekly_Sales"], test_sw["pred_seasonal_naive"], test_sw["weight"])
wmae_ma = wmae(test_sw["Weekly_Sales"], test_sw["pred_moving_avg"], test_sw["weight"])
print(f"\n=== Forecast holdout (last 12 weeks), WMAE (holiday weeks weighted 5x, competition metric) ===")
print(f"Seasonal-naive (same week last year): WMAE = {wmae_sn:,.0f}")
print(f"4-week moving average:                WMAE = {wmae_ma:,.0f}")
better = "Seasonal-naive" if wmae_sn < wmae_ma else "4-week moving average"
print(f"Better baseline: {better}")

test_sw.to_csv(f"{OUT}/forecast_holdout.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(["Seasonal-naive\n(same wk last yr)", "4-week\nmoving avg"], [wmae_sn, wmae_ma], color=["#2E86AB", "#A23B72"])
ax.set_ylabel("WMAE ($, lower=better)")
ax.set_title("Forecast baseline comparison (12-week holdout)")
plt.tight_layout()
plt.savefig(f"{OUT}/forecast_comparison.png", dpi=150)
plt.close()

stats = {
    "rows": len(df),
    "stores": int(df["Store"].nunique()),
    "depts": int(df["Dept"].nunique()),
    "date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
    "holiday_uplift_pct": round(float(holiday_uplift), 4),
    "markdown_uplift_nonholiday_pct": round(float(uplift_nonholiday), 4),
    "markdown_uplift_holiday_pct": round(float(uplift_holiday), 4),
    "wmae_seasonal_naive": round(float(wmae_sn), 1),
    "wmae_moving_avg": round(float(wmae_ma), 1),
    "better_baseline": better,
}
with open(f"{OUT}/headline_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("\n=== Headline stats ===")
print(json.dumps(stats, indent=2))
