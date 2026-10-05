"""
Weather Forecasting: Traditional Regression vs Neural Networks
================================================================
Task   : Forecast next-day mean temperature (and optionally other variables).
Steps  : 1) load data  2) feature engineering  3) time-aware split
         4) train & tune models  5) compare  6) plots + saved results

Usage
-----
  python weather_forecast.py                      # runs on synthetic data
  python weather_forecast.py --csv data.csv --date-col date --target temp
  python weather_forecast.py --csv data.csv --date-col date --target temp --horizon 3

Any CSV with one row per day (or hour) and a date column works. Numeric columns
(humidity, pressure, wind, rainfall...) are used automatically as predictors.
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def make_synthetic_weather(n_days=3650, seed=RANDOM_STATE):
    """Realistic-ish daily weather: seasonality + autocorrelated noise +
    physically related variables (humidity, pressure, wind, rain, cloud)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-01", periods=n_days, freq="D")
    doy = dates.dayofyear.values
    season = np.sin(2 * np.pi * (doy - 110) / 365.25)

    # AR(1) "weather system" noise shared by variables
    noise = np.zeros(n_days)
    for i in range(1, n_days):
        noise[i] = 0.75 * noise[i - 1] + rng.normal(0, 1)

    pressure = 1013 - 4 * noise + rng.normal(0, 1.5, n_days)
    cloud = np.clip(50 + 18 * noise + rng.normal(0, 12, n_days), 0, 100)
    humidity = np.clip(60 + 8 * -season + 0.25 * cloud + rng.normal(0, 5, n_days), 10, 100)
    wind = np.abs(8 + 2.5 * noise + rng.normal(0, 2, n_days))
    rain = np.where(cloud > 70, rng.gamma(2, 3, n_days), 0.0)
    temp = (18 + 11 * season + 1.2 * noise - 0.03 * (cloud - 50)
            + rng.normal(0, 1.2, n_days))

    return pd.DataFrame({"date": dates, "temp": temp, "humidity": humidity,
                         "pressure": pressure, "wind": wind, "cloud": cloud,
                         "rain": rain})


def load_data(args):
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.date_col not in df.columns or args.target not in df.columns:
            raise SystemExit(f"Column not found. Your CSV has these columns:\n{list(df.columns)}\n"
                             f"Use --date-col and --target with exact names.")
        df[args.date_col] = pd.to_datetime(df[args.date_col], dayfirst=args.dayfirst, errors="coerce")
        df = df.dropna(subset=[args.date_col])
        df = df.drop(columns=[c for c in args.drop if c in df.columns])
        df = df.rename(columns={args.date_col: "date", args.target: "temp"})
        df = df.sort_values("date").reset_index(drop=True)
    else:
        print("[info] No CSV given -> using synthetic data.")
        df = make_synthetic_weather()
    # keep numeric columns + date only, drop columns that never change
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[["date"] + num]
    df = df.drop(columns=[c for c in num if c != "temp" and df[c].nunique() <= 1])

    # hourly / sub-daily data -> convert to ONE row per day (daily averages)
    step = df["date"].diff().median()
    if step < pd.Timedelta(days=1):
        print(f"[info] Data is every {step}, converting to daily averages.")
        df = df.set_index("date").resample("D").mean().reset_index()
    # time-based interpolation for small gaps, then drop leftovers
    df = df.set_index("date").interpolate(method="time", limit=3).reset_index()
    return df.dropna().reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def engineer_features(df, horizon=1, lags=(1, 2, 3, 7), windows=(3, 7, 14)):
    """Build a supervised-learning table.

    Every feature uses ONLY information available at day t (no leakage);
    target is temp at day t + horizon.
    """
    out = pd.DataFrame({"date": df["date"]})
    base_cols = [c for c in df.columns if c != "date"]

    # (a) raw current-day values
    for c in base_cols:
        out[c] = df[c]

    # (b) cyclical calendar features - Dec 31 and Jan 1 are neighbours
    doy = df["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    # (c) lag features - memory of the atmosphere
    for c in base_cols:
        for l in lags:
            out[f"{c}_lag{l}"] = df[c].shift(l)

    # (d) rolling statistics - trend & variability
    for c in base_cols:
        for w in windows:
            out[f"{c}_roll_mean{w}"] = df[c].rolling(w).mean()
        out[f"{c}_roll_std7"] = df[c].rolling(7).std()

    # (e) change features - is pressure falling? is it warming up?
    for c in base_cols:
        out[f"{c}_diff1"] = df[c].diff(1)
    if "pressure" in df:
        out["pressure_diff3"] = df["pressure"].diff(3)

    # (f) target
    out["target"] = df["temp"].shift(-horizon)
    return out.dropna().reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. MODELS
# ----------------------------------------------------------------------------
def build_models():
    """Returns {name: (pipeline, param_grid)}. Grids are kept small on purpose."""
    scale = lambda est: Pipeline([("scaler", StandardScaler()), ("model", est)])
    return {
        "Linear Regression": (scale(LinearRegression()), {}),
        "Ridge": (scale(Ridge()), {"model__alpha": [0.1, 1, 10, 100]}),
        "Lasso": (scale(Lasso(max_iter=10000)), {"model__alpha": [0.001, 0.01, 0.1]}),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [200], "max_depth": [8, 16], "min_samples_leaf": [2, 5]},
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1], "max_depth": [3]},
        ),
        "Neural Net (MLP)": (
    scale(MLPRegressor(max_iter=2000, early_stopping=False,
                       random_state=RANDOM_STATE)),
    {"model__hidden_layer_sizes": [(16,), (32,), (32, 16)],
     "model__alpha": [0.1, 1, 10],
     "model__learning_rate_init": [1e-3]},
),
    }


def evaluate(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)}


# ----------------------------------------------------------------------------
# OPTIONAL: LSTM (needs PyTorch: pip install torch)
# ----------------------------------------------------------------------------
def train_lstm(feat_df, feature_cols, split_idx, seq_len=14, epochs=40):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[skip] PyTorch not installed -> LSTM skipped (pip install torch).")
        return None

    torch.manual_seed(RANDOM_STATE)
    scaler = StandardScaler().fit(feat_df.loc[:split_idx - 1, feature_cols])
    X = scaler.transform(feat_df[feature_cols]).astype("float32")
    y = feat_df["target"].values.astype("float32")
    y_mu, y_sd = y[:split_idx].mean(), y[:split_idx].std()
    yn = (y - y_mu) / y_sd

    def seqs(lo, hi):
        xs = [X[i - seq_len:i] for i in range(max(lo, seq_len), hi)]
        ys = [yn[i - 1] for i in range(max(lo, seq_len), hi)]  # target aligned to last step
        return torch.tensor(np.array(xs)), torch.tensor(np.array(ys)).unsqueeze(1)

    Xtr, ytr = seqs(0, split_idx)
    Xte, yte = seqs(split_idx, len(X) + 1)

    class Net(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lstm = nn.LSTM(d, 64, num_layers=2, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.fc(o[:, -1])

    net = Net(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(net(Xtr[idx]), ytr[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte).numpy().ravel() * y_sd + y_mu
    true = yte.numpy().ravel() * y_sd + y_mu
    return true, pred


# ----------------------------------------------------------------------------
# 4. MAIN PIPELINE
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--target", default="temp", help="column to forecast")
    ap.add_argument("--horizon", type=int, default=1, help="days ahead")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out", default="results")
    ap.add_argument("--lstm", action="store_true", help="also train an LSTM (needs torch)")
    ap.add_argument("--drop", nargs="*", default=[], help="columns to ignore, e.g. --drop FeelsLikeC HeatIndexC")
    ap.add_argument("--dayfirst", action="store_true", help="dates look like 25-12-2020 (day first)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- data & features
    df = load_data(args)
    feat = engineer_features(df, horizon=args.horizon)
    feature_cols = [c for c in feat.columns if c not in ("date", "target")]
    print(f"Rows: {len(feat)} | Features: {len(feature_cols)} | Horizon: {args.horizon} day(s)")

    # ---- chronological split (NEVER shuffle time series)
    split = int(len(feat) * (1 - args.test_size))
    train, test = feat.iloc[:split], feat.iloc[split:]
    Xtr, ytr = train[feature_cols], train["target"]
    Xte, yte = test[feature_cols], test["target"]
    print(f"Train: {len(train)} rows | Test: {len(test)} rows "
          f"(test starts {test['date'].iloc[0].date()})")

    results, preds, fitted = [], {}, {}

    # ---- baselines
    baselines = {
        "Persistence (today=tomorrow)": test["temp"].values,
        "Climatology (train mean by month)": test["date"].dt.month.map(
            train.groupby(train["date"].dt.month)["target"].mean()).values,
    }
    for name, p in baselines.items():
        preds[name] = p
        results.append({"Model": name, **evaluate(yte, p), "Best params": "-"})

    # ---- ML models with time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=4)
    for name, (est, grid) in build_models().items():
        print(f"Training {name} ...")
        if grid:
            gs = GridSearchCV(est, grid, cv=tscv,
                              scoring="neg_mean_absolute_error", n_jobs=-1)
            gs.fit(Xtr, ytr)
            best, params = gs.best_estimator_, gs.best_params_
        else:
            best, params = est.fit(Xtr, ytr), {}
        p = best.predict(Xte)
        preds[name], fitted[name] = p, best
        results.append({"Model": name, **evaluate(yte, p), "Best params": str(params)})

    # ---- optional LSTM
    if args.lstm:
        out = train_lstm(feat.reset_index(drop=True), feature_cols, split)
        if out is not None:
            true, p = out
            results.append({"Model": "LSTM", **evaluate(true, p), "Best params": "2x64, seq=14"})
            preds["LSTM"] = np.concatenate([np.full(len(yte) - len(p), np.nan), p])

    # ---- results table
    res = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    res.to_csv(f"{args.out}/model_comparison.csv", index=False)
    print("\n=== MODEL COMPARISON (sorted by RMSE, lower is better) ===")
    print(res[["Model", "MAE", "RMSE", "R2"]].round(3).to_string(index=False))

    # ---- plots
    dates = test["date"].values
    best_name = res[~res["Model"].str.contains("Persistence|Climatology")].iloc[0]["Model"]

    # 1. model comparison
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    r = res.sort_values("RMSE", ascending=False)
    ax[0].barh(r["Model"], r["RMSE"], color="steelblue"); ax[0].set_title("RMSE (lower = better)")
    ax[1].barh(r["Model"], r["MAE"], color="darkorange"); ax[1].set_title("MAE (lower = better)")
    plt.tight_layout(); plt.savefig(f"{args.out}/1_model_comparison.png", dpi=130); plt.close()

    # 2. actual vs predicted (last 120 days)
    n = min(120, len(yte))
    plt.figure(figsize=(12, 4.5))
    plt.plot(dates[-n:], yte.values[-n:], "k", lw=2, label="Actual")
    for m in ["Linear Regression", "Random Forest", "Neural Net (MLP)"]:
        plt.plot(dates[-n:], preds[m][-n:], lw=1.2, label=m)
    plt.title(f"Forecast vs Actual (last {n} test days)"); plt.legend(); plt.ylabel("Temperature")
    plt.tight_layout(); plt.savefig(f"{args.out}/2_forecast_vs_actual.png", dpi=130); plt.close()

    # 3. scatter + residuals for best model
    p = preds[best_name]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(yte, p, s=8, alpha=0.5)
    lo, hi = yte.min(), yte.max(); ax[0].plot([lo, hi], [lo, hi], "r--")
    ax[0].set_xlabel("Actual"); ax[0].set_ylabel("Predicted"); ax[0].set_title(f"{best_name}: predicted vs actual")
    ax[1].hist(yte.values - p, bins=40, color="seagreen"); ax[1].set_title("Residuals")
    plt.tight_layout(); plt.savefig(f"{args.out}/3_best_model_diagnostics.png", dpi=130); plt.close()

    # 4. feature importance (tree model)
    if "Random Forest" in fitted:
        imp = pd.Series(fitted["Random Forest"].feature_importances_, index=feature_cols)
        imp.nlargest(15).sort_values().plot.barh(figsize=(8, 5.5), color="purple")
        plt.title("Random Forest - top 15 features"); plt.tight_layout()
        plt.savefig(f"{args.out}/4_feature_importance.png", dpi=130); plt.close()

    # 5. ablation: does feature engineering help?
    raw_cols = [c for c in feature_cols if not any(k in c for k in ("lag", "roll", "diff", "sin", "cos"))]
    ab = []
    for label, cols in [("Raw features only", raw_cols), ("Engineered features", feature_cols)]:
        m = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=10))]).fit(Xtr[cols], ytr)
        ab.append({"Feature set": label, "n_features": len(cols),
                   **evaluate(yte, m.predict(Xte[cols]))})
    ab = pd.DataFrame(ab); ab.to_csv(f"{args.out}/feature_ablation.csv", index=False)
    print("\n=== FEATURE ENGINEERING ABLATION (Ridge) ===")
    print(ab.round(3).to_string(index=False))

    print(f"\nBest ML model: {best_name}. Outputs saved in ./{args.out}/")


if __name__ == "__main__":
    main()
