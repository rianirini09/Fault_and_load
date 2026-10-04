"""
train_load_forecast.py  (VERSI 2: prediksi langsung 24 jam / "direct multi-horizon")
----------------------------------------------------------------------------------
Berbeda dari versi sebelumnya (1 model dipakai berulang 24x, galat menumpuk),
di sini dilatih 24 MODEL TERPISAH:
    model jam+1  -> khusus memprediksi beban 1 jam ke depan
    model jam+2  -> khusus memprediksi beban 2 jam ke depan
    ...
    model jam+24 -> khusus memprediksi beban 24 jam ke depan
Tiap model melihat data "sekarang" yang sama (hasil input form), sehingga tidak
ada galat yang menumpuk dari prediksi jam sebelumnya.

Input yang dipakai (sama dengan form di web):
    beban sekarang, 1 jam lalu, 2 jam lalu, jam sama kemarin, jam sama minggu lalu,
    rata-rata beban 24 jam terakhir, prakiraan cuaca umum (rata-rata 24 jam ke depan),
    info kalender (Holiday_ID, holiday, school), serta jam/hari/bulan dari jam target.

Cuaca 3 kota di dataset dirata-ratakan jadi 1 set (T2M, QV2M, TQL, W2M).

Interval prediksi: diambil dari sebaran galat relatif (persentil 10 dan 90) tiap
horizon pada data uji, jadi lebar interval ikut melebar untuk jam yang lebih jauh.

Cara pakai:
    python train_load_forecast.py --csv continuous_dataset.csv --out models/load_forecast_model.pkl
"""

import argparse
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZON = 24
TARGET_COL = "nat_demand"
WEATHER_VARS = ["T2M", "QV2M", "TQL", "W2M"]
CITIES = ["toc", "san", "dav"]

FEATURE_COLS = (
    WEATHER_VARS
    + ["Holiday_ID", "holiday", "school"]
    + ["hour", "dayofweek", "month", "is_weekend"]
    + ["d0", "d1", "d2", "d24", "d168", "m24"]
)


def load_hourly(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    full = df.asfreq("h")
    missing = len(full) - len(df)
    if missing > 0:
        print(f"      Peringatan: {missing} jam hilang, diisi dengan interpolasi.")
        full = full.interpolate(limit=6).ffill().bfill()
    full = full.reset_index()
    for v in WEATHER_VARS:  # rata-rata 3 kota -> 1 set cuaca
        full[v] = full[[f"{v}_{c}" for c in CITIES]].mean(axis=1)
    return full


def train(csv_path: str, out_path: str, test_days: int = 30):
    print(f"[1/4] Membaca {csv_path} ...")
    df = load_hourly(csv_path)
    y = df[TARGET_COL].values
    n = len(y)
    times = df["datetime"].values

    cs = np.concatenate([[0], np.cumsum(y)])

    def rmean(i, k):
        return (cs[i + 1] - cs[i + 1 - k]) / k

    def wmean24(v, i):  # rata-rata cuaca 24 jam ke depan (i+1 .. i+24)
        c = np.concatenate([[0], np.cumsum(df[v].values)])
        return (c[i + 1 + HORIZON] - c[i + 1]) / HORIZON

    def features(i, h):
        tt = pd.DatetimeIndex(times[i + h])
        dow = tt.dayofweek.values
        F = {v: wmean24(v, i) for v in WEATHER_VARS}
        F.update({
            "Holiday_ID": df["Holiday_ID"].values[i],
            "holiday": df["holiday"].values[i],
            "school": df["school"].values[i],
            "hour": tt.hour.values, "dayofweek": dow, "month": tt.month.values,
            "is_weekend": (dow >= 5).astype(int),
            "d0": y[i], "d1": y[i - 1], "d2": y[i - 2],
            "d24": y[i - 24], "d168": y[i - 168], "m24": rmean(i, 24),
        })
        return pd.DataFrame(F)[FEATURE_COLS]

    cutoff = df["datetime"].max() - pd.Timedelta(days=test_days)
    idx = np.arange(168, n - HORIZON)
    train_i = idx[times[idx + HORIZON] <= np.datetime64(cutoff)]
    test_i = idx[times[idx] > np.datetime64(cutoff)]
    print(f"[2/4] Titik awal latih: {len(train_i)} | titik awal uji ({test_days} hari terakhir): {len(test_i)}")

    print("[3/4] Melatih 24 model (jam+1 ... jam+24) ...")
    models, mape_h, rel_q10, rel_q90 = [], [], [], []
    for h in range(1, HORIZON + 1):
        m = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.08, max_depth=6,
            l2_regularization=0.1, random_state=42,
        )
        m.fit(features(train_i, h), y[train_i + h])
        pred = m.predict(features(test_i, h))
        actual = y[test_i + h]
        rel = (actual - pred) / pred
        models.append(m)
        mape_h.append(float(np.mean(np.abs(actual - pred) / actual) * 100))
        rel_q10.append(float(np.percentile(rel, 10)))
        rel_q90.append(float(np.percentile(rel, 90)))
        if h in (1, 6, 12, 24):
            print(f"      jam+{h:<2d}: MAPE {mape_h[-1]:.2f}%")

    mape_24 = float(np.mean(mape_h))
    print(f"[4/4] MAPE rata-rata 24 jam: {mape_24:.2f}%  (jam+1: {mape_h[0]:.2f}%, jam+24: {mape_h[-1]:.2f}%)")

    bundle = {
        "version": "direct-24h",
        "models": models,
        "feature_cols": FEATURE_COLS,
        "rel_q10": rel_q10,
        "rel_q90": rel_q90,
        "mape": round(mape_24, 2),
        "mape_1h": round(mape_h[0], 2),
        "mape_by_horizon": [round(v, 2) for v in mape_h],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "default_capacity_mw": round(float(y.max()) * 1.15, 1),
        # default_capacity_mw hanya tebakan (1.15x beban puncak historis).
        # Isi kapasitas riil di form untuk hasil status yang benar.
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"\nModel disimpan ke: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="continuous_dataset.csv")
    parser.add_argument("--out", default="models/load_forecast_model.pkl")
    parser.add_argument("--test-days", type=int, default=30)
    args = parser.parse_args()
    train(args.csv, args.out, args.test_days)