from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ============================================================
# LOAD MODELS - FAULT DETECTION (sudah ada, tidak diubah)
# ============================================================
MODEL_DIR = "models"
try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model_stage1 = joblib.load(os.path.join(MODEL_DIR, "model_stage1.pkl"))
    model_stage2 = joblib.load(os.path.join(MODEL_DIR, "model_stage2.pkl"))
    model_stage3 = joblib.load(os.path.join(MODEL_DIR, "model_stage3.pkl"))
    print("[INFO] Semua model fault detection (stage 1-3) berhasil dimuat.")
except FileNotFoundError as e:
    print(f"[ERROR] Model fault detection tidak ditemukan: {e}")
    exit()

RAW_FEATURES = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

FAULT_TYPE_LABEL = {
    'LG':   'Single Line-to-Ground (LG)',
    'LL':   'Line-to-Line (LL)',
    'LLG':  'Double Line-to-Ground (LLG)',
    'LLL':  'Three Phase (LLL)',
    'LLLG': 'Three Phase to Ground (LLLG)',
}


# ============================================================
# LOAD MODEL - LOAD FORECASTING (baru)
# ============================================================
LOAD_FORECAST_MODEL_PATH = os.path.join(MODEL_DIR, "load_forecast_model.pkl")

try:
    load_forecast_bundle = joblib.load(LOAD_FORECAST_MODEL_PATH)
    print("[INFO] Model load forecasting berhasil dimuat.")
except FileNotFoundError as e:
    load_forecast_bundle = None
    print(f"[WARNING] Model load forecasting belum ada: {e}. "
          f"Jalankan train_load_forecast.py dulu untuk membuatnya.")

# Harus identik dengan yang dipakai saat training (train_load_forecast.py)
WEATHER_COLS = ["T2M", "QV2M", "TQL", "W2M"]   # rata-rata 3 kota
CALENDAR_COLS = ["Holiday_ID", "holiday", "school"]
LOAD_HORIZON = 24


def compute_fault_features(data: dict) -> np.ndarray:
    """Hitung 14 fitur: 6 raw + 8 turunan. Identik dengan train_model.py."""
    Ia = float(data['Ia']); Ib = float(data['Ib']); Ic = float(data['Ic'])
    Va = float(data['Va']); Vb = float(data['Vb']); Vc = float(data['Vc'])

    a = np.exp(1j * 2 * np.pi / 3)
    a2 = a ** 2

    I0 = abs((Ia + Ib + Ic) / 3)
    I1 = abs((Ia + a*Ib + a2*Ic) / 3)
    I2 = abs((Ia + a2*Ib + a*Ic) / 3)
    V0 = abs((Va + Vb + Vc) / 3)
    V1 = abs((Va + a*Vb + a2*Vc) / 3)
    V2 = abs((Va + a2*Vb + a*Vc) / 3)

    I0_I1 = I0 / (I1 + 1e-9)
    V0_V1 = V0 / (V1 + 1e-9)

    return np.array([[Ia, Ib, Ic, Va, Vb, Vc,
                      I0, I1, I2, V0, V1, V2,
                      I0_I1, V0_V1]])


def classify_status(peak_pred: float, capacity_mw: float) -> dict:
    ratio = peak_pred / capacity_mw if capacity_mw else 0
    if ratio >= 1.0:
        label, level = "Kritis", "critical"
    elif ratio >= 0.85:
        label, level = "Waspada", "warning"
    else:
        label, level = "Aman", "ok"
    return {"label": label, "level": level, "ratio_pct": round(ratio * 100, 1)}


def run_manual_load_forecast(bundle: dict, payload: dict) -> dict:
    """
    Prediksi beban 24 jam ke depan dari input form.

    Memakai 24 model terpisah (jam+1 ... jam+24): tiap jam diprediksi LANGSUNG dari
    data "sekarang" yang kamu isi, tanpa memakai hasil prediksi jam sebelumnya,
    sehingga galat tidak menumpuk.

    payload:
      target_datetime  : waktu "sekarang"
      current_demand, demand_1h_ago, demand_2h_ago,
      demand_24h_ago (jam sama kemarin), demand_168h_ago (jam sama minggu lalu),
      roll_mean_24 (rata-rata beban 24 jam terakhir, WAJIB),
      weather  : {T2M, QV2M, TQL, W2M}  -> prakiraan rata-rata untuk 24 jam ke depan
      calendar : {Holiday_ID, holiday, school}
      capacity_mw (opsional)
    """
    if payload.get("roll_mean_24") in (None, ""):
        raise KeyError("roll_mean_24")

    start_time = pd.to_datetime(payload["target_datetime"])
    h_now = float(payload["current_demand"])
    h_1 = float(payload["demand_1h_ago"])
    h_2 = float(payload["demand_2h_ago"])
    d24 = float(payload["demand_24h_ago"])
    d168 = float(payload["demand_168h_ago"])
    m24 = float(payload["roll_mean_24"])
    weather = {c: float(payload["weather"][c]) for c in WEATHER_COLS}
    calendar = {c: payload["calendar"][c] for c in CALENDAR_COLS}
    capacity_mw = payload.get("capacity_mw") or bundle["default_capacity_mw"]

    forecast_rows = []
    for h in range(1, LOAD_HORIZON + 1):
        target_time = start_time + pd.Timedelta(hours=h)
        row = {
            **weather, **calendar,
            "hour": target_time.hour,
            "dayofweek": target_time.dayofweek,
            "month": target_time.month,
            "is_weekend": int(target_time.dayofweek >= 5),
            "d0": h_now, "d1": h_1, "d2": h_2, "d24": d24, "d168": d168, "m24": m24,
        }
        X_row = pd.DataFrame([row])[bundle["feature_cols"]]
        q50 = max(float(bundle["models"][h - 1].predict(X_row)[0]), 0)
        q10 = max(q50 * (1 + bundle["rel_q10"][h - 1]), 0)
        q90 = q50 * (1 + bundle["rel_q90"][h - 1])
        forecast_rows.append({"datetime": target_time, "q10": q10, "q50": q50, "q90": q90})

    peak_row = max(forecast_rows, key=lambda r: r["q50"])
    status = classify_status(peak_row["q50"], capacity_mw)

    actual_history = [
        {"datetime": (start_time - pd.Timedelta(hours=2)).isoformat(), "value": round(h_2, 1)},
        {"datetime": (start_time - pd.Timedelta(hours=1)).isoformat(), "value": round(h_1, 1)},
        {"datetime": start_time.isoformat(), "value": round(h_now, 1)},
    ]

    return {
        "current_demand_mw": round(h_now, 1),
        "current_time": start_time.isoformat(),
        "predicted_peak_mw": round(peak_row["q50"], 1),
        "peak_hour": peak_row["datetime"].strftime("%H:%M"),
        "mape_pct": bundle["mape"],
        "capacity_mw": capacity_mw,
        "status": status,
        "actual_history": actual_history,
        "forecast": [
            {"datetime": r["datetime"].isoformat(), "q10": round(r["q10"], 1),
             "q50": round(r["q50"], 1), "q90": round(r["q90"], 1)}
            for r in forecast_rows
        ],
    }


# ============================================================
# ROUTES - FAULT DETECTION (sudah ada, tidak diubah)
# ============================================================
@app.route("/")
def index():
    return render_template("index.html", active_page="fault")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        for f in RAW_FEATURES:
            if f not in data or data[f] == "" or data[f] is None:
                return jsonify({"error": f"Field '{f}' wajib diisi."}), 400

        X = compute_fault_features(data)
        X_scaled = scaler.transform(X)

        pred1 = model_stage1.predict(X_scaled)[0]
        prob1 = max(model_stage1.predict_proba(X_scaled)[0])

        result = {
            "stage1": {
                "label": pred1,
                "description": "Healthy (Tidak ada gangguan)" if pred1 == "healthy"
                               else "Faulty (Terdeteksi gangguan)",
                "confidence": round(float(prob1) * 100, 2),
            },
            "stage2": None,
            "stage3": None,
        }

        if pred1 == "healthy":
            return jsonify({"success": True, "result": result})

        pred2 = model_stage2.predict(X_scaled)[0]
        prob2 = max(model_stage2.predict_proba(X_scaled)[0])

        result["stage2"] = {
            "label": pred2,
            "description": "Symmetrical Fault (Gangguan Simetris)"
                           if pred2 == "symmetrical"
                           else "Unsymmetrical Fault (Gangguan Tidak Simetris)",
            "confidence": round(float(prob2) * 100, 2),
        }

        pred3 = model_stage3.predict(X_scaled)[0]
        prob3 = max(model_stage3.predict_proba(X_scaled)[0])

        result["stage3"] = {
            "label": pred3,
            "description": FAULT_TYPE_LABEL.get(pred3, pred3),
            "confidence": round(float(prob3) * 100, 2),
        }

        return jsonify({"success": True, "result": result})

    except ValueError as e:
        return jsonify({"error": f"Input harus berupa angka. Detail: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500


# ============================================================
# ROUTES - LOAD FORECASTING (baru)
# ============================================================
@app.route("/load-forecast")
def load_forecast_page():
    return render_template("load_forecast.html", active_page="load")


@app.route("/api/load-forecast/predict", methods=["POST"])
def load_forecast_predict():
    if load_forecast_bundle is None:
        return jsonify({"error": "Model load forecasting belum tersedia. "
                                  "Jalankan train_load_forecast.py terlebih dahulu."}), 503
    try:
        payload = request.get_json()
        result = run_manual_load_forecast(load_forecast_bundle, payload)
        return jsonify({"success": True, "result": result})
    except KeyError as e:
        return jsonify({"error": f"Field {e} wajib diisi."}), 400
    except ValueError as e:
        return jsonify({"error": f"Input harus berupa angka. Detail: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Gagal menghasilkan prediksi: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)