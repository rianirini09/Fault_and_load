# ============================================================
# BAGIAN B — LOAD MODELS & FUNGSI MACHINE LEARNING
# ============================================================

import joblib
import numpy as np
import pandas as pd
import os


# ============================================================
# LOAD MODELS - FAULT DETECTION
# ============================================================
MODEL_DIR = "models"

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model_stage1 = joblib.load(os.path.join(MODEL_DIR, "model_stage1.pkl"))
    model_stage2 = joblib.load(os.path.join(MODEL_DIR, "model_stage2.pkl"))
    model_stage3 = joblib.load(os.path.join(MODEL_DIR, "model_stage3.pkl"))

except FileNotFoundError as e:
    scaler = None
    model_stage1 = None
    model_stage2 = None
    model_stage3 = None


RAW_FEATURES = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

FAULT_TYPE_LABEL = {
    'LG':   'Single Line-to-Ground (LG)',
    'LL':   'Line-to-Line (LL)',
    'LLG':  'Double Line-to-Ground (LLG)',
    'LLL':  'Three Phase (LLL)',
    'LLLG': 'Three Phase to Ground (LLLG)',
}


# ============================================================
# LOAD MODEL - LOAD FORECASTING
# ============================================================
LOAD_FORECAST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "load_forecast_model.pkl"
)

try:
    load_forecast_bundle = joblib.load(LOAD_FORECAST_MODEL_PATH)

except FileNotFoundError as e:
    load_forecast_bundle = None


# Harus identik dengan yang dipakai saat training
WEATHER_COLS = ["T2M", "QV2M", "TQL", "W2M"]
CALENDAR_COLS = ["Holiday_ID", "holiday", "school"]
LOAD_HORIZON = 24


# ============================================================
# FAULT DETECTION
# ============================================================

def compute_fault_features(data: dict) -> np.ndarray:
    """Hitung 14 fitur: 6 raw + 8 turunan. Identik dengan train_model.py."""
    Ia = float(data['Ia'])
    Ib = float(data['Ib'])
    Ic = float(data['Ic'])

    Va = float(data['Va'])
    Vb = float(data['Vb'])
    Vc = float(data['Vc'])

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

    return np.array([[
        Ia, Ib, Ic,
        Va, Vb, Vc,
        I0, I1, I2,
        V0, V1, V2,
        I0_I1, V0_V1
    ]])


# ============================================================
# LOAD FORECASTING
# ============================================================

def classify_status(peak_pred: float, capacity_mw: float) -> dict:
    ratio = peak_pred / capacity_mw if capacity_mw else 0

    if ratio >= 1.0:
        label, level = "Kritis", "critical"

    elif ratio >= 0.85:
        label, level = "Waspada", "warning"

    else:
        label, level = "Aman", "ok"

    return {
        "label": label,
        "level": level,
        "ratio_pct": round(ratio * 100, 1)
    }


def run_manual_load_forecast(bundle: dict, payload: dict) -> dict:
    """
    Prediksi beban 24 jam ke depan dari input form.

    Memakai 24 model terpisah (jam+1 ... jam+24): tiap jam diprediksi LANGSUNG dari
    data "sekarang" yang kamu isi, tanpa memakai hasil prediksi jam sebelumnya,
    sehingga galat tidak menumpuk.
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

    weather = {
        c: float(payload["weather"][c])
        for c in WEATHER_COLS
    }

    calendar = {
        c: payload["calendar"][c]
        for c in CALENDAR_COLS
    }

    capacity_mw = (
        payload.get("capacity_mw")
        or bundle["default_capacity_mw"]
    )

    forecast_rows = []

    for h in range(1, LOAD_HORIZON + 1):

        target_time = start_time + pd.Timedelta(hours=h)

        row = {
            **weather,
            **calendar,

            "hour": target_time.hour,
            "dayofweek": target_time.dayofweek,
            "month": target_time.month,
            "is_weekend": int(target_time.dayofweek >= 5),

            "d0": h_now,
            "d1": h_1,
            "d2": h_2,
            "d24": d24,
            "d168": d168,
            "m24": m24,
        }

        X_row = pd.DataFrame([row])[
            bundle["feature_cols"]
        ]

        q50 = max(
            float(
                bundle["models"][h - 1].predict(X_row)[0]
            ),
            0
        )

        q10 = max(
            q50 * (
                1 + bundle["rel_q10"][h - 1]
            ),
            0
        )

        q90 = q50 * (
            1 + bundle["rel_q90"][h - 1]
        )

        forecast_rows.append({
            "datetime": target_time,
            "q10": q10,
            "q50": q50,
            "q90": q90
        })

    peak_row = max(
        forecast_rows,
        key=lambda r: r["q50"]
    )

    status = classify_status(
        peak_row["q50"],
        capacity_mw
    )

    actual_history = [
        {
            "datetime": (
                start_time -
                pd.Timedelta(hours=2)
            ).isoformat(),
            "value": round(h_2, 1)
        },

        {
            "datetime": (
                start_time -
                pd.Timedelta(hours=1)
            ).isoformat(),
            "value": round(h_1, 1)
        },

        {
            "datetime": start_time.isoformat(),
            "value": round(h_now, 1)
        }
    ]

    return {
        "current_demand_mw": round(
            h_now,
            1
        ),

        "current_time": start_time.isoformat(),

        "predicted_peak_mw": round(
            peak_row["q50"],
            1
        ),

        "peak_hour": peak_row[
            "datetime"
        ].strftime("%H:%M"),

        "mape_pct": bundle["mape"],

        "capacity_mw": capacity_mw,

        "status": status,

        "actual_history": actual_history,

        "forecast": [
            {
                "datetime": r[
                    "datetime"
                ].isoformat(),

                "q10": round(
                    r["q10"],
                    1
                ),

                "q50": round(
                    r["q50"],
                    1
                ),

                "q90": round(
                    r["q90"],
                    1
                )
            }

            for r in forecast_rows
        ],
    }



# ============================================================
# BAGIAN C — UI STREAMLIT
# ============================================================

import streamlit as st


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="ElectroMind",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "🛡️ Fault Detection"

if "fault_result" not in st.session_state:
    st.session_state.fault_result = None


# ============================================================
# FUNGSI RESET FAULT DETECTION
# ============================================================

def reset_fault_detection():

    st.session_state.fault_result = None

    st.session_state.fault_Ia = None
    st.session_state.fault_Ib = None
    st.session_state.fault_Ic = None
    st.session_state.fault_Va = None
    st.session_state.fault_Vb = None
    st.session_state.fault_Vc = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        font-size: 22px;
        font-weight: 700;
        padding: 10px 0 20px 0;
    ">
        ⚡ElectroMind
    </div>
    """,
    unsafe_allow_html=True
)


page = st.sidebar.radio(
    "Menu",
    [
        "🛡️ Fault Detection",
        "📈 Load Forecasting"
    ],
    key="page"
)


# ============================================================
# STYLE TAMBAHAN
# ============================================================

st.markdown(
    """
    <style>

    /* Menghilangkan animasi/transisi yang membuat
       perpindahan tampilan terasa lambat */
    *,
    *::before,
    *::after {
        transition: none !important;
        animation: none !important;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #888;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    .stage-title {
        font-size: 18px;
        font-weight: 700;
    }

    .stage-label {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .stage-description {
        color: #888;
        margin-bottom: 10px;
    }

    .stat-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        min-height: 110px;
    }

    .stat-label {
        font-size: 14px;
        color: #888;
    }

    .stat-value {
        font-size: 25px;
        font-weight: 700;
        margin-top: 8px;
    }

    .status-ok {
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 600;
        background: rgba(76, 175, 80, 0.15);
    }

    .status-warning {
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 600;
        background: rgba(255, 193, 7, 0.15);
    }

    .status-critical {
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 600;
        background: rgba(244, 67, 54, 0.15);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1 — FAULT DETECTION
# ============================================================

if page == "🛡️ Fault Detection":

    st.markdown(
        '<div class="main-title">⚡ Sistem Deteksi Gangguan Saluran Listrik</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Klasifikasi Bertingkat Berbasis Machine Learning</div>',
        unsafe_allow_html=True
    )

    left_col, right_col = st.columns(2)


    # ========================================================
    # INPUT FAULT DETECTION
    # ========================================================

    with left_col:

        st.subheader("Input Parameter")

        Ia = st.number_input(
            "Arus Fasa A — Ia (A)",
            value=None,
            placeholder="-151.2918124",
            format="%.10f",
            key="fault_Ia"
        )

        Ib = st.number_input(
            "Arus Fasa B — Ib (A)",
            value=None,
            placeholder="-9.677451563",
            format="%.10f",
            key="fault_Ib"
        )

        Ic = st.number_input(
            "Arus Fasa C — Ic (A)",
            value=None,
            placeholder="85.80016226",
            format="%.10f",
            key="fault_Ic"
        )

        Va = st.number_input(
            "Tegangan Fasa A — Va (pu)",
            value=None,
            placeholder="0.400749853",
            format="%.10f",
            key="fault_Va"
        )

        Vb = st.number_input(
            "Tegangan Fasa B — Vb (pu)",
            value=None,
            placeholder="-0.132934945",
            format="%.10f",
            key="fault_Vb"
        )

        Vc = st.number_input(
            "Tegangan Fasa C — Vc (pu)",
            value=None,
            placeholder="-0.267814907",
            format="%.10f",
            key="fault_Vc"
        )


        # ----------------------------------------------------
        # BUTTON
        # ----------------------------------------------------

        predict_fault = st.button(
            "🔍 Prediksi",
            type="primary",
            use_container_width=True,
            key="predict_fault_btn"
        )

        reset_fault = st.button(
            "↺ Reset",
            use_container_width=True,
            key="reset_fault_btn",
            on_click=reset_fault_detection
        )


    # ========================================================
    # PROSES PREDIKSI FAULT
    # ========================================================

    if predict_fault:

        if None in [Ia, Ib, Ic, Va, Vb, Vc]:

            st.session_state.fault_result = {
                "error":
                    "❌ Semua parameter harus diisi."
            }

        elif (
            scaler is None
            or model_stage1 is None
            or model_stage2 is None
            or model_stage3 is None
        ):

            st.session_state.fault_result = {
                "error":
                    "❌ Model fault detection belum berhasil dimuat."
            }

        else:

            try:

                data = {
                    "Ia": Ia,
                    "Ib": Ib,
                    "Ic": Ic,
                    "Va": Va,
                    "Vb": Vb,
                    "Vc": Vc
                }

                X = compute_fault_features(
                    data
                )

                X_scaled = scaler.transform(
                    X
                )

                pred1 = model_stage1.predict(
                    X_scaled
                )[0]

                prob1 = max(
                    model_stage1.predict_proba(
                        X_scaled
                    )[0]
                )

                result = {
                    "stage1": {
                        "label": pred1,

                        "description":
                            "Healthy (Tidak ada gangguan)"
                            if pred1 == "healthy"
                            else
                            "Faulty (Terdeteksi gangguan)",

                        "confidence":
                            round(
                                float(prob1) * 100,
                                2
                            )
                    },

                    "stage2": None,
                    "stage3": None
                }


                # ------------------------------------------------
                # STAGE 2
                # ------------------------------------------------

                if pred1 != "healthy":

                    pred2 = model_stage2.predict(
                        X_scaled
                    )[0]

                    prob2 = max(
                        model_stage2.predict_proba(
                            X_scaled
                        )[0]
                    )

                    result["stage2"] = {
                        "label": pred2,

                        "description":
                            "Symmetrical Fault (Gangguan Simetris)"
                            if pred2 == "symmetrical"
                            else
                            "Unsymmetrical Fault (Gangguan Tidak Simetris)",

                        "confidence":
                            round(
                                float(prob2) * 100,
                                2
                            )
                    }


                    # ------------------------------------------------
                    # STAGE 3
                    # ------------------------------------------------

                    pred3 = model_stage3.predict(
                        X_scaled
                    )[0]

                    prob3 = max(
                        model_stage3.predict_proba(
                            X_scaled
                        )[0]
                    )

                    result["stage3"] = {
                        "label": pred3,

                        "description":
                            FAULT_TYPE_LABEL.get(
                                pred3,
                                pred3
                            ),

                        "confidence":
                            round(
                                float(prob3) * 100,
                                2
                            )
                    }


                st.session_state.fault_result = result


            except ValueError as e:

                st.session_state.fault_result = {
                    "error":
                        f"❌ Input harus berupa angka. Detail: {str(e)}"
                }

            except Exception as e:

                st.session_state.fault_result = {
                    "error":
                        f"❌ Terjadi kesalahan: {str(e)}"
                }


    # ========================================================
    # HASIL FAULT DETECTION
    # ========================================================

    with right_col:

        st.subheader("Hasil Prediksi Bertingkat")

        result = st.session_state.fault_result


        if result:

            # ------------------------------------------------
            # ERROR
            # ------------------------------------------------

            if "error" in result:

                st.error(
                    result["error"]
                )


            # ------------------------------------------------
            # HASIL
            # ------------------------------------------------

            else:

                # =================================================
                # STAGE 1
                # =================================================

                st.markdown(
                    f"""
                    <div class="result-card">

                    <div class="stage-title">
                    1 — Status Sistem
                    </div>

                    <div class="stage-label">
                    {result["stage1"]["label"].upper()}
                    </div>

                    <div class="stage-description">
                    {result["stage1"]["description"]}
                    </div>

                    <b>
                    Confidence:
                    {result["stage1"]["confidence"]:.2f}%
                    </b>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # =================================================
                # STAGE 2
                # =================================================

                if result["stage2"]:

                    st.markdown(
                        f"""
                        <div class="result-card">

                        <div class="stage-title">
                        2 — Kategori Gangguan
                        </div>

                        <div class="stage-label">
                        {result["stage2"]["label"].upper()}
                        </div>

                        <div class="stage-description">
                        {result["stage2"]["description"]}
                        </div>

                        <b>
                        Confidence:
                        {result["stage2"]["confidence"]:.2f}%
                        </b>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # =================================================
                # STAGE 3
                # =================================================

                if result["stage3"]:

                    st.markdown(
                        f"""
                        <div class="result-card">

                        <div class="stage-title">
                        3 — Tipe Gangguan
                        </div>

                        <div class="stage-label">
                        {result["stage3"]["label"]}
                        </div>

                        <div class="stage-description">
                        {result["stage3"]["description"]}
                        </div>

                        <b>
                        Confidence:
                        {result["stage3"]["confidence"]:.2f}%
                        </b>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================
# PAGE 2 — LOAD FORECASTING
# ============================================================

elif page == "📈 Load Forecasting":

    st.markdown(
        '<div class="main-title">📈 Prediksi Beban Listrik</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Load Forecasting 24 Jam ke Depan</div>',
        unsafe_allow_html=True
    )

    left_col, right_col = st.columns(2)


    # ========================================================
    # INPUT LOAD FORECASTING
    # ========================================================

    with left_col:

        st.subheader("Input Parameter")

        target_datetime = st.datetime_input(
            'Waktu "Sekarang" (titik awal prediksi)'
        )

        st.markdown(
            "### Riwayat Beban Terakhir (MW)"
        )

        current_demand = st.number_input(
            "Beban saat ini",
            value=None,
            placeholder="1013.6"
        )

        demand_1h_ago = st.number_input(
            "1 jam sebelumnya",
            value=None,
            placeholder="1040.0"
        )

        demand_2h_ago = st.number_input(
            "2 jam sebelumnya",
            value=None,
            placeholder="1068.3"
        )

        demand_24h_ago = st.number_input(
            "Jam sama, kemarin",
            value=None,
            placeholder="1005.2"
        )

        demand_168h_ago = st.number_input(
            "Jam sama, minggu lalu",
            value=None,
            placeholder="998.7"
        )

        roll_mean_24 = st.number_input(
            "Rata-rata beban 24 jam terakhir",
            value=None,
            placeholder="1195.1"
        )


        # ====================================================
        # CUACA
        # ====================================================

        st.markdown(
            "### Prakiraan Cuaca "
            "(rata-rata untuk 24 jam ke depan)"
        )

        T2M = st.number_input(
            "Suhu (°C)",
            value=None,
            placeholder="26.3"
        )

        QV2M = st.number_input(
            "Kelembaban Spesifik",
            value=None,
            placeholder="0.0177",
            format="%.6f"
        )

        TQL = st.number_input(
            "Awan/Curah (TQL)",
            value=None,
            placeholder="0.11"
        )

        W2M = st.number_input(
            "Kecepatan Angin",
            value=None,
            placeholder="8.0"
        )


        # ====================================================
        # KALENDER & KAPASITAS
        # ====================================================

        st.markdown(
            "### Kalender & Kapasitas"
        )

        Holiday_ID = st.number_input(
            "Kode Hari Libur (Holiday_ID)",
            min_value=0,
            step=1,
            value=0
        )

        capacity_mw = st.number_input(
            "Kapasitas Sistem, MW (opsional)",
            min_value=0.0,
            value=None,
            placeholder="kosongkan = otomatis"
        )

        is_holiday = st.checkbox(
            "Hari Libur Nasional"
        )

        is_school = st.checkbox(
            "Hari Sekolah Aktif",
            value=True
        )


        # ====================================================
        # BUTTON PREDIKSI
        # ====================================================

        predict_load = st.button(
            "🔮 Prediksi Beban",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # HASIL LOAD FORECASTING
    # ========================================================

    with right_col:

        st.subheader("Hasil Prediksi")

        if predict_load:

            required_values = [
                current_demand,
                demand_1h_ago,
                demand_2h_ago,
                demand_24h_ago,
                demand_168h_ago,
                roll_mean_24,
                T2M,
                QV2M,
                TQL,
                W2M
            ]


            if any(
                value is None
                for value in required_values
            ):

                st.error(
                    "❌ Semua parameter wajib diisi."
                )


            elif load_forecast_bundle is None:

                st.error(
                    "❌ Model load forecasting belum tersedia."
                )


            else:

                try:

                    payload = {

                        "target_datetime":
                            target_datetime,

                        "current_demand":
                            current_demand,

                        "demand_1h_ago":
                            demand_1h_ago,

                        "demand_2h_ago":
                            demand_2h_ago,

                        "demand_24h_ago":
                            demand_24h_ago,

                        "demand_168h_ago":
                            demand_168h_ago,

                        "roll_mean_24":
                            roll_mean_24,

                        "capacity_mw":
                            capacity_mw,

                        "weather": {
                            "T2M": T2M,
                            "QV2M": QV2M,
                            "TQL": TQL,
                            "W2M": W2M
                        },

                        "calendar": {
                            "Holiday_ID":
                                Holiday_ID,

                            "holiday":
                                1 if is_holiday
                                else 0,

                            "school":
                                1 if is_school
                                else 0
                        }
                    }


                    result = run_manual_load_forecast(
                        load_forecast_bundle,
                        payload
                    )


                    # =================================================
                    # STAT CARDS
                    # =================================================

                    stat1, stat2 = st.columns(2)
                    stat3, stat4 = st.columns(2)


                    with stat1:

                        st.metric(
                            "Beban saat ini",
                            f'{result["current_demand_mw"]:.0f} MW'
                        )


                    with stat2:

                        st.metric(
                            "Puncak diprediksi",
                            f'{result["predicted_peak_mw"]:.0f} MW'
                        )


                    with stat3:

                        st.metric(
                            "Jam puncak",
                            result["peak_hour"]
                        )


                    with stat4:

                        st.metric(
                            "MAPE rata-rata 24 jam",
                            f'{result["mape_pct"]}%'
                        )


                    # =================================================
                    # CHART
                    # =================================================

                    st.markdown(
                        "### Grafik Beban 24 Jam"
                    )


                    chart_history = pd.DataFrame(
                        result["actual_history"]
                    )

                    chart_forecast = pd.DataFrame(
                        result["forecast"]
                    )


                    chart_history["datetime"] = pd.to_datetime(
                        chart_history["datetime"]
                    )

                    chart_forecast["datetime"] = pd.to_datetime(
                        chart_forecast["datetime"]
                    )


                    chart_history = chart_history[
                        ["datetime", "value"]
                    ].rename(
                        columns={
                            "value": "Aktual"
                        }
                    )


                    chart_forecast = chart_forecast[
                        [
                            "datetime",
                            "q50"
                        ]
                    ].rename(
                        columns={
                            "q50": "Prediksi"
                        }
                    )


                    chart_df = pd.merge(
                        chart_history,
                        chart_forecast,
                        on="datetime",
                        how="outer"
                    ).sort_values(
                        "datetime"
                    )


                    chart_df = chart_df.set_index(
                        "datetime"
                    )


                    st.line_chart(
                        chart_df[
                            ["Aktual", "Prediksi"]
                        ]
                    )


                    # =================================================
                    # INTERVAL Q10-Q90
                    # =================================================

                    st.markdown(
                        "### Interval Prediksi"
                    )


                    forecast_full = pd.DataFrame(
                        result["forecast"]
                    )


                    forecast_full["datetime"] = pd.to_datetime(
                        forecast_full["datetime"]
                    )


                    interval_df = forecast_full[
                        [
                            "datetime",
                            "q10",
                            "q50",
                            "q90"
                        ]
                    ].set_index(
                        "datetime"
                    )


                    interval_df = interval_df.rename(
                        columns={
                            "q10": "Batas Bawah",
                            "q50": "Prediksi",
                            "q90": "Batas Atas"
                        }
                    )


                    st.line_chart(
                        interval_df
                    )


                    # =================================================
                    # STATUS
                    # =================================================

                    status = result["status"]


                    if status["level"] == "ok":

                        st.success(
                            f'🟢 Status: {status["label"]} — '
                            f'Rasio puncak {status["ratio_pct"]}% '
                            f'dari kapasitas '
                            f'({result["capacity_mw"]:.0f} MW)'
                        )


                    elif status["level"] == "warning":

                        st.warning(
                            f'🟡 Status: {status["label"]} — '
                            f'Rasio puncak {status["ratio_pct"]}% '
                            f'dari kapasitas '
                            f'({result["capacity_mw"]:.0f} MW), '
                            f'siapkan cadangan'
                        )


                    else:

                        st.error(
                            f'🔴 Status: {status["label"]} — '
                            f'Rasio puncak {status["ratio_pct"]}% '
                            f'dari kapasitas '
                            f'({result["capacity_mw"]:.0f} MW), '
                            f'siapkan cadangan'
                        )


                except KeyError as e:

                    st.error(
                        f"❌ Field {e} wajib diisi."
                    )


                except ValueError as e:

                    st.error(
                        f"❌ Input harus berupa angka. "
                        f"Detail: {str(e)}"
                    )


                except Exception as e:

                    st.error(
                        f"❌ Gagal menghasilkan prediksi: "
                        f"{str(e)}"
                    )
