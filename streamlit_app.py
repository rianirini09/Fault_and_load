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
import base64
from datetime import datetime
 
 
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
# KONSTANTA NAVIGASI
# ============================================================
 
PAGE_DASHBOARD = "🏠 Dashboard"
PAGE_FAULT = "🛡️ Fault Detection"
PAGE_LOAD = "📈 Load Forecasting"
 
 
# ============================================================
# SESSION STATE
# ============================================================
 
if "page" not in st.session_state:
    st.session_state.page = PAGE_DASHBOARD
 
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
# FUNGSI BANTU UI DASHBOARD (khusus tampilan)
# ============================================================
 
def go_to_page(target_page: str):
    """Pindah halaman dari tombol di dashboard."""
    st.session_state.page = target_page
 
 
def _flat(html: str) -> str:
    """Ratakan HTML supaya tidak terbaca sebagai code block oleh markdown."""
    return " ".join(line.strip() for line in html.splitlines() if line.strip())
 
 
def _svg_uri(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"
 
 
def _image_uri(filename: str, fallback_svg: str) -> str:
    """
    Pakai gambar asli dari folder assets/ bila ada
    (assets/fault_detection.png, assets/load_forecasting.png),
    kalau tidak ada pakai ilustrasi bawaan.
    """
    path = os.path.join("assets", filename)
 
    if os.path.exists(path):
        ext = os.path.splitext(filename)[1].lower().replace(".", "")
        ext = "jpeg" if ext == "jpg" else ext
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{ext};base64,{encoded}"
 
    return _svg_uri(fallback_svg)
 
 
BOLT_PATH = "M13 2 4 14h6l-1 8 9-12h-6z"
 
SVG_BOLT_LIGHT = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
    f'<path d="{BOLT_PATH}" fill="#5aa9ff"/></svg>'
)
 
SVG_BOLT_BLUE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
    f'<path d="{BOLT_PATH}" fill="#1565f0"/></svg>'
)
 
SVG_CHART_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="#ffffff" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M3 20V9l5 4 5-7 4 4 4-6"/>'
    '<path d="M3 20h18"/><path d="M3 20V9h3v11M10 20v-5h3v5"/></svg>'
)
 
SVG_TOWERS_SIDEBAR = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 120" '
    'fill="none" stroke="#3f78c9" stroke-width="1" opacity="0.8">'
    '<path d="M130 10 L118 110 M130 10 L142 110 M118 110 L142 110 '
    'M122 70 L138 70 M120 90 L140 90 M125 40 L135 40 '
    'M122 70 L140 90 M138 70 L120 90 M126 40 L138 70 M134 40 L122 70 '
    'M112 30 L148 30 M116 22 L144 22"/>'
    '<path d="M60 50 L52 110 M60 50 L68 110 M52 110 L68 110 '
    'M54 85 L66 85 M54 85 L68 110 M66 85 L52 110 M50 62 L70 62"/>'
    '<path d="M0 90 Q60 60 130 30 T200 45 M0 100 Q70 75 130 22 T200 38"/>'
    '</svg>'
)
 
SVG_FAULT_IMAGE = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 306">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#14306f"/>
      <stop offset="0.45" stop-color="#3a5db0"/>
      <stop offset="0.75" stop-color="#d98a6c"/>
      <stop offset="1" stop-color="#f7b267"/>
    </linearGradient>
  </defs>
  <rect width="320" height="306" fill="url(#sky)"/>
  <g stroke="#0a1428" stroke-width="1.2" fill="none" opacity="0.85">
    <path d="M0 150 Q120 190 200 105 T320 120"/>
    <path d="M0 165 Q120 205 200 120 T320 138"/>
    <path d="M0 180 Q120 218 200 135 T320 156"/>
    <path d="M0 120 Q100 150 200 80 T320 70"/>
  </g>
  <path d="M0 150 Q120 190 200 105 T320 120" stroke="#6fb0ff" stroke-width="0.8" fill="none" opacity="0.6"/>
  <g stroke="#0a1428" stroke-width="3" fill="none" stroke-linecap="round">
    <path d="M200 20 L178 290 M200 20 L222 290 M178 290 L222 290"/>
    <path d="M186 150 L214 150 M182 200 L218 200 M190 100 L210 100 M193 70 L207 70"/>
    <path d="M190 100 L214 150 M210 100 L186 150 M186 150 L218 200 M214 150 L182 200"/>
    <path d="M160 80 L240 80 M168 105 L232 105"/>
    <path d="M200 20 L200 8"/>
  </g>
  <g stroke="#0a1428" stroke-width="2.4" fill="none" stroke-linecap="round">
    <path d="M58 170 L44 290 M58 170 L72 290 M44 290 L72 290"/>
    <path d="M48 230 L68 230 M50 205 L66 205"/>
    <path d="M48 230 L72 290 M68 230 L44 290"/>
    <path d="M36 185 L80 185"/>
  </g>
  <path d="M0 296 Q80 284 160 292 T320 288 L320 306 L0 306Z" fill="#0a1428"/>
</svg>
"""
 
SVG_LOAD_IMAGE = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 306">
  <defs>
    <linearGradient id="sky2" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#cfe0fa"/>
      <stop offset="1" stop-color="#eef4fd"/>
    </linearGradient>
    <linearGradient id="gnd" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#6c93c9"/>
      <stop offset="1" stop-color="#3b6aa8"/>
    </linearGradient>
  </defs>
  <rect width="320" height="306" fill="url(#sky2)"/>
  <g fill="#b9cdea">
    <rect x="190" y="110" width="42" height="150"/>
    <rect x="236" y="80" width="36" height="180"/>
    <rect x="276" y="125" width="44" height="135"/>
    <rect x="150" y="150" width="34" height="110"/>
  </g>
  <g fill="#e8f0fb" opacity="0.9">
    <rect x="198" y="122" width="6" height="8"/><rect x="210" y="122" width="6" height="8"/>
    <rect x="198" y="142" width="6" height="8"/><rect x="210" y="142" width="6" height="8"/>
    <rect x="244" y="94" width="6" height="8"/><rect x="256" y="94" width="6" height="8"/>
    <rect x="244" y="114" width="6" height="8"/><rect x="256" y="114" width="6" height="8"/>
    <rect x="284" y="140" width="6" height="8"/><rect x="298" y="140" width="6" height="8"/>
  </g>
  <g stroke="#3b6aa8" stroke-width="2.4" fill="none" stroke-linecap="round">
    <path d="M70 40 L40 270 M70 40 L100 270 M40 270 L100 270"/>
    <path d="M52 170 L88 170 M46 215 L94 215 M58 120 L82 120 M62 85 L78 85"/>
    <path d="M58 120 L88 170 M82 120 L52 170 M52 170 L94 215 M88 170 L46 215"/>
    <path d="M30 70 L110 70 M38 95 L102 95"/>
  </g>
  <rect x="0" y="262" width="320" height="44" fill="url(#gnd)"/>
  <g fill="#3b6aa8" opacity="0.8">
    <rect x="10" y="245" width="60" height="30"/>
    <rect x="25" y="232" width="8" height="14"/><rect x="45" y="232" width="8" height="14"/>
  </g>
  <polyline points="170,150 205,128 235,98 262,86 285,62 308,44"
            stroke="#1d5fe0" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <polygon points="308,44 296,46 303,56" fill="#1d5fe0"/>
  <g fill="#ffffff" opacity="0.85">
    <rect x="236" y="262" width="12" height="30"/>
    <rect x="256" y="248" width="12" height="44"/>
    <rect x="276" y="232" width="12" height="60"/>
    <rect x="296" y="214" width="12" height="78"/>
  </g>
</svg>
"""
 
 
# ============================================================
# SIDEBAR
# ============================================================
 
st.sidebar.markdown(
    _flat(
        f"""
        <div class="brand">
            <img class="brand-icon" src="{_svg_uri(SVG_BOLT_LIGHT)}"/>
            <div>
                <div class="brand-name">ElectroMind</div>
                <div class="brand-sub">Electrical Analysis Platform</div>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True
)
 
 
page = st.sidebar.radio(
    "Menu",
    [
        PAGE_DASHBOARD,
        PAGE_FAULT,
        PAGE_LOAD
    ],
    key="page",
    label_visibility="collapsed"
)
 
 
st.sidebar.markdown(
    _flat(
        f"""
        <div class="side-spacer"></div>
        <img class="side-towers" src="{_svg_uri(SVG_TOWERS_SIDEBAR)}"/>
        <div class="side-tagline">
            Analisis Cerdas<br>untuk Sistem Kelistrikan<br>yang Lebih Andal
        </div>
        <div class="side-version">v1.0.0</div>
        """
    ),
    unsafe_allow_html=True
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
 
    /* ---------------- SIDEBAR (navy) ---------------- */
 
    section[data-testid="stSidebar"] {
        background: #0b2a5b;
    }
 
    section[data-testid="stSidebar"] > div {
        background: #0b2a5b;
    }
 
    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 4px 22px 4px;
    }
 
    .brand-icon {
        width: 38px;
        height: 38px;
    }
 
    .brand-name {
        font-size: 20px;
        font-weight: 700;
        color: #5aa9ff;
        line-height: 1.1;
    }
 
    .brand-sub {
        font-size: 11.5px;
        color: #cfe0fa;
    }
 
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 6px;
    }
 
    section[data-testid="stSidebar"] label[data-baseweb="radio"] {
        width: 100%;
        margin: 0;
        padding: 11px 14px;
        border-radius: 10px;
        cursor: pointer;
    }
 
    /* Hapus lingkaran radio bawaan Streamlit */
    section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:not(:has([data-testid="stMarkdownContainer"])),
    section[data-testid="stSidebar"] label[data-baseweb="radio"] > span:not(:has([data-testid="stMarkdownContainer"])) {
        display: none !important;
    }
 
    /* Teks menu putih */
    section[data-testid="stSidebar"] label[data-baseweb="radio"] p,
    section[data-testid="stSidebar"] label[data-baseweb="radio"] span,
    section[data-testid="stSidebar"] label[data-baseweb="radio"] div {
        color: #ffffff !important;
        font-size: 15px;
        margin: 0;
    }
 
    section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
        background: rgba(255,255,255,0.08);
    }
 
    section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
        background: #1565f0;
    }
 
    /* Sembunyikan tombol Deploy & menu titik tiga bawaan Streamlit */
    [data-testid="stAppDeployButton"],
    [data-testid="stToolbarActions"],
    [data-testid="stMainMenu"],
    .stDeployButton,
    #MainMenu {
        display: none !important;
    }
 
    header[data-testid="stHeader"] {
        background: transparent;
    }
 
    .side-spacer {
        height: 6vh;
    }
 
    .side-towers {
        width: 100%;
        opacity: 0.9;
    }
 
    .side-tagline {
        color: #cfe0fa;
        font-size: 14px;
        line-height: 1.45;
        padding: 6px 4px 0 4px;
    }
 
    .side-version {
        color: #9db8e3;
        font-size: 12px;
        padding: 22px 4px 0 4px;
    }
 
    </style>
    """,
    unsafe_allow_html=True
)
 
 
# ============================================================
# PAGE 0 — DASHBOARD
# ============================================================
 
if page == PAGE_DASHBOARD:
 
    # CSS khusus dashboard (hanya aktif di halaman ini)
    st.markdown(
        """
        <style>
 
        .stApp {
            background: #f3f7fd;
        }
 
        .block-container {
            padding-top: 2.4rem;
            max-width: 1280px;
        }
 
        .dash-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 22px;
        }
 
        .dash-title {
            font-size: 38px;
            font-weight: 800;
            color: #0f2a5c;
            line-height: 1.15;
            margin-bottom: 8px;
        }
 
        .dash-welcome {
            font-size: 20px;
            color: #0f2a5c;
        }
 
        .dash-sub {
            font-size: 15px;
            color: #4a6fb5;
        }
 
        .dash-meta {
            display: flex;
            align-items: center;
            gap: 28px;
            padding-top: 10px;
            color: #0f2a5c;
            font-size: 15px;
        }
 
        .dash-avatar {
            display: inline-block;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: #a9c2ea;
            vertical-align: middle;
            margin-right: 10px;
            text-align: center;
            line-height: 30px;
            font-weight: 700;
            color: #ffffff;
        }
 
        .dash-user {
            font-weight: 600;
        }
 
        /* Kartu fitur */
        div[class*="st-key-card_"] {
            background: #ffffff;
            border: 1px solid #dfe8f6;
            border-radius: 14px;
            padding: 12px 14px;
        }
 
        .card-head {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 18px;
        }
 
        .card-icon {
            width: 46px;
            height: 46px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
 
        .card-icon img {
            width: 24px;
            height: 24px;
        }
 
        .card-icon-blue {
            background: #bcd7fb;
        }
 
        .card-icon-green {
            background: #0fa36b;
        }
 
        .card-title {
            font-size: 18px;
            font-weight: 700;
            color: #0f2a5c;
        }
 
        .card-text {
            font-size: 15px;
            line-height: 1.55;
            color: #1b2f57;
            margin-bottom: 14px;
        }
 
        .card-image {
            width: 100%;
            border-radius: 12px;
            display: block;
        }
 
        /* Tombol "Pelajari Lebih Lanjut" */
        .st-key-go_fault button,
        .st-key-go_load button {
            border: none;
            border-radius: 8px;
            padding: 6px 18px;
        }
 
        .st-key-go_fault button {
            background: #1565f0;
        }
 
        .st-key-go_load button {
            background: #0fa36b;
        }
 
        .st-key-go_fault button p,
        .st-key-go_load button p {
            color: #ffffff;
            font-size: 14px;
            font-weight: 600;
        }
 
        .st-key-go_fault button:hover {
            background: #0f52c4;
        }
 
        .st-key-go_load button:hover {
            background: #0c8556;
        }
 
        </style>
        """,
        unsafe_allow_html=True
    )
 
 
    # ========================================================
    # HEADER
    # ========================================================
 
    st.markdown(
        _flat(
            f"""
            <div class="dash-header">
                <div>
                    <div class="dash-title">Dashboard</div>
                    <div class="dash-welcome">Selamat datang di ElectroMind!</div>
                    <div class="dash-sub">
                        Platform pembelajaran dan analisis sistem kelistrikan
                        menggunakan Machine Learning.
                    </div>
                </div>
                <div class="dash-meta">
                    <div>📅&nbsp; {datetime.now().strftime("%d %b %Y")}</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True
    )
 
 
    # ========================================================
    # KARTU FITUR
    # ========================================================
 
    card_left, card_right = st.columns(2, gap="medium")
 
 
    # ---------------- Fault Detection ----------------
 
    with card_left:
 
        with st.container(key="card_fault"):
 
            text_col, img_col = st.columns([1.1, 1], gap="medium")
 
            with text_col:
 
                st.markdown(
                    _flat(
                        f"""
                        <div class="card-head">
                            <div class="card-icon card-icon-blue">
                                <img src="{_svg_uri(SVG_BOLT_BLUE)}"/>
                            </div>
                            <div class="card-title">Apa itu Fault Detection?</div>
                        </div>
                        <div class="card-text">
                            Fault Detection adalah proses mengidentifikasi adanya
                            gangguan pada sistem tenaga listrik, seperti gangguan
                            hubung singkat antar fasa maupun ke tanah, menggunakan
                            teknik analisis data dan machine learning.
                        </div>
                        """
                    ),
                    unsafe_allow_html=True
                )
 
                st.button(
                    "Pelajari Lebih Lanjut →",
                    key="go_fault",
                    on_click=go_to_page,
                    args=(PAGE_FAULT,)
                )
 
            with img_col:
 
                st.markdown(
                    f'<img class="card-image" src="'
                    f'{_image_uri("fault_detection.png", SVG_FAULT_IMAGE)}"/>',
                    unsafe_allow_html=True
                )
 
 
    # ---------------- Load Forecasting ----------------
 
    with card_right:
 
        with st.container(key="card_load"):
 
            text_col, img_col = st.columns([1.1, 1], gap="medium")
 
            with text_col:
 
                st.markdown(
                    _flat(
                        f"""
                        <div class="card-head">
                            <div class="card-icon card-icon-green">
                                <img src="{_svg_uri(SVG_CHART_ICON)}"/>
                            </div>
                            <div class="card-title">Apa itu Load Forecasting?</div>
                        </div>
                        <div class="card-text">
                            Load Forecasting adalah proses memprediksi beban
                            listrik di masa depan berdasarkan data historis dan
                            faktor-faktor yang mempengaruhi, sehingga membantu
                            perencanaan dan pengambilan keputusan pada sistem
                            tenaga listrik.
                        </div>
                        """
                    ),
                    unsafe_allow_html=True
                )
 
                st.button(
                    "Pelajari Lebih Lanjut →",
                    key="go_load",
                    on_click=go_to_page,
                    args=(PAGE_LOAD,)
                )
 
            with img_col:
 
                st.markdown(
                    f'<img class="card-image" src="'
                    f'{_image_uri("load_forecasting.png", SVG_LOAD_IMAGE)}"/>',
                    unsafe_allow_html=True
                )
 
 
# ============================================================
# PAGE 1 — FAULT DETECTION
# ============================================================
 
elif page == PAGE_FAULT:
 
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
 
elif page == PAGE_LOAD:
 
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
 
