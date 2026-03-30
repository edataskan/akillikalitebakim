import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import cv2
import random
import os
from datetime import datetime
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ──────────────────────────────────────────────────────────
ERROR_LOG_PATH = "error_logs.csv"
ERROR_LOG_COLUMNS = [
    "zaman", "simülasyon_adımı", "hata_çeşidi", "açıklama",
    "sensör_simülasyon_durumu", "yolo_sonuç",
    "nesne_sıcaklık_c", "tork_nm", "devir_rpm",
]

SENSOR_MODEL_PATH  = "sensor_anomaly_model.pkl"
VISION_MODEL_PATH  = "final_model_colab.pt"
TEST_IMAGE_FOLDER  = "test_image"

# Colour tokens (Plotly-compatible)
CLR_NORMAL   = "#1D9E75"   # teal-400
CLR_WARNING  = "#BA7517"   # amber-400
CLR_CRITICAL = "#E24B4A"   # red-400
CLR_GRID     = "rgba(255,255,255,0.06)"

# ──────────────────────────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kalite Kontrol Merkezi",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────
# STARTUP POPUP
# ──────────────────────────────────────────────────────────
if "popup_closed" not in st.session_state:
    st.session_state.popup_closed = False

if "run_sim_toggle" not in st.session_state:
    st.session_state.run_sim_toggle = False

dialog_decorator = None
if hasattr(st, "dialog"):
    dialog_decorator = st.dialog
elif hasattr(st, "experimental_dialog"):
    dialog_decorator = st.experimental_dialog

if dialog_decorator is not None:
    @dialog_decorator("HAVELSAN SUİT PROGRAMI -BETEDATA PROJESİ")
    def startup_popup():
        st.markdown("""
        <style>
            [data-testid="stModal"] button[aria-label="Close"] {
                display: none !important;
            }
            div[data-testid="stModalCloseButton"] {
                display: none !important;
            }
        </style>
        <div style='text-align: center; margin-bottom: 16px;'>
            <h3 style='color: #f0ede8; margin-bottom: 4px; font-weight: 600;'>Akıllı Kalite Kontrol Sistemi Otonom Hata Tespiti </h3>
            
        </div>
        """, unsafe_allow_html=True)
        
        st.info("Bu sistem, üretim hatalarını **insan müdahalesi gerektirmeksizin** gerçek zamanlı olarak tespit etmek için geliştirilmiştir.")
        
        st.markdown("""
        <div style='background: #1a1a1a; padding: 14px; border-radius: 8px; border: 1px solid rgba(29, 158, 117, 0.3); margin-bottom: 10px;'>
            <h5 style='color: #1D9E75; margin-top: 0; font-size: 15px; margin-bottom: 6px;'>📡 Katman 1: Sensör Analizi</h5>
            <p style='font-size: 13px; color: #ccc; line-height: 1.4; margin: 0;'>Sıcaklık, tork ve devir verileri <b>Random Forest</b> yapay zekasıyla sürekli izlenir. Kritik anomalide kamera motoru tetiklenir.</p>
        </div>
        
        <div style='background: #1a1a1a; padding: 14px; border-radius: 8px; border: 1px solid rgba(226, 75, 74, 0.3); margin-bottom: 12px;'>
            <h5 style='color: #E24B4A; margin-top: 0; font-size: 15px; margin-bottom: 6px;'>📷 Katman 2: Görsel Analiz</h5>
            <p style='font-size: 13px; color: #ccc; line-height: 1.4; margin: 0;'>Kamera devreye girer ve <b>YOLOv8n</b> AI modeli baskı yüzeyini tarayarak <i>stringing, warping, blob</i> hatalarını lokalize eder.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("💡 **Dashboard**, her iki katmanı birleştirerek tek ekranda tam hakimiyet sunar.")
        
        st.markdown("<div style='text-align: center; margin-top: 15px; margin-bottom: 10px; font-size: 13px; color: #888;'>Simülasyon başlatma komutunu onaylayın</div>", unsafe_allow_html=True)
        if st.button("Simülasyonu Başlat ", type="primary", use_container_width=True):
            st.session_state.run_sim_toggle = True
            st.session_state.popup_closed = True
            st.rerun()

    if not st.session_state.popup_closed:
        startup_popup()
else:
    if not st.session_state.popup_closed:
        st.info("betadata projesi: Lütfen simülasyonu kontrol panelinden başlatın.")
        st.session_state.popup_closed = True

# Global CSS – applied once
st.markdown("""
<style>
/* ── Typography & palette ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Remove default Streamlit chrome ─────────────────── */
#MainMenu, header, footer { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ── App header ───────────────────────────────────────── */
.app-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1rem 0 1.25rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.5rem;
}
.app-header .badge {
    font-size: 11px; font-weight: 600; letter-spacing: .05em;
    text-transform: uppercase;
    background: rgba(29,158,117,.15); color: #1D9E75;
    border: 1px solid rgba(29,158,117,.3);
    border-radius: 20px; padding: 3px 10px;
}
.app-header h1 {
    font-size: 1.25rem; font-weight: 600;
    color: #f0ede8; margin: 0; letter-spacing: -.01em;
}
.app-header p { font-size: .8rem; color: #888; margin: 0; }

/* ── Section label ────────────────────────────────────── */
.section-label {
    font-size: 10px; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; color: #666;
    margin: 0 0 .6rem;
}

/* ── KPI cards ────────────────────────────────────────── */
.kpi-card {
    background: #1a1a1a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: .9rem 1.1rem;
    position: relative; overflow: hidden;
}
.kpi-card .kpi-label {
    font-size: 11px; color: #666;
    font-weight: 500; letter-spacing: .04em;
    text-transform: uppercase; margin-bottom: 6px;
}
.kpi-card .kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem; font-weight: 500;
    color: #f0ede8; line-height: 1;
}
.kpi-card .kpi-unit {
    font-size: .75rem; color: #666; margin-left: 4px;
}
.kpi-card .kpi-delta {
    font-size: .75rem; margin-top: 6px;
}
.kpi-card .kpi-bar {
    position: absolute; bottom: 0; left: 0; height: 3px;
    border-radius: 0 0 10px 10px;
    transition: width .4s ease;
}

/* ── Status pill ──────────────────────────────────────── */
.status-pill {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 13px; font-weight: 600;
    border-radius: 24px; padding: 8px 16px;
    letter-spacing: .01em;
}
.status-normal   { background: rgba(29,158,117,.12); color: #1D9E75; border: 1px solid rgba(29,158,117,.25); }
.status-anomaly  { background: rgba(226,75,74,.12);  color: #E24B4A; border: 1px solid rgba(226,75,74,.3);  }
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block;
}
.dot-normal  { background: #1D9E75; animation: pulse-green 2s infinite; }
.dot-anomaly { background: #E24B4A; animation: pulse-red   1s infinite; }
@keyframes pulse-green { 0%,100%{box-shadow:0 0 0 0 rgba(29,158,117,.4)} 50%{box-shadow:0 0 0 5px rgba(29,158,117,0)} }
@keyframes pulse-red   { 0%,100%{box-shadow:0 0 0 0 rgba(226,75,74,.6)}  50%{box-shadow:0 0 0 6px rgba(226,75,74,0)} }

/* ── Alert box ────────────────────────────────────────── */
.alert-banner {
    border-radius: 8px; padding: 12px 16px;
    font-size: 13px; font-weight: 500;
    display: flex; align-items: flex-start; gap: 10px;
    line-height: 1.5;
}
.alert-critical { background: rgba(226,75,74,.1); border: 1px solid rgba(226,75,74,.3); color: #e88; }
.alert-ok       { background: rgba(29,158,117,.08); border: 1px solid rgba(29,158,117,.2); color: #6dc; }
.alert-info     { background: rgba(55,138,221,.08); border: 1px solid rgba(55,138,221,.2); color: #7bc; }

/* ── Control panel card ───────────────────────────────── */
.control-card {
    background: #161616;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.25rem;
}

/* ── Log table tweaks ─────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Tab styling ──────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px; font-weight: 500;
    padding: 8px 18px;
    border-radius: 8px 8px 0 0;
    color: #666;
}
.stTabs [aria-selected="true"] {
    color: #f0ede8 !important;
    background: rgba(255,255,255,0.05) !important;
}

/* ── Streamlit toggle override ────────────────────────── */
.stToggle { margin: 0 !important; }

/* ── Stat boxes in log tab ────────────────────────────── */
.stat-box {
    background: #1a1a1a; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: .9rem 1.1rem; text-align: center;
}
.stat-box .stat-n { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 500; color: #f0ede8; }
.stat-box .stat-l { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: .08em; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────
import base64

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

logo_b64 = get_base64_image("logo.png")
logo_html = f'<img src="data:image/png;base64,{logo_b64}" width="65" style="border-radius:4px; object-fit:contain;" />' if logo_b64 else '<span style="font-size:32px;">🎓</span>'

st.markdown(f"""
<div class="app-header">
    <div style="font-size:26px">🏭</div>
    <div>
        <h1 style="font-size:18px; margin-bottom:4px; color:#f0ede8;">BETEDATA HAVELSAN SUIT PROGRAMI - AKILLI KALİTE KONTROL SİSTEMİ</h1>
    </div>
    <div style="margin-left:auto; display:flex; align-items:center; gap:14px; text-align:right;">
        <div style="line-height: 1.4;">
            <div style="font-size: 15px; font-weight: 600; color: #1D9E75; letter-spacing: 0.05em;">MUĞLA SITKI KOÇMAN ÜNİVERSİTESİ</div>
            <div style="font-size: 15px; color: #888; font-weight: 500;">Betül YENİTOPCU & Eda TAŞKAN</div>
        </div>
        {logo_html}
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    try:
        s_model = joblib.load(SENSOR_MODEL_PATH)
        v_model = YOLO(VISION_MODEL_PATH)
        return s_model, v_model
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None, None

sensor_model, vision_model = load_all_models()

if sensor_model and vision_model:
    st.toast("✅ YOLOv8 + Sensör modelleri yüklendi", icon="✅")


# ──────────────────────────────────────────────────────────
# DATA / SENSOR FUNCTIONS
# ──────────────────────────────────────────────────────────
def get_sensor_data():
    ambient_c = random.uniform(24, 26)

    if random.random() < 0.20:
        object_c       = random.uniform(95, 105)
        x_g            = random.uniform(7.0, 9.0)
        y_g            = random.uniform(7.0, 9.0)
        z_g            = random.uniform(10.0, 12.0)
        status         = "CRITICAL"
        simulated_torque = random.uniform(75.0, 90.0)
        simulated_rpm  = random.uniform(1100, 1200)
    else:
        object_c       = random.uniform(35, 45)
        x_g            = random.uniform(0.1, 0.5)
        y_g            = random.uniform(0.1, 0.5)
        z_g            = random.uniform(0.9, 1.1)
        status         = "NORMAL"
        simulated_torque = random.uniform(35.0, 45.0)
        simulated_rpm  = 1500.0

    data = pd.DataFrame([[
        ambient_c + 273.15, object_c + 273.15,
        simulated_rpm, simulated_torque, 180.0
    ]], columns=[
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ])

    return data, status, (ambient_c, object_c, x_g, y_g, z_g, simulated_torque, simulated_rpm)


def analyze_image_with_yolo():
    if not os.path.exists(TEST_IMAGE_FOLDER):
        return None, "Klasör Yok", 0
    files = [f for f in os.listdir(TEST_IMAGE_FOLDER) if f.endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        return None, "Resim Yok", 0
    img_path = os.path.join(TEST_IMAGE_FOLDER, random.choice(files))
    results   = vision_model(img_path, conf=0.4, verbose=False)
    result_obj = results[0]
    if len(result_obj.boxes) > 0:
        label      = "Defect"
        confidence = float(result_obj.boxes[0].conf)
    else:
        label      = "Normal"
        confidence = 0.95
    annotated_frame = cv2.cvtColor(result_obj.plot(), cv2.COLOR_BGR2RGB)
    return annotated_frame, label, confidence


# ──────────────────────────────────────────────────────────
# ERROR LOG HELPERS
# ──────────────────────────────────────────────────────────
def _ensure_error_log_file():
    if not os.path.exists(ERROR_LOG_PATH):
        pd.DataFrame(columns=ERROR_LOG_COLUMNS).to_csv(ERROR_LOG_PATH, index=False, encoding="utf-8-sig")


def classify_error_event(sim_status, yolo_label, image_ok):
    if not image_ok:
        return (
            "Sensör anomalisi – görüntü analizi yok",
            "Anomali algılandı; test görüntüsü veya klasör bulunamadı.",
        )
    if yolo_label == "Defect":
        if sim_status == "CRITICAL":
            return (
                "Doğrulanmış hata (kritik sensör + görsel kusur)",
                "Modelin işaretlediği kritik profil YOLO ile kusur olarak doğrulandı.",
            )
        return (
            "Görsel kusur (beklenmeyen sensör profili)",
            "YOLO kusur gösteriyor; simülasyon temel profili NORMAL sınıfında.",
        )
    if sim_status == "CRITICAL":
        return (
            "Sensör kritik – görüntü temiz",
            "Kritik telemetri var; YOLO bu karede kusur görmüyor (takip önerilir).",
        )
    return (
        "Sensör uyarısı – görüntü temiz",
        "Sensör modeli anomali dedi; YOLO bu örnekte kusur raporlamadı.",
    )


def append_error_log(step_idx, sim_status, yolo_label, image_ok, obj_c, torque_nm, rpm):
    _ensure_error_log_file()
    hata_cesidi, aciklama = classify_error_event(sim_status, yolo_label, image_ok)
    row = {
        "zaman":                 datetime.now().isoformat(timespec="seconds"),
        "simülasyon_adımı":      step_idx,
        "hata_çeşidi":           hata_cesidi,
        "açıklama":              aciklama,
        "sensör_simülasyon_durumu": sim_status,
        "yolo_sonuç":            yolo_label if image_ok else "-",
        "nesne_sıcaklık_c":      round(obj_c, 2),
        "tork_nm":               round(torque_nm, 2),
        "devir_rpm":             round(rpm, 2),
    }
    pd.DataFrame([row]).to_csv(ERROR_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")


def load_error_logs():
    if not os.path.exists(ERROR_LOG_PATH):
        return pd.DataFrame(columns=ERROR_LOG_COLUMNS)
    try:
        return pd.read_csv(ERROR_LOG_PATH, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=ERROR_LOG_COLUMNS)


# ──────────────────────────────────────────────────────────
# CHART HELPERS (consistent dark theme)
# ──────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=28, b=0),
    font=dict(family="IBM Plex Sans", size=12, color="#999"),
    xaxis=dict(showgrid=True, gridcolor=CLR_GRID, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor=CLR_GRID, zeroline=False),
    legend=dict(orientation="h", y=-0.12, x=0, font_size=11),
    title_font=dict(size=13, color="#ccc"),
)

SENSOR_COLORS = {
    "Ortam Sıc. (°C)":  "#378ADD",
    "Nesne Sıc. (°C)":  "#E24B4A",
    "X-G":              "#EF9F27",
    "Y-G":              "#BA7517",
    "Z-G":              "#1D9E75",
}


def build_telemetry_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    col_map = {
        "Ambient": "Ortam Sıc. (°C)",
        "Object":  "Nesne Sıc. (°C)",
        "X-G":     "X-G",
        "Y-G":     "Y-G",
        "Z-G":     "Z-G",
    }
    for raw, label in col_map.items():
        if raw not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["Time"], y=df[raw],
            mode="lines",
            name=label,
            line=dict(color=SENSOR_COLORS[label], width=1.6),
            hovertemplate=f"<b>{label}</b><br>Adım: %{{x}}<br>Değer: %{{y:.2f}}<extra></extra>",
        ))
    layout = DARK_LAYOUT.copy()
    layout["title"] = "Gerçek Zamanlı Telemetri"
    fig.update_layout(**layout)
    return fig


def build_gauge(value: float, title: str, min_v: float, max_v: float,
                warn: float, crit: float, unit: str = "") -> go.Figure:
    if value >= crit:
        bar_color = CLR_CRITICAL
    elif value >= warn:
        bar_color = CLR_WARNING
    else:
        bar_color = CLR_NORMAL

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix=unit, font=dict(size=22, color="#f0ede8", family="IBM Plex Mono")),
        title=dict(text=title, font=dict(size=12, color="#888")),
        gauge=dict(
            axis=dict(range=[min_v, max_v], tickcolor="#555", tickfont=dict(size=10, color="#666")),
            bar=dict(color=bar_color, thickness=0.22),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[min_v, warn], color="rgba(255,255,255,0.04)"),
                dict(range=[warn, crit],  color="rgba(186,117,23,0.08)"),
                dict(range=[crit, max_v], color="rgba(226,75,74,0.1)"),
            ],
            threshold=dict(line=dict(color=CLR_CRITICAL, width=2), value=crit),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=36, b=0),
        height=180,
        font=dict(family="IBM Plex Sans"),
    )
    return fig


# ──────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────
tab_live, tab_logs = st.tabs(["📡  Canlı İzleme", "📋  Hata Kayıtları & İstatistik"])


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 1 — LIVE MONITORING                                 ║
# ╚══════════════════════════════════════════════════════════╝
with tab_live:

    # ── Left sidebar: controls + KPIs ──────────────────────
    ctrl_col, chart_col, cam_col = st.columns([1, 2.2, 1.8])

    with ctrl_col:
        st.markdown('<p class="section-label">Kontrol</p>', unsafe_allow_html=True)
        run_sim = st.toggle("Simülasyonu Başlat/Durdur", key="run_sim_toggle")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Anlık Değerler</p>', unsafe_allow_html=True)

        kpi_amb    = st.empty()
        kpi_obj    = st.empty()
        kpi_torque = st.empty()
        kpi_rpm    = st.empty()

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Sistem Durumu</p>', unsafe_allow_html=True)
        status_box = st.empty()

    with chart_col:
        st.markdown('<p class="section-label">Telemetri Grafiği</p>', unsafe_allow_html=True)
        chart_ph   = st.empty()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Ölçüm Göstergeler</p>', unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        with g1: gauge_temp  = st.empty()
        with g2: gauge_tork  = st.empty()
        with g3: gauge_rpm   = st.empty()

    with cam_col:
        st.markdown('<p class="section-label">Görsel Analiz</p>', unsafe_allow_html=True)
        cam_ph    = st.empty()
        alert_ph  = st.empty()

    # ── KPI renderer helper ─────────────────────────────────
    def render_kpi(placeholder, label, value, unit, pct, color):
        placeholder.markdown(f"""
        <div class="kpi-card" style="margin-bottom:8px">
            <div class="kpi-label">{label}</div>
            <div>
                <span class="kpi-value">{value:.1f}</span>
                <span class="kpi-unit">{unit}</span>
            </div>
            <div class="kpi-bar" style="width:{min(pct,100):.0f}%;background:{color}"></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Default idle states & Renderer ──────────────────────
    if "last_sim_state" not in st.session_state:
        st.session_state.last_sim_state = None

    def render_dashboard(state, render_key=""):
        amb, obj, sim_torque, sim_rpm = state["amb"], state["obj"], state["sim_torque"], state["sim_rpm"]
        history_df = state["history_df"]
        sensor_pred = state["sensor_pred"]
        proc_img, label, conf = state.get("proc_img"), state.get("label"), state.get("conf", 0.0)

        # KPI cards
        render_kpi(kpi_amb,    "Ortam Sıcaklığı", amb,       "°C", (amb-20)/20*100,     CLR_NORMAL)
        render_kpi(kpi_obj,    "Nesne Sıcaklığı", obj,       "°C", (obj-20)/90*100,
                   CLR_CRITICAL if obj > 80 else CLR_WARNING if obj > 55 else CLR_NORMAL)
        render_kpi(kpi_torque, "Tork",            sim_torque, "Nm", sim_torque/90*100,
                   CLR_CRITICAL if sim_torque > 70 else CLR_WARNING if sim_torque > 55 else CLR_NORMAL)
        render_kpi(kpi_rpm,    "Devir",           sim_rpm,    "RPM", sim_rpm/1500*100, CLR_NORMAL)

        # Telemetry chart
        chart_ph.plotly_chart(
            build_telemetry_chart(history_df),
            use_container_width=True,
            key=f"telemetry_{render_key}",
        )

        # Gauges
        gauge_temp.plotly_chart(build_gauge(obj, "Nesne Sıc.", 0, 120, 55, 80, "°C"),
                                use_container_width=True, key=f"gauge_temp_{render_key}")
        gauge_tork.plotly_chart(build_gauge(sim_torque, "Tork", 0, 90, 55, 70, " Nm"),
                                use_container_width=True, key=f"gauge_tork_{render_key}")
        gauge_rpm.plotly_chart(build_gauge(sim_rpm, "Devir", 0, 1600, 1300, 1500, " RPM"),
                               use_container_width=True, key=f"gauge_rpm_{render_key}")

        # Status & vision
        if sensor_pred == 1:
            status_box.markdown("""
            <div class="status-pill status-anomaly">
                <span class="status-dot dot-anomaly"></span>
                ANOMALİ TESPİT EDİLDİ
            </div>""", unsafe_allow_html=True)

            if proc_img is not None:
                cam_ph.image(proc_img, caption="YOLOv8 Hata Lokalizasyonu", use_column_width=True)
                if label == "Defect":
                    alert_ph.markdown(f"""
                    <div class="alert-banner alert-critical">
                        🚨 <strong>KRİTİK HATA:</strong> {label} — Güven: %{conf*100:.1f}
                    </div>""", unsafe_allow_html=True)
                else:
                    alert_ph.markdown(f"""
                    <div class="alert-banner alert-ok">
                        ✅ Görsel kontrol temiz — Güven: %{conf*100:.1f}
                    </div>""", unsafe_allow_html=True)
            else:
                cam_ph.markdown("""
                <div class="alert-banner alert-info">
                    ⚠️ Görüntü analizi yapılamadı — klasör/resim kontrolü yapın.
                </div>""", unsafe_allow_html=True)
        else:
            status_box.markdown("""
            <div class="status-pill status-normal">
                <span class="status-dot dot-normal"></span>
                Sistem Normal
            </div>""", unsafe_allow_html=True)
            cam_ph.markdown("""
            <div class="alert-banner alert-info" style="margin-top:8px">
                📷 Kamera bekleniyor — anomali tespit edildiğinde analiz başlar.
            </div>""", unsafe_allow_html=True)
            alert_ph.empty()


    # ── Handle stopped state ────────────────────────────────
    if not run_sim:
        if st.session_state.last_sim_state is not None:
            render_dashboard(st.session_state.last_sim_state, render_key="idle")
        else:
            status_box.markdown("""
            <div class="status-pill" style="background:rgba(255,255,255,0.04);color:#555;border:1px solid rgba(255,255,255,0.07)">
                <span class="status-dot" style="background:#444"></span>
                Bekleniyor
            </div>""", unsafe_allow_html=True)
            cam_ph.markdown("""
            <div class="alert-banner alert-info" style="margin-top:8px">
                📷 Simülasyon başlatıldığında görüntü analizi yapılır.
            </div>""", unsafe_allow_html=True)

    # ── Simulation loop ─────────────────────────────────────
    if run_sim:
        history_df = pd.DataFrame(columns=["Time", "Ambient", "Object", "X-G", "Y-G", "Z-G"])
        if st.session_state.last_sim_state is not None and not st.session_state.last_sim_state["history_df"].empty:
            history_df = st.session_state.last_sim_state["history_df"]

        for i in range(100):
            data, sim_status, physical_vals = get_sensor_data()
            amb, obj, xg, yg, zg, sim_torque, sim_rpm = physical_vals

            sensor_pred = sensor_model.predict(data)[0] if sensor_model else 0

            new_row = pd.DataFrame({"Time": [i + (history_df["Time"].max() + 1 if not history_df.empty else 0)], 
                                    "Ambient": [amb], "Object": [obj],
                                    "X-G": [xg], "Y-G": [yg], "Z-G": [zg]})
            history_df = pd.concat([history_df, new_row], ignore_index=True)
            recent_history = history_df.tail(30).copy()

            proc_img, label, conf = None, "", 0.0
            image_ok = False
            
            if sensor_pred == 1:
                proc_img, label, conf = analyze_image_with_yolo()
                image_ok = proc_img is not None
                yolo_for_log = label if label in ("Defect", "Normal") else "Normal"
                append_error_log(i, sim_status, yolo_for_log, image_ok, obj, sim_torque, sim_rpm)

            # Store state in session_state before rendering
            st.session_state.last_sim_state = {
                "amb": amb, "obj": obj, "sim_torque": sim_torque, "sim_rpm": sim_rpm,
                "history_df": recent_history,
                "sensor_pred": sensor_pred,
                "proc_img": proc_img,
                "label": label,
                "conf": conf
            }

            render_dashboard(st.session_state.last_sim_state, render_key=str(i))

            if sensor_pred == 1 and proc_img is not None:
                time.sleep(3.0)
            else:
                time.sleep(0.8)


# ╔══════════════════════════════════════════════════════════╗
# ║  TAB 2 — ERROR LOG & STATISTICS                          ║
# ╚══════════════════════════════════════════════════════════╝
with tab_logs:
    log_df = load_error_logs()

    if log_df.empty:
        st.markdown("""
        <div class="alert-banner alert-info" style="margin-top:16px">
            📋 Henüz kayıtlı anomali olayı yok. Canlı İzleme sekmesinde simülasyonu başlatın.
        </div>""", unsafe_allow_html=True)
    else:
        # ── Summary row ─────────────────────────────────────
        son_zaman = log_df["zaman"].iloc[-1] if "zaman" in log_df.columns else "-"
        defect_count = (log_df.get("yolo_sonuç", pd.Series()) == "Defect").sum()
        critical_count = (log_df.get("sensör_simülasyon_durumu", pd.Series()) == "CRITICAL").sum()

        s1, s2, s3, s4 = st.columns(4)
        for col, n, label in [
            (s1, len(log_df),       "Toplam Kayıt"),
            (s2, critical_count,    "Kritik Sensör"),
            (s3, defect_count,      "Görsel Kusur"),
            (s4, son_zaman,         "Son Kayıt"),
        ]:
            col.markdown(f"""
            <div class="stat-box">
                <div class="stat-n">{n}</div>
                <div class="stat-l">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── Charts ──────────────────────────────────────────
        if "hata_çeşidi" in log_df.columns and log_df["hata_çeşidi"].notna().any():
            counts = log_df["hata_çeşidi"].value_counts().reset_index()
            counts.columns = ["hata_çeşidi", "sayı"]

            pie_col, bar_col = st.columns(2)

            with pie_col:
                st.markdown('<p class="section-label">Hata Çeşidi Dağılımı</p>', unsafe_allow_html=True)
                fig_pie = px.pie(
                    counts, names="hata_çeşidi", values="sayı",
                    color_discrete_sequence=["#378ADD", "#E24B4A", "#1D9E75", "#EF9F27", "#7F77DD"],
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                                      textfont_size=11, hole=0.38)
                pie_layout = DARK_LAYOUT.copy()
                pie_layout.update(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=260)
                fig_pie.update_layout(**pie_layout)
                st.plotly_chart(fig_pie, use_container_width=True)

            with bar_col:
                st.markdown('<p class="section-label">Olay Sayıları</p>', unsafe_allow_html=True)
                fig_bar = px.bar(
                    counts, x="sayı", y="hata_çeşidi", orientation="h",
                    labels={"sayı": "Olay sayısı", "hata_çeşidi": ""},
                    color="sayı",
                    color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"],
                )
                bar_layout = DARK_LAYOUT.copy()
                bar_layout.update(
                    showlegend=False, coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=10, b=0), height=260,
                )
                fig_bar.update_layout(**bar_layout)
                st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Temperature scatter over time ────────────────────
        if "nesne_sıcaklık_c" in log_df.columns and "zaman" in log_df.columns:
            st.markdown('<p class="section-label">Nesne Sıcaklığı Zaman Serisi</p>', unsafe_allow_html=True)
            fig_ts = px.scatter(
                log_df.sort_values("zaman"),
                x="zaman", y="nesne_sıcaklık_c",
                color="sensör_simülasyon_durumu",
                color_discrete_map={"CRITICAL": CLR_CRITICAL, "NORMAL": CLR_NORMAL},
                labels={"zaman": "Zaman", "nesne_sıcaklık_c": "Nesne Sıc. (°C)",
                        "sensör_simülasyon_durumu": "Durum"},
            )
            ts_layout = DARK_LAYOUT.copy()
            ts_layout.update(margin=dict(l=0, r=0, t=10, b=0), height=200)
            fig_ts.update_layout(**ts_layout)
            st.plotly_chart(fig_ts, use_container_width=True)

        # ── Data table ──────────────────────────────────────
        st.markdown('<p class="section-label">Tüm Kayıtlar</p>', unsafe_allow_html=True)
        st.dataframe(
            log_df.sort_values("zaman", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "zaman":                  st.column_config.TextColumn("Zaman"),
                "simülasyon_adımı":       st.column_config.NumberColumn("Adım", format="%d"),
                "nesne_sıcaklık_c":       st.column_config.NumberColumn("Nesne Sıc. (°C)", format="%.1f"),
                "tork_nm":                st.column_config.NumberColumn("Tork (Nm)", format="%.1f"),
                "devir_rpm":              st.column_config.NumberColumn("Devir (RPM)", format="%.0f"),
                "sensör_simülasyon_durumu": st.column_config.TextColumn("Sensör Durumu"),
                "yolo_sonuç":             st.column_config.TextColumn("YOLO"),
                "hata_çeşidi":            st.column_config.TextColumn("Hata Çeşidi"),
                "açıklama":               st.column_config.TextColumn("Açıklama"),
            },
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("🗑️  Tüm kayıtları sil", type="secondary"):
            if os.path.exists(ERROR_LOG_PATH):
                os.remove(ERROR_LOG_PATH)
            st.success("Tüm kayıtlar silindi.")
            st.rerun()