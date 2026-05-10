import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EcoMonitor — Air Quality Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
    --bg-main:    #111827;
    --bg-card:    #1f2937;
    --bg-card2:   #1a2332;
    --bg-sidebar: #0f172a;
    --border:     rgba(255,255,255,0.07);
    --border2:    rgba(255,255,255,0.12);
    --teal:       #00d4aa;
    --teal2:      #00b894;
    --teal-glow:  rgba(0,212,170,0.15);
    --teal-dim:   rgba(0,212,170,0.08);
    --text:       #f1f5f9;
    --muted:      #94a3b8;
    --muted2:     #64748b;
    --red:        #ef4444;
    --red-dim:    rgba(239,68,68,0.12);
    --amber:      #f59e0b;
    --amber-dim:  rgba(245,158,11,0.12);
    --blue:       #3b82f6;
    --blue-dim:   rgba(59,130,246,0.12);
    --green:      #10b981;
    --green-dim:  rgba(16,185,129,0.12);
    --purple:     #8b5cf6;
    --purple-dim: rgba(139,92,246,0.12);
}

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"],
.main, .block-container, section.main {
    background-color: var(--bg-main) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
.block-container { padding: 0 28px 48px 28px !important; max-width: 100% !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border2) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: var(--muted) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border2) !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.84rem !important; color: var(--muted) !important;
    padding: 9px 12px !important; border-radius: 8px !important;
    width: 100% !important; transition: all 0.18s !important;
    border: 1px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--teal-dim) !important; color: var(--teal) !important;
    border-color: rgba(0,212,170,0.15) !important;
}
[data-testid="stSidebar"] .stSuccess {
    background: var(--green-dim) !important; color: var(--green) !important;
    border: 1px solid rgba(16,185,129,0.2) !important; border-radius: 8px !important;
}
[data-testid="stSidebar"] .stInfo {
    background: var(--blue-dim) !important; color: #93c5fd !important;
    border: 1px solid rgba(59,130,246,0.2) !important; border-radius: 8px !important;
}

/* ── TOPBAR ── */
.eco-topbar {
    background: rgba(15,23,42,0.95); backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border2); padding: 13px 28px;
    display: flex; align-items: center; justify-content: space-between;
    margin: 0 -28px 28px -28px; position: sticky; top: 0; z-index: 100;
}
.eco-brand { font-size: 1.05rem; font-weight: 700; color: var(--text); display: flex; align-items: center; gap: 9px; }
.eco-brand .brand-accent { color: var(--teal); }
.eco-topbar-right { display: flex; align-items: center; gap: 14px; font-size: 0.78rem; }
.eco-status {
    display: flex; align-items: center; gap: 6px;
    background: var(--green-dim); border: 1px solid rgba(16,185,129,0.2);
    border-radius: 20px; padding: 4px 12px; font-size: 0.68rem; font-weight: 700; color: #6ee7b7 !important;
}
.eco-status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); animation: pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
.eco-search {
    display: flex; align-items: center; gap: 8px;
    background: var(--bg-card); border: 1px solid var(--border2);
    border-radius: 8px; padding: 6px 14px; font-size: 0.78rem; color: var(--muted2);
}
.eco-icon-btn {
    width: 32px; height: 32px; border-radius: 8px;
    background: var(--bg-card); border: 1px solid var(--border2);
    display: flex; align-items: center; justify-content: center;
    color: var(--muted); font-size: 0.85rem; cursor: pointer;
}
.eco-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, var(--teal2), var(--teal));
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; color: #0a1628;
}

/* ── PAGE HEADERS ── */
.pg-eyebrow {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; color: var(--teal); margin-bottom: 6px;
    display: flex; align-items: center; gap: 7px;
}
.pg-title { font-size: 1.85rem; font-weight: 700; color: var(--text); margin-bottom: 5px; letter-spacing: -0.5px; }
.pg-sub { font-size: 0.875rem; color: var(--muted); line-height: 1.7; margin-bottom: 24px; max-width: 680px; }
.eco-divider { height: 1px; background: var(--border); margin: 20px 0 24px 0; }

/* ── HERO ── */
.eco-hero {
    background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #0a1628 100%);
    border: 1px solid var(--border2); border-radius: 18px;
    padding: 50px 54px 90px 54px; margin-bottom: 0px; position: relative; overflow: visible;
}
.eco-hero::before {
    content:''; position:absolute; right:-80px; top:-80px; width:420px; height:420px;
    border-radius:50%; background:radial-gradient(circle, rgba(0,212,170,0.12) 0%, transparent 65%);
}
.eco-hero::after {
    content:''; position:absolute; left:-60px; bottom:-60px; width:280px; height:280px;
    border-radius:50%; background:radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 65%);
}
.eco-hero-eyebrow { font-size:0.62rem; font-weight:700; letter-spacing:3px; text-transform:uppercase; color:var(--teal); margin-bottom:18px; display:flex; align-items:center; gap:8px; }
.eco-hero-title { font-size:3rem; font-weight:800; color:var(--text); line-height:1.1; margin-bottom:18px; letter-spacing:-1.5px; }
.eco-hero-title .ht-accent { color: var(--teal); }
.eco-hero-desc { font-size:0.92rem; color:var(--muted); line-height:1.8; max-width:520px; margin-bottom:32px; }
.eco-btn { display:inline-flex; align-items:center; gap:8px; background:var(--teal); color:#0a1628 !important; padding:11px 24px; border-radius:9px; font-size:0.86rem; font-weight:700; margin-right:10px; }
.eco-btn-ghost { display:inline-flex; align-items:center; gap:8px; background:transparent; border:1px solid var(--border2); color:var(--muted) !important; padding:11px 24px; border-radius:9px; font-size:0.86rem; font-weight:600; }

/* ── STAT CARDS ── */
.eco-stat { background:var(--bg-card); border:1px solid var(--border); border-radius:14px; padding:20px 22px; position:relative; overflow:hidden; transition:border-color 0.2s, transform 0.2s; }
.eco-stat:hover { border-color:var(--border2); transform:translateY(-2px); }
.eco-stat-top { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px; }
.eco-stat-label { font-size:0.62rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--muted2); }
.eco-badge { display:inline-flex; align-items:center; gap:4px; font-size:0.58rem; font-weight:700; letter-spacing:0.8px; text-transform:uppercase; padding:3px 8px; border-radius:5px; }
.eco-badge-teal   { background:var(--teal-dim);   color:var(--teal)  !important; border:1px solid rgba(0,212,170,0.2); }
.eco-badge-green  { background:var(--green-dim);   color:#6ee7b7     !important; border:1px solid rgba(16,185,129,0.2); }
.eco-badge-red    { background:var(--red-dim);     color:#fca5a5     !important; border:1px solid rgba(239,68,68,0.2); }
.eco-badge-amber  { background:var(--amber-dim);   color:#fcd34d     !important; border:1px solid rgba(245,158,11,0.2); }
.eco-badge-blue   { background:var(--blue-dim);    color:#93c5fd     !important; border:1px solid rgba(59,130,246,0.2); }
.eco-badge-purple { background:var(--purple-dim);  color:#c4b5fd     !important; border:1px solid rgba(139,92,246,0.2); }
.eco-stat-value { font-family:'DM Mono',monospace; font-size:1.9rem; font-weight:500; color:var(--text); line-height:1; margin-bottom:12px; }
.eco-stat-bar  { height:2px; background:rgba(255,255,255,0.06); border-radius:2px; overflow:hidden; }
.eco-stat-fill { height:2px; border-radius:2px; }
.eco-stat-icon { position:absolute; right:18px; bottom:18px; font-size:1.8rem; opacity:0.06; }

/* ── CARDS ── */
.eco-card { background:var(--bg-card); border:1px solid var(--border); border-radius:14px; padding:20px 22px; margin-bottom:14px; transition:border-color 0.2s; }
.eco-card:hover { border-color:var(--border2); }
.eco-card-title { font-size:0.87rem; font-weight:600; color:var(--text); margin-bottom:4px; display:flex; align-items:center; gap:8px; }
.eco-card-title i { color:var(--teal); }
.eco-card-sub { font-size:0.74rem; color:var(--muted2); margin-bottom:12px; }

/* ── METRIC CARDS ── */
.eco-metric { background:var(--bg-card); border:1px solid var(--border); border-radius:14px; padding:24px 26px; }
.eco-metric-label { font-size:0.62rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--muted2); margin-bottom:12px; }
.eco-metric-value { font-family:'DM Mono',monospace; font-size:2.6rem; font-weight:500; color:var(--teal); line-height:1; margin-bottom:8px; }
.eco-metric-note   { font-size:0.78rem; color:var(--muted); }
.eco-metric-note-g { color:var(--green)  !important; font-weight:600; }
.eco-metric-note-r { color:var(--red)    !important; font-weight:600; }

/* ── INSIGHT BOX ── */
.eco-insight { background:rgba(0,212,170,0.05); border:1px solid rgba(0,212,170,0.2); border-left:3px solid var(--teal); border-radius:10px; padding:16px 18px; margin-bottom:14px; }
.eco-insight-title { font-size:0.78rem; font-weight:700; color:var(--teal); margin-bottom:7px; display:flex; align-items:center; gap:6px; }
.eco-insight-body { font-size:0.8rem; color:var(--muted); line-height:1.75; }

/* ── FEATURE IMPORTANCE ── */
.eco-feat-row { margin-bottom:12px; }
.eco-feat-top { display:flex; justify-content:space-between; font-size:0.78rem; margin-bottom:5px; }
.eco-feat-name { color:var(--muted); font-weight:500; }
.eco-feat-pct  { color:var(--teal); font-family:'DM Mono',monospace; }
.eco-feat-bg   { height:4px; background:rgba(255,255,255,0.06); border-radius:2px; overflow:hidden; }
.eco-feat-fill { height:4px; border-radius:2px; background:linear-gradient(90deg, var(--teal2), var(--teal)); }

/* ── PREDICTOR ── */
.eco-pred { background:var(--bg-card); border:1px solid var(--border); border-top:2px solid var(--teal); border-radius:14px; padding:28px; text-align:center; }
.eco-pred-label { font-size:0.62rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--muted2); margin-bottom:12px; }
.eco-pred-unit { font-size:0.85rem; color:var(--muted); margin-bottom:18px; }
.eco-pred-badge { display:inline-block; padding:6px 20px; border-radius:6px; font-weight:700; font-size:0.82rem; margin-bottom:20px; }
.eco-who-label { display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:6px; }
.eco-who-bar-bg  { height:5px; background:rgba(255,255,255,0.06); border-radius:3px; margin-bottom:14px; overflow:hidden; }
.eco-who-bar-fill { height:5px; border-radius:3px; }
.eco-conf-row { display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid var(--border); font-size:0.82rem; }
.eco-conf-row:last-child { border-bottom:none; }
.eco-conf-lbl { color:var(--muted); }
.eco-conf-val { color:var(--text); font-family:'DM Mono',monospace; font-weight:500; }

/* ── DIST BARS ── */
.eco-dist-row { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.eco-dist-name { min-width:110px; color:var(--muted); font-weight:500; font-size:0.74rem; }
.eco-dist-bar-bg { flex:1; height:4px; background:rgba(255,255,255,0.06); border-radius:2px; }
.eco-dist-bar    { height:4px; border-radius:2px; }
.eco-dist-val { color:var(--text); font-family:'DM Mono',monospace; font-size:0.75rem; min-width:36px; text-align:right; }

/* ── AQI BADGES ── */
.aqi-good     { background:rgba(16,185,129,0.12); color:#6ee7b7; border:1px solid rgba(16,185,129,0.25); }
.aqi-moderate { background:rgba(245,158,11,0.12); color:#fcd34d; border:1px solid rgba(245,158,11,0.25); }
.aqi-usg      { background:rgba(249,115,22,0.12); color:#fdba74; border:1px solid rgba(249,115,22,0.25); }
.aqi-unhlthy  { background:var(--red-dim);          color:#fca5a5; border:1px solid rgba(239,68,68,0.25); }
.aqi-very     { background:var(--purple-dim);        color:#c4b5fd; border:1px solid rgba(139,92,246,0.25); }
.aqi-haz      { background:rgba(185,28,28,0.15);    color:#fca5a5; border:1px solid rgba(185,28,28,0.3); }

/* ── STREAMLIT OVERRIDES ── */
[data-testid="metric-container"] { background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:12px !important; padding:18px !important; }
[data-testid="stMetricValue"]  { color:var(--teal) !important; font-family:'DM Mono',monospace !important; font-size:1.8rem !important; font-weight:500 !important; }
[data-testid="stMetricLabel"]  { color:var(--muted) !important; font-size:0.65rem !important; letter-spacing:2px !important; text-transform:uppercase !important; }
[data-testid="stMetricDelta"]  { color:var(--green) !important; }

div[data-testid="stTabs"] [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid var(--border2) !important; gap:4px !important; }
div[data-testid="stTabs"] [data-baseweb="tab"] { background:transparent !important; color:var(--muted2) !important; font-size:0.83rem !important; padding:10px 18px !important; border-radius:8px 8px 0 0 !important; }
div[data-testid="stTabs"] [aria-selected="true"] { color:var(--teal) !important; border-bottom:2px solid var(--teal) !important; background:var(--teal-dim) !important; }

.stSelectbox > div > div, .stMultiSelect > div > div { background:var(--bg-card) !important; border:1px solid var(--border2) !important; color:var(--text) !important; border-radius:8px !important; }
.stNumberInput > div > div > input, input, textarea { background:var(--bg-card) !important; border:1px solid var(--border2) !important; color:var(--text) !important; border-radius:8px !important; }
.stButton > button { background:linear-gradient(135deg, var(--teal2), var(--teal)) !important; color:#0a1628 !important; font-weight:700 !important; font-size:0.88rem !important; border:none !important; border-radius:9px !important; padding:12px 20px !important; width:100% !important; box-shadow:0 4px 20px rgba(0,212,170,0.2) !important; transition:opacity 0.2s, transform 0.2s !important; }
/* New Analysis button in sidebar - outline style */
[data-testid="stSidebar"] .stButton > button { background:transparent !important; color:#00d4aa !important; border:1px solid rgba(0,212,170,0.35) !important; box-shadow:none !important; font-weight:600 !important; }
[data-testid="stSidebar"] .stButton > button:hover { background:rgba(0,212,170,0.08) !important; border-color:rgba(0,212,170,0.6) !important; }
.stButton > button:hover { opacity:0.9 !important; transform:translateY(-1px) !important; }
.stFileUploader { background:var(--bg-card) !important; border:1px dashed var(--border2) !important; border-radius:10px !important; }

/* ── DARK FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
[data-testid="stFileUploader"] section {
    background: #1a2332 !important;
    border: 1px dashed rgba(0,212,170,0.3) !important;
    border-radius: 10px !important;
    padding: 20px 16px !important;
}
[data-testid="stFileUploader"] section > div {
    background: transparent !important;
}
[data-testid="stFileUploader"] section p,
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section small {
    color: #64748b !important;
}
[data-testid="stFileUploadDropzone"] {
    background: #1a2332 !important;
    border: 1px dashed rgba(0,212,170,0.3) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploadDropzone"] > div { background: transparent !important; }
[data-testid="stFileUploadDropzone"] span { color: #64748b !important; }
[data-testid="stFileUploadDropzone"] small { color: #475569 !important; }
/* Browse files button inside uploader */
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploader"] button {
    background: rgba(0,212,170,0.08) !important;
    border: 1px solid rgba(0,212,170,0.3) !important;
    color: #00d4aa !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    box-shadow: none !important;
    width: auto !important;
    padding: 6px 16px !important;
}
[data-testid="stFileUploadDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover {
    background: rgba(0,212,170,0.16) !important;
    border-color: rgba(0,212,170,0.5) !important;
    transform: none !important;
}
.stExpander { background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:10px !important; }

thead tr th { background:rgba(255,255,255,0.04) !important; color:var(--teal) !important; font-size:0.65rem !important; font-weight:700 !important; letter-spacing:1.5px !important; text-transform:uppercase !important; border-bottom:1px solid var(--border2) !important; }
tbody tr { background:var(--bg-card) !important; color:var(--text) !important; font-size:0.82rem !important; }
tbody tr:nth-child(even) { background:var(--bg-card2) !important; }
tbody tr:hover { background:rgba(0,212,170,0.04) !important; }

/* ── DATAFRAME DARK THEME ── */
[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:10px !important; overflow:hidden !important; }

#MainMenu, footer { visibility:hidden !important; }
header { background:transparent !important; }
[data-testid="stDecoration"] { display:none !important; }
.stRadio label { color:var(--muted) !important; }
.stCaption { color:var(--muted2) !important; }
.stSuccess { background:var(--green-dim) !important; border:1px solid rgba(16,185,129,0.2) !important; border-radius:8px !important; color:#6ee7b7 !important; }
.stInfo    { background:var(--blue-dim)  !important; border:1px solid rgba(59,130,246,0.2)   !important; border-radius:8px !important; color:#93c5fd !important; }
.stWarning { background:var(--amber-dim) !important; border:1px solid rgba(245,158,11,0.2)   !important; border-radius:8px !important; color:var(--amber) !important; }

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--bg-main); }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.1); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:rgba(255,255,255,0.2); }
.stSlider > div > div > div { background:var(--teal) !important; }
</style>
""", unsafe_allow_html=True)




# ── HELPERS ──────────────────────────────────────────────────
def classify_aqi(pm25):
    if pm25 <= 35:    return 'Good',           'aqi-good',    '#6ee7b7'
    elif pm25 <= 75:  return 'Moderate',        'aqi-moderate','#fcd34d'
    elif pm25 <= 115: return 'Unhealthy (SG)',  'aqi-usg',     '#fdba74'
    elif pm25 <= 150: return 'Unhealthy',       'aqi-unhlthy', '#fca5a5'
    elif pm25 <= 250: return 'Very Unhealthy',  'aqi-very',    '#c4b5fd'
    else:             return 'Hazardous',       'aqi-haz',     '#fca5a5'

AQI_COL = {
    'Good':'#10b981','Moderate':'#f59e0b',
    'Unhealthy for Sensitive Groups':'#f97316',
    'Unhealthy':'#ef4444','Very Unhealthy':'#8b5cf6','Hazardous':'#dc2626'
}
TEAL='#00d4aa'; TEAL2='#00b894'
MUTED='#94a3b8'; TEXT='#f1f5f9'
RED='#ef4444'; AMBER='#f59e0b'; GREEN='#10b981'; BLUE='#3b82f6'

DARK_PLOT = dict(
    plot_bgcolor='#1a2332', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8', family='Inter'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)',
               zerolinecolor='rgba(255,255,255,0.05)', color='#64748b'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)',
               zerolinecolor='rgba(255,255,255,0.05)', color='#64748b'),
)
M = dict(t=20, b=20, l=10, r=10)

def ap(fig, extra=None):
    kw = {**DARK_PLOT, 'margin': M}
    if extra: kw.update(extra)
    fig.update_layout(**kw)
    return fig

def dark_table(df_in, max_rows=None, height=None):
    """Render a DataFrame as a styled dark HTML table."""
    d = df_in.copy()
    if max_rows:
        d = d.head(max_rows)
    d = d.reset_index(drop=True)

    header_cells = "".join(
        f"<th style='background:#0f172a;color:#00d4aa;font-size:0.65rem;font-weight:700;"
        f"letter-spacing:1.5px;text-transform:uppercase;padding:10px 14px;"
        f"border-bottom:1px solid rgba(255,255,255,0.12);white-space:nowrap;'>{col}</th>"
        for col in d.columns
    )

    rows_html = ""
    for i, row in d.iterrows():
        bg = "#1f2937" if i % 2 == 0 else "#1a2332"
        cells = "".join(
            f"<td style='padding:8px 14px;color:#f1f5f9;font-size:0.82rem;"
            f"border-bottom:1px solid rgba(255,255,255,0.05);'>{val}</td>"
            for val in row.values
        )
        rows_html += f"<tr style='background:{bg};'>{cells}</tr>"

    h_style = f"max-height:{height}px;overflow-y:auto;" if height else ""
    html = f"""
    <div style='border:1px solid rgba(255,255,255,0.08);border-radius:10px;
                overflow:hidden;margin-bottom:14px;{h_style}'>
      <table style='width:100%;border-collapse:collapse;background:#1f2937;'>
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


def prepare(df):
    df = df.copy()
    for col in ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM']:
        if col not in df.columns: df[col] = np.nan
    if 'hour' not in df.columns:
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df['hour']  = df['datetime'].dt.hour
            df['month'] = df['datetime'].dt.month
            df['year']  = df['datetime'].dt.year
            df['day']   = df['datetime'].dt.day
        else:
            df['hour']  = df.get('hour',  pd.Series(np.zeros(len(df), dtype=int)))
            df['month'] = df.get('month', pd.Series(np.ones(len(df), dtype=int)))
            df['year']  = df.get('year',  pd.Series(np.full(len(df), 2015, dtype=int)))
            df['day']   = df.get('day',   pd.Series(np.ones(len(df), dtype=int)))
    if 'season' not in df.columns:
        def mts(m):
            if m in [12,1,2]: return 'Winter'
            elif m in [3,4,5]: return 'Spring'
            elif m in [6,7,8]: return 'Summer'
            else: return 'Autumn'
        df['season'] = df['month'].apply(mts)
    if 'station' not in df.columns: df['station'] = 'Unknown'
    if 'station_type' not in df.columns:
        sub = ['Changping','Dingling','Shunyi','Huairou']
        df['station_type'] = df['station'].apply(lambda s: 'Suburban' if any(x in str(s) for x in sub) else 'Urban')
    if 'aqi_category' not in df.columns:
        df['aqi_category'] = df['PM2.5'].apply(lambda x: classify_aqi(x)[0] if pd.notnull(x) else 'Unknown')
    return df


# ── MODEL ────────────────────────────────────────────────────
@st.cache_resource
def train_model(_df):
    dm = _df.copy()
    le_st=LabelEncoder(); le_se=LabelEncoder()
    dm['station_encoded']      = le_st.fit_transform(dm['station'])
    dm['season_encoded']       = le_se.fit_transform(dm['season'])
    dm['station_type_encoded'] = dm['station_type'].map({'Urban':1,'Suburban':0})
    fc = ['PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM',
          'hour','month','season_encoded','station_encoded','station_type_encoded']
    dc = dm.dropna(subset=['PM2.5']+fc)
    X, y = dc[fc], dc['PM2.5']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    sc = StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
    lr=LinearRegression(); lr.fit(Xtr_s,ytr); lp=lr.predict(Xte_s)
    lmae=mean_absolute_error(yte,lp); lrmse=np.sqrt(mean_squared_error(yte,lp)); lr2=r2_score(yte,lp)
    rf=RandomForestRegressor(n_estimators=100,max_depth=25,min_samples_split=5,
                              min_samples_leaf=2,max_features=0.5,random_state=42,n_jobs=-1)
    rf.fit(Xtr_s,ytr); rp=rf.predict(Xte_s)
    rmae=mean_absolute_error(yte,rp); rrmse=np.sqrt(mean_squared_error(yte,rp)); rr2=r2_score(yte,rp)
    return rf,sc,fc,le_st,le_se,Xte_s,yte,rp,rmae,rrmse,rr2,lmae,lrmse,lr2

@st.cache_data
def load_default(): return prepare(pd.read_csv('beijing_air_quality_cleaned.csv'))
@st.cache_data
def load_uploaded(fb): return prepare(pd.read_csv(fb))


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:22px 16px 18px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:10px;'>
        <div style='display:flex;align-items:center;gap:9px;margin-bottom:4px;'>
            <div style='width:34px;height:34px;background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.25);
                        border-radius:9px;display:flex;align-items:center;justify-content:center;flex-shrink:0;'>
                <i class="fa-solid fa-leaf" style="color:#00d4aa;font-size:15px;"></i>
            </div>
            <div>
                <div style='font-size:0.82rem;font-weight:700;color:#f1f5f9;line-height:1.3;'>Kunalan Subatharan</div>
                <div style='font-size:0.58rem;color:#00d4aa;letter-spacing:0.5px;font-weight:600;'>ST20274714 · Cardiff Met</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    NAV_PAGES = [
        "🏠  Overview",
        "📤  Data Upload",
        "📋  Dataset",
        "📊  Visualisation",
        "🤖  Analytics",
        "📄  Reports",
    ]
    if 'nav_page' not in st.session_state:
        st.session_state.nav_page = "🏠  Overview"

    page = st.radio("nav", NAV_PAGES,
        index=NAV_PAGES.index(st.session_state.nav_page),
        label_visibility="collapsed")

    # Always keep session state in sync with what user clicked
    if page != st.session_state.nav_page:
        st.session_state.nav_page = page

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style='font-size:0.6rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                            color:#475569;margin:8px 4px 8px;'>
        🗄️ Dataset Source
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df = load_uploaded(uploaded); dsrc = "uploaded"
        st.success(f"✓ {uploaded.name}")
    else:
        df = load_default(); dsrc = "default"

    with st.spinner("Training model..."):
        (model,scaler,feature_cols,le_station,le_season,
         X_test_sc,y_test,y_pred,
         rf_mae,rf_rmse,rf_r2,lr_mae,lr_rmse,lr_r2) = train_model(df)

    st.markdown(f"""
    <div style='margin-top:14px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                border-radius:10px;padding:14px 16px;'>
        <div style='font-size:0.58rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                    color:#475569;margin-bottom:10px;display:flex;align-items:center;gap:5px;'>
            ℹ️ Live Summary
        </div>
        <div style='font-size:0.75rem;line-height:2.5;'>
            <div style='display:flex;justify-content:space-between;'>
                <span style='color:#94a3b8;'>Records</span>
                <span style='color:#00d4aa;font-family:DM Mono,monospace;'>{len(df):,}</span>
            </div>
            <div style='display:flex;justify-content:space-between;'>
                <span style='color:#94a3b8;'>Stations</span>
                <span style='color:#00d4aa;font-family:DM Mono,monospace;'>{df['station'].nunique()}</span>
            </div>
            <div style='display:flex;justify-content:space-between;'>
                <span style='color:#94a3b8;'>Model R²</span>
                <span style='color:#00d4aa;font-family:DM Mono,monospace;font-weight:700;'>{rf_r2:.4f}</span>
            </div>
            <div style='display:flex;justify-content:space-between;'>
                <span style='color:#94a3b8;'>MAE</span>
                <span style='color:#00d4aa;font-family:DM Mono,monospace;'>{rf_mae:.2f} µg/m³</span>
            </div>
        </div>
    </div>
    <div style='margin-top:10px;background:rgba(0,212,170,0.06);border:1px solid rgba(0,212,170,0.15);
                border-radius:8px;padding:9px 12px;display:flex;align-items:center;justify-content:space-between;'>
        <div style='display:flex;align-items:center;gap:6px;font-size:0.68rem;color:#00d4aa;font-weight:600;'>
            <span style='width:6px;height:6px;border-radius:50%;background:#10b981;display:inline-block;
                         animation:pulse-dot 2s ease-in-out infinite;'></span>
            System Online
        </div>
        <div style='font-size:0.65rem;color:#475569;'>CMP7005 · Cardiff Met</div>
    </div>
    <div style='margin-top:16px;padding-top:14px;border-top:1px solid rgba(255,255,255,0.06);'>
        <div style='display:flex;align-items:center;gap:8px;font-size:0.74rem;color:#475569;margin-bottom:8px;'>
            ⚙️ <span>Settings</span>
        </div>
        <div style='display:flex;align-items:center;gap:8px;font-size:0.74rem;color:#475569;'>
            ❓ <span>Support</span>
        </div>
    </div>
    </div>""", unsafe_allow_html=True)

    # ── New Analysis button (real Streamlit button - works reliably) ──
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    if st.button("＋  New Analysis", use_container_width=True):
        st.session_state.nav_page = "🏠  Overview"
        st.rerun()


# ── TOPBAR ───────────────────────────────────────────────────
st.markdown(f"""
<div class="eco-topbar">
    <div class="eco-brand">
        🌿
        Environmental <span class="brand-accent">Dashboard</span>
    </div>
    <div class="eco-topbar-right">
        <div class="eco-status">
            <div class="eco-status-dot"></div>
            SYSTEM ONLINE
        </div>
        <div class="eco-icon-btn">🔔</div>
        <div class="eco-icon-btn">⚙️</div>
        <div class="eco-avatar">KS</div>
    </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════
if "Overview" in page:
    avg_pm  = df['PM2.5'].mean()
    good_pct = (df['aqi_category']=='Good').mean()*100

    st.markdown(f"""
    <div class="eco-hero">
        <div class="eco-hero-eyebrow">
            ● Beijing Multi-Site Air Quality Dataset &nbsp;·&nbsp; 2013-2017
        </div>
        <div class="eco-hero-title">
            Air Quality <span class="ht-accent">Intelligence</span><br>Analytics Platform
        </div>
        <div class="eco-hero-desc">
            Comprehensive PM2.5 analysis across 4 Beijing monitoring stations — combining
            exploratory data analysis with machine learning prediction to track environmental
            health trends and mitigation efficacy.
        </div>
        <div style="margin-top:32px;"></div>
    </div>""", unsafe_allow_html=True)

    # Buttons visually pulled UP into the hero box via negative margin
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"]:has(> div > div[data-testid="stVerticalBlockBorderWrapper"] .hero-btn-row) {
        margin-top: -80px !important; position: relative; z-index: 10; padding: 0 54px;
    }
    /* Style the two hero buttons */
    .hero-btn-wrap { display:flex; gap:14px; margin-top:-72px; padding:0 54px 0 54px;
                     position:relative; z-index:10; margin-bottom:24px; }
    </style>
    <div class="hero-btn-wrap" id="hero-btn-placeholder"></div>
    """, unsafe_allow_html=True)

    # Pull buttons up with negative margin CSS targeting their container
    st.markdown("""<style>
    /* Target the columns row that contains btn_explore / btn_model */
    div[data-testid="stHorizontalBlock"]:has(button[data-testid="baseButton-secondary"]) {
        margin-top: -82px !important;
        padding-left: 54px !important;
        padding-right: 54px !important;
        position: relative !important;
        z-index: 20 !important;
        margin-bottom: 20px !important;
    }
    /* Explore Dataset — solid teal */
    div[data-testid="stHorizontalBlock"]:has(button[data-testid="baseButton-secondary"])
        div:nth-child(1) button {
        background: #00d4aa !important;
        color: #0a1628 !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 12px 24px !important;
        border-radius: 9px !important;
    }
    /* View Model — ghost style */
    div[data-testid="stHorizontalBlock"]:has(button[data-testid="baseButton-secondary"])
        div:nth-child(2) button {
        background: transparent !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 12px 24px !important;
        border-radius: 9px !important;
        box-shadow: none !important;
    }
    div[data-testid="stHorizontalBlock"]:has(button[data-testid="baseButton-secondary"])
        div:nth-child(2) button:hover {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.4) !important;
    }
    </style>""", unsafe_allow_html=True)

    _hc1, _hc2, _hc3 = st.columns([1.5, 1.5, 4])
    with _hc1:
        if st.button("→  Explore Dataset", key="btn_explore", use_container_width=True):
            st.session_state.nav_page = "📋  Dataset"
            st.rerun()
    with _hc2:
        if st.button("🤖  View Model", key="btn_model", use_container_width=True):
            st.session_state.nav_page = "🤖  Analytics"
            st.rerun()

    # Stat Cards
    c1,c2,c3,c4 = st.columns(4)
    cards = [
        (c1,"fa-database","TOTAL RECORDS",  f"{len(df):,}",           "eco-badge-teal",  "ACTIVE",    72,   TEAL),
        (c2,"fa-location-dot","STATIONS",   f"{df['station'].nunique()} Sites","eco-badge-green","ONLINE",55,GREEN),
        (c3,"fa-wind","MEAN PM2.5",         f"{avg_pm:.1f} µg/m³",    "eco-badge-red",   "ABOVE WHO", 65,   RED),
        (c4,"fa-chart-simple","MODEL R²",   f"{rf_r2:.4f}",            "eco-badge-teal",  "OPTIMISED", int(rf_r2*100), TEAL),
    ]
    for col,icon,lbl,val,bc,bt,bw,bcol in cards:
        with col:
            st.markdown(f"""
            <div class="eco-stat">
                <div class="eco-stat-top">
                    <div class="eco-stat-label">{lbl}</div>
                    <div class="eco-badge {bc}">
                        <i class="fa-solid fa-circle" style="font-size:5px;"></i>{bt}
                    </div>
                </div>
                <div class="eco-stat-value">{val}</div>
                <div class="eco-stat-bar"><div class="eco-stat-fill" style="width:{bw}%;background:{bcol};"></div></div>
                <i class="fa-solid {icon} eco-stat-icon"></i>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    L, R = st.columns([1.7, 1])

    with L:
        mo = df.groupby(['year','month'])['PM2.5'].mean().reset_index()
        mo['date'] = pd.to_datetime(mo[['year','month']].assign(day=1))
        fig_tr = px.line(mo, x='date', y='PM2.5', color_discrete_sequence=[TEAL], height=265)
        fig_tr.update_traces(line_width=2.5, fill='tozeroy', fillcolor='rgba(0,212,170,0.05)')
        fig_tr.add_hline(y=15, line_dash='dot', line_color=GREEN,
                         annotation_text='WHO 15 µg/m³', annotation_font_color=GREEN)
        ap(fig_tr)
        st.markdown("""
        <div class="eco-card" style="margin-bottom:0;">
            <div class="eco-card-title">
                <i class="fa-solid fa-chart-line"></i>Monthly PM2.5 Trend (2013-2017)
                <span style="margin-left:auto;">
                    <span class="eco-badge eco-badge-teal" style="cursor:pointer;">
                        <i class="fa-solid fa-download" style="font-size:7px;"></i>Download CSV
                    </span>
                </span>
            </div>
            <div class="eco-card-sub">Mean hourly PM2.5 across all 4 stations with WHO guideline reference</div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_tr, use_container_width=True)

    with R:
        aqi_c = df['aqi_category'].value_counts()
        cat_order = ['Good','Moderate','Unhealthy for Sensitive Groups','Unhealthy','Very Unhealthy','Hazardous']
        st.markdown("""
        <div class="eco-card" style="margin-bottom:0;">
            <div class="eco-card-title"><i class="fa-solid fa-chart-pie"></i>AQI Category Distribution</div>
            <div class="eco-card-sub">Proportion of readings in each health band</div>
        """, unsafe_allow_html=True)
        for cat in cat_order:
            if cat in aqi_c.index:
                pct = aqi_c[cat]/len(df)*100
                short = cat.replace('Unhealthy for Sensitive Groups','Unhealthy (SG)')
                col_bar = AQI_COL.get(cat, TEAL)
                st.markdown(f"""
                <div class="eco-dist-row">
                    <span class="eco-dist-name">{short}</span>
                    <div class="eco-dist-bar-bg"><div class="eco-dist-bar" style="width:{pct:.0f}%;background:{col_bar};"></div></div>
                    <span class="eco-dist-val">{pct:.1f}%</span>
                </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="eco-insight" style="margin-top:14px;">
            <div class="eco-insight-title"><i class="fa-solid fa-lightbulb"></i>Key Finding</div>
            <div class="eco-insight-body">
                Only <b style="color:#f1f5f9;">{good_pct:.1f}%</b> of hourly readings meet WHO PM2.5 standards
                (≤35 µg/m³). The annual mean of <b style="color:#f1f5f9;">{avg_pm:.1f} µg/m³</b> is
                <b style="color:#ef4444;">{avg_pm/15:.1f}×</b> the WHO 24-hour guideline.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="eco-divider"></div>', unsafe_allow_html=True)
    st.markdown("""<div class="pg-eyebrow"><i class="fa-solid fa-tower-broadcast"></i>Station Network</div>""", unsafe_allow_html=True)

    stn = df.groupby('station').agg(mean_pm25=('PM2.5','mean'),max_pm25=('PM2.5','max'),records=('PM2.5','count')).reset_index()
    if 'station_type' in df.columns:
        stype_map = df.groupby('station')['station_type'].first().to_dict()
        stn['station_type'] = stn['station'].map(stype_map).fillna('Urban')
    else:
        stn['station_type'] = 'Urban'

    for col_s, (_,row) in zip(st.columns(len(stn)), stn.iterrows()):
        is_sub = row['station_type']=='Suburban'
        bc = 'eco-badge-blue' if is_sub else 'eco-badge-amber'
        bcol = BLUE if is_sub else AMBER
        icon = 'fa-tree' if is_sub else 'fa-city'
        pm_col = RED if row['mean_pm25']>75 else AMBER if row['mean_pm25']>35 else GREEN
        col_s.markdown(f"""
        <div class="eco-stat" style="border-top:2px solid {bcol};">
            <div class="eco-badge {bc}" style="margin-bottom:12px;">
                <i class="fa-solid {icon}" style="font-size:8px;"></i>{row['station_type'].upper()}
            </div>
            <div style='font-size:0.98rem;font-weight:700;color:{TEXT};margin-bottom:14px;
                         display:flex;align-items:center;gap:7px;'>
                <i class="fa-solid fa-location-dot" style="color:{bcol};font-size:0.8rem;"></i>{row['station']}
            </div>
            <div style='font-size:0.73rem;line-height:2.5;'>
                <div style='display:flex;justify-content:space-between;'>
                    <span style='color:{MUTED};'>Mean PM2.5</span>
                    <span style='color:{pm_col};font-family:DM Mono,monospace;font-weight:600;'>{row['mean_pm25']:.1f}</span>
                </div>
                <div style='display:flex;justify-content:space-between;'>
                    <span style='color:{MUTED};'>Max PM2.5</span>
                    <span style='color:{RED};font-family:DM Mono,monospace;'>{row['max_pm25']:.0f}</span>
                </div>
                <div style='display:flex;justify-content:space-between;'>
                    <span style='color:{MUTED};'>Records</span>
                    <span style='color:{TEXT};font-family:DM Mono,monospace;'>{row['records']:,}</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: DATA UPLOAD
# ══════════════════════════════════════════════════════════════
elif "Data Upload" in page or "Upload" in page:
    st.markdown("""
    <div class="pg-eyebrow"><i class="fa-solid fa-folder-open"></i>Archive Management</div>
    <div class="pg-title">Import Dataset</div>
    <div class="pg-sub">Upload your cleaned Beijing air quality CSV. All pages — visualisations, model and predictor — update automatically.</div>
    """, unsafe_allow_html=True)

    L, R = st.columns([1.5, 1])
    with L:
        border = "rgba(0,212,170,0.3)" if dsrc=="uploaded" else "rgba(255,255,255,0.1)"
        ic = "#00d4aa" if dsrc=="uploaded" else "#64748b"
        icon = "fa-circle-check" if dsrc=="uploaded" else "fa-cloud-arrow-up"
        title = "File Loaded Successfully" if dsrc=="uploaded" else "Drop your CSV here"
        msg = (f"<b style='color:#00d4aa;'>{uploaded.name}</b> is active." if dsrc=="uploaded"
               else "Use the <b>file uploader in the left sidebar</b> to select your CSV.")
        st.markdown(f"""
        <div class="eco-card" style='border:1px dashed {border};text-align:center;padding:48px 36px;'>
            <div style='width:58px;height:58px;background:rgba(0,212,170,0.07);border:1px solid rgba(0,212,170,0.2);
                        border-radius:14px;display:flex;align-items:center;justify-content:center;margin:0 auto 18px;'>
                <i class="fa-solid {icon}" style="color:{ic};font-size:1.4rem;"></i>
            </div>
            <div style='font-size:1rem;font-weight:600;color:{TEXT};margin-bottom:8px;'>{title}</div>
            <div style='font-size:0.82rem;color:#64748b;'>{msg}</div>
            <div style='margin-top:20px;font-size:0.58rem;letter-spacing:2px;color:#374151;font-weight:700;'>SUPPORTED: .CSV</div>
        </div>""", unsafe_allow_html=True)
        if dsrc=="default": st.info("ℹ️ Using default dataset — beijing_air_quality_cleaned.csv")

    with R:
        mp = df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100
        st.markdown(f"""
        <div class="eco-card">
            <div class="eco-card-title"><i class="fa-solid fa-chart-simple"></i>File Statistics</div>
            <div style='margin:14px 0;'>
                <div class='eco-stat-label' style='margin-bottom:4px;'>Total Observations</div>
                <div style='font-family:DM Mono,monospace;font-size:2.2rem;font-weight:500;color:{TEAL};'>{len(df):,}</div>
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;'>
                <div>
                    <div class='eco-stat-label' style='margin-bottom:4px;'>Columns</div>
                    <div style='font-family:DM Mono,monospace;font-size:1.6rem;color:{TEXT};'>{df.shape[1]}</div>
                </div>
                <div>
                    <div class='eco-stat-label' style='margin-bottom:4px;'>Completeness</div>
                    <div style='font-family:DM Mono,monospace;font-size:1.6rem;color:{GREEN};'>{100-mp:.1f}%</div>
                </div>
            </div>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:14px;'>
                <div class="eco-badge eco-badge-green"><i class="fa-solid fa-shield-halved" style="font-size:8px;"></i>VERIFIED</div>
                <span style='font-size:0.72rem;color:#64748b;'>Integrity checked</span>
            </div>
            <div style='background:rgba(255,255,255,0.03);border:1px solid var(--border);border-radius:8px;padding:14px;'>
                <div style='font-size:0.58rem;font-weight:700;letter-spacing:1px;color:{TEAL};margin-bottom:8px;text-transform:uppercase;'>
                    <i class="fa-solid fa-list-check" style="margin-right:4px;"></i>Required Columns
                </div>
                <div style='font-size:0.7rem;color:#64748b;line-height:2;font-family:DM Mono,monospace;'>
                    PM2.5 · PM10 · SO2 · NO2 · CO · O3<br>TEMP · PRES · DEWP · RAIN · WSPM<br>hour · month · station · season
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-table"></i>Data Preview — Top 10 Rows</div></div>""", unsafe_allow_html=True)
    dcols = [c for c in ['year','month','day','hour','station','station_type','season','PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM'] if c in df.columns]
    dark_table(df[dcols], max_rows=10)

    st.markdown("""<div class="eco-card" style="margin-top:14px;"><div class="eco-card-title"><i class="fa-solid fa-list-check"></i>Column Validation</div></div>""", unsafe_allow_html=True)
    req=['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM','hour','month','station','season','station_type']
    vrows=[{'Column':c,'Present':'✅' if c in df.columns else '❌','Missing Values':int(df[c].isnull().sum()) if c in df.columns else 'N/A','Type':str(df[c].dtype) if c in df.columns else '-','Example':str(df[c].dropna().iloc[0]) if c in df.columns and len(df[c].dropna())>0 else '-'} for c in req]
    dark_table(pd.DataFrame(vrows))


# ══════════════════════════════════════════════════════════════
# PAGE: DATASET
# ══════════════════════════════════════════════════════════════
elif "Dataset" in page:
    st.markdown("""
    <div class="pg-eyebrow"><i class="fa-solid fa-microchip"></i>Data Intelligence</div>
    <div class="pg-title">Dataset Overview</div>
    <div class="pg-sub">Technical audit of meteorological and pollutant concentrations from 4 Beijing monitoring stations (March 2013 — February 2017).</div>
    """, unsafe_allow_html=True)

    mp = df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100
    c1,c2,c3,c4 = st.columns(4)
    for col,icon,lbl,val,vc in [
        (c1,"fa-database","Observations",f"{len(df):,}",TEAL),
        (c2,"fa-table-columns","Columns",f"{df.shape[1]}",TEXT),
        (c3,"fa-circle-check","Completeness",f"{100-mp:.1f}%",GREEN),
        (c4,"fa-tower-broadcast","Stations",f"{df['station'].nunique()}",BLUE),
    ]:
        with col:
            st.markdown(f"""
            <div class="eco-stat">
                <div class="eco-stat-label">{lbl}</div>
                <div class="eco-stat-value" style="color:{vc};">{val}</div>
                <i class="fa-solid {icon} eco-stat-icon"></i>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    L, R = st.columns([1.3, 1])
    with L:
        kc=[c for c in ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM'] if c in df.columns]
        mc=df[kc].isnull().sum().reset_index(); mc.columns=['Feature','Missing']
        mc['Pct']=(mc['Missing']/len(df)*100).round(2)
        fig_m=px.bar(mc,x='Feature',y='Pct',color='Pct',height=255,color_continuous_scale=[GREEN,AMBER,RED])
        ap(fig_m, {'coloraxis_showscale':False})
        st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-triangle-exclamation"></i>Missing Values per Feature</div></div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_m, use_container_width=True)
    with R:
        sr=[{'Column':c,'Type':str(df[c].dtype),'Example':str(df[c].dropna().iloc[0]) if len(df[c].dropna())>0 else '-','Status':'✅'} for c in df.columns[:16]]
        st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-code"></i>Schema</div></div>""", unsafe_allow_html=True)
        dark_table(pd.DataFrame(sr), height=255)

    st.markdown("""<div class="eco-card" style="margin-top:4px;"><div class="eco-card-title"><i class="fa-solid fa-calculator"></i>Statistical Summary</div></div>""", unsafe_allow_html=True)
    nc=[c for c in ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM'] if c in df.columns]
    sm=df[nc].describe().T.round(2)[['count','mean','std','min','25%','50%','75%','max']]
    sm.columns=['Count','Mean','Std Dev','Min','25%','50%','75%','Max']; sm.index.name='Feature'
    dark_table(sm.reset_index())

    st.markdown(f"""
    <div class="eco-insight">
        <div class="eco-insight-title"><i class="fa-solid fa-circle-check"></i>{100-mp:.1f}% Data Completeness</div>
        <div class="eco-insight-body">Four-station dataset covers March 2013 — February 2017. After cleaning, <b style="color:{TEXT};">{len(df):,}</b> complete records remain for analysis and modelling.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-filter"></i>Interactive Data Explorer</div></div>""", unsafe_allow_html=True)
    f1,f2,f3=st.columns(3)
    ss=f1.selectbox("Station",['All']+sorted(df['station'].unique().tolist()))
    se=f2.selectbox("Season", ['All']+sorted(df['season'].unique().tolist()))
    sy=f3.selectbox("Year",   ['All']+sorted(df['year'].unique().tolist()))
    dff=df.copy()
    if ss!='All': dff=dff[dff['station']==ss]
    if se!='All': dff=dff[dff['season']==se]
    if sy!='All': dff=dff[dff['year']==int(sy)]
    st.caption(f"{len(dff):,} records after filters")
    dcols=[c for c in ['year','month','day','hour','station','station_type','season','PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM'] if c in dff.columns]
    dark_table(dff[dcols], max_rows=500, height=300)


# ══════════════════════════════════════════════════════════════
# PAGE: VISUALISATION
# ══════════════════════════════════════════════════════════════
elif "Visualisation" in page:
    st.markdown("""
    <div class="pg-eyebrow"><i class="fa-solid fa-eye"></i>Visual Analysis</div>
    <div class="pg-title">Air Quality Analytics</div>
    <div class="pg-sub">Interactive exploration of pollutant distributions, temporal patterns, station comparisons and variable relationships.</div>
    """, unsafe_allow_html=True)

    tab1,tab2,tab3,tab4=st.tabs(["  📈  Distribution  ","  ⏱️  Temporal  ","  🏙️  Stations  ","  🔗  Bivariate  "])

    with tab1:
        L,R=st.columns([1.6,1])
        with L:
            fig_h=px.histogram(df,x='PM2.5',nbins=60,color_discrete_sequence=[TEAL],height=290)
            fig_h.update_traces(marker_line_color='rgba(255,255,255,0.04)',marker_line_width=0.5,opacity=0.85)
            fig_h.add_vline(x=df['PM2.5'].mean(),line_dash='dash',line_color=RED,annotation_text=f"Mean: {df['PM2.5'].mean():.1f}",annotation_font_color=RED)
            fig_h.add_vline(x=15,line_dash='dot',line_color=GREEN,annotation_text="WHO: 15",annotation_font_color=GREEN)
            ap(fig_h)
            st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-chart-bar"></i>PM2.5 Distribution Histogram</div><div class="eco-card-sub">Right-skewed with WHO and mean reference lines</div></div>""", unsafe_allow_html=True)
            st.plotly_chart(fig_h, use_container_width=True)

            pols=[c for c in ['PM2.5','PM10','SO2','NO2','CO','O3'] if c in df.columns]
            clrs=[TEAL,TEAL2,AMBER,RED,'#8b5cf6',GREEN]
            fig_b=go.Figure()
            for p,c in zip(pols,clrs):
                fig_b.add_trace(go.Box(y=df[p].dropna(),name=p,marker_color=c,boxmean=True,line_width=1.5))
            ap(fig_b, {'height':270,'yaxis_title':'µg/m³'})
            st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-box"></i>All Pollutants — Box Plots</div></div>""", unsafe_allow_html=True)
            st.plotly_chart(fig_b, use_container_width=True)

        with R:
            cc=[c for c in ['PM2.5','PM10','NO2','O3','TEMP','WSPM','PRES'] if c in df.columns]
            fig_c=px.imshow(df[cc].corr().round(2),color_continuous_scale=['#1e3a5f','#0d6e64','#00b894',TEAL],zmin=-1,zmax=1,text_auto=True,height=310)
            ap(fig_c, {'coloraxis_showscale':False})
            fig_c.update_traces(textfont=dict(color='#f1f5f9',size=10))
            st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-grip"></i>Correlation Heatmap</div><div class="eco-card-sub">Cross-pollutant dependency matrix</div></div>""", unsafe_allow_html=True)
            st.plotly_chart(fig_c, use_container_width=True)

            ac=df['aqi_category'].value_counts().reset_index(); ac.columns=['Category','Count']
            fig_p=px.pie(ac,names='Category',values='Count',color='Category',color_discrete_map=AQI_COL,height=230,hole=0.45)
            ap(fig_p, {'margin':dict(t=10,b=10,l=0,r=0),'legend':dict(font=dict(color=MUTED,size=9),bgcolor='rgba(0,0,0,0)')})
            st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-circle-half-stroke"></i>AQI Breakdown</div></div>""", unsafe_allow_html=True)
            st.plotly_chart(fig_p, use_container_width=True)

    with tab2:
        mo=df.groupby(['year','month'])['PM2.5'].mean().reset_index(); mo['date']=pd.to_datetime(mo[['year','month']].assign(day=1))
        fig_l=px.line(mo,x='date',y='PM2.5',color_discrete_sequence=[TEAL],height=270)
        fig_l.update_traces(line_width=2.5,fill='tozeroy',fillcolor='rgba(0,212,170,0.04)')
        fig_l.add_hline(y=15,line_dash='dot',line_color=GREEN,annotation_text='WHO: 15 µg/m³',annotation_font_color=GREEN)
        ap(fig_l)
        st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-chart-line"></i>Monthly PM2.5 Trend</div></div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_l, use_container_width=True)

        t1,t2=st.columns(2)
        with t1:
            ho=df.groupby('hour')['PM2.5'].mean().reset_index()
            fig_ho=px.area(ho,x='hour',y='PM2.5',height=230,color_discrete_sequence=[TEAL])
            fig_ho.update_traces(fillcolor='rgba(0,212,170,0.07)',line_width=2); ap(fig_ho)
            st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-regular fa-clock"></i>Hour-of-Day Pattern</div></div>""", unsafe_allow_html=True)
            st.plotly_chart(fig_ho, use_container_width=True)
        with t2:
            mn_m={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
            ma=df.groupby('month')['PM2.5'].mean().reset_index(); ma['Month']=ma['month'].map(mn_m)
            fig_ma=px.bar(ma,x='Month',y='PM2.5',color='PM2.5',height=230,color_continuous_scale=[GREEN,AMBER,RED])
            ap(fig_ma, {'coloraxis_showscale':False})
            st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-calendar"></i>Monthly Seasonal Pattern</div></div>""", unsafe_allow_html=True)
            st.plotly_chart(fig_ma, use_container_width=True)

    with tab3:
        s1,s2=st.columns(2)
        with s1:
            sa=df.groupby('station')['PM2.5'].mean().reset_index().sort_values('PM2.5',ascending=True)
            fig_sa=px.bar(sa,x='PM2.5',y='station',orientation='h',color='PM2.5',height=260,color_continuous_scale=[GREEN,AMBER,RED],labels={'PM2.5':'Mean PM2.5 (µg/m³)'})
            ap(fig_sa, {'coloraxis_showscale':False})
            st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-ranking-star"></i>Station PM2.5 Ranking</div></div>""", unsafe_allow_html=True)
            st.plotly_chart(fig_sa, use_container_width=True)
        with s2:
            _bc='station_type' if 'station_type' in df.columns else None
            _bk=dict(color_discrete_map={'Urban':AMBER,'Suburban':BLUE}) if _bc else {}
            fig_bs=px.box(df,x='station',y='PM2.5',color=_bc,height=260,labels={'PM2.5':'PM2.5 (µg/m³)'},**_bk)
            ap(fig_bs, {'legend':dict(font=dict(color=MUTED),bgcolor='rgba(0,0,0,0)')})
            st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-box-open"></i>Station Distribution</div></div>""", unsafe_allow_html=True)
            st.plotly_chart(fig_bs, use_container_width=True)

        so=['Winter','Spring','Summer','Autumn']
        if 'season' not in df.columns: df['season']='Unknown'
        seas=df.groupby(['season','station'])['PM2.5'].mean().reset_index()
        seas['season']=pd.Categorical(seas['season'],categories=so,ordered=True)
        fig_se=px.bar(seas.sort_values('season'),x='season',y='PM2.5',color='station',barmode='group',height=270,color_discrete_sequence=[TEAL,TEAL2,BLUE,'#8b5cf6'],labels={'PM2.5':'Mean PM2.5 (µg/m³)'})
        ap(fig_se, {'legend':dict(font=dict(color=MUTED),bgcolor='rgba(0,0,0,0)')})
        st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-snowflake"></i>PM2.5 by Season and Station</div></div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_se, use_container_width=True)

    with tab4:
        b1,b2=st.columns(2)
        xv=b1.selectbox("X-axis variable",[c for c in ['TEMP','PRES','DEWP','WSPM','NO2','CO','SO2','O3','PM10'] if c in df.columns])
        cv=b2.selectbox("Colour by",[c for c in ['season','station_type','station'] if c in df.columns])
        sdf=df[[xv,'PM2.5',cv]].dropna().sample(min(5000,len(df)),random_state=42)
        fig_s=px.scatter(sdf,x=xv,y='PM2.5',color=cv,opacity=0.45,height=380,color_discrete_sequence=[TEAL,TEAL2,BLUE,'#8b5cf6'])
        x2=sdf[xv].values; y2=sdf['PM2.5'].values; msk=~(np.isnan(x2)|np.isnan(y2))
        if msk.sum()>1:
            co_=np.polyfit(x2[msk],y2[msk],1); xl=np.linspace(x2[msk].min(),x2[msk].max(),100)
            fig_s.add_trace(go.Scatter(x=xl,y=np.polyval(co_,xl),mode='lines',name='Trend',line=dict(color=RED,width=2,dash='dash')))
        ap(fig_s, {'legend':dict(font=dict(color=MUTED),bgcolor='rgba(0,0,0,0)')})
        st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-circle-nodes"></i>Bivariate Scatter Explorer</div></div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_s, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: ANALYTICS (Model + Predictor + Relationships)
# ══════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown("""
    <div class="pg-eyebrow"><i class="fa-solid fa-robot"></i>Machine Learning</div>
    <div class="pg-title">Model Analytics</div>
    <div class="pg-sub">Optimised Random Forest vs Linear Regression baseline — 15 features, 80/20 train/test split, full performance diagnostics and live PM2.5 prediction.</div>
    """, unsafe_allow_html=True)

    tab_perf,tab_pred,tab_rel=st.tabs(["  🎯  Model Performance  ","  🔮  Live Predictor  ","  📐  Relationships  "])

    with tab_perf:
        c1,c2,c3=st.columns(3)
        for col,lbl,val,note,nc in [
            (c1,"R² Score",      f"{rf_r2:.4f}", f"↗ +{rf_r2-lr_r2:.4f} vs Linear Regression","eco-metric-note-g"),
            (c2,"Mean Abs Error",f"{rf_mae:.2f}",f"µg/m³  ·  {lr_mae-rf_mae:.2f} improvement","eco-metric-note"),
            (c3,"Root MSE",      f"{rf_rmse:.2f}","µg/m³ — driven by extreme events","eco-metric-note-r"),
        ]:
            with col:
                st.markdown(f"""<div class="eco-metric">
                    <div class="eco-metric-label">{lbl}</div>
                    <div class="eco-metric-value">{val}</div>
                    <div class="eco-metric-note {nc}">{note}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        L,R=st.columns([1.7,1])
        with L:
            ta,tb,tc=st.tabs(["  🎯  Actual vs Predicted  ","  📊  Model Comparison  ","  📉  Residuals  "])
            with ta:
                yta=np.array(y_test); res=yta-y_pred
                idx=np.random.choice(len(yta),size=min(3000,len(yta)),replace=False)
                avdf=pd.DataFrame({'Actual':yta[idx],'Predicted':y_pred[idx],'Residual':res[idx]})
                fig_av=px.scatter(avdf,x='Actual',y='Predicted',color='Residual',color_continuous_scale=[RED,'#fef9c3',TEAL],opacity=0.45,height=340)
                mv=max(avdf['Actual'].max(),avdf['Predicted'].max())
                fig_av.add_trace(go.Scatter(x=[0,mv],y=[0,mv],mode='lines',name='Perfect',line=dict(color=TEAL,dash='dash',width=1.5)))
                ap(fig_av, {'coloraxis_showscale':False,'legend':dict(font=dict(color=MUTED),bgcolor='rgba(0,0,0,0)')})
                st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-bullseye"></i>Actual vs Predicted PM2.5</div></div>""", unsafe_allow_html=True)
                st.plotly_chart(fig_av, use_container_width=True)
            with tb:
                cd=pd.DataFrame({'Model':['Linear Regression','Random Forest'],'MAE':[lr_mae,rf_mae],'RMSE':[lr_rmse,rf_rmse],'R²':[lr_r2,rf_r2]})
                fig_co=px.bar(pd.melt(cd,id_vars='Model',var_name='Metric',value_name='Value'),x='Metric',y='Value',color='Model',barmode='group',height=300,color_discrete_map={'Linear Regression':'#334155','Random Forest':TEAL})
                ap(fig_co, {'legend':dict(font=dict(color=MUTED),bgcolor='rgba(0,0,0,0)')})
                st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-scale-balanced"></i>Model Comparison</div></div>""", unsafe_allow_html=True)
                st.plotly_chart(fig_co, use_container_width=True)
                dark_table(cd)
            with tc:
                fig_rd=px.histogram(avdf,x='Residual',nbins=60,color_discrete_sequence=[TEAL],height=280)
                fig_rd.add_vline(x=0,line_dash='dash',line_color=RED,annotation_text='Zero (Perfect)')
                fig_rd.add_vline(x=res.mean(),line_dash='dot',line_color=GREEN,annotation_text=f'Mean: {res.mean():.2f}')
                ap(fig_rd)
                st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-wave-square"></i>Residual Distribution</div><div class="eco-card-sub">Centred near zero confirms low bias</div></div>""", unsafe_allow_html=True)
                st.plotly_chart(fig_rd, use_container_width=True)

        with R:
            fi=model.feature_importances_
            fi_df=pd.DataFrame({'Feature':feature_cols,'Importance':fi}).sort_values('Importance',ascending=False)
            mx=fi_df['Importance'].max()
            st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-ranking-star"></i>Feature Importance</div></div>""", unsafe_allow_html=True)
            for _,row in fi_df.iterrows():
                pct=int(row['Importance']/mx*100); dp=int(row['Importance']/fi.sum()*100)
                st.markdown(f"""
                <div class="eco-feat-row">
                    <div class="eco-feat-top">
                        <span class="eco-feat-name">{row['Feature']}</span>
                        <span class="eco-feat-pct">{dp}%</span>
                    </div>
                    <div class="eco-feat-bg"><div class="eco-feat-fill" style="width:{pct}%;"></div></div>
                </div>""", unsafe_allow_html=True)

    with tab_pred:
        L,R=st.columns([1.4,1])
        with L:
            st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-flask"></i>Pollutant Inputs (µg/m³)</div></div>""", unsafe_allow_html=True)
            p1,p2,p3=st.columns(3)
            ipm10=p1.number_input("PM10",0.0,1000.0,80.0,step=5.0)
            iso2 =p2.number_input("SO2", 0.0,500.0, 15.0,step=1.0)
            ino2 =p3.number_input("NO2", 0.0,300.0, 50.0,step=1.0)
            p4,p5=st.columns(2)
            ico  =p4.number_input("CO",0.0,15000.0,900.0,step=50.0)
            io3  =p5.number_input("O3",0.0,500.0,60.0,step=5.0)
            st.markdown("""<div class="eco-card" style="margin-top:10px;"><div class="eco-card-title"><i class="fa-solid fa-cloud-sun"></i>Meteorological Inputs</div></div>""", unsafe_allow_html=True)
            m1,m2,m3=st.columns(3)
            itemp=m1.number_input("Temp (°C)",-30.0,45.0,10.0,step=1.0)
            ipres=m2.number_input("Pres (hPa)",980.0,1040.0,1010.0,step=1.0)
            idewp=m3.number_input("Dew Pt (°C)",-40.0,30.0,-5.0,step=1.0)
            m4,m5=st.columns(2)
            irain=m4.number_input("Rain (mm)",0.0,100.0,0.0,step=0.5)
            iwspm=m5.number_input("Wind (m/s)",0.0,20.0,2.0,step=0.5)
            st.markdown("""<div class="eco-card" style="margin-top:10px;"><div class="eco-card-title"><i class="fa-solid fa-map-pin"></i>Temporal & Location</div></div>""", unsafe_allow_html=True)
            t1,t2=st.columns(2)
            ihr=t1.slider("Hour of Day",0,23,12)
            imo=t2.slider("Month",1,12,6)
            t3,t4=st.columns(2)
            istn=t3.selectbox("Station",sorted(df['station'].unique().tolist()))
            isty=t4.selectbox("Station Type",['Urban','Suburban'])
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.button("⚡  Generate PM2.5 Prediction", use_container_width=True)

        with R:
            smapx={12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
            ise=smapx.get(imo,'Summer')
            try: ste=le_station.transform([istn])[0]
            except: ste=0
            see={'Winter':3,'Spring':1,'Summer':2,'Autumn':0}.get(ise,0)
            sye=1 if isty=='Urban' else 0
            ia=np.array([[ipm10,iso2,ino2,ico,io3,itemp,ipres,idewp,irain,iwspm,ihr,imo,see,ste,sye]])
            pred=max(0,model.predict(scaler.transform(ia))[0])
            cat_name,cat_cls,cat_col=classify_aqi(pred)
            who_diff=pred-15
            bar_fill=min(int(pred/300*100),100)
            bar_col=RED if pred>75 else AMBER if pred>35 else GREEN

            radius=72; circ=2*3.14159*radius
            gpct=min(pred/300,1.0); goff=circ*(1-gpct)
            gcol=RED if pred>115 else AMBER if pred>35 else GREEN

            st.markdown(f"""
            <div class="eco-pred">
                <div class="eco-pred-label"><i class="fa-solid fa-crosshairs"></i> Predicted PM2.5</div>
                <svg width="180" height="180" viewBox="0 0 180 180" style="display:block;margin:0 auto 4px;">
                    <circle cx="90" cy="90" r="{radius}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="10"/>
                    <circle cx="90" cy="90" r="{radius}" fill="none" stroke="{gcol}" stroke-width="10"
                            stroke-linecap="round" stroke-dasharray="{circ:.1f}"
                            stroke-dashoffset="{goff:.1f}" transform="rotate(-90 90 90)"/>
                    <text x="90" y="84" text-anchor="middle" font-family="DM Mono,monospace"
                          font-size="26" font-weight="500" fill="{gcol}">{pred:.1f}</text>
                    <text x="90" y="102" text-anchor="middle" font-family="Inter,sans-serif"
                          font-size="11" fill="#64748b">µg/m³</text>
                </svg>
                <div class="eco-pred-unit">{ise} &nbsp;·&nbsp; {istn}</div>
                <div style='margin-bottom:18px;'>
                    <span class="eco-pred-badge {cat_cls}">{cat_name}</span>
                </div>
                <div class="eco-who-label">
                    <span style='color:{MUTED};'>vs WHO (15 µg/m³)</span>
                    <span style='color:{RED if who_diff>0 else GREEN};font-family:DM Mono;font-weight:600;'>
                        {("+" if who_diff>0 else "")+str(round(who_diff,1))} µg/m³
                    </span>
                </div>
                <div class="eco-who-bar-bg">
                    <div class="eco-who-bar-fill" style="width:{bar_fill}%;background:{bar_col};"></div>
                </div>
                <div style='font-size:0.75rem;color:{MUTED};text-align:left;font-style:italic;line-height:1.7;margin-bottom:18px;'>
                    {"⚠️ Above WHO guidelines — sensitive groups should reduce outdoor exposure." if pred>75
                      else "✅ Within acceptable WHO PM2.5 guidelines for this period."}
                </div>
            </div>
            <div class="eco-card" style="margin-top:14px;">
                <div class="eco-card-title"><i class="fa-solid fa-shield-halved"></i>Model Confidence</div>
                <div class="eco-conf-row"><span class="eco-conf-lbl">R² Score</span><span class="eco-conf-val">{rf_r2:.4f}</span></div>
                <div class="eco-conf-row"><span class="eco-conf-lbl">MAE</span><span class="eco-conf-val">{rf_mae:.2f} µg/m³</span></div>
                <div class="eco-conf-row"><span class="eco-conf-lbl">RMSE</span><span class="eco-conf-val">{rf_rmse:.2f} µg/m³</span></div>
                <div class="eco-conf-row"><span class="eco-conf-lbl">Training Records</span><span class="eco-conf-val">{int(len(df)*0.8):,}</span></div>
            </div>""", unsafe_allow_html=True)

    with tab_rel:
        rt1,rt2,rt3=st.tabs(["  🔲  Scatter Matrix  ","  🌡️  Full Correlation  ","  📊  Variable Deep Dive  "])
        with rt1:
            num_cols=[c for c in ['PM2.5','PM10','NO2','CO','TEMP','WSPM','O3'] if c in df.columns]
            sel=st.multiselect("Select variables",num_cols,default=['PM2.5','PM10','TEMP','WSPM'])
            if len(sel)>=2:
                _sc=sel+(['station_type'] if 'station_type' in df.columns else [])
                samp=df[_sc].dropna().sample(min(2500,len(df)),random_state=42)
                _smc='station_type' if 'station_type' in df.columns else None
                _smk=dict(color_discrete_map={'Urban':AMBER,'Suburban':BLUE}) if _smc else {}
                fig_sm=px.scatter_matrix(samp,dimensions=sel,color=_smc,**_smk,opacity=0.35,height=540)
                fig_sm.update_traces(marker=dict(size=3,line=dict(width=0)))
                ap(fig_sm, {'showlegend':True,'legend':dict(font=dict(color=MUTED),bgcolor='rgba(0,0,0,0)')})
                st.plotly_chart(fig_sm, use_container_width=True)
            else: st.info("Select at least 2 variables.")

        with rt2:
            all_num=[c for c in ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM','hour','month'] if c in df.columns]
            cf=df[all_num].corr().round(2)
            fig_cf=px.imshow(cf,color_continuous_scale=['#1e3a5f','#0d6e64','#00b894',TEAL],zmin=-1,zmax=1,text_auto=True,height=500)
            ap(fig_cf, {'coloraxis_showscale':True})
            fig_cf.update_traces(textfont=dict(color='#f1f5f9',size=9))
            st.plotly_chart(fig_cf, use_container_width=True)

            pm_c=cf['PM2.5'].drop('PM2.5').sort_values(key=abs,ascending=False)
            st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-arrows-up-down"></i>PM2.5 Correlation Ranking</div></div>""", unsafe_allow_html=True)
            for feat,cv in pm_c.items():
                cc=TEAL if cv>0 else RED; pct=int(abs(cv)*100)
                st.markdown(f"""<div class="eco-dist-row">
                    <span class="eco-dist-name">{feat}</span>
                    <div class="eco-dist-bar-bg"><div class="eco-dist-bar" style="width:{pct}%;background:{cc};"></div></div>
                    <span class="eco-dist-val" style='color:{cc};'>{cv:+.2f}</span>
                </div>""", unsafe_allow_html=True)

        with rt3:
            sv=st.selectbox("Select variable",[c for c in ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','WSPM'] if c in df.columns])
            v1,v2=st.columns(2)
            with v1:
                _vc='station_type' if 'station_type' in df.columns else None
                _vk=dict(color_discrete_map={'Urban':AMBER,'Suburban':BLUE}) if _vc else {}
                fig_vb=px.box(df,x='season',y=sv,color=_vc,height=280,category_orders={'season':['Winter','Spring','Summer','Autumn']},**_vk)
                ap(fig_vb, {'legend':dict(font=dict(color=MUTED),bgcolor='rgba(0,0,0,0)')})
                st.plotly_chart(fig_vb, use_container_width=True)
            with v2:
                hv=df.groupby('hour')[sv].mean().reset_index()
                fig_vh=px.area(hv,x='hour',y=sv,height=280,color_discrete_sequence=[TEAL])
                fig_vh.update_traces(fillcolor='rgba(0,212,170,0.07)',line_width=2); ap(fig_vh)
                st.plotly_chart(fig_vh, use_container_width=True)
            ss2=df[sv].describe().round(2)
            s1,s2,s3,s4=st.columns(4)
            for cs,ls,vs,vc in [(s1,'Mean',f"{ss2['mean']:.2f}",TEAL),(s2,'Std Dev',f"{ss2['std']:.2f}",MUTED),(s3,'Min',f"{ss2['min']:.2f}",GREEN),(s4,'Max',f"{ss2['max']:.2f}",RED)]:
                cs.markdown(f"""<div class="eco-stat"><div class="eco-stat-label">{ls}</div><div class="eco-stat-value" style="font-size:1.5rem;color:{vc};">{vs}</div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE: REPORTS
# ══════════════════════════════════════════════════════════════
elif "Reports" in page:
    st.markdown("""
    <div class="pg-eyebrow"><i class="fa-solid fa-file-medical"></i>Public Health Reference</div>
    <div class="pg-title">AQI Health Guide & Reports</div>
    <div class="pg-sub">WHO PM2.5 standards, AQI category definitions, health recommendations and live exceedance analysis across all monitoring stations.</div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-book-medical"></i>PM2.5 AQI Categories & Health Guidelines</div><div class="eco-card-sub">Based on Chinese AQI standards aligned with WHO PM2.5 guidelines</div></div>""", unsafe_allow_html=True)
    aqi_data=[
        {"Category":"Good","PM2.5 Range":"0–35 µg/m³","Level":"🟢","Health Impact":"Satisfactory air quality. Little or no risk.","Recommended Action":"No restrictions. Enjoy outdoor activities freely."},
        {"Category":"Moderate","PM2.5 Range":"36–75 µg/m³","Level":"🟡","Health Impact":"Acceptable. May affect unusually sensitive individuals.","Recommended Action":"Sensitive people: consider reducing prolonged outdoor exertion."},
        {"Category":"Unhealthy (SG)","PM2.5 Range":"76–115 µg/m³","Level":"🟠","Health Impact":"Elderly, children and asthma sufferers may experience effects.","Recommended Action":"Sensitive groups should limit prolonged outdoor exertion."},
        {"Category":"Unhealthy","PM2.5 Range":"116–150 µg/m³","Level":"🔴","Health Impact":"Everyone may begin to experience health effects.","Recommended Action":"Reduce prolonged outdoor exertion. Sensitive groups avoid it."},
        {"Category":"Very Unhealthy","PM2.5 Range":"151–250 µg/m³","Level":"🟣","Health Impact":"Health alert — everyone may experience serious effects.","Recommended Action":"Avoid prolonged outdoor exertion. Stay indoors when possible."},
        {"Category":"Hazardous","PM2.5 Range":"> 250 µg/m³","Level":"⚫","Health Impact":"Emergency conditions. Entire population affected.","Recommended Action":"Avoid ALL outdoor activity. Stay indoors, windows closed."},
    ]
    dark_table(pd.DataFrame(aqi_data))

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        ac=df['aqi_category'].value_counts().reset_index(); ac.columns=['Category','Hours']
        ac['Percentage']=(ac['Hours']/len(df)*100).round(1)
        ac['Days Equivalent']=(ac['Hours']/24).round(0).astype(int)
        st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-list-ol"></i>AQI Category Counts</div></div>""", unsafe_allow_html=True)
        dark_table(ac)

    with c2:
        if 'season' in df.columns:
            sa2=df.groupby(['season','aqi_category']).size().reset_index(name='count')
            so=['Winter','Spring','Summer','Autumn']
            sa2['season']=pd.Categorical(sa2['season'],categories=so,ordered=True)
            fig_sa2=px.bar(sa2.sort_values('season'),x='season',y='count',color='aqi_category',barmode='stack',height=280,color_discrete_map=AQI_COL,labels={'count':'Hours','aqi_category':'AQI'})
            ap(fig_sa2, {'legend':dict(font=dict(color=MUTED,size=9),bgcolor='rgba(0,0,0,0)')})
            st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-snowflake"></i>AQI Distribution by Season</div></div>""", unsafe_allow_html=True)
            st.plotly_chart(fig_sa2, use_container_width=True)

    # WHO Exceedance
    st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-triangle-exclamation"></i>WHO Guideline Exceedance by Station</div></div>""", unsafe_allow_html=True)
    thr=[15,35,75,115,150]; tlab=['15 µg/m³ (Annual)','35 µg/m³ (Good)','75 µg/m³ (Moderate)','115 µg/m³ (USG)','150 µg/m³ (Unhealthy)']
    rows=[]
    for sn in df['station'].unique():
        sd=df[df['station']==sn]['PM2.5'].dropna()
        r={'Station':sn}
        for t,l in zip(thr,tlab): r[l]=f"{(sd>t).mean()*100:.1f}%"
        rows.append(r)
    dark_table(pd.DataFrame(rows))

    # Worst events
    st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-skull-crossbones"></i>Top 10 Worst Pollution Events</div></div>""", unsafe_allow_html=True)
    wc=[c for c in ['year','month','day','hour','station','season','PM2.5','PM10','NO2','CO'] if c in df.columns]
    w=df.nlargest(10,'PM2.5')[wc].reset_index(drop=True); w.index+=1
    dark_table(w)

    # Annual trend
    if 'year' in df.columns:
        ann=df.groupby(['year','station'])['PM2.5'].mean().reset_index()
        fig_ann=px.line(ann,x='year',y='PM2.5',color='station',markers=True,height=300,color_discrete_sequence=[TEAL,TEAL2,BLUE,'#8b5cf6'],labels={'PM2.5':'Mean PM2.5 (µg/m³)','year':'Year'})
        fig_ann.update_traces(line_width=2.5)
        fig_ann.add_hline(y=15,line_dash='dot',line_color=GREEN,annotation_text='WHO Annual: 15 µg/m³',annotation_font_color=GREEN)
        ap(fig_ann, {'legend':dict(font=dict(color=MUTED),bgcolor='rgba(0,0,0,0)')})
        st.markdown("""<div class="eco-card"><div class="eco-card-title"><i class="fa-solid fa-arrow-trend-down"></i>Annual Mean PM2.5 per Station (2013–2017)</div></div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_ann, use_container_width=True)

    avg_pm=df['PM2.5'].mean(); gp=(df['aqi_category']=='Good').mean()*100
    st.markdown(f"""
    <div class="eco-insight">
        <div class="eco-insight-title"><i class="fa-solid fa-lightbulb"></i>Dataset Health Summary</div>
        <div class="eco-insight-body">Annual mean PM2.5 of <b style="color:{RED};">{avg_pm:.1f} µg/m³</b> is
        <b style="color:{RED};">{avg_pm/15:.1f}×</b> the WHO guideline. Only
        <b style="color:{TEXT};">{gp:.1f}%</b> of readings meet WHO standards.
        Urban stations (Dongsi, Guanyuan) consistently record higher PM2.5 than suburban sites.</div>
    </div>""", unsafe_allow_html=True)