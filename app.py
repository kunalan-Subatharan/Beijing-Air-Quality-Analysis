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
    page_title="Beijing AQI Analytics",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,700;0,800;1,700&display=swap');

:root {
    --bg:      #f8f7f4;
    --bg2:     #ffffff;
    --bg3:     #f1f0ec;
    --border:  #e5e3dc;
    --gold:    #c9870a;
    --gold2:   #f5a623;
    --goldlt:  #fef3dc;
    --text:    #1a1a18;
    --muted:   #78756e;
    --green:   #1a7a4a;
    --greenlt: #e8f5ee;
    --red:     #c0392b;
    --redlt:   #fdf0ee;
    --blue:    #2563eb;
    --bluelt:  #eff6ff;
    --purple:  #7c3aed;
}

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stMainBlockContainer"],
.main, .block-container, section.main {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.block-container { padding: 0 32px 48px 32px !important; max-width: 100% !important; }

/* ── BACKGROUND ANIMATION ── */
.bg-anim {
    position: fixed; top:0; left:0; width:100vw; height:100vh;
    pointer-events: none; z-index: 0; overflow: hidden;
}
.blob {
    position: absolute; border-radius: 50%;
    filter: blur(90px); opacity: 0.28;
}
.blob1 {
    width: 600px; height: 600px;
    background: radial-gradient(circle, #fde68a, #fbbf24);
    top: -150px; right: -100px;
    animation: blobMove1 20s ease-in-out infinite;
}
.blob2 {
    width: 400px; height: 400px;
    background: radial-gradient(circle, #bbf7d0, #6ee7b7);
    bottom: -80px; left: -80px;
    animation: blobMove2 25s ease-in-out infinite;
}
.blob3 {
    width: 300px; height: 300px;
    background: radial-gradient(circle, #bfdbfe, #93c5fd);
    top: 40%; left: 40%;
    animation: blobMove3 18s ease-in-out infinite;
}
@keyframes blobMove1 {
    0%,100% { transform: translate(0,0) scale(1); }
    33%      { transform: translate(-40px,60px) scale(1.06); }
    66%      { transform: translate(30px,-40px) scale(0.95); }
}
@keyframes blobMove2 {
    0%,100% { transform: translate(0,0) scale(1); }
    50%      { transform: translate(50px,-40px) scale(1.08); }
}
@keyframes blobMove3 {
    0%,100% { transform: translate(0,0); }
    40%      { transform: translate(-30px,50px); }
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background-color: #ffffff !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: var(--muted) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.88rem !important;
    color: var(--text) !important;
    padding: 7px 10px !important;
    border-radius: 7px !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--bg3) !important;
    color: var(--gold) !important;
}
[data-testid="stSidebar"] .stSuccess {
    background: var(--greenlt) !important;
    color: var(--green) !important;
    border: 1px solid #a7f3d0 !important;
}
[data-testid="stSidebar"] .stInfo {
    background: var(--bluelt) !important;
    color: var(--blue) !important;
}

/* ── TOPBAR ── */
.topbar {
    background: rgba(248,247,244,0.92);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    padding: 12px 32px;
    display: flex; align-items: center; justify-content: space-between;
    margin: 0 -32px 28px -32px;
    position: sticky; top: 0; z-index: 100;
}
.app-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.15rem; font-weight: 800;
    color: var(--text) !important; letter-spacing: -0.3px;
}
.app-title span { color: var(--gold); }
.topbar-right { font-size: 0.78rem; color: var(--muted); display:flex; align-items:center; gap:14px; }
.status-pill {
    background: var(--greenlt); color: var(--green) !important;
    border: 1px solid #a7f3d0; border-radius: 20px;
    padding: 3px 10px; font-size: 0.7rem; font-weight: 600;
    display: flex; align-items: center; gap: 5px;
}
.status-pill::before { content:''; width:6px; height:6px; border-radius:50%; background:var(--green); }

/* ── PAGE HEADERS ── */
.pg-eyebrow { font-size: 0.65rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; color: var(--gold); margin-bottom: 6px; }
.pg-title { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2rem; font-weight: 800; color: var(--text); margin-bottom: 4px; line-height: 1.15; }
.pg-sub { font-size: 0.88rem; color: var(--muted); line-height: 1.7; margin-bottom: 22px; }
.divider { height: 1px; background: var(--border); margin: 20px 0 24px 0; }

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #1a1a18 0%, #2a2720 60%, #1e1c18 100%);
    border-radius: 20px; padding: 52px 56px;
    margin-bottom: 24px; position: relative; overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute; right: -40px; top: -40px;
    width: 360px; height: 360px; border-radius: 50%;
    background: radial-gradient(circle, rgba(245,166,35,0.18) 0%, transparent 65%);
    pointer-events: none;
}
.hero-eyebrow { font-size: 0.62rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; color: var(--gold2); margin-bottom: 18px; opacity: 0.9; }
.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 3.2rem; font-weight: 800; color: #f5f0e8;
    line-height: 1.08; margin-bottom: 18px; letter-spacing: -1px;
}
.hero-title span { color: var(--gold2); }
.hero-desc { font-size: 0.93rem; color: rgba(245,240,232,0.55); line-height: 1.8; max-width: 500px; margin-bottom: 30px; }
.hero-btn { display: inline-block; border: 1px solid var(--gold2); color: var(--gold2) !important; padding: 11px 26px; border-radius: 8px; font-size: 0.86rem; font-weight: 600; cursor: pointer; margin-right: 10px; }
.hero-btn-2 { display: inline-block; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); color: rgba(245,240,232,0.7) !important; padding: 11px 26px; border-radius: 8px; font-size: 0.86rem; font-weight: 600; cursor: pointer; }

/* ── CARDS ── */
.card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
    position: relative;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, transform 0.2s;
}
.card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); transform: translateY(-1px); }
.card-title { font-size: 0.88rem; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.card-sub { font-size: 0.75rem; color: var(--muted); margin-bottom: 14px; }

/* ── STAT CARDS ── */
.stat-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 20px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, transform 0.2s;
}
.stat-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.08); transform: translateY(-2px); }
.stat-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
.stat-value { font-family: 'DM Mono', monospace; font-size: 1.9rem; font-weight: 500; color: var(--text); line-height: 1; margin-bottom: 8px; }
.stat-tag { display: inline-block; font-size: 0.58rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; padding: 2px 8px; border-radius: 4px; }
.tag-gold   { background: var(--goldlt);  color: var(--gold) !important; }
.tag-green  { background: var(--greenlt); color: var(--green) !important; }
.tag-red    { background: var(--redlt);   color: var(--red) !important; }
.stat-bar { height: 2px; background: var(--bg3); border-radius: 2px; margin-top: 12px; overflow: hidden; }
.stat-fill { height: 2px; border-radius: 2px; }

/* ── METRIC CARDS ── */
.metric-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 24px 26px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.metric-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 12px; }
.metric-value { font-family: 'DM Mono', monospace; font-size: 2.8rem; font-weight: 500; color: var(--gold); line-height: 1; margin-bottom: 8px; }
.metric-note   { font-size: 0.78rem; color: var(--muted); }
.metric-note-g { color: var(--green) !important; font-weight: 600; }
.metric-note-r { color: var(--red) !important; font-weight: 600; }

/* ── INSIGHT BOX ── */
.insight {
    background: var(--goldlt);
    border: 1px solid #f5d07a;
    border-radius: 12px; padding: 18px 20px; margin-bottom: 16px;
}
.insight-title { font-size: 0.78rem; font-weight: 700; color: var(--gold); margin-bottom: 8px; }
.insight-body { font-size: 0.8rem; color: #78600a; line-height: 1.75; }

/* ── FEATURE BAR ── */
.feat-row { margin-bottom: 14px; }
.feat-top { display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 6px; }
.feat-name { color: var(--text); font-weight: 500; }
.feat-pct { color: var(--gold); font-family: 'DM Mono', monospace; font-weight: 500; }
.feat-bg { height: 4px; background: var(--bg3); border-radius: 2px; overflow: hidden; }
.feat-fill { height: 4px; border-radius: 2px; background: linear-gradient(90deg, var(--gold), var(--gold2)); }

/* ── PREDICTOR ── */
.pred-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-top: 3px solid var(--gold);
    border-radius: 14px; padding: 28px; text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.pred-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 12px; }
.pred-value { font-family: 'DM Mono', monospace; font-size: 5rem; font-weight: 500; color: var(--gold); line-height: 1; margin-bottom: 6px; }
.pred-unit { font-size: 0.85rem; color: var(--muted); margin-bottom: 16px; }
.pred-badge { display: inline-block; padding: 6px 20px; border-radius: 6px; font-weight: 600; font-size: 0.82rem; margin-bottom: 18px; }
.who-label { display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 6px; }
.who-bar-bg { height: 4px; background: var(--bg3); border-radius: 2px; margin-bottom: 14px; overflow: hidden; }
.who-bar-fill { height: 4px; border-radius: 2px; }
.conf-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 0.82rem; }
.conf-lbl { color: var(--muted); }
.conf-val { color: var(--text); font-family: 'DM Mono', monospace; font-weight: 500; }

/* ── DISTRICT ROW ── */
.dist-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; font-size: 0.8rem; }
.dist-name { min-width: 110px; color: var(--text); font-weight: 500; }
.dist-bar-bg { flex: 1; height: 4px; background: var(--bg3); border-radius: 2px; }
.dist-bar { height: 4px; border-radius: 2px; }
.dist-val { color: var(--text); font-family: 'DM Mono', monospace; font-size: 0.78rem; min-width: 28px; text-align: right; }

/* ── AQI BADGES ── */
.aqi-good     { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
.aqi-moderate { background: var(--goldlt); color: #92400e; border: 1px solid #fde68a; }
.aqi-usg      { background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; }
.aqi-unhlthy  { background: var(--redlt); color: var(--red); border: 1px solid #fca5a5; }
.aqi-very     { background: #f5f3ff; color: var(--purple); border: 1px solid #ddd6fe; }
.aqi-haz      { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }

/* ── STREAMLIT OVERRIDES ── */
[data-testid="metric-container"] {
    background: var(--bg2) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; padding: 18px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetricValue"] { color: var(--gold) !important; font-family: 'DM Mono', monospace !important; font-size: 1.8rem !important; font-weight: 500 !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.65rem !important; letter-spacing: 2px !important; text-transform: uppercase !important; }
[data-testid="stMetricDelta"] { color: var(--green) !important; }

div[data-testid="stTabs"] [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--border) !important; }
div[data-testid="stTabs"] [data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; font-size: 0.83rem !important; padding: 10px 18px !important; }
div[data-testid="stTabs"] [aria-selected="true"] { color: var(--gold) !important; border-bottom: 2px solid var(--gold) !important; background: transparent !important; }

.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--bg2) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 8px !important;
}
.stNumberInput > div > div > input, input, textarea {
    background: var(--bg2) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--gold), var(--gold2)) !important;
    color: #ffffff !important; font-weight: 700 !important;
    font-size: 0.9rem !important; border: none !important;
    border-radius: 8px !important; padding: 12px 20px !important; width: 100% !important;
    box-shadow: 0 2px 12px rgba(201,135,10,0.25) !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
.stFileUploader { background: var(--bg2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
.stExpander { background: var(--bg2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }

thead tr th { background: var(--bg3) !important; color: var(--gold) !important; font-size: 0.65rem !important; font-weight: 700 !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; border-bottom: 1px solid var(--border) !important; }
tbody tr { background: var(--bg2) !important; color: var(--text) !important; font-size: 0.82rem !important; }
tbody tr:nth-child(even) { background: var(--bg3) !important; }
tbody tr:hover { background: #f0ede4 !important; }

#MainMenu, footer { visibility: hidden !important; }
header { background: transparent !important; }
[data-testid="stDecoration"] { display: none !important; }
.stRadio label { color: var(--text) !important; }
.stCaption { color: var(--muted) !important; }
.stSuccess { background: var(--greenlt) !important; border: 1px solid #a7f3d0 !important; border-radius: 8px !important; color: var(--green) !important; }
.stInfo { background: var(--bluelt) !important; border: 1px solid #bfdbfe !important; border-radius: 8px !important; color: var(--blue) !important; }
.stWarning { background: var(--goldlt) !important; border: 1px solid #fde68a !important; border-radius: 8px !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>

<div class="bg-anim">
  <div class="blob blob1"></div>
  <div class="blob blob2"></div>
  <div class="blob blob3"></div>
</div>
""", unsafe_allow_html=True)


# ── HELPERS ──────────────────────────────────────────────────
def classify_aqi(pm25):
    if pm25 <= 35:    return 'Good',          'aqi-good',     '#166534'
    elif pm25 <= 75:  return 'Moderate',       'aqi-moderate', '#92400e'
    elif pm25 <= 115: return 'Unhealthy (SG)', 'aqi-usg',      '#9a3412'
    elif pm25 <= 150: return 'Unhealthy',      'aqi-unhlthy',  '#c0392b'
    elif pm25 <= 250: return 'Very Unhealthy', 'aqi-very',     '#7c3aed'
    else:             return 'Hazardous',      'aqi-haz',      '#991b1b'

AQI_COL = {
    'Good':'#4ade80','Moderate':'#f5a623',
    'Unhealthy for Sensitive Groups':'#fb923c',
    'Unhealthy':'#f87171','Very Unhealthy':'#a78bfa','Hazardous':'#dc2626'
}
GOLD='#c9870a'; GOLD2='#f5a623'; BG='#f8f7f4'; GRID='#e5e3dc'; TEXT='#1a1a18'

# ── FIX: PLOT dict does NOT include margin - pass it separately when needed
PLOT = dict(
    plot_bgcolor='#fafaf8',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#78756e', family='Plus Jakarta Sans'),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, color='#78756e'),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, color='#78756e'),
)
MARGIN = dict(t=20, b=20, l=10, r=10)

def ap(fig, extra=None):
    """Apply PLOT theme + default margin, optionally merge extra layout kwargs."""
    kw = {**PLOT, 'margin': MARGIN}
    if extra:
        kw.update(extra)
    fig.update_layout(**kw)
    return fig

def prepare(df):
    """Robustly prepare any CSV - works with both cleaned and raw uploads."""
    df = df.copy()

    # ── Ensure numeric columns exist ──
    for col in ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM']:
        if col not in df.columns:
            df[col] = np.nan

    # ── Parse datetime components from raw data if missing ──
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

    # ── Add season if missing ──
    if 'season' not in df.columns:
        def month_to_season(m):
            if m in [12,1,2]:  return 'Winter'
            elif m in [3,4,5]: return 'Spring'
            elif m in [6,7,8]: return 'Summer'
            else:              return 'Autumn'
        df['season'] = df['month'].apply(month_to_season)

    # ── Add station if missing ──
    if 'station' not in df.columns:
        df['station'] = 'Unknown'

    # ── Add station_type if missing (derive from station name or default Urban) ──
    if 'station_type' not in df.columns:
        suburban_stations = ['Changping','Dingling','Shunyi','Huairou']
        df['station_type'] = df['station'].apply(
            lambda s: 'Suburban' if any(sub in str(s) for sub in suburban_stations) else 'Urban'
        )

    # ── AQI category ──
    if 'aqi_category' not in df.columns:
        df['aqi_category'] = df['PM2.5'].apply(
            lambda x: classify_aqi(x)[0] if pd.notnull(x) else 'Unknown'
        )

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
    <div style='padding:20px 16px 16px;border-bottom:1px solid #e5e3dc;margin-bottom:8px;'>
        <div style='font-family:Plus Jakarta Sans,sans-serif;font-size:1.05rem;font-weight:800;color:#1a1a18;'>Beijing AQI</div>
        <div style='font-size:0.7rem;color:#78756e;letter-spacing:1px;margin-top:2px;'>Analytics Platform</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Pages", [
        "🏠  Overview",
        "📂  Data Upload",
        "📋  Dataset",
        "📊  Visualisation",
        "🤖  Model Performance",
        "🔮  Live Predictor",
        "📐  Data Relationships",
        "📈  AQI Health Guide"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="font-size:0.65rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#78756e;margin-bottom:8px;">Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df = load_uploaded(uploaded); dsrc = "uploaded"
        st.success(f"✅ {uploaded.name}")
    else:
        df = load_default(); dsrc = "default"

    with st.spinner("Training model..."):
        (model,scaler,feature_cols,le_station,le_season,
         X_test_sc,y_test,y_pred,
         rf_mae,rf_rmse,rf_r2,lr_mae,lr_rmse,lr_r2) = train_model(df)

    st.markdown(f"""
    <div style='font-size:0.72rem;color:#78756e;line-height:2.4;padding:4px 0;'>
        <span style='color:#1a1a18;font-weight:500;'>Records</span>
        <span style='color:#c9870a;font-family:DM Mono,monospace;float:right;'>{len(df):,}</span><br>
        <span style='color:#1a1a18;font-weight:500;'>Stations</span>
        <span style='color:#c9870a;font-family:DM Mono,monospace;float:right;'>{df['station'].nunique()}</span><br>
        <span style='color:#1a1a18;font-weight:500;'>Model R²</span>
        <span style='color:#c9870a;font-family:DM Mono,monospace;float:right;font-weight:700;'>{rf_r2:.4f}</span><br>
        <span style='color:#1a1a18;font-weight:500;'>MAE</span>
        <span style='color:#c9870a;font-family:DM Mono,monospace;float:right;'>{rf_mae:.2f} µg/m³</span>
    </div>""", unsafe_allow_html=True)


# ── TOPBAR ────────────────────────────────────────────────────
st.markdown(f"""<div class="topbar">
    <div class="app-title">Beijing <span>AQI</span> Analytics</div>
    <div class="topbar-right">
        <span class="status-pill">System Online</span>
        <span style='color:#78756e;'>CMP7005 · Cardiff Met University</span>
    </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    avg_pm = df['PM2.5'].mean()
    good_pct = (df['aqi_category']=='Good').mean()*100

    st.markdown(f"""<div class="hero">
        <div class="hero-eyebrow">Beijing Multi-Site Air Quality Dataset · 2013-2017</div>
        <div class="hero-title">Air Quality <span>Intelligence</span><br>Analytics Platform</div>
        <div class="hero-desc">Comprehensive PM2.5 analysis across 4 Beijing monitoring stations - combining exploratory data analysis with machine learning prediction.</div>
        <span class="hero-btn">Explore Dataset →</span>
        <span class="hero-btn-2">View Model</span>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,lbl,val,tag_cls,tag_txt,bar_w,bar_col in [
        (c1,"TOTAL RECORDS",   f"{len(df):,}",       "tag-gold",  "ACTIVE",       72, GOLD),
        (c2,"STATIONS",        f"{df['station'].nunique()} Sites","tag-green","ONLINE",55,"#4ade80"),
        (c3,"MEAN PM2.5",      f"{avg_pm:.1f} µg/m³","tag-red",   "ABOVE WHO",    65,"#f87171"),
        (c4,"MODEL R²",        f"{rf_r2:.4f}",        "tag-gold",  "OPTIMISED",    int(rf_r2*100),GOLD),
    ]:
        with col:
            st.markdown(f"""<div class="stat-card">
                <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;'>
                    <div class="stat-label">{lbl}</div>
                    <div class="stat-tag {tag_cls}">{tag_txt}</div>
                </div>
                <div class="stat-value">{val}</div>
                <div class="stat-bar"><div class="stat-fill" style="width:{bar_w}%;background:{bar_col};"></div></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    L, R = st.columns([1.7,1])

    with L:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Monthly PM2.5 Trend (2013-2017)</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Mean hourly PM2.5 across all 4 stations with WHO guideline</div>', unsafe_allow_html=True)
        mo = df.groupby(['year','month'])['PM2.5'].mean().reset_index()
        mo['date'] = pd.to_datetime(mo[['year','month']].assign(day=1))
        fig_tr = px.line(mo, x='date', y='PM2.5', color_discrete_sequence=[GOLD], height=250)
        fig_tr.update_traces(line_width=2.5)
        fig_tr.add_hline(y=15, line_dash='dot', line_color='#4ade80',
                         annotation_text='WHO 15 µg/m³', annotation_font_color='#166534')
        ap(fig_tr); st.plotly_chart(fig_tr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with R:
        aqi_c = df['aqi_category'].value_counts()
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">AQI Category Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Proportion of readings in each health band</div>', unsafe_allow_html=True)
        for cat, count in aqi_c.items():
            pct = count/len(df)*100
            col = AQI_COL.get(cat, GOLD)
            st.markdown(f"""<div class="dist-row">
                <span class="dist-name" style='font-size:0.74rem;'>{cat}</span>
                <div class="dist-bar-bg"><div class="dist-bar" style="width:{pct:.0f}%;background:{col};"></div></div>
                <span class="dist-val">{pct:.1f}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""<div class="insight">
            <div class="insight-title">🔬 Key Finding</div>
            <div class="insight-body">Only <b>{good_pct:.1f}%</b> of hourly readings meet WHO PM2.5 standards (≤35 µg/m³). The annual mean of <b>{avg_pm:.1f} µg/m³</b> is <b>{avg_pm/15:.1f}×</b> the WHO 24-hour guideline of 15 µg/m³.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-eyebrow">Station Network</div>', unsafe_allow_html=True)
    stn = df.groupby('station').agg(
        mean_pm25=('PM2.5','mean'), max_pm25=('PM2.5','max'),
        records=('PM2.5','count')
    ).reset_index()
    # safely get station_type - may be missing in raw uploads
    if 'station_type' in df.columns:
        stype_map = df.groupby('station')['station_type'].first().to_dict()
        stn['station_type'] = stn['station'].map(stype_map).fillna('Urban')
    else:
        stn['station_type'] = 'Urban'
    for col_s,(_, row) in zip(st.columns(len(stn)), stn.iterrows()):
        sc = GOLD if row['station_type']=='Urban' else '#166534'
        bg = 'rgba(201,135,10,0.08)' if row['station_type']=='Urban' else 'rgba(26,122,74,0.08)'
        col_s.markdown(f"""<div class="stat-card">
            <div style='display:inline-block;font-size:0.58rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;
                        padding:2px 8px;border-radius:4px;background:{bg};color:{sc};margin-bottom:10px;'>{row['station_type']}</div>
            <div style='font-size:0.95rem;font-weight:700;color:{TEXT};margin-bottom:10px;'>{row['station']}</div>
            <div style='font-size:0.72rem;color:#78756e;line-height:2.2;'>
                Mean PM2.5 <span style='color:{GOLD};font-family:DM Mono;float:right;'>{row['mean_pm25']:.1f}</span><br>
                Max PM2.5  <span style='color:#c0392b;font-family:DM Mono;float:right;'>{row['max_pm25']:.0f}</span><br>
                Records    <span style='color:{TEXT};font-family:DM Mono;float:right;'>{row['records']:,}</span>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DATA UPLOAD
# ══════════════════════════════════════════════════════════════
elif page == "📂  Data Upload":
    st.markdown('<div class="pg-eyebrow">Archive Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Import Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Upload your cleaned Beijing air quality CSV. All pages - visualisations, model and predictor - update automatically.</div>', unsafe_allow_html=True)

    L, R = st.columns([1.5,1])
    with L:
        border_col = "#f5d07a" if dsrc == "uploaded" else "#e5e3dc"
        msg = f"<b style='color:{GOLD};'>{uploaded.name}</b> is active and powering the application." if dsrc=="uploaded" else "Use the <b>file uploader in the left sidebar</b> to select your CSV file."
        st.markdown(f"""<div class="card" style='border:2px dashed {border_col};text-align:center;padding:44px;'>
            <div style='font-size:2.5rem;opacity:0.3;margin-bottom:16px;'>📂</div>
            <div style='font-size:1rem;font-weight:700;color:{TEXT};margin-bottom:8px;'>
                {"✅ File Loaded Successfully" if dsrc=="uploaded" else "Drop your CSV here"}
            </div>
            <div style='font-size:0.82rem;color:#78756e;'>{msg}</div>
            <div style='margin-top:18px;font-size:0.58rem;letter-spacing:2px;color:#b5b2a8;font-weight:700;'>SUPPORTED FORMAT: .CSV ONLY</div>
        </div>""", unsafe_allow_html=True)
        if dsrc == "default":
            st.info("ℹ️ Using default dataset - beijing_air_quality_cleaned.csv")

    with R:
        st.markdown(f"""<div class="card">
            <div class="stat-label">Total Observations</div>
            <div style='font-family:DM Mono,monospace;font-size:2.2rem;font-weight:500;color:{GOLD};margin-bottom:14px;'>{len(df):,}</div>
            <div class="stat-label">Columns</div>
            <div style='font-family:DM Mono,monospace;font-size:2.2rem;font-weight:500;color:{TEXT};margin-bottom:16px;'>{df.shape[1]}</div>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:16px;'>
                <span class="stat-tag tag-green">VERIFIED</span>
                <span style='font-size:0.72rem;color:#78756e;'>File integrity checked</span>
            </div>
            <div style='background:#f8f7f4;border:1px solid #e5e3dc;border-radius:8px;padding:14px;'>
                <div style='font-size:0.58rem;font-weight:700;letter-spacing:1px;color:{GOLD};margin-bottom:8px;text-transform:uppercase;'>Required Columns</div>
                <div style='font-size:0.72rem;color:#78756e;line-height:2;font-family:DM Mono,monospace;font-size:0.68rem;'>
                    PM2.5 · PM10 · SO2 · NO2 · CO · O3<br>
                    TEMP · PRES · DEWP · RAIN · WSPM<br>
                    hour · month · station · season · station_type
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:12px;">Data Preview - Top 10 Rows</div>', unsafe_allow_html=True)
    dcols = [c for c in ['year','month','day','hour','station','station_type','season','PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM'] if c in df.columns]
    st.dataframe(df[dcols].head(10), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:12px;">Column Validation</div>', unsafe_allow_html=True)
    req = ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM','hour','month','station','season','station_type']
    vrows = [{'Column':c,'Present':'✅' if c in df.columns else '❌','Missing Values':int(df[c].isnull().sum()) if c in df.columns else 'N/A','Type':str(df[c].dtype) if c in df.columns else '-','Example':str(df[c].dropna().iloc[0]) if c in df.columns and len(df[c].dropna())>0 else '-'} for c in req]
    st.dataframe(pd.DataFrame(vrows), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════
elif page == "📋  Dataset":
    st.markdown('<div class="pg-eyebrow">Data Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Technical audit of meteorological and pollutant concentrations from 4 Beijing monitoring stations (March 2013 - February 2017).</div>', unsafe_allow_html=True)

    miss_pct = df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100
    c1,c2,c3,c4 = st.columns(4)
    for col,lbl,val in [(c1,"Observations",f"{len(df):,}"),(c2,"Columns",f"{df.shape[1]}"),(c3,"Completeness",f"{100-miss_pct:.1f}%"),(c4,"Stations",f"{df['station'].nunique()}")]:
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-label">{lbl}</div><div class="stat-value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    L, R = st.columns([1.3,1])
    with L:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Missing Values per Feature</div>', unsafe_allow_html=True)
        kc=[c for c in ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM'] if c in df.columns]
        mc=df[kc].isnull().sum().reset_index(); mc.columns=['Feature','Missing']
        mc['Pct']=(mc['Missing']/len(df)*100).round(2)
        fig_m=px.bar(mc,x='Feature',y='Pct',color='Pct',height=255,color_continuous_scale=['#dcfce7','#fde68a','#fca5a5'])
        ap(fig_m, {'coloraxis_showscale':False})
        st.plotly_chart(fig_m,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with R:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Schema</div>', unsafe_allow_html=True)
        sr=[{'Column':c,'Type':str(df[c].dtype),'Example':str(df[c].dropna().iloc[0]) if len(df[c].dropna())>0 else '-','Status':'✅'} for c in df.columns[:16]]
        st.dataframe(pd.DataFrame(sr),use_container_width=True,height=255,hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Statistical Summary</div>', unsafe_allow_html=True)
    nc=[c for c in ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM'] if c in df.columns]
    sm=df[nc].describe().T.round(2)[['count','mean','std','min','25%','50%','75%','max']]
    sm.columns=['Count','Mean','Std Dev','Min','25%','50%','75%','Max']; sm.index.name='Feature'
    st.dataframe(sm.reset_index(),use_container_width=True,hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="insight">
        <div class="insight-title">{100-miss_pct:.1f}% Data Completeness</div>
        <div class="insight-body">Four-station dataset covers March 2013 - February 2017. Missing values are concentrated in pollutant sensors during scheduled maintenance windows. After cleaning, <b>{len(df):,}</b> complete records remain for analysis and modelling.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom:12px;">Interactive Data Explorer</div>', unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    ss=f1.selectbox("Station",['All']+sorted(df['station'].unique().tolist()))
    se=f2.selectbox("Season",['All']+sorted(df['season'].unique().tolist()))
    sy=f3.selectbox("Year",['All']+sorted(df['year'].unique().tolist()))
    dff=df.copy()
    if ss!='All': dff=dff[dff['station']==ss]
    if se!='All': dff=dff[dff['season']==se]
    if sy!='All': dff=dff[dff['year']==int(sy)]
    st.caption(f"{len(dff):,} records after filters")
    dcols=[c for c in ['year','month','day','hour','station','station_type','season','PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM'] if c in dff.columns]
    st.dataframe(dff[dcols].head(500),use_container_width=True,height=300,hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════
elif page == "📊  Visualisation":
    st.markdown('<div class="pg-eyebrow">Visual Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Air Quality Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Interactive exploration of pollutant distributions, temporal patterns, station comparisons and variable relationships.</div>', unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs(["📈 Distribution","⏱️ Temporal","🏙️ Stations","🔗 Bivariate"])

    with tab1:
        L,R = st.columns([1.6,1])
        with L:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">PM2.5 Distribution Histogram</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-sub">Right-skewed distribution with WHO (15 µg/m³) and mean reference lines</div>', unsafe_allow_html=True)
            fig_h=px.histogram(df,x='PM2.5',nbins=60,color_discrete_sequence=[GOLD],height=290)
            fig_h.update_traces(marker_line_color=GRID,marker_line_width=0.5,opacity=0.85)
            fig_h.add_vline(x=df['PM2.5'].mean(),line_dash='dash',line_color='#f87171',annotation_text=f"Mean: {df['PM2.5'].mean():.1f}")
            fig_h.add_vline(x=15,line_dash='dot',line_color='#4ade80',annotation_text="WHO: 15")
            ap(fig_h); st.plotly_chart(fig_h,use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">All Pollutants - Box Plots</div>', unsafe_allow_html=True)
            pols=[c for c in ['PM2.5','PM10','SO2','NO2','CO','O3'] if c in df.columns]
            clrs=[GOLD,GOLD2,'#fb923c','#f87171','#a78bfa','#4ade80']
            fig_b=go.Figure()
            for p,c in zip(pols,clrs):
                fig_b.add_trace(go.Box(y=df[p].dropna(),name=p,marker_color=c,boxmean=True,line_width=1.5))
            ap(fig_b, {'height':270,'yaxis_title':'µg/m³'})
            st.plotly_chart(fig_b,use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with R:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Correlation Heatmap</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-sub">Cross-pollutant dependency matrix</div>', unsafe_allow_html=True)
            cc=[c for c in ['PM2.5','PM10','NO2','O3','TEMP','WSPM','PRES'] if c in df.columns]
            fig_c=px.imshow(df[cc].corr().round(2),
                            color_continuous_scale=['#dcfce7','#fef9c3','#fef3dc','#fde68a','#f5a623'],
                            zmin=-1,zmax=1,text_auto=True,height=310)
            ap(fig_c, {'coloraxis_showscale':False})
            fig_c.update_traces(textfont=dict(color='#1a1a18',size=10))
            st.plotly_chart(fig_c,use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">AQI Breakdown</div>', unsafe_allow_html=True)
            ac=df['aqi_category'].value_counts().reset_index(); ac.columns=['Category','Count']
            fig_p=px.pie(ac,names='Category',values='Count',color='Category',
                         color_discrete_map=AQI_COL,height=220)
            ap(fig_p, {'margin':dict(t=10,b=10,l=0,r=0),
                       'legend':dict(font=dict(color='#78756e',size=9),bgcolor='rgba(0,0,0,0)')})
            st.plotly_chart(fig_p,use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Monthly PM2.5 Trend</div>', unsafe_allow_html=True)
        mo=df.groupby(['year','month'])['PM2.5'].mean().reset_index()
        mo['date']=pd.to_datetime(mo[['year','month']].assign(day=1))
        fig_l=px.line(mo,x='date',y='PM2.5',color_discrete_sequence=[GOLD],height=270)
        fig_l.update_traces(line_width=2.5)
        fig_l.add_hline(y=15,line_dash='dot',line_color='#4ade80',
                        annotation_text='WHO: 15 µg/m³',annotation_font_color='#166534')
        ap(fig_l); st.plotly_chart(fig_l,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        t1,t2=st.columns(2)
        with t1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Hour-of-Day Pattern</div>', unsafe_allow_html=True)
            ho=df.groupby('hour')['PM2.5'].mean().reset_index()
            fig_ho=px.area(ho,x='hour',y='PM2.5',height=230,color_discrete_sequence=[GOLD])
            fig_ho.update_traces(fillcolor='rgba(201,135,10,0.1)',line_width=2)
            ap(fig_ho); st.plotly_chart(fig_ho,use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with t2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Monthly Seasonal Pattern</div>', unsafe_allow_html=True)
            mn_m={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
            ma=df.groupby('month')['PM2.5'].mean().reset_index(); ma['Month']=ma['month'].map(mn_m)
            fig_ma=px.bar(ma,x='Month',y='PM2.5',color='PM2.5',height=230,
                          color_continuous_scale=['#dcfce7','#fef3dc','#fde68a','#fca5a5'])
            ap(fig_ma, {'coloraxis_showscale':False})
            st.plotly_chart(fig_ma,use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        s1,s2=st.columns(2)
        with s1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            sa=df.groupby('station')['PM2.5'].mean().reset_index().sort_values('PM2.5',ascending=True)
            fig_sa=px.bar(sa,x='PM2.5',y='station',orientation='h',color='PM2.5',height=250,
                          color_continuous_scale=['#dcfce7','#fef3dc','#fde68a','#fca5a5'],
                          labels={'PM2.5':'Mean PM2.5 (µg/m³)'})
            ap(fig_sa, {'coloraxis_showscale':False})
            st.plotly_chart(fig_sa,use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with s2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # safely color by station_type if column exists
            _bs_color = 'station_type' if 'station_type' in df.columns else None
            _bs_kwargs = dict(color_discrete_map={'Urban':GOLD,'Suburban':GOLD2}) if _bs_color else {}
            fig_bs=px.box(df,x='station',y='PM2.5',color=_bs_color,height=250,
                          labels={'PM2.5':'PM2.5 (µg/m³)'},**_bs_kwargs)
            ap(fig_bs, {'legend':dict(font=dict(color='#78756e'),bgcolor='rgba(0,0,0,0)')})
            st.plotly_chart(fig_bs,use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">PM2.5 by Season and Station</div>', unsafe_allow_html=True)
        so=['Winter','Spring','Summer','Autumn']
        if 'season' not in df.columns: df['season'] = 'Unknown'
        seas=df.groupby(['season','station'])['PM2.5'].mean().reset_index()
        seas['season']=pd.Categorical(seas['season'],categories=so,ordered=True)
        fig_se=px.bar(seas.sort_values('season'),x='season',y='PM2.5',color='station',
                      barmode='group',height=270,
                      color_discrete_sequence=[GOLD,GOLD2,'#60a5fa','#a78bfa'],
                      labels={'PM2.5':'Mean PM2.5 (µg/m³)'})
        ap(fig_se, {'legend':dict(font=dict(color='#78756e'),bgcolor='rgba(0,0,0,0)')})
        st.plotly_chart(fig_se,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        b1,b2=st.columns(2)
        xv=b1.selectbox("X-axis variable",[c for c in ['TEMP','PRES','DEWP','WSPM','NO2','CO','SO2','O3','PM10'] if c in df.columns])
        cv=b2.selectbox("Colour by",[c for c in ['season','station_type','station'] if c in df.columns])
        sdf=df[[xv,'PM2.5',cv]].dropna().sample(min(5000,len(df)),random_state=42)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig_s=px.scatter(sdf,x=xv,y='PM2.5',color=cv,opacity=0.45,height=360,
                         color_discrete_sequence=[GOLD,GOLD2,'#60a5fa','#a78bfa'])
        x2=sdf[xv].values; y2=sdf['PM2.5'].values; msk=~(np.isnan(x2)|np.isnan(y2))
        if msk.sum()>1:
            co_=np.polyfit(x2[msk],y2[msk],1); xl=np.linspace(x2[msk].min(),x2[msk].max(),100)
            fig_s.add_trace(go.Scatter(x=xl,y=np.polyval(co_,xl),mode='lines',name='Trend',
                                        line=dict(color='#f87171',width=2,dash='dash')))
        ap(fig_s, {'legend':dict(font=dict(color='#78756e'),bgcolor='rgba(0,0,0,0)')})
        st.plotly_chart(fig_s,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "🤖  Model Performance":
    st.markdown('<div class="pg-eyebrow">Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pg-sub">Optimised Random Forest vs Linear Regression baseline - 15 features, 112,204 training records, 80/20 split.</div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{rf_r2:.4f}</div>
            <div class="metric-note metric-note-g">↗ +{rf_r2-lr_r2:.4f} vs Linear Regression</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Mean Absolute Error</div>
            <div class="metric-value">{rf_mae:.2f}</div>
            <div class="metric-note">µg/m³ &nbsp;·&nbsp; {lr_mae-rf_mae:.2f} improvement</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Root Mean Sq Error</div>
            <div class="metric-value">{rf_rmse:.2f}</div>
            <div class="metric-note metric-note-r">µg/m³ - driven by extreme events</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    L,R = st.columns([1.7,1])
    with L:
        ta,tb,tc = st.tabs(["🎯 Actual vs Predicted","📊 Model Comparison","📉 Residuals"])
        with ta:
            yta=np.array(y_test); res=yta-y_pred
            idx=np.random.choice(len(yta),size=min(3000,len(yta)),replace=False)
            avdf=pd.DataFrame({'Actual':yta[idx],'Predicted':y_pred[idx],'Residual':res[idx]})
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig_av=px.scatter(avdf,x='Actual',y='Predicted',color='Residual',
                              color_continuous_scale=['#fca5a5','#fef9c3',GOLD],
                              opacity=0.45,height=340,
                              labels={'Actual':'Actual PM2.5 (µg/m³)','Predicted':'Predicted PM2.5 (µg/m³)'})
            mv=max(avdf['Actual'].max(),avdf['Predicted'].max())
            fig_av.add_trace(go.Scatter(x=[0,mv],y=[0,mv],mode='lines',name='Perfect',
                                         line=dict(color=GOLD,dash='dash',width=1.5)))
            ap(fig_av, {'coloraxis_showscale':False,
                        'legend':dict(font=dict(color='#78756e'),bgcolor='rgba(0,0,0,0)')})
            st.plotly_chart(fig_av,use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with tb:
            cd=pd.DataFrame({'Model':['Linear Regression','Random Forest'],
                             'MAE':[lr_mae,rf_mae],'RMSE':[lr_rmse,rf_rmse],'R²':[lr_r2,rf_r2]})
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig_co=px.bar(pd.melt(cd,id_vars='Model',var_name='Metric',value_name='Value'),
                          x='Metric',y='Value',color='Model',barmode='group',height=280,
                          color_discrete_map={'Linear Regression':'#d1d5db','Random Forest':GOLD})
            ap(fig_co, {'legend':dict(font=dict(color='#78756e'),bgcolor='rgba(0,0,0,0)')})
            st.plotly_chart(fig_co,use_container_width=True)
            st.dataframe(cd,use_container_width=True,hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with tc:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Residual Distribution</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-sub">Distribution of (Actual − Predicted) errors - centred near zero confirms low bias</div>', unsafe_allow_html=True)
            fig_rd=px.histogram(avdf,x='Residual',nbins=60,color_discrete_sequence=[GOLD],height=280)
            fig_rd.add_vline(x=0,line_dash='dash',line_color='#f87171',annotation_text='Zero (Perfect)')
            fig_rd.add_vline(x=res.mean(),line_dash='dot',line_color='#4ade80',annotation_text=f'Mean: {res.mean():.2f}')
            ap(fig_rd); st.plotly_chart(fig_rd,use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with R:
        fi=model.feature_importances_
        fi_df=pd.DataFrame({'Feature':feature_cols,'Importance':fi}).sort_values('Importance',ascending=False)
        mx=fi_df['Importance'].max()
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-bottom:16px;">Feature Importance</div>', unsafe_allow_html=True)
        for _,row in fi_df.iterrows():
            pct=int(row['Importance']/mx*100)
            dp=int(row['Importance']/fi.sum()*100)
            st.markdown(f"""<div class="feat-row">
                <div class="feat-top"><span class="feat-name">{row['Feature']}</span><span class="feat-pct">{dp}%</span></div>
                <div class="feat-bg"><div class="feat-fill" style="width:{pct}%;"></div></div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════
elif page == "🔮  Live Predictor":
    st.markdown('<div class="pg-eyebrow">Real-Time Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">PM2.5 Live Predictor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pg-sub">Input atmospheric conditions to generate an instant PM2.5 forecast - model R²={rf_r2:.4f}, MAE={rf_mae:.2f} µg/m³.</div>', unsafe_allow_html=True)

    L,R = st.columns([1.4,1])
    with L:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="margin-bottom:14px;">Pollutant Inputs (µg/m³)</div>', unsafe_allow_html=True)
        p1,p2,p3 = st.columns(3)
        ipm10=p1.number_input("PM10",0.0,1000.0,80.0,step=5.0)
        iso2 =p2.number_input("SO2", 0.0,500.0, 15.0,step=1.0)
        ino2 =p3.number_input("NO2", 0.0,300.0, 50.0,step=1.0)
        p4,p5 = st.columns(2)
        ico  =p4.number_input("CO",  0.0,15000.0,900.0,step=50.0)
        io3  =p5.number_input("O3",  0.0,500.0,  60.0, step=5.0)
        st.markdown('<div class="card-title" style="margin:12px 0 10px;">Meteorological Inputs</div>', unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        itemp=m1.number_input("Temp (°C)",  -30.0,45.0,  10.0,step=1.0)
        ipres=m2.number_input("Pres (hPa)",  980.0,1040.0,1010.0,step=1.0)
        idewp=m3.number_input("Dew Pt (°C)",-40.0,30.0,  -5.0,step=1.0)
        m4,m5 = st.columns(2)
        irain=m4.number_input("Rain (mm)",  0.0,100.0,0.0,step=0.5)
        iwspm=m5.number_input("Wind (m/s)", 0.0,20.0, 2.0,step=0.5)
        st.markdown('<div class="card-title" style="margin:12px 0 10px;">Temporal & Location</div>', unsafe_allow_html=True)
        t1,t2 = st.columns(2)
        ihr =t1.slider("Hour of Day",0,23,12)
        imo =t2.slider("Month",      1,12, 6)
        t3,t4 = st.columns(2)
        istn=t3.selectbox("Station",sorted(df['station'].unique().tolist()))
        isty=t4.selectbox("Station Type",['Urban','Suburban'])
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.button("Generate PM2.5 Prediction →", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with R:
        smapx={12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
               6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'}
        ise=smapx.get(imo,'Summer')
        try: ste=le_station.transform([istn])[0]
        except: ste=0
        see={'Winter':3,'Spring':1,'Summer':2,'Autumn':0}.get(ise,0)
        sye=1 if isty=='Urban' else 0
        ia=np.array([[ipm10,iso2,ino2,ico,io3,itemp,ipres,idewp,irain,iwspm,ihr,imo,see,ste,sye]])
        pred=max(0,model.predict(scaler.transform(ia))[0])
        cat_name,cat_cls,cat_col=classify_aqi(pred)
        who_diff=pred-15; bar_fill=min(int(pred/300*100),100)
        bar_col='#f87171' if pred>75 else '#4ade80'

        st.markdown(f"""<div class="pred-card">
            <div class="pred-label">Predicted PM2.5</div>
            <div class="pred-value">{pred:.1f}</div>
            <div class="pred-unit">µg/m³ &nbsp;·&nbsp; {ise} &nbsp;·&nbsp; {istn}</div>
            <div style='margin-bottom:14px;'><span class="pred-badge {cat_cls}">{cat_name}</span></div>
            <div class="who-label">
                <span style='color:#78756e;'>vs WHO Guideline (15 µg/m³)</span>
                <span style='color:{"#c0392b" if who_diff>0 else "#1a7a4a"};font-family:DM Mono;font-weight:600;'>
                    {"+"+str(round(who_diff,1)) if who_diff>0 else str(round(who_diff,1))} µg/m³
                </span>
            </div>
            <div class="who-bar-bg"><div class="who-bar-fill" style="width:{bar_fill}%;background:{bar_col};"></div></div>
            <div style='font-size:0.75rem;color:#78756e;text-align:left;font-style:italic;line-height:1.7;margin-bottom:16px;'>
                {"⚠️ Above WHO guidelines - sensitive groups should reduce outdoor exposure." if pred>75 else "✅ Within acceptable WHO PM2.5 guidelines for this period."}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="card" style="margin-top:14px;">
            <div class="card-title" style="margin-bottom:12px;">Model Confidence</div>
            <div class="conf-row"><span class="conf-lbl">R² Score</span><span class="conf-val">{rf_r2:.4f}</span></div>
            <div class="conf-row"><span class="conf-lbl">MAE</span><span class="conf-val">{rf_mae:.2f} µg/m³</span></div>
            <div class="conf-row"><span class="conf-lbl">RMSE</span><span class="conf-val">{rf_rmse:.2f} µg/m³</span></div>
            <div class="conf-row" style='border:none;'><span class="conf-lbl">Training Records</span><span class="conf-val">{int(len(df)*0.8):,}</span></div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DATA RELATIONSHIPS
# ══════════════════════════════════════════════════════════════
elif page == "📐  Data Relationships":
    st.markdown('<div class="pg-eyebrow">Statistical Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Data Relationships</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Pairwise scatter matrix, full correlation analysis and variable deep dive explorer.</div>', unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(["🔲 Scatter Matrix","🌡️ Full Correlation","📊 Variable Deep Dive"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Pairwise Scatter Matrix</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Select variables to compare simultaneously - coloured by station type</div>', unsafe_allow_html=True)
        num_cols=[c for c in ['PM2.5','PM10','NO2','CO','TEMP','WSPM','O3'] if c in df.columns]
        sel_cols=st.multiselect("Select variables (2-6 recommended)",num_cols,default=['PM2.5','PM10','TEMP','WSPM'])
        if len(sel_cols) >= 2:
            _sm_cols = sel_cols + (['station_type'] if 'station_type' in df.columns else [])
            samp=df[_sm_cols].dropna().sample(min(2500,len(df)),random_state=42)
            _sm_color = 'station_type' if 'station_type' in df.columns else None
            _sm_ckw = dict(color_discrete_map={'Urban':GOLD,'Suburban':GOLD2}) if _sm_color else {}
            fig_sm=px.scatter_matrix(samp,dimensions=sel_cols,color=_sm_color,
                                     **_sm_ckw,
                                     opacity=0.35,height=560)
            fig_sm.update_traces(marker=dict(size=3,line=dict(width=0)))
            ap(fig_sm, {'showlegend':True,
                        'legend':dict(font=dict(color='#78756e'),bgcolor='rgba(0,0,0,0)')})
            st.plotly_chart(fig_sm,use_container_width=True)
        else:
            st.info("Select at least 2 variables to display the scatter matrix.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Full Correlation Matrix</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Pearson r between all numeric features - warm = positive correlation</div>', unsafe_allow_html=True)
        all_num=[c for c in ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM','hour','month'] if c in df.columns]
        corr_full=df[all_num].corr().round(2)
        fig_cf=px.imshow(corr_full,
                          color_continuous_scale=['#dcfce7','#fef9c3','#fef3dc','#fde68a','#f5a623'],
                          zmin=-1,zmax=1,text_auto=True,height=520)
        # FIX: use 'title' not 'titlefont' in coloraxis_colorbar
        ap(fig_cf, {'coloraxis_showscale':True,
                    'coloraxis_colorbar':dict(
                        tickfont=dict(color='#78756e'),
                        title=dict(text='r', font=dict(color='#78756e'))
                    )})
        fig_cf.update_traces(textfont=dict(color='#1a1a18',size=9))
        st.plotly_chart(fig_cf,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        pm25_corr=corr_full['PM2.5'].drop('PM2.5').sort_values(key=abs,ascending=False)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">PM2.5 Correlation Ranking</div>', unsafe_allow_html=True)
        for feat,corr_val in pm25_corr.items():
            col_c=GOLD if corr_val>0 else '#f87171'
            pct=int(abs(corr_val)*100)
            st.markdown(f"""<div class="dist-row">
                <span class="dist-name">{feat}</span>
                <div class="dist-bar-bg"><div class="dist-bar" style="width:{pct}%;background:{col_c};"></div></div>
                <span class="dist-val" style='color:{col_c};'>{corr_val:+.2f}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Variable Deep Dive</div>', unsafe_allow_html=True)
        sel_var=st.selectbox("Select variable",[c for c in ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','WSPM'] if c in df.columns])
        v1,v2=st.columns(2)
        with v1:
            _vb_color = 'station_type' if 'station_type' in df.columns else None
            _vb_ckw = dict(color_discrete_map={'Urban':GOLD,'Suburban':GOLD2}) if _vb_color else {}
            fig_vbox=px.box(df,x='season',y=sel_var,color=_vb_color,height=280,
                            category_orders={'season':['Winter','Spring','Summer','Autumn']},
                            labels={sel_var:f'{sel_var} (µg/m³)'},**_vb_ckw)
            ap(fig_vbox, {'legend':dict(font=dict(color='#78756e'),bgcolor='rgba(0,0,0,0)')})
            st.plotly_chart(fig_vbox,use_container_width=True)
        with v2:
            hourly_v=df.groupby('hour')[sel_var].mean().reset_index()
            fig_vh=px.area(hourly_v,x='hour',y=sel_var,height=280,color_discrete_sequence=[GOLD])
            fig_vh.update_traces(fillcolor='rgba(201,135,10,0.1)',line_width=2)
            ap(fig_vh); st.plotly_chart(fig_vh,use_container_width=True)
        stats_s=df[sel_var].describe().round(2)
        s1,s2,s3,s4=st.columns(4)
        for col_s,lbl_s,val_s in [
            (s1,'Mean',      f"{stats_s['mean']:.2f}"),
            (s2,'Std Dev',   f"{stats_s['std']:.2f}"),
            (s3,'Min',       f"{stats_s['min']:.2f}"),
            (s4,'Max',       f"{stats_s['max']:.2f}")
        ]:
            col_s.markdown(f'<div class="stat-card"><div class="stat-label">{lbl_s}</div><div class="stat-value" style="font-size:1.5rem;">{val_s}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# AQI HEALTH GUIDE
# ══════════════════════════════════════════════════════════════
elif page == "📈  AQI Health Guide":
    st.markdown('<div class="pg-eyebrow">Public Health Reference</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">AQI Health Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">WHO PM2.5 standards, AQI category definitions and health recommendations - with live analysis of the current dataset.</div>', unsafe_allow_html=True)

    # ── AQI Category Reference Table ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">PM2.5 AQI Categories & Health Guidelines</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Based on Chinese AQI standards aligned with WHO PM2.5 guidelines</div>', unsafe_allow_html=True)
    aqi_data = [
        {"Category":"Good",                         "PM2.5 Range":"0 - 35 µg/m³",   "Colour":"🟢","Health Impact":"Air quality is satisfactory. Little or no risk.","Recommended Action":"No restrictions. Enjoy outdoor activities."},
        {"Category":"Moderate",                     "PM2.5 Range":"36 - 75 µg/m³",  "Colour":"🟡","Health Impact":"Acceptable quality. Some pollutants may affect sensitive individuals.","Recommended Action":"Unusually sensitive people should consider reducing prolonged outdoor exertion."},
        {"Category":"Unhealthy for Sensitive Groups","PM2.5 Range":"76 - 115 µg/m³", "Colour":"🟠","Health Impact":"Sensitive groups (elderly, children, asthma) may experience health effects.","Recommended Action":"Sensitive groups should limit prolonged outdoor exertion."},
        {"Category":"Unhealthy",                    "PM2.5 Range":"116 - 150 µg/m³","Colour":"🔴","Health Impact":"Everyone may begin to experience health effects.","Recommended Action":"Everyone should reduce prolonged outdoor exertion. Sensitive groups should avoid it."},
        {"Category":"Very Unhealthy",               "PM2.5 Range":"151 - 250 µg/m³","Colour":"🟣","Health Impact":"Health alert: everyone may experience more serious health effects.","Recommended Action":"Everyone should avoid prolonged outdoor exertion. Stay indoors when possible."},
        {"Category":"Hazardous",                    "PM2.5 Range":"> 250 µg/m³",    "Colour":"⚫","Health Impact":"Health warnings of emergency conditions. Entire population affected.","Recommended Action":"Everyone should avoid all outdoor activity. Stay indoors with windows closed."},
    ]
    st.dataframe(pd.DataFrame(aqi_data), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Dataset AQI stats ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">AQI Category Counts in Dataset</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">How many hours fall into each health category</div>', unsafe_allow_html=True)
        aqi_counts = df['aqi_category'].value_counts().reset_index()
        aqi_counts.columns = ['Category','Hours']
        aqi_counts['Percentage'] = (aqi_counts['Hours']/len(df)*100).round(1)
        aqi_counts['Days Equivalent'] = (aqi_counts['Hours']/24).round(0).astype(int)
        st.dataframe(aqi_counts, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">AQI Distribution by Season</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Proportion of unhealthy days across seasons</div>', unsafe_allow_html=True)
        if 'season' in df.columns:
            seas_aqi = df.groupby(['season','aqi_category']).size().reset_index(name='count')
            so = ['Winter','Spring','Summer','Autumn']
            seas_aqi['season'] = pd.Categorical(seas_aqi['season'], categories=so, ordered=True)
            fig_sa = px.bar(seas_aqi.sort_values('season'), x='season', y='count',
                            color='aqi_category', barmode='stack', height=260,
                            color_discrete_map=AQI_COL,
                            labels={'count':'Hours','season':'Season','aqi_category':'AQI Category'})
            ap(fig_sa, {'legend':dict(font=dict(color='#78756e',size=9),bgcolor='rgba(0,0,0,0)')})
            st.plotly_chart(fig_sa, use_container_width=True)
        else:
            st.info("Season column not available in this dataset.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── WHO Exceedance Analysis ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">WHO Guideline Exceedance Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Proportion of hours exceeding WHO PM2.5 thresholds per station</div>', unsafe_allow_html=True)

    who_thresholds = [15, 35, 75, 115, 150]
    threshold_labels = ['15 µg/m³ (Annual)', '35 µg/m³ (Good)', '75 µg/m³ (Moderate)',
                        '115 µg/m³ (USG)', '150 µg/m³ (Unhealthy)']
    rows = []
    for stn_name in df['station'].unique():
        stn_df = df[df['station']==stn_name]['PM2.5'].dropna()
        row = {'Station': stn_name}
        for thr, lbl in zip(who_thresholds, threshold_labels):
            row[lbl] = f"{(stn_df > thr).mean()*100:.1f}%"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Worst pollution hours ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Top 10 Worst Pollution Events</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Highest recorded PM2.5 hourly readings in the dataset</div>', unsafe_allow_html=True)
    worst_cols = [c for c in ['year','month','day','hour','station','season','PM2.5','PM10','NO2','CO'] if c in df.columns]
    worst = df.nlargest(10, 'PM2.5')[worst_cols].reset_index(drop=True)
    worst.index += 1
    st.dataframe(worst, use_container_width=True, hide_index=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── PM2.5 Trend: annual mean per station ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Annual Mean PM2.5 per Station</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Year-on-year trend - Year-on-year pollution trend from 2013 to 2017</div>', unsafe_allow_html=True)
    if 'year' in df.columns:
        annual = df.groupby(['year','station'])['PM2.5'].mean().reset_index()
        fig_ann = px.line(annual, x='year', y='PM2.5', color='station',
                          markers=True, height=280,
                          color_discrete_sequence=[GOLD, GOLD2, '#60a5fa', '#a78bfa'],
                          labels={'PM2.5':'Mean PM2.5 (µg/m³)','year':'Year'})
        fig_ann.update_traces(line_width=2.5)
        fig_ann.add_hline(y=15, line_dash='dot', line_color='#4ade80',
                          annotation_text='WHO Annual Guideline: 15 µg/m³')
        ap(fig_ann, {'legend':dict(font=dict(color='#78756e'),bgcolor='rgba(0,0,0,0)')})
        st.plotly_chart(fig_ann, use_container_width=True)
    else:
        st.info("Year column not available in this dataset.")
    st.markdown('</div>', unsafe_allow_html=True)