import json
import sqlite3
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "cia_real_warehouse.db"

st.set_page_config(
    page_title="CIA | Crime Intelligence Analysis",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&family=Oswald:wght@500;600;700&display=swap');

:root {
    color-scheme: dark;
    --bg: #050505;
    --panel: #101010;
    --panel-2: #171717;
    --line: #343434;
    --muted: #999999;
    --text: #eeeeee;
    --cyan: #d9363e;
    --violet: #8f2028;
    --red: #e04444;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 78% -12%, rgba(130, 20, 28, .16), transparent 34%),
        radial-gradient(ellipse at 8% 18%, rgba(105, 14, 22, .13), transparent 28%),
        radial-gradient(ellipse at 92% 78%, rgba(160, 25, 35, .055), transparent 25%),
        linear-gradient(145deg, #050505 0%, #0b0b0b 48%, #080808 100%);
    color: var(--text);
}

/* Subtle atmospheric glow */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(217, 54, 62, .018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(217, 54, 62, .018) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: linear-gradient(to bottom, black, transparent 75%);
}

[data-testid="stHeader"] {
    background: rgba(5, 7, 13, .72);
    border-bottom: 1px solid rgba(130, 144, 168, .08);
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(14, 14, 14, .99), rgba(5, 5, 5, .99));
    border-right: 1px solid rgba(217, 54, 62, .14);
}

[data-testid="stSidebar"] > div:first-child {
    background: transparent;
}

.block-container {
    max-width: 1600px;
    padding-top: 2.1rem;
    padding-bottom: 3.5rem;
    position: relative;
    z-index: 1;
}

h1, h2, h3 {
    font-family: 'Oswald', 'Manrope', sans-serif !important;
    color: #f3f6fc !important;
    letter-spacing: -.045em;
}

.cia-main-title {
    margin: 0 0 10px;
    color: #f3f3f3;
    font-family: 'Oswald', 'Manrope', sans-serif !important;
    font-size: clamp(2.45rem, 4.4vw, 3.55rem);
    font-weight: 600;
    letter-spacing: -.025em;
    line-height: 1.08;
}
.cia-main-title .crime-word {
    color: #e3343e;
    font-family: 'Oswald', 'Manrope', sans-serif !important;
    font-weight: 700;
    letter-spacing: .015em;
    text-transform: none;
    text-shadow: 0 0 18px rgba(227, 52, 62, .12);
}


h1 {
    font-size: clamp(2.2rem, 4vw, 3.15rem) !important;
    line-height: 1.12 !important;
    font-weight: 800 !important;
    text-shadow: 0 0 34px rgba(217, 54, 62, .10);
}

h2 {
    font-size: 1.55rem !important;
    font-weight: 750 !important;
}

h3 {
    font-size: 1.12rem !important;
    font-weight: 700 !important;
}

p, label, .stMarkdown {
    color: #b9c4d6;
}

.cia-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--cyan);
    font-family: 'DM Mono', monospace;
    font-size: .70rem;
    font-weight: 500;
    letter-spacing: .22em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.cia-brand::before {
    content: "";
    display: inline-block;
    width: 23px;
    height: 1px;
    background: linear-gradient(90deg, var(--cyan), var(--violet));
    box-shadow: 0 0 10px rgba(217, 54, 62, .7);
}

.cia-subtitle {
    max-width: 850px;
    color: #8e9bb1;
    font-size: .94rem;
    line-height: 1.8;
    margin-top: -5px;
    margin-bottom: 24px;
}

[data-testid="stMetric"] {
    background:
        linear-gradient(135deg, rgba(25, 25, 25, .96), rgba(10, 10, 10, .98));
    border: 1px solid rgba(217, 54, 62, .20);
    border-radius: 15px;
    padding: 20px 21px;
    box-shadow: 0 12px 34px rgba(0, 0, 0, .22);
    transition: border-color .2s ease, transform .2s ease;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(217, 54, 62, .55);
    transform: translateY(-2px);
}

[data-testid="stMetricLabel"] {
    color: #91a0b7 !important;
    font-size: .76rem !important;
    letter-spacing: .06em;
    text-transform: uppercase;
}

[data-testid="stMetricValue"] {
    color: #f4f7ff !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -.05em;
}

[data-testid="stMetricDelta"] {
    color: var(--cyan) !important;
}

[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"] {
    background: rgba(14, 14, 14, .58);
    border: 1px solid rgba(130, 130, 130, .20);
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 14px 35px rgba(0, 0, 0, .15);
}

[data-testid="stPlotlyChart"] {
    padding: 5px;
}

.cia-panel {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(24, 24, 24, .95), rgba(10, 10, 10, .98));
    border: 1px solid rgba(130, 130, 130, .24);
    border-radius: 16px;
    padding: 22px 24px;
    margin: 10px 0 16px;
    box-shadow: 0 14px 38px rgba(0, 0, 0, .18);
}

.cia-panel::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 1px;
    width: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 107, 107, .60), rgba(217, 54, 62, .55), transparent);
}

.cia-panel-title {
    color: #eef5ff;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: -.02em;
    margin-bottom: 7px;
}

.cia-panel-text {
    color: #94a4bc;
    font-size: .87rem;
    line-height: 1.85;
}

/* Primary actions: calm teal, designed for an analytical tool rather than an alarm console */
div.stButton > button {
    background: #351013;
    color: #c4fff4;
    border: 1px solid #8f2028;
    border-radius: 10px;
    min-height: 43px;
    font-weight: 750;
    box-shadow: 0 4px 14px rgba(0, 0, 0, .16);
    transition: background .18s ease, border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}

div.stButton > button:hover {
    background: #4a1519;
    border-color: #d9363e;
    color: #ffffff;
    box-shadow: 0 0 18px rgba(217, 54, 62, .14);
    transform: translateY(-1px);
}

div.stButton > button:focus-visible,
div.stDownloadButton > button:focus-visible {
    outline: 2px solid #8ef3dc;
    outline-offset: 3px;
}

/* Export actions: slightly quieter than primary actions, but still easy to find */
div.stDownloadButton > button {
    background: #2b1012;
    color: #b8fff2;
    border: 1px solid #237c79;
    border-radius: 10px;
    min-height: 42px;
    font-weight: 700;
    box-shadow: 0 3px 12px rgba(0, 0, 0, .13);
    transition: background .18s ease, border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}

div.stDownloadButton > button:hover {
    background: #401417;
    border-color: #d9363e;
    color: #ffffff;
    box-shadow: 0 0 16px rgba(217, 54, 62, .12);
    transform: translateY(-1px);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #101722;
    border-color: #2b3a53;
    border-radius: 9px;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #8493aa;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 7px 9px;
    border-radius: 8px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(217, 54, 62, .06);
}

[data-testid="stSidebar"] h1 {
    color: var(--cyan) !important;
}

hr {
    border-color: rgba(120, 141, 172, .18);
    margin: 1.5rem 0;
}

[data-testid="stCaptionContainer"] {
    color: #77879f;
}

.stAlert {
    border-radius: 12px;
}

[data-testid="stExpander"] {
    background: rgba(12, 18, 31, .70);
    border: 1px solid rgba(130, 130, 130, .20);
    border-radius: 12px;
}


/* Topic-specific archive / investigation briefing */
.cia-briefing {
    position: relative; overflow: hidden;
    background: radial-gradient(circle at 82% 45%, rgba(217,54,62,.10), transparent 25%),
      radial-gradient(circle at 8% 0%, rgba(170, 45, 52,.055), transparent 30%),
      linear-gradient(125deg, rgba(24,24,24,.98), rgba(10,10,10,.98));
    border: 1px solid rgba(130,130,130,.24); border-radius: 18px;
    margin: 14px 0 24px; padding: 20px 24px 15px;
    box-shadow: 0 18px 45px rgba(0,0,0,.24);
}
.cia-briefing::before {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background-image: linear-gradient(rgba(217,54,62,.035) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(217,54,62,.035) 1px, transparent 1px);
    background-size: 26px 26px; mask-image: linear-gradient(90deg, black, transparent 88%);
}
.cia-briefing-top, .cia-briefing-footer {
    display: flex; justify-content: space-between; align-items: center;
    gap: 12px; position: relative; z-index: 1;
}
.cia-briefing-kicker, .cia-briefing-status, .cia-briefing-label, .cia-briefing-footer {
    font-family: 'DM Mono', monospace; letter-spacing: .11em; font-size: .65rem;
}
.cia-briefing-kicker { color: #d9363e; }
.cia-pulse {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: #d9363e; margin-right: 7px;
    box-shadow: 0 0 0 4px rgba(217,54,62,.12), 0 0 13px rgba(217,54,62,.45);
}
.cia-briefing-status {
    border: 1px solid rgba(217,54,62,.30); color: #9cf3e3;
    background: rgba(217,54,62,.065); border-radius: 5px; padding: 6px 9px;
}
.cia-briefing-grid {
    display: grid; grid-template-columns: minmax(0, 1fr) 210px;
    align-items: center; gap: 18px; position: relative; z-index: 1;
    min-height: 205px; padding: 15px 0 11px;
}
.cia-briefing-label { color: #8b9bb2; margin-bottom: 10px; }
.cia-briefing-heading {
    color: #f3f6fc; font-size: clamp(1.65rem, 3vw, 2.35rem);
    line-height: 1.12; font-weight: 800; letter-spacing: -.055em;
}
.cia-briefing-heading em { color: #d9363e; font-style: normal; }
.cia-briefing-copy { color: #9aa9bd; font-size: .86rem; line-height: 1.75; max-width: 620px; margin-top: 12px; }
.cia-briefing-footer {
    border-top: 1px solid rgba(130,130,130,.17); padding-top: 12px;
    color: #74859c; flex-wrap: wrap;
}
.cia-radar {
    position: relative; width: 190px; height: 190px; justify-self: center;
    border-radius: 50%; border: 1px solid rgba(217,54,62,.22);
    background: radial-gradient(circle, rgba(217,54,62,.06), rgba(7,12,20,.1) 70%);
    box-shadow: 0 0 38px rgba(217,54,62,.06), inset 0 0 25px rgba(217,54,62,.04);
}
.cia-radar-ring { position:absolute; border:1px solid rgba(217,54,62,.18); border-radius:50%; inset:12%; }
.cia-radar-ring.ring-two { inset:27%; }
.cia-radar-ring.ring-three { inset:42%; }
.cia-radar-cross { position:absolute; background:rgba(217,54,62,.16); }
.cia-radar-cross.cross-x { height:1px; left:4%; right:4%; top:50%; }
.cia-radar-cross.cross-y { width:1px; top:4%; bottom:4%; left:50%; }
.cia-radar-sweep {
    position:absolute; inset:7%; border-radius:50%;
    background:conic-gradient(from 0deg, transparent 0 68%, rgba(217,54,62,.2) 83%, transparent 100%);
}
.cia-radar-dot { position:absolute; width:6px; height:6px; border-radius:50%; background:#e04444; box-shadow:0 0 10px rgba(255,107,107,.9); }
.cia-radar-dot.dot-one { top:28%; left:62%; }
.cia-radar-dot.dot-two { top:62%; left:31%; background:#d9363e; box-shadow:0 0 10px rgba(217,54,62,.9); }
.cia-radar-dot.dot-three { top:69%; left:68%; background:#d6a0a0; box-shadow:0 0 10px rgba(214,160,160,.9); }
.cia-radar-center { position:absolute; width:5px; height:5px; border-radius:50%; background:#f4f7ff; left:calc(50% - 2.5px); top:calc(50% - 2.5px); }
@media (max-width: 700px) {
    .cia-briefing { padding: 17px; }
    .cia-briefing-grid { grid-template-columns: 1fr; }
    .cia-radar { width: 145px; height: 145px; margin: 8px auto; }
    .cia-briefing-footer { gap: 8px; }
}


/* Use the same condensed display font across dashboard headings */
.cia-section-title,
.cia-panel-title,
.cia-briefing-heading,
.cia-toolstrip-title {
    font-family: 'Oswald', 'Manrope', sans-serif !important;
}

/* Keep the sidebar brand consistent with the display typography */
[data-testid="stSidebar"] h1 {
    font-family: 'Oswald', 'Manrope', sans-serif !important;
}

/* Consistent investigation-console treatment across every workspace */
.cia-section-heading { display:flex; align-items:flex-start; gap:12px; margin:22px 0 8px; }
.cia-section-mark { display:block; width:4px; min-width:4px; height:27px; margin-top:3px; border-radius:4px; background:linear-gradient(180deg,#d9363e,#8f2028); box-shadow:0 0 13px rgba(217,54,62,.18); }
.cia-section-title { color:#f1f5fc; font-size:1.18rem; font-weight:800; letter-spacing:-.035em; line-height:1.35; }
.cia-section-subtitle { color:#8392a9; font-size:.82rem; line-height:1.7; margin-top:3px; }
.cia-toolstrip { display:flex; align-items:center; gap:13px; padding:14px 16px; margin:12px 0 18px; border:1px solid rgba(217,54,62,.18); border-radius:13px; background:linear-gradient(100deg,rgba(30,18,19,.88),rgba(14,14,14,.82)); box-shadow:inset 3px 0 0 rgba(217,54,62,.62); }
.cia-toolstrip-icon { display:flex; align-items:center; justify-content:center; flex:0 0 36px; height:36px; border-radius:10px; color:#d9363e; font-size:1.35rem; background:rgba(217,54,62,.09); border:1px solid rgba(217,54,62,.18); }
.cia-toolstrip-title { color:#eaf1fc; font-size:.86rem; font-weight:800; letter-spacing:-.01em; }
.cia-toolstrip-copy { color:#8c9bb1; font-size:.75rem; line-height:1.55; margin-top:3px; }
.cia-toolstrip-tag { margin-left:auto; white-space:nowrap; color:#d9363e; font-family:'DM Mono',monospace; font-size:.60rem; letter-spacing:.10em; border:1px solid rgba(217,54,62,.22); background:rgba(217,54,62,.05); border-radius:5px; padding:6px 8px; }
.cia-schema-map { display:grid; grid-template-columns:1fr minmax(120px,.75fr) 1fr; gap:12px; align-items:center; margin:14px 0 22px; padding:18px; border:1px solid rgba(130,130,130,.18); border-radius:15px; background:linear-gradient(135deg,rgba(22,22,22,.92),rgba(9,9,9,.9)); }
.cia-schema-side { display:grid; gap:8px; }
.cia-schema-kicker { color:#d9363e; font-family:'DM Mono',monospace; font-size:.62rem; letter-spacing:.14em; margin-bottom:2px; }
.cia-schema-table { display:flex; flex-direction:column; gap:4px; padding:11px 12px; border:1px solid rgba(143,32,40,.22); border-radius:9px; background:rgba(143,32,40,.045); }
.cia-schema-table b { color:#e7edf8; font-size:.77rem; overflow-wrap:anywhere; }
.cia-schema-table small { color:#8191a9; font-size:.68rem; }
.fact-side .cia-schema-table { border-color:rgba(217,54,62,.24); background:rgba(217,54,62,.045); }
.cia-schema-link { text-align:center; color:#9aa9bd; }
.cia-schema-link span { font-family:'DM Mono',monospace; font-size:.59rem; letter-spacing:.12em; color:#8393aa; }
.cia-schema-link div { color:#d9363e; font-size:2.1rem; line-height:1.5; }
.cia-schema-link small { display:block; color:#708199; font-size:.62rem; line-height:1.5; }

/* Navigation and form controls */
[data-testid="stSidebar"] [data-testid="stRadio"] > label { color:#8293aa !important; font-family:'DM Mono',monospace; font-size:.66rem !important; letter-spacing:.14em; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] { gap:5px; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label { border:1px solid transparent; padding:10px 11px; transition:all .18s ease; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) { background:linear-gradient(90deg,rgba(217,54,62,.13),rgba(217,54,62,.025)); border-color:rgba(217,54,62,.22); box-shadow:inset 2px 0 #d9363e; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p { color:#dce6f5 !important; font-size:.82rem; font-weight:650; }
[data-testid="stWidgetLabel"] p { color:#9aa9bd !important; font-size:.77rem !important; font-weight:650; }
[data-baseweb="select"] * { color:#e3eaf6 !important; }
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea { color:#edf3ff !important; }
[data-testid="stDataFrame"] { border-color:rgba(130,130,130,.23); box-shadow:0 12px 32px rgba(0,0,0,.13); }
[data-testid="stDataFrame"] [role="columnheader"] { background:rgba(217,54,62,.07) !important; }
[data-testid="stPlotlyChart"] { border-color:rgba(130,130,130,.19); background:linear-gradient(145deg,rgba(13,19,32,.7),rgba(8,12,21,.55)); }
[data-testid="stExpander"] summary { color:#e4ecf8; font-weight:700; }
.stAlert p { color:inherit; }
.stDownloadButton button { letter-spacing:.01em; }
@media (max-width: 760px) {
  .cia-toolstrip { align-items:flex-start; }
  .cia-toolstrip-tag { display:none; }
  .cia-schema-map { grid-template-columns:1fr; }
  .cia-schema-link div { transform:rotate(90deg); line-height:1.1; }
  .cia-schema-link small { margin-bottom:4px; }
}


/* Final polish: clearer focus, calmer navigation, and consistent chart rhythm */
[data-testid="stSidebar"] [data-testid="stRadio"] label:focus-within {
    outline: 2px solid rgba(217,54,62,.72);
    outline-offset: 2px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
    background: linear-gradient(90deg, rgba(217,54,62,.13), rgba(217,54,62,.035));
    border: 1px solid rgba(217,54,62,.20);
    box-shadow: inset 2px 0 #d9363e;
}
[data-testid="stPlotlyChart"] {
    transition: border-color .18s ease, box-shadow .18s ease;
}
[data-testid="stPlotlyChart"]:hover {
    border-color: rgba(217,54,62,.28);
    box-shadow: 0 15px 38px rgba(0,0,0,.20);
}
div.stButton > button:active, div.stDownloadButton > button:active {
    transform: translateY(0);
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        transition-duration: .01ms !important;
        animation-duration: .01ms !important;
        scroll-behavior: auto !important;
    }
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def get_connection():
    return sqlite3.connect(DB_PATH)


@st.cache_data
def load_data(query, params=()):
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def format_number(value):
    return f"{int(value or 0):,}"


def heading(title, subtitle=None):
    subtitle_html = f'<div class="cia-section-subtitle">{subtitle}</div>' if subtitle else ''
    st.markdown(
        f'<div class="cia-section-heading"><span class="cia-section-mark"></span>'
        f'<div><div class="cia-section-title">{title}</div>{subtitle_html}</div></div>',
        unsafe_allow_html=True,
    )


def styled_chart(fig, height=None):
    """Apply the CIA chart system consistently across all Plotly visualizations."""
    fig.update_layout(
        template="plotly_dark",
        colorway=["#d9363e", "#a7a7a7", "#8f2028", "#e6a0a0", "#6f6f6f", "#f05a5a"],
        paper_bgcolor="rgba(7,10,18,0)",
        plot_bgcolor="rgba(7,10,18,0)",
        font=dict(family="Manrope, sans-serif", color="#dce7f7", size=12),
        margin=dict(l=14, r=18, t=38, b=16),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(color="#b9c4d6", size=11),
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            itemsizing="constant",
        ),
        hoverlabel=dict(
            bgcolor="#111c2b", bordercolor="#d9363e",
            font=dict(family="Manrope, sans-serif", color="#edf7ff", size=12),
            namelength=-1,
        ),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(130,130,130,.10)",
            zerolinecolor="rgba(130,130,130,.22)",
            ticks="outside", tickcolor="rgba(130,130,130,.25)",
            tickfont=dict(color="#aebbd0", size=11),
            title_font=dict(color="#dce7f7", size=12),
            automargin=True,
            showline=False,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(130,130,130,.08)",
            zerolinecolor="rgba(130,130,130,.22)",
            ticks="outside", tickcolor="rgba(130,130,130,.25)",
            tickfont=dict(color="#aebbd0", size=11),
            title_font=dict(color="#dce7f7", size=12),
            automargin=True,
            showline=False,
        ),
        hovermode="closest",
    )
    # Make chart backgrounds transparent and outlines subtle, including chart types
    # whose traces are generated by Plotly Express.
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    if height:
        fig.update_layout(height=height)
    return fig


@st.cache_data
def get_years():
    return load_data("SELECT year FROM Dim_Date ORDER BY year")["year"].astype(int).tolist()


@st.cache_data
def get_states():
    return load_data("SELECT DISTINCT state FROM Dim_Location ORDER BY state")["state"].dropna().tolist()


@st.cache_data
def get_crime_types():
    return load_data("SELECT crime_type FROM Dim_Crime_Type ORDER BY crime_type")["crime_type"].dropna().tolist()


def get_districts(state="All"):
    if state != "All":
        df = load_data("SELECT DISTINCT district FROM Dim_Location WHERE state = ? ORDER BY district", (state,))
    else:
        df = load_data("SELECT DISTINCT district FROM Dim_Location ORDER BY district")
    return df["district"].dropna().tolist()


def normalize_state(name):
    value = " ".join(str(name).strip().lower().replace("&", "and").split())
    aliases = {
        "orissa": "odisha", "uttaranchal": "uttarakhand",
        "nct of delhi": "delhi", "national capital territory of delhi": "delhi",
        "pondicherry": "puducherry",
        "andaman and nicobar islands": "andaman and nicobar islands",
    }
    return aliases.get(value, value)


@st.cache_data(ttl=86400)
def load_india_geojson():
    url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/india_states.geojson"
    request = Request(url, headers={"User-Agent": "CIA-Crime-Analytics/1.0"})
    with urlopen(request, timeout=20) as response:
        geojson = json.loads(response.read().decode("utf-8"))
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        name = props.get("ST_NM") or props.get("state") or props.get("NAME_1") or props.get("name")
        if name:
            props["cia_state"] = normalize_state(name)
    return geojson



# Census 2011 population benchmarks from India's official census tables.
# These are fixed reference populations, not annual estimates for every crime year.
CENSUS_2011_POPULATION = {
    "andhra pradesh": 84580777, "arunachal pradesh": 1383727,
    "assam": 31205576, "bihar": 104099452, "chhattisgarh": 25545198,
    "goa": 1458545, "gujarat": 60439692, "haryana": 25351462,
    "himachal pradesh": 6864602, "jammu and kashmir": 12541302,
    "jharkhand": 32988134, "karnataka": 61095297, "kerala": 33406061,
    "madhya pradesh": 72626809, "maharashtra": 112374333,
    "manipur": 2855794, "meghalaya": 2966889, "mizoram": 1097206,
    "nagaland": 1978502, "odisha": 41974218, "punjab": 27743338,
    "rajasthan": 68548437, "sikkim": 610577, "tamil nadu": 72147030,
    "tripura": 3673917, "uttar pradesh": 199812341, "uttarakhand": 10086292,
    "west bengal": 91276115, "delhi": 16787941, "puducherry": 1247953,
    "andaman and nicobar islands": 380581, "chandigarh": 1055450,
    "dadra and nagar haveli": 343709, "daman and diu": 243247,
    "lakshadweep": 64473, "telangana": 35193978,
}


def population_for_state(state):
    key = normalize_state(state)
    aliases = {
        "orissa": "odisha", "uttaranchal": "uttarakhand",
        "nct of delhi": "delhi",
        "national capital territory of delhi": "delhi",
        "andaman and nicobar": "andaman and nicobar islands",
        "jammu & kashmir": "jammu and kashmir",
    }
    return CENSUS_2011_POPULATION.get(aliases.get(key, key))


def render_india_map(selected_year="All"):
    heading("India Crime Heatmap", "Compare total reported counts with population-adjusted reported-crime rates.")
    metric = st.radio(
        "Heatmap metric",
        ["Total reported crimes", "Reported crimes per 100,000 people"],
        horizontal=True,
        key="overview_heatmap_metric",
    )
    if metric == "Reported crimes per 100,000 people":
        st.caption(
            "Rate = reported IPC counts ÷ Census 2011 population × 100,000. "
            "Census 2011 population is a fixed benchmark, so rates for years other than 2011 are approximate."
        )

    try:
        geojson = load_india_geojson()
    except Exception as error:
        st.warning("The map boundary file could not be downloaded. Check your internet connection.")
        with st.expander("Technical details"):
            st.code(str(error))
        return

    year_filter, params = ("", ())
    if selected_year != "All":
        year_filter, params = ("WHERE d.year = ?", (selected_year,))

    data = load_data(f"""
        SELECT l.state, SUM(f.total_ipc_crimes) AS reported_crimes,
               COUNT(DISTINCT l.district) AS districts_represented
        FROM Fact_District_Totals f
        JOIN Dim_Date d ON f.date_id=d.date_id
        JOIN Dim_Location l ON f.location_id=l.location_id
        {year_filter}
        GROUP BY l.state
    """, params)

    if data.empty:
        st.warning("No records are available for this period.")
        return

    data["map_state"] = data["state"].map(normalize_state)
    geo_names = {f.get("properties", {}).get("cia_state") for f in geojson.get("features", [])}
    matched = data[data["map_state"].isin(geo_names)].copy()

    if metric == "Reported crimes per 100,000 people":
        matched["population_2011"] = matched["state"].map(population_for_state)
        unmatched_population = matched[matched["population_2011"].isna()]
        matched = matched[matched["population_2011"].notna() & (matched["population_2011"] > 0)].copy()
        matched["crime_rate_per_100k"] = matched["reported_crimes"] / matched["population_2011"] * 100000
        if not unmatched_population.empty:
            st.warning(
                "Some states/UTs could not be matched to the Census 2011 population reference and are omitted from the rate map."
            )

    if matched.empty:
        st.warning("No state names could be matched to the map and population reference.")
        st.dataframe(data[["state", "reported_crimes"]], use_container_width=True, hide_index=True)
        return

    if metric == "Reported crimes per 100,000 people":
        colour_column = "crime_rate_per_100k"
        colour_label = "Reported crimes per 100,000 people"
        hover_data = {
            "crime_rate_per_100k": ":.2f",
            "reported_crimes": ":,",
            "population_2011": ":,",
            "districts_represented": True,
            "map_state": False,
        }
        labels = {
            "crime_rate_per_100k": "Reported crimes per 100,000 people",
            "reported_crimes": "Reported IPC counts",
            "population_2011": "Census 2011 population",
            "districts_represented": "Districts represented",
        }
        tick_format = ".2f"
    else:
        colour_column = "reported_crimes"
        colour_label = "Reported IPC counts"
        hover_data = {
            "reported_crimes": ":,",
            "districts_represented": True,
            "map_state": False,
        }
        labels = {
            "reported_crimes": "Reported IPC counts",
            "districts_represented": "Districts represented",
        }
        tick_format = ","

    fig = px.choropleth(
        matched, geojson=geojson, locations="map_state",
        featureidkey="properties.cia_state", color=colour_column,
        color_continuous_scale=[[0, "#18090B"], [.25, "#390D12"], [.5, "#67121B"], [.75, "#A51F2B"], [1, "#E3343E"]],
        hover_name="state",
        hover_data=hover_data,
        labels=labels,
    )
    fig.update_geos(fitbounds="locations", visible=False, showcountries=True, countrycolor="#526780", showcoastlines=False, bgcolor="rgba(0,0,0,0)")
    fig.update_traces(
        marker_line_color="#6f555b", marker_line_width=.7,
        colorbar=dict(title=colour_label, tickformat=tick_format, thickness=14, outlinewidth=0),
    )
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=5, b=0))
    st.plotly_chart(styled_chart(fig, 620), use_container_width=True, config={"displaylogo": False, "scrollZoom": False})

    if metric == "Total reported crimes":
        st.caption("Colour intensity represents summed reported counts, not population-adjusted rates or live reports.")
    else:
        st.caption(
            "Rate values use Census 2011 population benchmarks. They describe reported counts per 100,000 residents, "
            "not the true likelihood of victimisation or all crime."
        )
    unmatched = data[~data["map_state"].isin(geo_names)]
    if not unmatched.empty:
        with st.expander(f"States not matched to boundaries ({len(unmatched)})"):
            st.dataframe(unmatched[["state", "reported_crimes"]], use_container_width=True, hide_index=True)


# Validate database
if not DB_PATH.exists():
    st.error("Database not found: cia_real_warehouse.db. Keep it in the same folder as app.py.")
    st.stop()

try:
    existing_tables = load_data("SELECT name FROM sqlite_master WHERE type='table'")["name"].tolist()
    required_tables = ["Dim_Date", "Dim_Location", "Dim_Crime_Type", "Fact_Crime_Stats", "Fact_District_Totals"]
    missing = [t for t in required_tables if t not in existing_tables]
    if missing:
        st.error("Missing database tables: " + ", ".join(missing))
        st.stop()
    years, states, crime_types = get_years(), get_states(), get_crime_types()
except Exception as error:
    st.error(f"Unable to read the warehouse: {error}")
    st.stop()

if not years:
    st.error("The warehouse contains no year records. Run python etl.py first.")
    st.stop()


# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 6px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px">
        <div style="display:flex;align-items:center;justify-content:center;width:43px;height:43px;border:1px solid rgba(217,54,62,.55);border-radius:7px;clip-path:polygon(50% 0%,100% 50%,50% 100%,0% 50%);background:linear-gradient(145deg,rgba(217,54,62,.18),rgba(255,107,107,.08));font-size:1.55rem;color:#d9363e;box-shadow:0 0 22px rgba(217,54,62,.08)">⌕</div>
        <div><div style="font-size:1.55rem;font-weight:800;line-height:1.1;color:#f8fafc;letter-spacing:-.06em">CIA<span style="color:#d9363e">.</span></div><div style="font-family:'DM Mono',monospace;color:#8fa1ba;font-size:.61rem;letter-spacing:.14em;margin-top:4px">INTELLIGENCE LAB</div></div>
      </div>
      <div style="color:#b9c9dc;font-size:.78rem;font-weight:700;letter-spacing:.04em">CRIME INTELLIGENCE ANALYSIS</div>
      <div style="color:#8295b0;font-size:.73rem;margin-top:6px">Historical Crime Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("WORKSPACE", ["Overview", "Crime Analytics", "District Records", "Preprocessing", "Data Warehouse", "Data Mining"])
    st.divider()
    st.caption(f"Period: {min(years)}–{max(years)}")
    st.caption(f"State / UT labels: {len(states)}")
    st.caption("Storage: SQLite")
    st.caption("Source: dstrIPC_1.csv")
    st.divider()
    st.caption("Academic Data Warehousing & Data Mining Project")


# OVERVIEW
if page == "Overview":
    st.markdown('<div class="cia-brand">INTELLIGENCE DASHBOARD / OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="cia-main-title"><span class="crime-word">Crime</span> Intelligence Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<div class="cia-subtitle">Explore historical Indian IPC crime statistics through regional comparisons, category trends, and district-level records.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cia-briefing">
      <div class="cia-briefing-top">
        <span class="cia-briefing-kicker"><span class="cia-pulse"></span> ARCHIVE INTELLIGENCE / INDIA</span>
        <span class="cia-briefing-status">HISTORICAL DATASET</span>
      </div>
      <div class="cia-briefing-grid">
        <div>
          <div class="cia-briefing-label">ANALYTICAL OBJECTIVE</div>
          <div class="cia-briefing-heading">Read the patterns.<br><em>Understand the context.</em></div>
          <div class="cia-briefing-copy">Explore historical reported crime counts by year, state, district, and category. Use the map, trends, and district records to understand how the dataset is distributed.</div>
        </div>
        <div class="cia-radar" aria-hidden="true">
          <div class="cia-radar-ring ring-one"></div><div class="cia-radar-ring ring-two"></div>
          <div class="cia-radar-ring ring-three"></div><div class="cia-radar-cross cross-x"></div>
          <div class="cia-radar-cross cross-y"></div><div class="cia-radar-sweep"></div>
          <div class="cia-radar-dot dot-one"></div><div class="cia-radar-dot dot-two"></div>
          <div class="cia-radar-dot dot-three"></div><div class="cia-radar-center"></div>
        </div>
      </div>
      <div class="cia-briefing-footer"><span>01 / OBSERVE</span><span>02 / COMPARE</span><span>03 / DISCOVER PATTERNS</span></div>
    </div>
    """, unsafe_allow_html=True)

    totals = load_data("SELECT SUM(total_ipc_crimes) AS total_crimes, COUNT(*) AS record_count FROM Fact_District_Totals")
    total_crimes = int(totals.iloc[0]["total_crimes"] or 0)
    record_count = int(totals.iloc[0]["record_count"] or 0)
    location_count = int(load_data("SELECT COUNT(*) AS n FROM Dim_Location").iloc[0]["n"])
    state_count = int(load_data("SELECT COUNT(DISTINCT state) AS n FROM Dim_Location").iloc[0]["n"])

    a, b, c, d = st.columns(4)
    a.metric("Summed Reported IPC Counts", format_number(total_crimes))
    b.metric("States / UT Labels", format_number(state_count))
    c.metric("District Labels", format_number(location_count))
    d.metric("District-Year Records", format_number(record_count))
    st.caption("Summed counts add district-reported totals across the period; they are not deduplicated individual incidents.")
    st.divider()

    map_year = st.selectbox("Heatmap period", ["All"] + years, key="overview_map_year")
    render_india_map(map_year)
    st.divider()

    yearly = load_data("""
        SELECT d.year, SUM(f.total_ipc_crimes) AS total_crimes
        FROM Fact_District_Totals f JOIN Dim_Date d ON f.date_id=d.date_id
        GROUP BY d.year ORDER BY d.year
    """)
    heading("Historical Crime Trend", "Year-wise sum of reported district-level IPC counts.")
    if not yearly.empty:
        fig = px.line(yearly, x="year", y="total_crimes", markers=True, labels={"year": "Year", "total_crimes": "Reported IPC counts"})
        fig.update_traces(
            line=dict(color="#d9363e", width=3),
            marker=dict(size=7, color="#d9363e", line=dict(color="#171717", width=1.5)),
            fill="tozeroy", fillcolor="rgba(217,54,62,.08)",
            hovertemplate="<b>Year %{x}</b><br>Reported counts: %{y:,.0f}<extra></extra>",
        )
        st.plotly_chart(styled_chart(fig, 380), use_container_width=True)

    left, right = st.columns(2)
    with left:
        heading("Leading Crime Categories", "Top 10 categories by summed counts.")
        cats = load_data("""
            SELECT c.crime_type, SUM(f.crime_count) AS crime_count
            FROM Fact_Crime_Stats f JOIN Dim_Crime_Type c ON f.crime_type_id=c.crime_type_id
            GROUP BY c.crime_type ORDER BY crime_count DESC LIMIT 10
        """)
        if not cats.empty:
            fig = px.bar(cats.sort_values("crime_count"), x="crime_count", y="crime_type", orientation="h", labels={"crime_count": "Reported count", "crime_type": "Crime category"})
            fig.update_traces(marker_color="#d9363e", marker_line_color="rgba(217,54,62,.28)", marker_line_width=1,
                              hovertemplate="<b>%{y}</b><br>Reported counts: %{x:,.0f}<extra></extra>")
            fig.update_layout(bargap=.28)
            st.plotly_chart(styled_chart(fig, 390), use_container_width=True)
    with right:
        heading("State / UT Comparison", "Top 10 by summed reported IPC counts.")
        top_states = load_data("""
            SELECT l.state, SUM(f.total_ipc_crimes) AS crime_count
            FROM Fact_District_Totals f JOIN Dim_Location l ON f.location_id=l.location_id
            GROUP BY l.state ORDER BY crime_count DESC LIMIT 10
        """)
        if not top_states.empty:
            fig = px.bar(top_states.sort_values("crime_count"), x="crime_count", y="state", orientation="h", labels={"crime_count": "Reported IPC counts", "state": "State / UT"})
            fig.update_traces(marker_color="#b5b5b5", marker_line_color="rgba(170, 45, 52,.28)", marker_line_width=1,
                              hovertemplate="<b>%{y}</b><br>Reported IPC counts: %{x:,.0f}<extra></extra>")
            fig.update_layout(bargap=.28)
            st.plotly_chart(styled_chart(fig, 390), use_container_width=True)
    st.markdown('<div class="cia-panel"><div class="cia-panel-title">About this dashboard</div><div class="cia-panel-text">CIA combines a historical crime data warehouse, SQL aggregation, OLAP-style analysis, visual analytics, and unsupervised machine learning.</div></div>', unsafe_allow_html=True)


# CRIME ANALYTICS
elif page == "Crime Analytics":
    st.markdown('<div class="cia-brand">ANALYTICS / EXPLORATION</div>', unsafe_allow_html=True)
    st.title("Crime Analytics")
    st.markdown('<div class="cia-subtitle">Filter historical data and compare regional and crime-category patterns.</div>', unsafe_allow_html=True)
    st.markdown('<div class="cia-toolstrip"><div class="cia-toolstrip-icon">⌘</div><div><div class="cia-toolstrip-title">Investigation filters</div><div class="cia-toolstrip-copy">Set the time period and geography first. All charts below update from the selected scope.</div></div><div class="cia-toolstrip-tag">FILTER CONSOLE</div></div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        selected_year = st.selectbox("Year", ["All"] + years, key="analytics_year")
    with f2:
        selected_state = st.selectbox("State / UT", ["All"] + states, key="analytics_state")
    districts = get_districts(selected_state)
    with f3:
        selected_district = st.selectbox("District", ["All"] + districts, key="analytics_district")
    with f4:
        selected_category = st.selectbox("Crime category", ["All"] + crime_types, key="analytics_category")

    conditions, params = [], []
    if selected_year != "All":
        conditions.append("d.year = ?"); params.append(selected_year)
    if selected_state != "All":
        conditions.append("l.state = ?"); params.append(selected_state)
    if selected_district != "All":
        conditions.append("l.district = ?"); params.append(selected_district)
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    trend = load_data(f"""
        SELECT d.year, SUM(f.total_ipc_crimes) AS total_crimes
        FROM Fact_District_Totals f JOIN Dim_Date d ON f.date_id=d.date_id
        JOIN Dim_Location l ON f.location_id=l.location_id
        {where_clause} GROUP BY d.year ORDER BY d.year
    """, tuple(params))
    heading("Reported Crime Trend", "Based on the separate total IPC field.")
    if not trend.empty:
        fig = px.line(trend, x="year", y="total_crimes", markers=True, labels={"year": "Year", "total_crimes": "Reported IPC counts"})
        fig.update_traces(
            line=dict(color="#d9363e", width=3),
            marker=dict(size=7, color="#d9363e", line=dict(color="#171717", width=1.5)),
            fill="tozeroy", fillcolor="rgba(217,54,62,.08)",
            hovertemplate="<b>Year %{x}</b><br>Reported counts: %{y:,.0f}<extra></extra>",
        )
        st.plotly_chart(styled_chart(fig, 350), use_container_width=True)
    else:
        st.warning("No records match the selected filters.")

    cat_conditions, cat_params = list(conditions), list(params)
    if selected_category != "All":
        cat_conditions.append("c.crime_type = ?"); cat_params.append(selected_category)
    cat_where = "WHERE " + " AND ".join(cat_conditions) if cat_conditions else ""
    category_data = load_data(f"""
        SELECT c.crime_type, SUM(f.crime_count) AS crime_count
        FROM Fact_Crime_Stats f JOIN Dim_Date d ON f.date_id=d.date_id
        JOIN Dim_Location l ON f.location_id=l.location_id
        JOIN Dim_Crime_Type c ON f.crime_type_id=c.crime_type_id
        {cat_where} GROUP BY c.crime_type ORDER BY crime_count DESC
    """, tuple(cat_params))
    heading("Crime Category Comparison")
    if not category_data.empty:
        plot_data = category_data.head(15).sort_values("crime_count")
        fig = px.bar(plot_data, x="crime_count", y="crime_type", orientation="h", labels={"crime_count": "Reported category counts", "crime_type": "Crime category"})
        fig.update_traces(marker_color="#d9363e", marker_line_color="rgba(217,54,62,.28)", marker_line_width=1,
                          hovertemplate="<b>%{y}</b><br>Reported counts: %{x:,.0f}<extra></extra>")
        fig.update_layout(bargap=.24)
        st.plotly_chart(styled_chart(fig, 480), use_container_width=True)
        st.download_button("Download category analysis", category_data.to_csv(index=False).encode("utf-8"), "cia_category_analysis.csv", "text/csv")

    heading("District Comparison", "Top districts under the selected filters.")
    district_data = load_data(f"""
        SELECT l.state, l.district, SUM(f.total_ipc_crimes) AS total_crimes
        FROM Fact_District_Totals f JOIN Dim_Date d ON f.date_id=d.date_id
        JOIN Dim_Location l ON f.location_id=l.location_id
        {where_clause} GROUP BY l.state,l.district ORDER BY total_crimes DESC LIMIT 25
    """, tuple(params))
    if not district_data.empty:
        st.dataframe(district_data, use_container_width=True, hide_index=True)


# DISTRICT RECORDS
elif page == "District Records":
    st.markdown('<div class="cia-brand">RECORDS / DATASET EXPLORER</div>', unsafe_allow_html=True)
    st.title("District Records")
    st.markdown('<div class="cia-subtitle">Search, inspect, and export historical district-year records.</div>', unsafe_allow_html=True)
    st.markdown('<div class="cia-toolstrip"><div class="cia-toolstrip-icon">⌕</div><div><div class="cia-toolstrip-title">Record lookup</div><div class="cia-toolstrip-copy">Search place names or narrow the archive to a specific year. Export the matching rows when needed.</div></div><div class="cia-toolstrip-tag">DATA EXPLORER</div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        search_text = st.text_input("Search state or district", placeholder="Enter a state or district name...")
    with c2:
        record_year = st.selectbox("Year", ["All"] + years, key="records_year")

    conditions, params = [], []
    if record_year != "All":
        conditions.append("d.year = ?"); params.append(record_year)
    if search_text.strip():
        conditions.append("(LOWER(l.state) LIKE ? OR LOWER(l.district) LIKE ?)")
        pattern = f"%{search_text.strip().lower()}%"; params.extend([pattern, pattern])
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    records = load_data(f"""
        SELECT d.year AS Year, l.state AS State_UT, l.district AS District,
               f.total_ipc_crimes AS Total_IPC_Crimes
        FROM Fact_District_Totals f JOIN Dim_Date d ON f.date_id=d.date_id
        JOIN Dim_Location l ON f.location_id=l.location_id
        {where_clause} ORDER BY d.year DESC,l.state,l.district
    """, tuple(params))
    st.metric("Matching district-year records", format_number(len(records)))
    st.dataframe(records, use_container_width=True, hide_index=True, height=480)
    st.download_button("Download CSV", records.to_csv(index=False).encode("utf-8"), "cia_district_records.csv", "text/csv")
    st.caption("Each row is an aggregated district-year record, not an individual FIR or incident.")


# DATA WAREHOUSE
elif page == "Data Warehouse":
    st.markdown('<div class="cia-brand">ARCHITECTURE / DATA WAREHOUSE</div>', unsafe_allow_html=True)
    st.title("Data Warehouse")
    st.markdown('<div class="cia-subtitle">Explore the stored crime tables, table statistics, and OLAP-style operations.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="cia-panel"><div class="cia-panel-title">Warehouse overview</div>
    <div class="cia-panel-text">Historical crime records are stored in a SQLite database. The ETL pipeline loads and organizes the source CSV into tables for district totals and crime-category analysis. Use the table statistics and OLAP tools below to explore the stored data.</div></div>
    """, unsafe_allow_html=True)

    tables = ["Dim_Date", "Dim_Location", "Dim_Crime_Type", "Fact_Crime_Stats", "Fact_District_Totals"]
    stats = []
    for table in tables:
        n = int(load_data(f"SELECT COUNT(*) AS n FROM {table}").iloc[0]["n"])
        stats.append({"Table": table, "Rows": n, "Type": "Fact" if table.startswith("Fact_") else "Dimension"})
    heading("Warehouse Table Statistics")
    st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        heading("Dimension Preview")
        dimension = st.selectbox("Choose dimension", ["Dim_Date", "Dim_Location", "Dim_Crime_Type"])
        st.dataframe(load_data(f"SELECT * FROM {dimension} LIMIT 100"), use_container_width=True, hide_index=True)
    with right:
        heading("Fact Table Preview")
        fact = st.selectbox("Choose fact table", ["Fact_District_Totals", "Fact_Crime_Stats"])
        if fact == "Fact_District_Totals":
            preview = load_data("""SELECT d.year,l.state,l.district,f.total_ipc_crimes FROM Fact_District_Totals f JOIN Dim_Date d ON f.date_id=d.date_id JOIN Dim_Location l ON f.location_id=l.location_id ORDER BY d.year DESC,l.state,l.district LIMIT 100""")
        else:
            preview = load_data("""SELECT d.year,l.state,l.district,c.crime_type,f.crime_count FROM Fact_Crime_Stats f JOIN Dim_Date d ON f.date_id=d.date_id JOIN Dim_Location l ON f.location_id=l.location_id JOIN Dim_Crime_Type c ON f.crime_type_id=c.crime_type_id ORDER BY d.year DESC,l.state,l.district LIMIT 100""")
        st.dataframe(preview, use_container_width=True, hide_index=True)

    st.divider()
    heading("OLAP Operations")
    with st.expander("1. Roll-up — summarize by year"):
        df = load_data("""SELECT d.year,SUM(f.total_ipc_crimes) AS total_crimes FROM Fact_District_Totals f JOIN Dim_Date d ON f.date_id=d.date_id GROUP BY d.year ORDER BY d.year""")
        st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander("2. Slice — select a single year"):
        yr = st.selectbox("Select year", years, key="warehouse_slice_year")
        df = load_data("""SELECT l.state,l.district,f.total_ipc_crimes FROM Fact_District_Totals f JOIN Dim_Date d ON f.date_id=d.date_id JOIN Dim_Location l ON f.location_id=l.location_id WHERE d.year=? ORDER BY f.total_ipc_crimes DESC LIMIT 100""", (yr,))
        st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander("3. Drill-down — compare states by year"):
        df = load_data("""SELECT d.year,l.state,SUM(f.total_ipc_crimes) AS total_crimes FROM Fact_District_Totals f JOIN Dim_Date d ON f.date_id=d.date_id JOIN Dim_Location l ON f.location_id=l.location_id GROUP BY d.year,l.state ORDER BY d.year,total_crimes DESC LIMIT 200""")
        st.dataframe(df, use_container_width=True, hide_index=True)



# PREPROCESSING
elif page == "Preprocessing":
    st.markdown('<div class="cia-brand">DATA QUALITY / PREPARATION</div>', unsafe_allow_html=True)
    st.title("Data Preprocessing")
    st.markdown('<div class="cia-subtitle">Inspect and prepare the warehouse data before running data-mining algorithms.</div>', unsafe_allow_html=True)
    st.info("The source contains aggregated district-year crime counts, not individual incident records. Cleaning and transformations below are applied to an analysis copy; the SQLite warehouse is not modified.")

    @st.cache_data
    def load_preprocessing_data():
        return load_data("""
            SELECT l.state, l.district, d.year, c.crime_type, f.crime_count
            FROM Fact_Crime_Stats f
            JOIN Dim_Location l ON f.location_id=l.location_id
            JOIN Dim_Date d ON f.date_id=d.date_id
            JOIN Dim_Crime_Type c ON f.crime_type_id=c.crime_type_id
        """)

    long_df = load_preprocessing_data()
    if long_df.empty:
        st.warning("No warehouse records were found. Run the ETL pipeline first.")
        st.stop()

    original_rows = len(long_df)
    original_missing = int(long_df.isna().sum().sum())
    duplicate_rows = int(long_df.duplicated().sum())
    invalid_counts = int((pd.to_numeric(long_df["crime_count"], errors="coerce") < 0).sum())

    clean = long_df.copy()
    clean["state"] = clean["state"].astype("string").str.strip().str.upper()
    clean["district"] = clean["district"].astype("string").str.strip().str.upper()
    clean["crime_type"] = clean["crime_type"].astype("string").str.strip()
    clean["year"] = pd.to_numeric(clean["year"], errors="coerce")
    clean["crime_count"] = pd.to_numeric(clean["crime_count"], errors="coerce")
    clean = clean.drop_duplicates()
    clean = clean.dropna(subset=["state", "district", "year", "crime_type"])
    clean = clean[clean["crime_count"].notna() & (clean["crime_count"] >= 0)]
    clean["crime_count"] = clean["crime_count"].fillna(0)
    clean["year"] = clean["year"].astype(int)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows loaded", format_number(original_rows))
    m2.metric("Duplicate rows", format_number(duplicate_rows))
    m3.metric("Missing cells", format_number(original_missing))
    m4.metric("Negative counts", format_number(invalid_counts))

    heading("Preprocessing steps")
    st.markdown("""
    - **Type conversion:** year and crime counts are converted to numeric values.
    - **Text normalization:** state, district and crime-category labels are trimmed; state and district labels are uppercased for consistency.
    - **Duplicate removal:** exact duplicate fact rows are removed from the analysis copy.
    - **Missing / invalid handling:** rows missing identifiers are excluded; missing or negative crime counts are excluded because a negative count is not meaningful.
    - **Feature reshaping:** long-format crime records can be pivoted into one row per district-year and one column per crime category.
    """)
    k1, k2, k3 = st.columns(3)
    k1.metric("Rows after cleaning", format_number(len(clean)))
    k2.metric("Rows removed", format_number(original_rows - len(clean)))
    k3.metric("Remaining missing cells", format_number(int(clean.isna().sum().sum())))

    with st.expander("Inspect data quality by column"):
        quality = pd.DataFrame({
            "Column": long_df.columns,
            "Data type": [str(long_df[c].dtype) for c in long_df.columns],
            "Missing values": [int(long_df[c].isna().sum()) for c in long_df.columns],
            "Distinct values": [int(long_df[c].nunique(dropna=True)) for c in long_df.columns],
        })
        st.dataframe(quality, use_container_width=True, hide_index=True)

    heading("Feature engineering")
    wide = clean.pivot_table(
        index=["state", "district", "year"],
        columns="crime_type", values="crime_count", aggfunc="sum", fill_value=0
    ).reset_index()
    wide.columns.name = None
    feature_cols = [c for c in wide.columns if c not in ["state", "district", "year"]]
    wide["TOTAL_REPORTED_CRIMES"] = wide[feature_cols].sum(axis=1)
    denom = wide["TOTAL_REPORTED_CRIMES"].replace(0, pd.NA)
    proportion = wide[feature_cols].div(denom, axis=0).fillna(0)
    proportion.columns = [f"SHARE_{c}" for c in feature_cols]
    engineered = pd.concat([wide, proportion], axis=1)

    e1, e2, e3 = st.columns(3)
    e1.metric("District-year profiles", format_number(len(wide)))
    e2.metric("Crime category features", format_number(len(feature_cols)))
    e3.metric("Engineered share features", format_number(len(feature_cols)))
    st.caption("Share features represent each category's proportion of that district-year's total reported counts. They are descriptive features, not population-adjusted crime rates.")
    st.dataframe(engineered.head(30), use_container_width=True, hide_index=True)

    st.download_button(
        "Download cleaned long-format data",
        clean.to_csv(index=False).encode("utf-8"),
        "cia_cleaned_crime_data.csv",
        "text/csv",
        type="primary",
    )
    st.download_button(
        "Download engineered district-year features",
        engineered.to_csv(index=False).encode("utf-8"),
        "cia_engineered_features.csv",
        "text/csv",
    )


# DATA MINING
elif page == "Data Mining":
    mining_tab, comparison_tab = st.tabs(["K-Means Analysis", "Compare Algorithms"])

    with mining_tab:
        st.markdown('<div class="cia-brand">MACHINE LEARNING / DATA MINING</div>', unsafe_allow_html=True)
        st.title("District Crime Pattern Clustering")
        st.markdown('<div class="cia-subtitle">Discover groups of districts with similar historical crime-category profiles using K-Means.</div>', unsafe_allow_html=True)

        st.markdown('<div class="cia-toolstrip"><div class="cia-toolstrip-icon">◎</div><div><div class="cia-toolstrip-title">Pattern discovery pipeline</div><div class="cia-toolstrip-copy">Standardize category counts → group district profiles with K-Means → visualize similarity using PCA.</div></div><div class="cia-toolstrip-tag">UNSUPERVISED ML</div></div>', unsafe_allow_html=True)
        st.info("K-Means is unsupervised learning. Clusters summarize similarities in reported historical counts; they do not predict individual criminality or personal risk.")

        @st.cache_data
        def load_clustering_data():
            raw = load_data("""
                SELECT l.state,l.district,c.crime_type,SUM(f.crime_count) AS crime_count
                FROM Fact_Crime_Stats f JOIN Dim_Location l ON f.location_id=l.location_id
                JOIN Dim_Crime_Type c ON f.crime_type_id=c.crime_type_id
                GROUP BY l.state,l.district,c.crime_type
            """)
            if raw.empty:
                return pd.DataFrame()
            wide = raw.pivot_table(index=["state", "district"], columns="crime_type", values="crime_count", aggfunc="sum", fill_value=0).reset_index()
            wide.columns.name = None
            return wide

        cluster_data = load_clustering_data()
        if cluster_data.empty:
            st.warning("No category data is available for clustering.")
            st.stop()

        identity = ["state", "district"]
        feature_cols = [c for c in cluster_data.columns if c not in identity]
        X = cluster_data[feature_cols].fillna(0).astype(float)
        X = X.loc[:, X.nunique() > 1]

        if len(cluster_data) < 3 or X.shape[1] == 0:
            st.warning("At least three district profiles and one varying category are required.")
            st.stop()

        controls1, controls2 = st.columns([1, 2])
        with controls1:
            k = st.slider("Number of clusters (K)", 2, min(6, len(cluster_data)), min(3, min(6, len(cluster_data))))
        with controls2:
            st.markdown('<div class="cia-panel"><div class="cia-panel-title">What this analysis shows</div><div class="cia-panel-text">Districts are grouped using their historical category-count profiles. The profile table below identifies categories that are comparatively high or low in each cluster.</div></div>', unsafe_allow_html=True)

        if st.button("Run K-Means Analysis", type="primary"):
            scaler = StandardScaler()
            scaled = scaler.fit_transform(X)
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(scaled)
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(scaled)

            results = cluster_data[identity].copy()
            results["Cluster"] = labels + 1
            results["Component 1"] = coords[:, 0]
            results["Component 2"] = coords[:, 1]

            # Per-cluster standardized feature means make profiles interpretable.
            scaled_df = pd.DataFrame(scaled, columns=X.columns)
            scaled_df["Cluster"] = labels + 1
            cluster_profiles = scaled_df.groupby("Cluster").mean()
            overall_mean = pd.Series(0.0, index=X.columns)

            st.session_state["cia_cluster_results"] = results
            st.session_state["cia_cluster_labels"] = labels
            st.session_state["cia_cluster_features"] = X.copy()
            st.session_state["cia_cluster_profiles"] = cluster_profiles
            st.session_state["cia_cluster_variance"] = pca.explained_variance_ratio_
            st.session_state["cia_cluster_k"] = k

        if "cia_cluster_results" not in st.session_state:
            st.markdown("""
            <div class="cia-panel"><div class="cia-panel-title">How the algorithm works</div>
            <div class="cia-panel-text">1. Aggregate category counts for each district across the dataset period.<br>
            2. Standardize categories so large-count categories do not automatically dominate.<br>
            3. Use K-Means to group similar profiles.<br>
            4. Use PCA only to display the high-dimensional groups on a 2D chart.<br>
            5. Interpret each group using the cluster profile analysis below.</div></div>
            """, unsafe_allow_html=True)
            st.write("Choose K and click **Run K-Means Analysis**.")
        else:
            results = st.session_state["cia_cluster_results"]
            labels = st.session_state["cia_cluster_labels"]
            X_saved = st.session_state["cia_cluster_features"]
            profiles = st.session_state["cia_cluster_profiles"]
            variance = st.session_state["cia_cluster_variance"]
            actual_k = st.session_state["cia_cluster_k"]

            sizes = results.groupby("Cluster").size().reset_index(name="District profiles")
            largest_cluster = int(sizes["District profiles"].max())
            smallest_cluster = int(sizes["District profiles"].min())
            m1, m2, m3 = st.columns(3)
            m1.metric("District profiles analysed", format_number(len(results)))
            m2.metric("Clusters generated", actual_k)
            m3.metric("Largest cluster size", format_number(largest_cluster))

            heading("Cluster Visualization", "Each point is a district profile. PCA projects all crime-category features into two dimensions for plotting.")
            fig = px.scatter(
                results, x="Component 1", y="Component 2",
                color=results["Cluster"].astype(str),
                hover_data=["state", "district", "Cluster"],
                labels={"color": "Cluster"},
                category_orders={"color": [str(i) for i in range(1, actual_k + 1)]},
            )
            cluster_colors = ["#d9363e", "#b5b5b5", "#d6a0a0", "#b75b61", "#e04444", "#8f2028"]
            fig.update_traces(marker=dict(size=9, opacity=.88, line=dict(color="rgba(7,10,18,.75)", width=1)),
                              hovertemplate="<b>%{customdata[0]}</b><br>District: %{customdata[1]}<br>Cluster: %{customdata[2]}<extra></extra>")
            for i, trace in enumerate(fig.data):
                trace.marker.color = cluster_colors[i % len(cluster_colors)]
            fig.update_layout(
                xaxis_title=f"Principal Component 1 ({variance[0]*100:.1f}% of variance)",
                yaxis_title=f"Principal Component 2 ({variance[1]*100:.1f}% of variance)",
                legend_title="Cluster",
            )
            st.plotly_chart(styled_chart(fig, 530), use_container_width=True)
            st.caption("The axes are PCA projections, not crime categories. Use the cluster profiles below to understand what differentiates groups.")

            col1, col2 = st.columns([1, 2])
            with col1:
                heading("Cluster Sizes")
                sizes["Share of profiles (%)"] = (sizes["District profiles"] / len(results) * 100).round(1)
                st.dataframe(sizes, use_container_width=True, hide_index=True)
            with col2:
                heading("What the clusters mean")
                st.markdown("""
                - **Cluster** is the group assigned by K-Means.
                - **District profiles** is the number of state-district profiles in that group.
                - **PCA components** are visualization coordinates, not interpretable crime counts.
                - **Standardized category profile** shows whether a category is above or below the dataset average within each cluster.
                """)

            heading("Cluster Crime Profiles", "Positive standardized values mean that category counts are higher than the overall district-profile average; negative values mean lower than average.")
            selected_cluster = st.selectbox("Inspect cluster", list(range(1, actual_k + 1)), key="cluster_profile_select")
            profile = profiles.loc[selected_cluster].sort_values(ascending=False)
            top_high = profile.head(5).sort_values()
            top_low = profile.tail(5).sort_values()

            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**Categories comparatively higher in this cluster**")
                high_df = top_high.rename("Standardized average").reset_index()
                high_df.columns = ["Crime category", "Standardized average"]
                st.dataframe(high_df, use_container_width=True, hide_index=True)
                fig = px.bar(
                    high_df, x="Standardized average", y="Crime category",
                    orientation="h", color="Standardized average",
                    color_continuous_scale="Blues",
                    labels={"Standardized average": "Standardized average", "Crime category": "Crime category"},
                )
                fig.update_traces(
                    marker_line_color="rgba(180,205,230,.22)", marker_line_width=1,
                    hovertemplate="<b>%{y}</b><br>Standardized average: %{x:.4f}<extra></extra>",
                )
                fig.update_layout(
                    bargap=.3,
                    xaxis=dict(zeroline=True, zerolinecolor="rgba(220,231,247,.45)", zerolinewidth=1),
                    coloraxis_colorbar=dict(title="Standardized average", thickness=10, len=.72),
                )
                st.plotly_chart(styled_chart(fig, 320), use_container_width=True)
            with p2:
                st.markdown("**Categories comparatively lower in this cluster**")
                low_df = top_low.rename("Standardized average").reset_index()
                low_df.columns = ["Crime category", "Standardized average"]
                st.dataframe(low_df, use_container_width=True, hide_index=True)
                fig = px.bar(
                    low_df, x="Standardized average", y="Crime category",
                    orientation="h", color="Standardized average",
                    color_continuous_scale="Reds_r",
                    labels={"Standardized average": "Standardized average", "Crime category": "Crime category"},
                )
                fig.update_traces(
                    marker_line_color="rgba(230,195,190,.22)", marker_line_width=1,
                    hovertemplate="<b>%{y}</b><br>Standardized average: %{x:.4f}<extra></extra>",
                )
                fig.update_layout(
                    bargap=.3,
                    xaxis=dict(zeroline=True, zerolinecolor="rgba(220,231,247,.45)", zerolinewidth=1),
                    coloraxis_colorbar=dict(title="Standardized average", thickness=10, len=.72),
                )
                st.plotly_chart(styled_chart(fig, 320), use_container_width=True)

            # Profile narrative
            high_names = ", ".join(top_high.sort_values(ascending=False).head(3).index.tolist())
            st.markdown(f"""
            <div class="cia-panel">
              <div class="cia-panel-title">Plain-English interpretation: Cluster {selected_cluster}</div>
              <div class="cia-panel-text">
                Compared with the overall district-profile average, this cluster has relatively higher standardized counts in
                <b>{high_names}</b>. This describes the historical data pattern only; it does not establish causes or adjust for population.
              </div>
            </div>
            """, unsafe_allow_html=True)

            heading("District Assignments")
            show = results.sort_values(["Cluster", "state", "district"])
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.download_button("Download Cluster Assignments", show.to_csv(index=False).encode("utf-8"), "cia_district_clusters.csv", "text/csv")

            # Averages in original units provide a more concrete count-based comparison.
            raw_profile = X_saved.copy()
            raw_profile["Cluster"] = labels + 1
            means = raw_profile.groupby("Cluster").mean()
            with st.expander("View average reported counts by category for each cluster"):
                chosen = st.selectbox("Cluster for raw-count table", list(range(1, actual_k + 1)), key="raw_cluster_select")
                raw_top = means.loc[chosen].sort_values(ascending=False).head(10).rename("Average summed count").reset_index()
                raw_top.columns = ["Crime category", "Average summed count"]
                st.dataframe(raw_top, use_container_width=True, hide_index=True)



        # ALGORITHM LAB

    with comparison_tab:
        st.markdown('<div class="cia-brand">MODEL COMPARISON / UNSUPERVISED LEARNING</div>', unsafe_allow_html=True)
        st.title("Clustering Algorithm Lab")
        st.markdown('<div class="cia-subtitle">Compare clustering methods on the same cleaned district crime profiles.</div>', unsafe_allow_html=True)
        st.info("These algorithms discover similarities in historical aggregated counts. Cluster labels are not crime-type labels and do not predict individual crime or personal risk.")

        @st.cache_data
        def load_algorithm_data():
            raw = load_data("""
                SELECT l.state, l.district, c.crime_type, SUM(f.crime_count) AS crime_count
                FROM Fact_Crime_Stats f
                JOIN Dim_Location l ON f.location_id=l.location_id
                JOIN Dim_Crime_Type c ON f.crime_type_id=c.crime_type_id
                GROUP BY l.state, l.district, c.crime_type
            """)
            if raw.empty:
                return pd.DataFrame()
            wide = raw.pivot_table(
                index=["state", "district"], columns="crime_type",
                values="crime_count", aggfunc="sum", fill_value=0
            ).reset_index()
            wide.columns.name = None
            return wide

        cluster_data = load_algorithm_data()
        if cluster_data.empty:
            st.warning("No data available. Run the ETL pipeline first.")
            st.stop()

        id_cols = ["state", "district"]
        crime_features = [c for c in cluster_data.columns if c not in id_cols]
        Xraw = cluster_data[crime_features].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
        Xraw = Xraw.loc[:, Xraw.nunique() > 1]
        crime_features = list(Xraw.columns)

        if len(cluster_data) < 4 or len(crime_features) < 1:
            st.warning("At least four district profiles and one varying crime category are required.")
            st.stop()

        left, right = st.columns(2)
        with left:
            algorithm = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "Agglomerative clustering"])
            feature_mode = st.selectbox("Feature representation", [
                "Standardized crime counts", "Crime-category proportions"
            ])
        with right:
            if algorithm == "K-Means":
                param_k = st.slider("Number of clusters (K)", 2, min(8, len(cluster_data)-1), min(3, min(8, len(cluster_data)-1)))
                eps = 0.8
                min_samples = 5
            elif algorithm == "DBSCAN":
                eps = st.slider("DBSCAN epsilon", 0.1, 5.0, 1.2, 0.1)
                min_samples = st.slider("Minimum samples", 2, 15, 5)
                param_k = 3
            else:
                param_k = st.slider("Number of clusters", 2, min(8, len(cluster_data)-1), min(3, min(8, len(cluster_data)-1)))
                eps = 0.8
                min_samples = 5

        if feature_mode == "Crime-category proportions":
            totals = Xraw.sum(axis=1).replace(0, 1)
            Xinput = Xraw.div(totals, axis=0)
        else:
            Xinput = Xraw.copy()

        scaler = StandardScaler()
        Xscaled = scaler.fit_transform(Xinput)
        st.caption(f"Profiles: {len(cluster_data):,} · Crime features: {Xinput.shape[1]} · Scaling: StandardScaler")

        if st.button("Run selected algorithm", type="primary"):
            if algorithm == "K-Means":
                model = KMeans(n_clusters=param_k, random_state=42, n_init=10)
                labels = model.fit_predict(Xscaled)
            elif algorithm == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(Xscaled)
            else:
                model = AgglomerativeClustering(n_clusters=param_k, linkage="ward")
                labels = model.fit_predict(Xscaled)

            unique_labels = sorted(set(labels))
            non_noise = labels != -1
            evaluable_labels = set(labels[non_noise])
            if len(evaluable_labels) > 1 and int(non_noise.sum()) > len(evaluable_labels):
                try:
                    score = float(silhouette_score(Xscaled[non_noise], labels[non_noise]))
                except Exception:
                    score = None
            else:
                score = None

            results = cluster_data[id_cols].copy()
            results["Cluster"] = ["Noise / outlier" if x == -1 else f"Cluster {x+1}" for x in labels]
            pca_n = min(2, Xscaled.shape[1], Xscaled.shape[0])
            if pca_n >= 2:
                coords = PCA(n_components=2, random_state=42).fit_transform(Xscaled)
                results["Component 1"] = coords[:, 0]
                results["Component 2"] = coords[:, 1]
                fig = px.scatter(
                    results, x="Component 1", y="Component 2", color="Cluster",
                    hover_data=["state", "district"], title=f"{algorithm} — PCA view"
                )
                fig.update_layout(legend_title="Assigned group")
                st.plotly_chart(styled_chart(fig, 500), use_container_width=True)

            noise_count = int((labels == -1).sum())
            cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
            a, b, c, d = st.columns(4)
            a.metric("District profiles", format_number(len(results)))
            b.metric("Clusters found", cluster_count)
            c.metric("Noise / outliers", format_number(noise_count))
            d.metric("Silhouette score", f"{score:.3f}" if score is not None else "N/A")

            st.caption("Silhouette score ranges from -1 to 1; higher generally indicates better separation. It is unavailable when the algorithm returns fewer than two evaluable clusters or otherwise cannot be scored. For DBSCAN, noise points are excluded from this score.")
            counts = results["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="District profiles")
            counts["Share (%)"] = (counts["District profiles"] / len(results) * 100).round(1)
            col_a, col_b = st.columns([1, 1.5])
            with col_a:
                heading("Cluster sizes")
                st.dataframe(counts, use_container_width=True, hide_index=True)
            with col_b:
                heading("How to interpret the output")
                if algorithm == "DBSCAN":
                    st.markdown("- DBSCAN groups nearby profiles in feature space.\n- **Noise / outlier** means the profile was not assigned to a dense cluster; it does not mean a district is dangerous.")
                elif algorithm == "K-Means":
                    st.markdown("- K-Means assigns every district profile to one of K groups.\n- Try different K values and compare silhouette scores; do not choose solely by cluster size.")
                else:
                    st.markdown("- Agglomerative clustering merges similar profiles into a hierarchy.\n- The selected cluster count controls where the hierarchy is cut.")
            heading("District assignments")
            st.dataframe(results.sort_values(["Cluster", "state", "district"]), use_container_width=True, hide_index=True)
            st.download_button(
                "Download assignments",
                results.to_csv(index=False).encode("utf-8"),
                "cia_clustering_assignments.csv",
                "text/csv",
            )
        else:
            st.markdown("""
            <div class="cia-panel"><div class="cia-panel-title">What this lab compares</div>
            <div class="cia-panel-text">
            <b>K-Means</b> partitions profiles into K groups; <b>DBSCAN</b> identifies dense groups and possible noise points;
            <b>Agglomerative clustering</b> builds groups by repeatedly merging similar profiles.
            Choose an algorithm and feature representation, then run it to see cluster assignments and evaluation metrics.
            </div></div>
            """, unsafe_allow_html=True)


# Footer
st.divider()
st.markdown("""
<div style="text-align:center;padding:8px 0 18px">
  <div style="color:#7dd3fc;font-weight:800;letter-spacing:2px;font-size:.8rem">CIA · CRIME INTELLIGENCE ANALYSIS</div>
  <div style="color:#74869f;font-size:.75rem;margin-top:5px">Historical reported IPC statistics · SQLite Data Warehouse · Data Warehousing & Data Mining Project</div>
</div>
""", unsafe_allow_html=True)
