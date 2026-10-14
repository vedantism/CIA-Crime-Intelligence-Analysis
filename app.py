import json
import sqlite3
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    color-scheme: dark;
    --bg: #060810;
    --panel: #0d1320;
    --panel-2: #111a2a;
    --line: #26334a;
    --muted: #8290a8;
    --text: #e8edf7;
    --cyan: #6ee7f9;
    --violet: #a78bfa;
    --red: #fb7185;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 75% -12%, rgba(100, 49, 130, .22), transparent 34%),
        radial-gradient(ellipse at 8% 18%, rgba(20, 100, 126, .14), transparent 28%),
        radial-gradient(ellipse at 90% 78%, rgba(125, 35, 61, .09), transparent 25%),
        linear-gradient(145deg, #05070d 0%, #090e19 48%, #080a12 100%);
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
        linear-gradient(rgba(110, 231, 249, .018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(110, 231, 249, .018) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: linear-gradient(to bottom, black, transparent 75%);
}

[data-testid="stHeader"] {
    background: rgba(5, 7, 13, .72);
    border-bottom: 1px solid rgba(130, 144, 168, .08);
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(12, 17, 30, .99), rgba(6, 9, 17, .99));
    border-right: 1px solid rgba(110, 231, 249, .14);
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
    font-family: 'Manrope', sans-serif !important;
    color: #f3f6fc !important;
    letter-spacing: -.045em;
}

h1 {
    font-size: clamp(2.2rem, 4vw, 3.15rem) !important;
    line-height: 1.12 !important;
    font-weight: 800 !important;
    text-shadow: 0 0 34px rgba(110, 231, 249, .10);
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
    box-shadow: 0 0 10px rgba(110, 231, 249, .7);
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
        linear-gradient(135deg, rgba(19, 29, 47, .96), rgba(9, 14, 25, .98));
    border: 1px solid rgba(117, 139, 172, .22);
    border-radius: 15px;
    padding: 20px 21px;
    box-shadow: 0 12px 34px rgba(0, 0, 0, .22);
    transition: border-color .2s ease, transform .2s ease;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(110, 231, 249, .45);
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
    background: rgba(10, 15, 26, .58);
    border: 1px solid rgba(110, 130, 160, .20);
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
    background: linear-gradient(135deg, rgba(18, 27, 44, .95), rgba(9, 14, 25, .98));
    border: 1px solid rgba(110, 130, 160, .24);
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
    background: linear-gradient(90deg, transparent, rgba(110, 231, 249, .65), rgba(167, 139, 250, .55), transparent);
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

div.stButton > button,
div.stDownloadButton > button {
    background: linear-gradient(110deg, #164e63, #3730a3);
    color: #f7fbff;
    border: 1px solid rgba(110, 231, 249, .38);
    border-radius: 10px;
    min-height: 43px;
    font-weight: 700;
    box-shadow: 0 5px 18px rgba(20, 75, 110, .18);
    transition: all .2s ease;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: linear-gradient(110deg, #0e7490, #5b21b6);
    border-color: rgba(110, 231, 249, .75);
    color: white;
    box-shadow: 0 0 22px rgba(110, 231, 249, .14);
    transform: translateY(-1px);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #0d1524;
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
    background: rgba(110, 231, 249, .06);
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
    border: 1px solid rgba(110, 130, 160, .20);
    border-radius: 12px;
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
    st.subheader(title)
    if subtitle:
        st.markdown(f'<div class="cia-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def styled_chart(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(7,10,18,0.2)",
        plot_bgcolor="rgba(7,10,18,0.2)",
        font=dict(family="Arial, sans-serif", color="#dce7f7", size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
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


def render_india_map(selected_year="All"):
    heading("India Crime Heatmap", "State-wise historical reported IPC counts. Hover over a state for details.")
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

    if matched.empty:
        st.warning("Dataset state names did not match the map boundaries.")
        st.dataframe(data[["state", "reported_crimes"]], use_container_width=True, hide_index=True)
        return

    fig = px.choropleth(
        matched, geojson=geojson, locations="map_state",
        featureidkey="properties.cia_state", color="reported_crimes",
        color_continuous_scale=[[0, "#172554"], [.25, "#1d4ed8"], [.5, "#0891b2"], [.75, "#f59e0b"], [1, "#ef4444"]],
        hover_name="state",
        hover_data={"reported_crimes": ":,", "districts_represented": True, "map_state": False},
        labels={"reported_crimes": "Reported IPC counts", "districts_represented": "Districts represented"},
    )
    fig.update_geos(fitbounds="locations", visible=False, showcountries=True, countrycolor="#526780", showcoastlines=False, bgcolor="rgba(0,0,0,0)")
    fig.update_traces(marker_line_color="#8094af", marker_line_width=.7, colorbar=dict(title="Reported counts", tickformat=",", thickness=14, outlinewidth=0))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=5, b=0))
    st.plotly_chart(styled_chart(fig, 620), use_container_width=True, config={"displaylogo": False, "scrollZoom": False})
    st.caption("Color intensity represents reported counts, not population-adjusted rates or live reports.")
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
      <div style="font-size:2.2rem;filter:drop-shadow(0 0 12px rgba(110,231,249,.35))">◈</div>
      <div style="font-size:1.55rem;font-weight:800;color:#f8fafc">CIA</div>
      <div style="color:#7dd3fc;font-size:.82rem;font-weight:600">CRIME INTELLIGENCE ANALYSIS</div>
      <div style="color:#8295b0;font-size:.75rem;margin-top:5px">Historical Crime Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("WORKSPACE", ["Overview", "Crime Analytics", "District Records", "Data Warehouse", "Data Mining"])
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
    st.title("Crime Intelligence Analysis")
    st.markdown('<div class="cia-subtitle">A unified view of historical Indian IPC crime statistics, regional patterns, and category-level trends.</div>', unsafe_allow_html=True)

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
        fig.update_traces(line=dict(width=3), marker=dict(size=7))
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
            st.plotly_chart(styled_chart(fig, 390), use_container_width=True)
    st.markdown('<div class="cia-panel"><div class="cia-panel-title">About this dashboard</div><div class="cia-panel-text">CIA combines a historical crime data warehouse, SQL aggregation, OLAP-style analysis, visual analytics, and unsupervised machine learning.</div></div>', unsafe_allow_html=True)


# CRIME ANALYTICS
elif page == "Crime Analytics":
    st.markdown('<div class="cia-brand">ANALYTICS / EXPLORATION</div>', unsafe_allow_html=True)
    st.title("Crime Analytics")
    st.markdown('<div class="cia-subtitle">Filter historical data and compare regional and crime-category patterns.</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="cia-subtitle">Inspect the relational star schema and explore OLAP-style operations.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="cia-panel"><div class="cia-panel-title">Warehouse architecture</div>
    <div class="cia-panel-text"><b>Dimensions:</b> Dim_Date, Dim_Location, Dim_Crime_Type<br>
    <b>Facts:</b> Fact_Crime_Stats and Fact_District_Totals<br>
    <b>Storage:</b> SQLite relational database<br><b>ETL:</b> Extract, transform, and load from the source CSV</div></div>
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


# DATA MINING
elif page == "Data Mining":
    st.markdown('<div class="cia-brand">MACHINE LEARNING / DATA MINING</div>', unsafe_allow_html=True)
    st.title("District Crime Pattern Clustering")
    st.markdown('<div class="cia-subtitle">Discover groups of districts with similar historical crime-category profiles using K-Means.</div>', unsafe_allow_html=True)

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
        fig.update_traces(marker=dict(size=8, opacity=.82))
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
            fig = px.bar(high_df, x="Standardized average", y="Crime category", orientation="h", color="Standardized average", color_continuous_scale="Blues")
            st.plotly_chart(styled_chart(fig, 320), use_container_width=True)
        with p2:
            st.markdown("**Categories comparatively lower in this cluster**")
            low_df = top_low.rename("Standardized average").reset_index()
            low_df.columns = ["Crime category", "Standardized average"]
            st.dataframe(low_df, use_container_width=True, hide_index=True)
            fig = px.bar(low_df, x="Standardized average", y="Crime category", orientation="h", color="Standardized average", color_continuous_scale="Reds_r")
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


# Footer
st.divider()
st.markdown("""
<div style="text-align:center;padding:8px 0 18px">
  <div style="color:#7dd3fc;font-weight:800;letter-spacing:2px;font-size:.8rem">CIA · CRIME INTELLIGENCE ANALYSIS</div>
  <div style="color:#74869f;font-size:.75rem;margin-top:5px">Historical reported IPC statistics · SQLite Data Warehouse · Data Warehousing & Data Mining Project</div>
</div>
""", unsafe_allow_html=True)
