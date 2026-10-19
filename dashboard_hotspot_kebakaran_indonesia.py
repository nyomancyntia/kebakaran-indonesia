# -*- coding: utf-8 -*-
"""Siaga Api — Dashboard Analisis Hotspot Kebakaran Indonesia.

Run with:
    pip install -r requirements_hotspot.txt
    streamlit run dashboard_hotspot_kebakaran_indonesia.py

The built-in dataset is synthetic demo data. For operational analysis, load a CSV
from NASA FIRMS or query the FIRMS Area API with your personal MAP_KEY.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from io import StringIO
from zoneinfo import ZoneInfo
import math

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

APP_TITLE = "Siaga Api"
FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
INDONESIA_BBOX = "94,-11,141,6"  # west,south,east,north
SENSOR_OPTIONS = {
    "VIIRS NOAA-21 (NRT)": "VIIRS_NOAA21_NRT",
    "VIIRS NOAA-20 (NRT)": "VIIRS_NOAA20_NRT",
    "VIIRS Suomi-NPP (NRT)": "VIIRS_SNPP_NRT",
    "MODIS (NRT)": "MODIS_NRT",
}

st.set_page_config(page_title="Siaga Api | Hotspot Indonesia", page_icon="🔥", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background: radial-gradient(ellipse at 80% 0%, #273044 0, #101722 35%, #0a1019 75%); color: #edf3fc; }
    [data-testid="stSidebar"] { background: #0e1622; border-right: 1px solid #263346; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1700px; }
    .brand-kicker { text-transform: uppercase; letter-spacing: .2em; color: #ff9a52; font-size: .72rem; font-weight: 800; }
    .hero-title { font-size: 2.2rem; line-height: 1.12; font-weight: 800; letter-spacing: -.045em; margin: .15rem 0 .45rem; color: #f4f7fc; }
    .hero-copy { color: #9eacc0; font-size: .94rem; margin-bottom: 1.2rem; }
    .demo-alert { padding: 14px 16px; border-radius: 13px; border: 1px solid #72502b; background: linear-gradient(90deg, #3a291d, #211b19); color: #ffddb9; margin: .6rem 0 1.15rem; }
    .live-alert { padding: 12px 16px; border-radius: 13px; border: 1px solid #235847; background: #10291f; color: #b6f1d3; margin: .6rem 0 1.15rem; }
    .section-title { color: #aab8cb; font-size: .76rem; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; margin: 1.4rem 0 .65rem; }
    .metric-card { background: linear-gradient(145deg, #172233, #101824); border: 1px solid #28374a; padding: 17px 18px; border-radius: 16px; min-height: 116px; }
    .metric-label { color: #9faec2; font-size: .78rem; font-weight: 600; }
    .metric-value { color: #f4f7fc; font-weight: 800; font-size: 1.95rem; letter-spacing: -.05em; margin-top: .32rem; line-height: 1.12; }
    .metric-note { color: #8292a9; font-size: .72rem; margin-top: .45rem; }
    div[data-testid="stMetric"] { background: linear-gradient(145deg, #172233, #101824); border: 1px solid #28374a; padding: 15px 16px; border-radius: 15px; }
    div[data-testid="stMetricLabel"] { color: #a4b3c6; }
    div[data-testid="stMetricValue"] { color: #f1f5fa; }
    div[data-testid="stMetricDelta"] { font-size: .75rem; }
    .panel { background: linear-gradient(160deg, #141f2e, #101722); border: 1px solid #263548; border-radius: 16px; padding: 15px; }
    .tiny-note { color: #8495ab; font-size: .78rem; line-height: 1.5; }
    .source-chip { display: inline-block; border-radius: 99px; padding: 5px 10px; background: #1b2a3a; border: 1px solid #304258; color: #c2d1e4; font-size: .74rem; font-weight: 700; }
    h2, h3 { color: #edf3fc !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background-color: #101a28; }
    .stButton button { border-radius: 10px; font-weight: 700; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def now_local() -> datetime:
    return datetime.now(ZoneInfo("Asia/Makassar"))


def classify_region(lat: float, lon: float) -> str:
    """Approximate geographic regions, not official administrative boundaries."""
    if not (-11.5 <= lat <= 6.5 and 93.0 <= lon <= 142.0):
        return "Di luar cakupan"
    if lon < 106.5 and lat < 6.5:
        return "Sumatra"
    if 105.0 <= lon <= 115.0 and -9.5 <= lat <= -5.0:
        return "Jawa"
    if 105.0 <= lon <= 117.5 and lat > -5.0:
        return "Kalimantan" if lon < 117.0 else "Sulawesi"
    if -11.0 <= lat <= -7.0 and 114.0 <= lon <= 130.0:
        return "Bali & Nusa Tenggara"
    if 118.0 <= lon <= 126.0 and lat > -7.0:
        return "Sulawesi" if lon < 124.5 else "Maluku"
    if lon >= 130.0:
        return "Papua"
    if lon >= 126.0:
        return "Maluku"
    if lon >= 115.0 and lat > -5.0:
        return "Kalimantan"
    if lon >= 115.0 and lat <= -5.0:
        return "Bali & Nusa Tenggara"
    return "Indonesia lainnya"


def confidence_label(value) -> str:
    if pd.isna(value):
        return "Tidak diketahui"
    s = str(value).strip().lower()
    if s in {"high", "h", "tinggi"}:
        return "Tinggi"
    if s in {"nominal", "n", "sedang", "medium", "moderate"}:
        return "Sedang"
    if s in {"low", "l", "rendah"}:
        return "Rendah"
    try:
        n = float(s.replace("%", ""))
        if n >= 80:
            return "Tinggi"
        if n >= 50:
            return "Sedang"
        return "Rendah"
    except (TypeError, ValueError):
        return "Tidak diketahui"


def normalize_data(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize common NASA FIRMS CSV field names to a stable internal schema."""
    if df is None or df.empty:
        raise ValueError("File atau respons tidak berisi data.")
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {
        "lat": "latitude", "lon": "longitude", "lng": "longitude",
        "date": "acq_date", "acquisition_date": "acq_date", "acquisition_time": "acq_time",
        "frp_mw": "frp", "confidence_level": "confidence", "day_night": "daynight",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
    for required in ("latitude", "longitude", "acq_date"):
        if required not in df.columns:
            raise ValueError(f"Kolom wajib '{required}' tidak ditemukan. Pastikan CSV memiliki latitude, longitude, dan acq_date.")
    out = pd.DataFrame()
    out["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    out["tanggal"] = pd.to_datetime(df["acq_date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    if "acq_time" in df:
        out["waktu"] = df["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    else:
        out["waktu"] = "—"
    if "confidence" in df:
        out["confidence"] = df["confidence"].map(confidence_label)
    else:
        out["confidence"] = "Tidak diketahui"
    out["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0) if "frp" in df else 0.0
    out["satellite"] = df["satellite"].astype(str) if "satellite" in df else source
    out["instrumen"] = df["instrument"].astype(str) if "instrument" in df else "—"
    if "daynight" in df:
        out["siang_malam"] = df["daynight"].astype(str).str.upper().map({"D": "Siang", "N": "Malam", "DAY": "Siang", "NIGHT": "Malam"}).fillna("Tidak diketahui")
    else:
        out["siang_malam"] = "Tidak diketahui"
    out["wilayah"] = [classify_region(float(lat), float(lon)) if pd.notna(lat) and pd.notna(lon) else "Tidak diketahui" for lat, lon in zip(out.latitude, out.longitude)]
    out["sumber"] = source
    out = out.dropna(subset=["latitude", "longitude", "tanggal"])
    out = out[out["latitude"].between(-90, 90) & out["longitude"].between(-180, 180)]
    out["tanggal"] = pd.to_datetime(out["tanggal"]).dt.date
    return out.reset_index(drop=True)


def create_demo_data(n: int = 650) -> pd.DataFrame:
    """Deterministic illustrative data. Do not interpret these rows as real detections."""
    import numpy as np
    rng = np.random.default_rng(20261019)
    clusters = [
        (-2.5, 102.0, 0.9, 1.7, 0.20),  # Sumatra
        (-0.6, 114.2, 1.0, 1.9, 0.30),  # Kalimantan
        (-7.0, 110.0, 0.6, 1.1, 0.09),  # Java
        (-8.4, 117.1, 0.5, 1.0, 0.06),  # Nusa Tenggara
        (-2.1, 121.0, 0.8, 1.1, 0.13),  # Sulawesi
        (-3.0, 130.0, 0.7, 1.2, 0.06),  # Maluku
        (-4.2, 137.0, 1.1, 1.2, 0.16),  # Papua
    ]
    weights = np.array([c[4] for c in clusters]); weights = weights / weights.sum()
    selected = rng.choice(len(clusters), n, p=weights)
    latitudes, longitudes, dates, times, conf, frp, dn, sats = [], [], [], [], [], [], [], []
    today = now_local().date()
    for idx in selected:
        clat, clon, lspread, ospread, _ = clusters[idx]
        lat = float(np.clip(rng.normal(clat, lspread), -10.5, 5.5))
        lon = float(np.clip(rng.normal(clon, ospread), 95, 140.5))
        latitudes.append(lat); longitudes.append(lon)
        dates.append(today - timedelta(days=int(rng.integers(0, 5))))
        times.append(f"{int(rng.integers(0, 2400)):04d}")
        cat = rng.choice(["Tinggi", "Sedang", "Rendah"], p=[0.18, 0.57, 0.25])
        conf.append(cat)
        frp.append(round(float(rng.lognormal(1.2 if cat == "Tinggi" else 0.3, 1.0)), 2))
        dn.append(str(rng.choice(["Siang", "Malam"])))
        sats.append(str(rng.choice(["NOAA-21", "NOAA-20", "Suomi-NPP"])))
    df = pd.DataFrame({"latitude": latitudes, "longitude": longitudes, "tanggal": dates, "waktu": times,
                       "confidence": conf, "frp": frp, "siang_malam": dn, "satellite": sats,
                       "instrumen": "Simulasi", "sumber": "SIMULASI"})
    df["wilayah"] = [classify_region(a, b) for a, b in zip(df.latitude, df.longitude)]
    return df


def fetch_firms(map_key: str, sensor: str, days: int) -> pd.DataFrame:
    url = f"{FIRMS_AREA_URL}/{map_key.strip()}/{sensor}/{INDONESIA_BBOX}/{int(days)}"
    response = requests.get(url, timeout=60, headers={"User-Agent": "SiagaApiDashboard/1.0"})
    response.raise_for_status()
    text = response.text.strip()
    if not text or text.lower().startswith(("error", "invalid", "unauthorized")):
        raise ValueError("NASA FIRMS mengembalikan pesan kesalahan. Periksa MAP_KEY dan pilih dataset yang tersedia.")
    try:
        df = pd.read_csv(StringIO(text))
    except Exception as exc:
        raise ValueError("Respons NASA FIRMS tidak terbaca sebagai CSV. Periksa MAP_KEY atau coba lagi.") from exc
    return normalize_data(df, f"NASA FIRMS · {sensor}")


def color_for_conf(conf: str) -> str:
    return {"Tinggi": "#ff4d4d", "Sedang": "#ffad43", "Rendah": "#63b6ff", "Tidak diketahui": "#b3bdca"}.get(conf, "#b3bdca")


def make_map(df: pd.DataFrame, show_satellite: bool) -> folium.Map:
    m = folium.Map(location=[-2.2, 118.0], zoom_start=4, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer("CartoDB dark_matter", name="Peta gelap", control=True).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Peta jalan", control=True, show=False).add_to(m)
    if show_satellite:
        try:
            date_for_layer = pd.to_datetime(df["tanggal"]).max().strftime("%Y-%m-%d") if not df.empty else now_local().strftime("%Y-%m-%d")
            gibs_url = ("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
                        f"VIIRS_NOAA21_Thermal_Anomalies_375m_All/default/{date_for_layer}/"
                        "GoogleMapsCompatible_Level9/{z}/{y}/{x}.png")
            folium.TileLayer(tiles=gibs_url, attr="NASA GIBS", name="Anomali termal VIIRS NOAA-21", 
                             overlay=True, control=True, opacity=0.72, max_zoom=9, max_native_zoom=9).add_to(m)
        except Exception:
            pass
    # Rendering all rows can make the map slow; statistics still use the full selected dataset.
    map_df = df if len(df) <= 3500 else df.sample(3500, random_state=42)
    points = folium.FeatureGroup(name=f"Titik hotspot pada peta ({len(map_df):,})", show=True)
    for row in map_df.itertuples(index=False):
        lat, lon = getattr(row, "latitude"), getattr(row, "longitude")
        conf = getattr(row, "confidence", "Tidak diketahui")
        frp_value = float(getattr(row, "frp", 0) or 0)
        radius = max(3, min(8, 3 + math.log1p(max(frp_value, 0)) * 0.65))
        popup = folium.Popup(
            f"<b>Deteksi hotspot</b><br>Tanggal: {getattr(row, 'tanggal')}<br>Waktu (UTC): {getattr(row, 'waktu', '—')}<br>"
            f"Wilayah (perkiraan): {getattr(row, 'wilayah')}<br>Confidence: {conf}<br>FRP: {frp_value:.2f} MW<br>"
            f"Satelit: {getattr(row, 'satellite', '—')}<br>Koordinat: {lat:.4f}, {lon:.4f}<br>"
            f"Sumber: {getattr(row, 'sumber', '—')}", max_width=290)
        folium.CircleMarker(location=[lat, lon], radius=radius, color=color_for_conf(conf),
                            fill=True, fill_color=color_for_conf(conf), fill_opacity=0.82,
                            weight=1, popup=popup, tooltip=f"{conf} · {getattr(row, 'wilayah')}").add_to(points)
    points.add_to(m)
    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    legend_html = """<div style='position: fixed; bottom: 22px; left: 22px; z-index: 9999; background:#101722ee;
    border:1px solid #39485d; border-radius:10px; padding:10px 12px; color:#eaf0f8; font:12px Arial;'>
    <b style='display:block;margin-bottom:7px'>Confidence deteksi</b>
    <div><span style='color:#ff4d4d'>●</span> Tinggi</div><div><span style='color:#ffad43'>●</span> Sedang</div>
    <div><span style='color:#63b6ff'>●</span> Rendah</div><div style='color:#aab6c7;font-size:10px;margin-top:5px'>Ukuran titik mengikuti FRP</div></div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def plot_style(fig, height: int = 300):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=35, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dfe8f4", family="Inter, sans-serif", size=11),
        title_font=dict(size=14, color="#edf3fc"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#2a3749", tickfont=dict(color="#92a2b8"))
    fig.update_yaxes(gridcolor="#253244", zerolinecolor="#253244", tickfont=dict(color="#92a2b8"))
    return fig


def fmt_int(value) -> str:
    return f"{int(value):,}".replace(",", ".")


# ---- Sidebar: data source and controls ----
st.sidebar.markdown("# 🔥 Siaga Api")
st.sidebar.caption("Dashboard analisis hotspot kebakaran Indonesia")
st.sidebar.divider()
source_mode = st.sidebar.radio("Sumber data", ["Demo simulasi", "NASA FIRMS API", "Impor CSV"], index=0)

if "dataset" not in st.session_state:
    st.session_state.dataset = create_demo_data()
    st.session_state.data_mode = "demo"
    st.session_state.dataset_label = "SIMULASI SINTETIS"

if source_mode == "Demo simulasi":
    if st.session_state.get("data_mode") != "demo":
        st.session_state.dataset = create_demo_data()
        st.session_state.data_mode = "demo"
        st.session_state.dataset_label = "SIMULASI SINTETIS"
    st.sidebar.info("Data demo dibuat secara sintetis untuk menunjukkan fitur dashboard. Bukan laporan kebakaran aktual.")
elif source_mode == "NASA FIRMS API":
    map_key = st.sidebar.text_input("NASA FIRMS MAP_KEY", type="password", help="MAP_KEY gratis dari NASA FIRMS; tidak ditulis ke file oleh aplikasi ini.")
    sensor_label = st.sidebar.selectbox("Dataset satelit", list(SENSOR_OPTIONS.keys()), index=0)
    api_days = st.sidebar.slider("Rentang data API", min_value=1, max_value=5, value=3, help="NASA FIRMS Area API membatasi permintaan NRT hingga 1–5 hari per permintaan.")
    if st.sidebar.button("↻ Tarik data terbaru", type="primary", use_container_width=True):
        if not map_key.strip():
            st.sidebar.error("Masukkan MAP_KEY terlebih dahulu.")
        else:
            with st.spinner("Mengambil data dari NASA FIRMS…"):
                try:
                    fresh = fetch_firms(map_key, SENSOR_OPTIONS[sensor_label], api_days)
                    st.session_state.dataset = fresh
                    st.session_state.data_mode = "live"
                    st.session_state.dataset_label = f"NASA FIRMS · {SENSOR_OPTIONS[sensor_label]}"
                    st.sidebar.success(f"Berhasil memuat {len(fresh):,} deteksi.".replace(",", "."))
                except Exception as exc:
                    st.sidebar.error(f"Data gagal dimuat: {exc}")
    st.sidebar.markdown("[Minta MAP_KEY di NASA FIRMS ↗](https://firms.modaps.eosdis.nasa.gov/api/area/)")
elif source_mode == "Impor CSV":
    uploaded = st.sidebar.file_uploader("Unggah CSV hotspot", type=["csv"])
    if uploaded is not None:
        try:
            imported = normalize_data(pd.read_csv(uploaded), "CSV pengguna")
            # Only replace the currently loaded dataset if content is different/newly uploaded.
            upload_fingerprint = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("upload_fingerprint") != upload_fingerprint:
                st.session_state.dataset = imported
                st.session_state.data_mode = "csv"
                st.session_state.dataset_label = f"CSV · {uploaded.name}"
                st.session_state.upload_fingerprint = upload_fingerprint
                st.sidebar.success(f"Berhasil membaca {len(imported):,} baris.".replace(",", "."))
        except Exception as exc:
            st.sidebar.error(f"CSV tidak dapat diproses: {exc}")
    st.sidebar.caption("Kolom minimal: latitude, longitude, acq_date. Kolom confidence, frp, satellite, acq_time dan daynight akan ikut digunakan bila tersedia.")

st.sidebar.divider()
st.sidebar.markdown("**Filter tampilan**")
st.sidebar.caption("Filter di bawah diterapkan pada peta, KPI, grafik, dan tabel.")

base_df = st.session_state.dataset.copy()
if base_df.empty:
    st.error("Belum ada data untuk dianalisis.")
    st.stop()
base_df["tanggal"] = pd.to_datetime(base_df["tanggal"], errors="coerce").dt.date
valid_dates = base_df["tanggal"].dropna()
min_date = min(valid_dates) if not valid_dates.empty else now_local().date()
max_date = max(valid_dates) if not valid_dates.empty else now_local().date()
if min_date > max_date:
    min_date, max_date = max_date, min_date
picked_dates = st.sidebar.date_input("Rentang tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(picked_dates, tuple) and len(picked_dates) == 2:
    start_date, end_date = picked_dates
elif isinstance(picked_dates, tuple) and len(picked_dates) == 1:
    start_date = end_date = picked_dates[0]
else:
    start_date = end_date = picked_dates
regions = sorted([r for r in base_df["wilayah"].dropna().unique().tolist() if r != "Di luar cakupan"])
selected_regions = st.sidebar.multiselect("Wilayah", options=regions, default=regions)
confidence_options = [c for c in ["Tinggi", "Sedang", "Rendah", "Tidak diketahui"] if c in base_df["confidence"].unique()]
selected_conf = st.sidebar.multiselect("Confidence", options=confidence_options, default=confidence_options)
dn_options = [c for c in ["Siang", "Malam", "Tidak diketahui"] if c in base_df["siang_malam"].unique()]
selected_dn = st.sidebar.multiselect("Waktu akuisisi", options=dn_options, default=dn_options)
show_satellite = st.sidebar.checkbox("Tampilkan layer anomali termal NASA GIBS", value=False, help="Layer visual eksternal. Memerlukan internet; ketersediaan tile bisa berbeda dari data hotspot yang sedang difilter.")
st.sidebar.caption("Waktu akuisisi dalam CSV FIRMS umumnya menggunakan UTC.")

filtered = base_df[(base_df["tanggal"] >= start_date) & (base_df["tanggal"] <= end_date)]
if selected_regions:
    filtered = filtered[filtered["wilayah"].isin(selected_regions)]
else:
    filtered = filtered.iloc[0:0]
if selected_conf:
    filtered = filtered[filtered["confidence"].isin(selected_conf)]
else:
    filtered = filtered.iloc[0:0]
if selected_dn:
    filtered = filtered[filtered["siang_malam"].isin(selected_dn)]
else:
    filtered = filtered.iloc[0:0]

# ---- Main page ----
st.markdown('<div class="brand-kicker">Indonesia · Pemantauan titik panas</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Analisis Hotspot Kebakaran</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-copy">Pantau persebaran deteksi termal, bandingkan pola wilayah, dan ekspor hasil sesuai filter analisis.</div>', unsafe_allow_html=True)

if st.session_state.data_mode == "demo":
    st.markdown('<div class="demo-alert">⚠️ <b>Mode demo aktif.</b> Titik dan angka yang tampil adalah data sintetis untuk mendemonstrasikan dashboard, bukan hotspot aktual. Muat data NASA FIRMS atau impor CSV untuk analisis nyata.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="live-alert">✓ <b>Sumber data:</b> {st.session_state.dataset_label} &nbsp;·&nbsp; Data yang dianalisis mengikuti rentang tanggal dan filter aktif.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Ringkasan indikator</div>', unsafe_allow_html=True)
num_hotspots = len(filtered)
high_count = int((filtered["confidence"] == "Tinggi").sum()) if num_hotspots else 0
frp_sum = float(filtered["frp"].sum()) if num_hotspots else 0
regions_count = int(filtered["wilayah"].nunique()) if num_hotspots else 0
avg_frp = float(filtered["frp"].mean()) if num_hotspots else 0
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Total deteksi", fmt_int(num_hotspots), help="Jumlah baris deteksi yang lolos filter, bukan jumlah kejadian kebakaran terkonfirmasi.")
with m2: st.metric("Confidence tinggi", fmt_int(high_count), f"{(high_count / num_hotspots * 100):.1f}% dari hasil" if num_hotspots else "Tidak ada data")
with m3: st.metric("Akumulasi FRP", f"{frp_sum:,.1f} MW".replace(",", "."), help="FRP adalah estimasi daya radiasi api pada titik deteksi, bukan luas area terbakar.")
with m4: st.metric("Wilayah terpantau", fmt_int(regions_count), help="Klasifikasi kawasan geografis perkiraan berdasarkan koordinat, bukan batas administrasi resmi.")

st.markdown('<div class="section-title">Peta persebaran dan kondisi wilayah</div>', unsafe_allow_html=True)
map_col, side_col = st.columns([1.8, 0.9], gap="medium")
with map_col:
    st.markdown("### Peta hotspot Indonesia")
    st.caption(f"{fmt_int(len(filtered))} deteksi sesuai filter · titik peta dapat diklik untuk detail koordinat dan atribut")
    if filtered.empty:
        st.info("Tidak ada hotspot yang cocok dengan filter saat ini. Ubah rentang tanggal atau pilihan wilayah.")
    else:
        map_obj = make_map(filtered, show_satellite)
        st_folium(map_obj, width=None, height=590, key=f"hotspot_map_{st.session_state.data_mode}_{len(filtered)}_{start_date}_{end_date}", returned_objects=[])
    st.markdown('<div class="tiny-note">Catatan: confidence mengacu pada tingkat keyakinan deteksi satelit. Hotspot bukan otomatis bukti kebakaran terkonfirmasi; koordinat tidak menggantikan verifikasi lapangan.</div>', unsafe_allow_html=True)
with side_col:
    st.markdown("### Deteksi per wilayah")
    if filtered.empty:
        st.info("Belum ada data pada filter aktif.")
    else:
        reg_df = filtered.groupby("wilayah", as_index=False).agg(jumlah=("latitude", "size"), frp=("frp", "sum")).sort_values("jumlah", ascending=True)
        fig_region = px.bar(reg_df, x="jumlah", y="wilayah", orientation="h", text="jumlah", title="Jumlah deteksi", color="jumlah", color_continuous_scale=["#7c5133", "#ffad43", "#ff5b41"])
        fig_region.update_traces(textposition="outside", cliponaxis=False)
        fig_region.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Deteksi")
        st.plotly_chart(plot_style(fig_region, max(300, 38 * len(reg_df) + 90)), use_container_width=True, config={"displayModeBar": False})
        if len(reg_df):
            top = reg_df.iloc[-1]
            st.markdown(f"<div class='panel'><div class='metric-label'>Wilayah terbanyak dalam filter</div><div style='font-size:1.35rem;font-weight:800;margin-top:5px'>{top['wilayah']}</div><div class='metric-note'>{fmt_int(top['jumlah'])} deteksi · FRP {top['frp']:.1f} MW</div></div>", unsafe_allow_html=True)

st.markdown('<div class="section-title">Tren dan komposisi deteksi</div>', unsafe_allow_html=True)
trend_col, conf_col, day_col = st.columns([1.3, 0.9, 0.9], gap="medium")
with trend_col:
    if filtered.empty:
        st.info("Tren harian akan tampil setelah data tersedia.")
    else:
        daily = filtered.groupby("tanggal", as_index=False).agg(jumlah=("latitude", "size"), frp=("frp", "sum"))
        daily["tanggal"] = pd.to_datetime(daily["tanggal"])
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=daily["tanggal"], y=daily["jumlah"], mode="lines+markers", name="Deteksi", line=dict(color="#ff9848", width=3), marker=dict(size=7), fill="tozeroy", fillcolor="rgba(255,152,72,.12)"))
        fig_trend.update_layout(title="Tren jumlah hotspot harian", xaxis_title="Tanggal", yaxis_title="Jumlah deteksi")
        st.plotly_chart(plot_style(fig_trend, 325), use_container_width=True, config={"displayModeBar": False})
with conf_col:
    st.markdown("### Confidence")
    if filtered.empty:
        st.info("Tidak ada data.")
    else:
        conf_df = filtered["confidence"].value_counts().rename_axis("confidence").reset_index(name="jumlah")
        color_map = {"Tinggi": "#ff514e", "Sedang": "#ffad43", "Rendah": "#68b9ff", "Tidak diketahui": "#8d9aab"}
        fig_conf = px.pie(conf_df, names="confidence", values="jumlah", hole=0.65, color="confidence", color_discrete_map=color_map)
        fig_conf.update_traces(textposition="inside", textinfo="percent", insidetextfont=dict(color="#101722"))
        fig_conf.update_layout(title="Komposisi", showlegend=True, legend=dict(orientation="h", y=-0.08, x=0), margin=dict(l=0, r=0, t=35, b=18))
        st.plotly_chart(plot_style(fig_conf, 325), use_container_width=True, config={"displayModeBar": False})
with day_col:
    st.markdown("### Siang / malam")
    if filtered.empty:
        st.info("Tidak ada data.")
    else:
        dn_df = filtered["siang_malam"].value_counts().rename_axis("periode").reset_index(name="jumlah")
        dn_map = {"Siang": "#f5b94c", "Malam": "#648ef5", "Tidak diketahui": "#8d9aab"}
        fig_dn = px.pie(dn_df, names="periode", values="jumlah", hole=0.65, color="periode", color_discrete_map=dn_map)
        fig_dn.update_traces(textposition="inside", textinfo="percent", insidetextfont=dict(color="#101722"))
        fig_dn.update_layout(title="Periode akuisisi", showlegend=True, legend=dict(orientation="h", y=-0.08, x=0), margin=dict(l=0, r=0, t=35, b=18))
        st.plotly_chart(plot_style(fig_dn, 325), use_container_width=True, config={"displayModeBar": False})

st.markdown('<div class="section-title">Tabel deteksi hotspot</div>', unsafe_allow_html=True)
if filtered.empty:
    st.info("Tidak ada baris yang sesuai dengan filter aktif.")
else:
    display_df = filtered.sort_values(["tanggal", "waktu"], ascending=[False, False]).copy()
    display_df["koordinat"] = display_df.apply(lambda r: f"{r.latitude:.4f}, {r.longitude:.4f}", axis=1)
    display_df["frp"] = pd.to_numeric(display_df["frp"], errors="coerce").round(2)
    show_cols = ["tanggal", "waktu", "wilayah", "koordinat", "confidence", "frp", "siang_malam", "satellite", "sumber"]
    show_cols = [c for c in show_cols if c in display_df.columns]
    st.dataframe(display_df[show_cols].rename(columns={
        "tanggal": "Tanggal", "waktu": "Waktu (UTC)", "wilayah": "Wilayah (perkiraan)", "koordinat": "Lat, Lon",
        "confidence": "Confidence", "frp": "FRP (MW)", "siang_malam": "Siang/Malam", "satellite": "Satelit", "sumber": "Sumber"
    }), use_container_width=True, hide_index=True, height=380)
    csv_export = display_df[show_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇ Unduh hasil filter (CSV)", data=csv_export, file_name=f"hotspot_indonesia_{start_date}_{end_date}.csv", mime="text/csv", type="primary")

with st.expander("Tentang data, metodologi, dan keterbatasan"):
    st.markdown(
        """
        - **NASA FIRMS** menyediakan deteksi anomali termal dari instrumen satelit. Hasilnya dapat mencakup kebakaran vegetasi, pembakaran terbuka, sumber panas lain, dan potensi false positive.
        - **Confidence** adalah kategori kualitas/keyakinan deteksi sebagaimana disediakan sensor; label dan skala dapat berbeda antarproduk.
        - **FRP (Fire Radiative Power)** adalah perkiraan daya radiasi api, bukan luas kebakaran atau ukuran dampak.
        - **Klasifikasi wilayah** di dashboard ini memakai rentang koordinat geografis kasar. Gunakan geocoding/batas administrasi resmi apabila dibutuhkan rekap provinsi/kabupaten.
        - Peta dasar dan layer NASA GIBS memerlukan koneksi internet. Layer GIBS adalah visualisasi satelit dan dapat berbeda tanggal/produk dari tabel FIRMS yang diimpor.
        - Data **demo** sepenuhnya sintetis. Jangan gunakan angka demo untuk laporan operasional, kebijakan, atau keputusan penanganan kebakaran.

        Sumber: [NASA FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/area/) · [Panduan FIRMS API](https://firms2.modaps.eosdis.nasa.gov/content/academy/data_api/firms_api_use.html)
        """
    )

st.divider()
st.caption(f"SIAGA API · Prototipe analisis | Sumber aktif: {st.session_state.dataset_label} | Waktu aplikasi: {now_local().strftime('%d %b %Y %H:%M WITA')}")
