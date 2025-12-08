from __future__ import annotations
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# views (two maps only)
from app.views.np_map_view import render as render_np      # accepts metric arg
from app.views.ratio_map_view import render as render_ratio

ROOT = Path(__file__).resolve().parents[1]
NP_FILE     = ROOT / "data_work" / "np_geocoded.csv"
PHYS_FILE   = ROOT / "data_work" / "phys_geocoded.csv"
DEM_FILE    = ROOT / "data_work" / "demographics.csv"     # optional
INCOME_FILE = ROOT / "data_work" / "income_by_fips.csv"   # optional

st.set_page_config(page_title="GA NP Dashboard", layout="wide")

# ---------- helpers ----------
@st.cache_data(show_spinner=False)
def load_ga_geojson():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    geo = requests.get(url, timeout=30).json()
    geo["features"] = [f for f in geo.get("features", []) if str(f.get("id","")).startswith("13")]
    rows = [{"county_fips": str(f.get("id","")).zfill(5),
             "county_name": f.get("properties",{}).get("NAME","")} for f in geo["features"]]
    names = pd.DataFrame(rows)
    return geo, names

@st.cache_data(show_spinner=False)
def load_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = df[df["status"].isin(["matched", "zip_fallback"])].copy()
    df["county_fips"] = df["county_fips"].str.zfill(5)
    return df.groupby("county_fips", as_index=False).size().rename(columns={"size":"count"})

@st.cache_data(show_spinner=False)
def load_optional_csv(path: Path, dtype=None) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=dtype)

def safe_ratio(numer, denom):
    x = pd.to_numeric(numer, errors="coerce")
    y = pd.to_numeric(denom, errors="coerce")
    return pd.Series(np.where((y > 0) & np.isfinite(y), x / y, np.nan))

# ---------- build master ----------
geo, names = load_ga_geojson()

np_counts   = load_points(NP_FILE).rename(columns={"count":"np_count"})
phys_counts = load_points(PHYS_FILE).rename(columns={"count":"doc_count"})

county_tbl = names.merge(np_counts, on="county_fips", how="left") \
                  .merge(phys_counts, on="county_fips", how="left")

county_tbl["np_count"]  = county_tbl["np_count"].fillna(0).astype(int)
county_tbl["doc_count"] = county_tbl["doc_count"].fillna(0).astype(int)
county_tbl["doc_np_ratio"] = safe_ratio(county_tbl["doc_count"], county_tbl["np_count"])

demo = load_optional_csv(DEM_FILE, dtype=str)
if not demo.empty:
    demo["county_fips"] = demo["county_fips"].str.zfill(5)
    for c in ["tot_pop","white_pct","black_pct","asian_pct","aian_pct","nhpi_pct","two_plus_pct","hispanic_pct"]:
        if c in demo.columns:
            demo[c] = pd.to_numeric(demo[c], errors="coerce")
    county_tbl = county_tbl.merge(
        demo[["county_fips","tot_pop","white_pct","black_pct","asian_pct","aian_pct","nhpi_pct","two_plus_pct","hispanic_pct"]],
        on="county_fips", how="left"
    )
    county_tbl["np_density_10k"] = np.where(
        (county_tbl["tot_pop"] > 0) & np.isfinite(county_tbl["tot_pop"]),
        county_tbl["np_count"] / county_tbl["tot_pop"] * 10000.0,
        np.nan
    )
else:
    county_tbl["np_density_10k"] = np.nan

income = load_optional_csv(INCOME_FILE, dtype=str)
if not income.empty and "median_income" in income.columns:
    income["county_fips"] = income["county_fips"].str.zfill(5)
    income["median_income"] = pd.to_numeric(income["median_income"], errors="coerce")
    county_tbl = county_tbl.merge(income[["county_fips","median_income"]], on="county_fips", how="left")

# ---------- UI ----------
st.title("Georgia Nurse Practitioners — Two Maps + Details")

with st.sidebar:
    st.header("Controls")
    show_outlines = st.checkbox("Show county outlines", value=True)
    if st.button("Clear cache"):
        st.cache_data.clear()
        rerun = getattr(st, "rerun", None)
        if callable(rerun):
            rerun()

if "selected_fips" not in st.session_state:
    st.session_state.selected_fips = county_tbl["county_fips"].iloc[0]

# two maps (side-by-side) + details on the right
col_maps, col_details = st.columns([2.2, 1.2])

with col_maps:
    left_metric = "np_density_10k" if county_tbl["np_density_10k"].notna().any() else "np_count"

    m1, m2 = st.columns(2)
    with m1:
        sel1 = render_np(geo=geo, county_tbl=county_tbl, show_outlines=show_outlines, metric=left_metric)
        if sel1: st.session_state.selected_fips = sel1
    with m2:
        sel2 = render_ratio(geo=geo, county_tbl=county_tbl, show_outlines=show_outlines)
        if sel2: st.session_state.selected_fips = sel2

# ---------- details panel ----------
with col_details:
    row = county_tbl.loc[county_tbl["county_fips"] == st.session_state.selected_fips].squeeze()

    st.subheader(f"{row['county_name']} County ({row['county_fips']})")

    a,b,c = st.columns(3)
    a.metric("NPs", f"{int(row['np_count']):,}")
    b.metric("Doctors", f"{int(row['doc_count']):,}")
    c.metric("Doctor:NP", f"{row['doc_np_ratio']:.2f}" if np.isfinite(row["doc_np_ratio"]) else "N/A")

    d1, d2 = st.columns(2)
    dens = f"{row['np_density_10k']:.2f}" if pd.notna(row["np_density_10k"]) else "N/A"
    d1.caption("NP density (per 10k)"); d1.write(dens)
    inc  = f"${int(row['median_income']):,}" if ("median_income" in row and pd.notna(row["median_income"])) else "N/A"
    d2.caption("Median household income"); d2.write(inc)

    st.markdown("### Racial / Ethnic Composition (%)")
    if ("white_pct" in row) and pd.notna(row["white_pct"]):
        labels = ["White","Black","Asian","AIAN","NHPI","Two+","Hispanic"]
        vals = [
            float(row.get("white_pct",0) or 0),
            float(row.get("black_pct",0) or 0),
            float(row.get("asian_pct",0) or 0),
            float(row.get("aian_pct",0) or 0),
            float(row.get("nhpi_pct",0) or 0),
            float(row.get("two_plus_pct",0) or 0),
            float(row.get("hispanic_pct",0) or 0),
        ]
        pie = go.Figure(data=[go.Pie(labels=labels, values=vals, hole=0.35)])
        pie.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=300, legend=dict(orientation="h"))
        st.plotly_chart(pie, use_container_width=True)
    else:
        st.info("No demographics available for this county.")

st.caption("Hover shows county name, NP count, and Doctor:NP ratio. Click either map to update the panel.")
