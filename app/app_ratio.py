# app/app_ratio.py
from pathlib import Path
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

# ---- paths ----
ROOT = Path(__file__).resolve().parents[1]
NP_FILE   = ROOT / "data_work" / "np_geocoded.csv"
PHYS_FILE = ROOT / "data_work" / "phys_geocoded.csv"

st.set_page_config(page_title="GA Doctor:NP Ratio", layout="wide")
st.title("Georgia — Doctor:NP Ratio by County")

# ---- 1) Load and filter data (keep only rows that mapped to a county) ----
def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = df[df["status"].isin(["matched", "zip_fallback"])].copy()
    df["county_fips"] = df["county_fips"].str.zfill(5)  # ensure 5-digit FIPS
    return df

np_df   = load_clean(NP_FILE)
phys_df = load_clean(PHYS_FILE)

# ---- 2) Aggregate to county level: counts of NPs and Physicians ----
np_counts = (
    np_df.groupby("county_fips", as_index=False)
         .size()
         .rename(columns={"size": "np_count"})
)
phys_counts = (
    phys_df.groupby("county_fips", as_index=False)
           .size()
           .rename(columns={"size": "phys_count"})
)

# Outer-join so counties that appear in one file but not the other still show up
county = pd.merge(np_counts, phys_counts, on="county_fips", how="outer").fillna(0)

# Ensure numeric types for safe math
county["np_count"] = county["np_count"].astype(int)
county["phys_count"] = county["phys_count"].astype(int)

# Doctor:NP ratio (avoid divide-by-zero by leaving None where np_count == 0)
county["doctor_np_ratio"] = county.apply(
    lambda r: (r["phys_count"] / r["np_count"]) if r["np_count"] > 0 else None,
    axis=1
)

# ---- 3) Get US counties GeoJSON and filter to Georgia (FIPS starts with "13") ----
url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
geo = requests.get(url, timeout=30).json()
geo["features"] = [f for f in geo["features"] if f["id"].startswith("13")]

# Build a tiny lookup so we can show readable county names on hover
name_rows = [
    {"county_fips": f["id"], "county_name": f["properties"].get("NAME", "")}
    for f in geo["features"]
]
names = pd.DataFrame(name_rows)
county = county.merge(names, on="county_fips", how="left")

# ---- 4) Draw the choropleth (sequential red/orange scale) ----
fig = px.choropleth(
    county,
    geojson=geo,
    locations="county_fips",
    featureidkey="id",
    color="doctor_np_ratio",
    color_continuous_scale="OrRd",
    hover_name="county_name",
    hover_data={
        "county_fips": True,
        "np_count": True,
        "phys_count": True,
        "doctor_np_ratio": True,
    },
    labels={
        "doctor_np_ratio": "Doctor:NP Ratio",
        "np_count": "NPs",
        "phys_count": "Doctors",
    },
)

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Doctor:NP ratio = physicians / NPs (blank if NP count is zero). "
    "Tip: keep this as a secondary view; your main map should be NP density per 10k."
)
