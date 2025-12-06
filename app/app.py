# app/app.py
from pathlib import Path
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

# --- basic setup ---
ROOT = Path(__file__).resolve().parents[1]
NP_FILE = ROOT / "data_work" / "np_geocoded.csv"

st.set_page_config(page_title="GA NP Map", layout="wide")
st.title("Georgia Nurse Practitioners — Count by County")

# --- 1) load your NP data ---
df = pd.read_csv(NP_FILE, dtype=str)
df = df[df["status"].isin(["matched", "zip_fallback"])].copy()
df["county_fips"] = df["county_fips"].str.zfill(5)

# aggregate to one row per county: number of NPs
county_counts = (
    df.groupby("county_fips", as_index=False)
      .size()
      .rename(columns={"size": "np_count"})
)

# --- 2) get a geojson for US counties and keep only Georgia (FIPS starts with '13') ---
url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
geo = requests.get(url, timeout=30).json()
geo["features"] = [f for f in geo["features"] if f["id"].startswith("13")]

# >>> NEW: build a tiny FIPS -> county NAME table from the geojson <<<
name_rows = [
    {"county_fips": feat["id"], "county_name": feat["properties"].get("NAME", "")}
    for feat in geo["features"]
]
names = pd.DataFrame(name_rows)

# >>> NEW: merge names into your aggregated table <<<
county_counts = county_counts.merge(names, on="county_fips", how="left")

# --- 3) draw the choropleth ---
fig = px.choropleth(
    county_counts,
    geojson=geo,
    locations="county_fips",
    featureidkey="id",
    color="np_count",
    color_continuous_scale="Blues",
    hover_name="county_name",                     # <<< NEW: show readable name
    hover_data={"county_fips": True, "np_count": True},
    labels={"np_count": "NPs"},
)

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

st.plotly_chart(fig, use_container_width=True)
st.caption("Shaded by number of NPs located (matched or ZIP-fallback). Hover shows county name, FIPS, and count.")
