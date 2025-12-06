# app/app_main.py
from pathlib import Path
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events

# -------- paths --------
ROOT = Path(__file__).resolve().parents[1]
NP_FILE    = ROOT / "data_work" / "np_geocoded.csv"
PHYS_FILE  = ROOT / "data_work" / "phys_geocoded.csv"
DEMO_FILE  = ROOT / "data_raw"  / "ga_county_demographics.csv"  # optional: population & race %

st.set_page_config(page_title="GA Nursing Workforce", layout="wide")
st.title("Georgia Nursing Workforce — County Dashboard")

# -------- helpers --------
def load_clean(path: Path) -> pd.DataFrame:
    """Read geocoded CSV, keep only rows mapped to a county, ensure 5-digit FIPS."""
    df = pd.read_csv(path, dtype=str)
    df = df[df["status"].isin(["matched", "zip_fallback"])].copy()
    df["county_fips"] = df["county_fips"].str.zfill(5)
    return df

def safe_div(num, den):
    try:
        num = float(num)
        den = float(den)
        return num / den if den > 0 else None
    except Exception:
        return None

# -------- 1) Load core data --------
np_df   = load_clean(NP_FILE)
phys_df = load_clean(PHYS_FILE)

np_counts = (np_df.groupby("county_fips", as_index=False)
               .size().rename(columns={"size": "np_count"}))
phys_counts = (phys_df.groupby("county_fips", as_index=False)
                 .size().rename(columns={"size": "phys_count"}))

county = pd.merge(np_counts, phys_counts, on="county_fips", how="outer").fillna(0)
county["np_count"]   = county["np_count"].astype(int)
county["phys_count"] = county["phys_count"].astype(int)
county["doctor_np_ratio"] = county.apply(lambda r: safe_div(r["phys_count"], r["np_count"]), axis=1)

# -------- 2) Optional: demographics (population & race %) --------
has_demo = DEMO_FILE.exists()
if has_demo:
    demo = pd.read_csv(DEMO_FILE, dtype={"county_fips": str})
    demo["county_fips"] = demo["county_fips"].str.zfill(5)
    # expected columns (use what you have): population, pct_white, pct_black, pct_hispanic
    # merge whatever exists; missing cols will remain NaN
    county = county.merge(demo, on="county_fips", how="left")

# If population exists, compute density per 10k; else we’ll use count as fallback
density_available = has_demo and ("population" in county.columns)
if density_available:
    county["np_density_per_10k"] = county.apply(
        lambda r: safe_div(r["np_count"] * 10000, r["population"]), axis=1
    )

# -------- 3) GeoJSON for counties (filter to GA) --------
url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
geo = requests.get(url, timeout=30).json()
geo["features"] = [f for f in geo["features"] if f["id"].startswith("13")]

# build FIPS -> name and attach
names = pd.DataFrame([{"county_fips": f["id"], "county_name": f["properties"].get("NAME", "")}
                      for f in geo["features"]])
county = county.merge(names, on="county_fips", how="left")

# -------- 4) UI controls (top row) --------
default_metric = "NP Density (per 10k)" if density_available else "NP Count"
metric = st.selectbox("Map:", [default_metric, "Doctor:NP Ratio"] if density_available
                               else ["NP Count", "Doctor:NP Ratio"])

left, right = st.columns([3.5, 2.5], gap="large")

# Decide color column + scale per selection
if metric == "Doctor:NP Ratio":
    color_col = "doctor_np_ratio"
    color_scale = "OrRd"
    labels = {"doctor_np_ratio": "Doctor:NP Ratio", "np_count": "NPs", "phys_count": "Doctors"}
else:
    if density_available:
        color_col = "np_density_per_10k"
        labels = {"np_density_per_10k": "NPs per 10k", "np_count": "NPs"}
    else:
        color_col = "np_count"
        labels = {"np_count": "NPs"}
    color_scale = "Blues"

# -------- 5) Map (left) --------
with left:
    # keep just the columns we need for plotting + hover/customdata
    plot_df = county[["county_fips", "county_name", "np_count", "phys_count",
                      "doctor_np_ratio"] + ([color_col] if color_col not in
                                              ["np_count", "phys_count", "doctor_np_ratio"] else [])].copy()

    fig = px.choropleth(
        plot_df,
        geojson=geo,
        locations="county_fips",
        featureidkey="id",
        color=color_col,
        color_continuous_scale=color_scale,
        hover_name="county_name",
        hover_data={
            "county_fips": True,
            "np_count": True,
            "phys_count": "Doctor:NP Ratio" != metric,  # keep hover light on the density view
            "doctor_np_ratio": True,
        },
        labels=labels,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    # IMPORTANT: attach fips as customdata so we can read it on click
    fig.update_traces(customdata=plot_df["county_fips"])

    # plot and capture click
    clicked = plotly_events(fig, click_event=True, hover_event=False,
                            select_event=False, override_height=620)

# -------- 6) Right panel: KPIs + demographics bar (on click) --------
with right:
    st.subheader("County details")
    selected_fips = None
    if clicked:
        # streamlit-plotly-events returns a list of points; grab the first
        point = clicked[0]
        # 'customdata' holds our county_fips
        selected_fips = point.get("customdata")

    if selected_fips is None:
        st.info("Click a county on the map to view details.")
    else:
        row = county.loc[county["county_fips"] == str(selected_fips)]
        if row.empty:
            st.warning("No data for that county.")
        else:
            r = row.iloc[0]
            # KPIs
            k1, k2, k3 = st.columns(3)
            if density_available:
                k1.metric("NP Density (per 10k)",
                          f"{(r['np_density_per_10k'] or 0):.2f}" if pd.notna(r.get("np_density_per_10k")) else "—")
            else:
                k1.metric("NP Count", int(r["np_count"]))

            k2.metric("Doctor:NP", f"{(r['doctor_np_ratio'] or 0):.2f}" if pd.notna(r["doctor_np_ratio"]) else "—")
            k3.metric("NPs", int(r["np_count"]))

            # demographics bar (if available)
            if has_demo and {"pct_white","pct_black","pct_hispanic"}.issubset(row.columns):
                bars = pd.DataFrame({
                    "Group": ["White", "Black", "Hispanic"],
                    "Percent": [r.get("pct_white"), r.get("pct_black"), r.get("pct_hispanic")]
                })
                bars = bars.dropna()
                if not bars.empty:
                    st.markdown("**Racial/Ethnic composition (%)**")
                    st.bar_chart(bars.set_index("Group"))
            else:
                st.caption("Add demographics CSV to show race % here (county_fips, pct_white, pct_black, pct_hispanic, population).")
