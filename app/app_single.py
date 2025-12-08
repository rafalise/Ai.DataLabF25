from __future__ import annotations
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Optional click-to-select
try:
    from streamlit_plotly_events import plotly_events
    HAS_EVENTS = True
except Exception:
    HAS_EVENTS = False

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
NP_FILE     = ROOT / "data_work" / "np_geocoded.csv"
PHYS_FILE   = ROOT / "data_work" / "phys_geocoded.csv"
DEM_FILE    = ROOT / "data_work" / "demographics.csv"     # optional
INCOME_FILE = ROOT / "data_work" / "income_by_fips.csv"   # optional

st.set_page_config(page_title="GA NP Dashboard (Single Map)", layout="wide")

# ---------- helpers ----------
@st.cache_data(show_spinner=False)
def load_ga_geojson():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    geo = requests.get(url, timeout=30).json()
    geo["features"] = [f for f in geo.get("features", []) if str(f.get("id","")).startswith("13")]
    rows = []
    for f in geo["features"]:
        rows.append({
            "county_fips": str(f.get("id","")).zfill(5),
            "county_name": f.get("properties",{}).get("NAME","")
        })
    return geo, pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def load_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = df[df["status"].isin(["matched", "zip_fallback"])].copy()
    df["county_fips"] = df["county_fips"].str.zfill(5)
    return (df.groupby("county_fips", as_index=False)
              .size().rename(columns={"size":"count"}))

@st.cache_data(show_spinner=False)
def load_optional_csv(path: Path, dtype=None) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=dtype)

def safe_ratio(numer, denom):
    x = pd.to_numeric(numer, errors="coerce")
    y = pd.to_numeric(denom, errors="coerce")
    out = np.where((y > 0) & np.isfinite(y), x / y, np.nan)
    return pd.Series(out)

def robust_range(s: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return (0.0, 1.0)
    lo = float(np.nanquantile(s, 0.05))
    hi = float(np.nanquantile(s, 0.95))
    if not np.isfinite(lo): lo = 0.0
    if (not np.isfinite(hi)) or hi <= lo: hi = lo + 1.0
    return (max(0.0, lo), hi)

def draw_ga_choropleth(*, df, geo, metric, palette, title, show_outlines):
    z = pd.to_numeric(df[metric], errors="coerce").fillna(0)
    npc = pd.to_numeric(df["np_count"], errors="coerce").fillna(0).astype(int)
    ratio = pd.to_numeric(df["doc_np_ratio"], errors="coerce").round(2)

    hover = (
        df["county_name"].fillna("") +
        "<br>NPs: " + npc.astype(str) +
        "<br>Doctor:NP: " + ratio.astype(str)
    )
    cmin, cmax = robust_range(z)

    fig = go.Figure(go.Choropleth(
        geojson=geo,
        featureidkey="id",
        locations=df["county_fips"],
        z=z,
        colorscale=palette,
        zmin=cmin, zmax=cmax,
        marker_line_width=(0.6 if show_outlines else 0),
        marker_line_color="#777",
        colorbar_title=("NPs" if metric=="np_count" else ("NP/10k" if metric=="np_density_10k" else "Doctor:NP")),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator",
    )
    fig.update_layout(
        title=title,
        height=600,                        # <-- big fixed height so it never collapses
        margin=dict(l=0,r=0,t=36,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo_bgcolor="rgba(0,0,0,0)",
    )
    return fig

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
        county_tbl["np_count"]/county_tbl["tot_pop"]*10000.0,
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
st.title("Georgia Nurse Practitioners — Single Map + Details")

with st.sidebar:
    st.header("Controls")
    show_outlines = st.checkbox("Show county outlines", value=True)

    # choose metric
    options = [("NP count","np_count"), ("Doctor:NP ratio","doc_np_ratio")]
    if county_tbl["np_density_10k"].notna().any():
        options.insert(1, ("NP density (per 10k)","np_density_10k"))
    label_to_key = {lbl:key for (lbl,key) in options}
    choice_lbl = st.radio("Color by:", [lbl for (lbl, _) in options], index=0)
    metric = label_to_key[choice_lbl]

    if st.button("Clear cache"):
        st.cache_data.clear()
        (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None))()

if "selected_fips" not in st.session_state:
    st.session_state.selected_fips = county_tbl["county_fips"].iloc[0]

# layout: big map left, details right
col_map, col_details = st.columns([2.6, 1.4])

with col_map:
    palette = "Blues" if metric in ("np_count","np_density_10k") else "OrRd"
    title   = "NP count" if metric=="np_count" else ("NP density (per 10k)" if metric=="np_density_10k" else "Doctor:NP ratio")

    fig = draw_ga_choropleth(
        df=county_tbl, geo=geo, metric=metric, palette=palette, title=title, show_outlines=show_outlines
    )

    if HAS_EVENTS:
        ev = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key=f"single_{metric}")
        if ev:
            loc = ev[0].get("location")
            if loc:
                st.session_state.selected_fips = str(loc).zfill(5)
            else:
                idx = int(ev[0].get("pointIndex", 0))
                st.session_state.selected_fips = county_tbl.iloc[idx]["county_fips"]
    else:
        st.plotly_chart(fig, use_container_width=True, key=f"single_{metric}")

# fallback picker if no click support (tiny helper under the map)
if not HAS_EVENTS:
    with col_map:
        opts = (county_tbl["county_fips"] + " — " + county_tbl["county_name"]).tolist()
        try:
            cur_idx = int(np.where(county_tbl["county_fips"] == st.session_state.selected_fips)[0][0])
        except Exception:
            cur_idx = 0
        picked = st.selectbox("Select county", opts, index=cur_idx, key="fallback_pick")
        st.session_state.selected_fips = picked.split(" — ")[0]

# ---------- details ----------
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
        fig_pie = go.Figure(go.Pie(labels=labels, values=vals, hole=0.35))
        fig_pie.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=300, legend=dict(orientation="h"))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No demographics available for this county.")

st.caption("Hover shows county name, NP count, and Doctor:NP ratio. Click the map to update the panel.")
