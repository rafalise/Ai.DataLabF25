from __future__ import annotations
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


def _robust_range(s: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return (0.0, 1.0)
    lo = float(np.nanquantile(s, 0.05))
    hi = float(np.nanquantile(s, 0.95))
    if not np.isfinite(lo): lo = 0.0
    if (not np.isfinite(hi)) or hi <= lo: hi = lo + 1.0
    return (max(0.0, lo), hi)


def render(*, geo, county_tbl: pd.DataFrame, show_outlines: bool, metric: str = "np_count"):
    """
    Classic GA choropleth (no basemap). Returns selected county_fips (or None).
    metric: "np_count" or "np_density_10k"
    """
    df = county_tbl.copy()

    # Ensure we have a name column
    if "county_name" not in df.columns:
        df["county_name"] = df.get("county_name_x", df.get("county_name_y", ""))

    # numeric Z
    if metric not in df.columns:
        df[metric] = 0
    z = pd.to_numeric(df[metric], errors="coerce").fillna(0)

    # hover text
    npc = pd.to_numeric(df["np_count"], errors="coerce").fillna(0).astype(int)
    ratio = pd.to_numeric(df["doc_np_ratio"], errors="coerce")
    hover = (
        df["county_name"].fillna("") +
        "<br>NPs: " + npc.astype(str) +
        "<br>Doctor:NP: " + ratio.round(2).astype(str)
    )

    cmin, cmax = _robust_range(z)

    fig = go.Figure(go.Choropleth(
        geojson=geo,
        featureidkey="id",
        locations=df["county_fips"],
        z=z,
        colorscale="Blues",
        zmin=cmin, zmax=cmax,
        marker_line_width=(0.6 if show_outlines else 0),
        marker_line_color="#777",
        colorbar_title="NPs" if metric == "np_count" else "NP/10k",
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.update_geos(
        fitbounds="locations",
        visible=False,                 # hide world layers, keep our polygons
        projection_type="mercator",
    )
    fig.update_layout(
        title=("NP density (per 10k)" if metric == "np_density_10k" else "NP count"),
        height=480,
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo_bgcolor="rgba(0,0,0,0)",
    )

    if HAS_EVENTS:
        ev = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key=f"np_{metric}_geo")
        if ev:
            loc = ev[0].get("location")
            if loc:
                return str(loc).zfill(5)
            idx = int(ev[0].get("pointIndex", 0))
            return df.iloc[idx]["county_fips"]
        return None
    else:
        st.plotly_chart(fig, use_container_width=True, key=f"np_{metric}_geo")
        return None
