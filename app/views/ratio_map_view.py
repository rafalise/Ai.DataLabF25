from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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


def render(*, geo, county_tbl: pd.DataFrame, show_outlines: bool):
    plot = county_tbl.copy()
    cmin, cmax = _robust_range(plot["doc_np_ratio"])

    fig = px.choropleth(
        plot,
        geojson=geo,
        locations="county_fips",
        featureidkey="id",
        color="doc_np_ratio",
        color_continuous_scale="OrRd",
        hover_name="county_name",
        hover_data={"county_fips": False, "np_count": True, "doc_np_ratio": True},
        labels={"np_count": "NPs", "doc_np_ratio": "Doctor:NP"},
        title="Doctor:NP ratio (higher = more doctors per NP)",
    )
    fig.update_coloraxes(cmin=cmin, cmax=cmax)
    fig.update_traces(
        marker_line_width=(0.6 if show_outlines else 0),
        marker_line_color="#777",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=36, b=0))

    if HAS_EVENTS:
        ev = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="ratio_map")
        if ev:
            loc = ev[0].get("location")
            if loc:
                return str(loc).zfill(5)
            idx = int(ev[0].get("pointIndex", 0))
            return plot.iloc[idx]["county_fips"]
        return None
    else:
        st.plotly_chart(fig, use_container_width=True, key="ratio_map")
        return None
