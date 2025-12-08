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

# 3x3 bivariate palette (blue ↗ × orange ↗). Feel free to tweak.
BIVAR_COLORS = {
    (0,0): "#d4dae6", (0,1): "#aebfd6", (0,2): "#8aa6c8",
    (1,0): "#e2c2b1", (1,1): "#b9a8a7", (1,2): "#6c8db8",
    (2,0): "#e99a6b", (2,1): "#c77d6a", (2,2): "#3f7fcb",
}

def _qbin(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    qs = np.nanquantile(x, [0, 1/3, 2/3, 1])
    # ensure strictly increasing
    for i in range(1, len(qs)):
        if qs[i] <= qs[i-1]:
            qs[i] = qs[i-1] + 1e-9
    return pd.cut(x, bins=qs, labels=[0,1,2], include_lowest=True).astype("int")

def render(*, geo, county_tbl: pd.DataFrame, show_outlines: bool) -> str | None:
    df = county_tbl.copy()
    # build two variables: density (or count) and ratio
    left = "np_density_10k" if df["np_density_10k"].notna().any() else "np_count"
    df["q_left"]  = _qbin(df[left])
    df["q_right"] = _qbin(df["doc_np_ratio"])

    # color per (q_left, q_right)
    df["bivar_color"] = [BIVAR_COLORS[(int(a), int(b))] for a, b in zip(df["q_left"], df["q_right"])]

    fig = px.choropleth(
        df,
        geojson=geo,
        locations="county_fips",
        featureidkey="id",
        color="bivar_color",
        hover_name="county_name",
        hover_data={"county_fips": False, "np_count": True, "doc_np_ratio": True},
        color_discrete_map="identity",
        title="Bivariate: NP density/count × Doctor:NP ratio",
    )
    fig.update_traces(marker_line_width=(0.6 if show_outlines else 0), marker_line_color="#777")
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=430, margin=dict(l=0,r=0,t=36,b=0), coloraxis_showscale=False)

    if HAS_EVENTS:
        ev = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="bivar_map")
        if ev:
            loc = ev[0].get("location")
            if loc: return str(loc).zfill(5)
            idx = int(ev[0].get("pointIndex", 0))
            return df.iloc[idx]["county_fips"]
        return None
    else:
        st.plotly_chart(fig, use_container_width=True, key="bivar_map")
        return None
