# app/views/combined_view.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

def render(county: pd.DataFrame, geo: dict, show_outlines: bool = True) -> None:
    st.header("Combined — Toggle what the colors mean")

    # Pick color metric
    has_density = ("np_density_per_10k" in county.columns) and county["np_density_per_10k"].notna().any()
    metric = st.radio(
        "Color by",
        ["NP density (per 10k)", "Doctor:NP ratio"] if has_density else ["NP count", "Doctor:NP ratio"],
        horizontal=True
    )

    if metric == "NP density (per 10k)":
        color_col = "np_density_per_10k"
        colorscale = "Blues"
        color_label = "NPs per 10k"
    elif metric == "NP count":
        color_col = "np_count"
        colorscale = "Blues"
        color_label = "NPs"
    else:
        color_col = "doctor_np_ratio"
        colorscale = "OrRd"
        color_label = "Doctor:NP"

    hover_cols = {
        "county_fips": True,
        "np_count": True,
        "phys_count": True,
    }
    if "np_density_per_10k" in county.columns:
        hover_cols["np_density_per_10k"] = ':.1f'
    if "doctor_np_ratio" in county.columns:
        hover_cols["doctor_np_ratio"] = ':.2f'

    fig = px.choropleth(
        county,
        geojson=geo,
        locations="county_fips",
        featureidkey="id",
        color=color_col,
        color_continuous_scale=colorscale,
        hover_name="county_name",
        hover_data=hover_cols,
        labels={color_col: color_label},
    )

    # GA only (no world/US frame)
    fig.update_geos(fitbounds="locations", visible=False)

    if show_outlines:
        fig.update_traces(marker_line_color="#444", marker_line_width=0.5)

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Use the toggle above to switch the color meaning. Hover a county for exact values; "
        "the details panel in other tabs can still be used for deeper breakdowns."
    )
