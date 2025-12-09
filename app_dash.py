# app_dash.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent
NP_FILE       = ROOT / "data_work" / "np_geocoded.csv"
PHYS_FILE     = ROOT / "data_work" / "phys_geocoded.csv"
DEM_FILE      = ROOT / "data_work" / "demographics.csv"          # optional (percents)
TOTPOP_FALLB  = ROOT / "data_raw"  / "censuscountiesclean.csv"   # fallback for pop + percents
CTYTYPE_FILE  = ROOT / "data_work" / "county_type_by_fips.csv"   # optional (type)
INCOME_FILE   = ROOT / "data_work" / "income_by_fips.csv"        # optional (median_income)

# ---------------- Helpers ----------------
def load_ga_geojson():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    geo = requests.get(url, timeout=30).json()
    geo["features"] = [f for f in geo.get("features", []) if str(f.get("id", "")).startswith("13")]
    names = pd.DataFrame(
        {"county_fips": [str(f["id"]).zfill(5) for f in geo["features"]],
         "county_name": [f["properties"].get("NAME", "") for f in geo["features"]]}
    )
    return geo, names

def load_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = df[df["status"].isin(["matched", "zip_fallback"])].copy()
    df["county_fips"] = df["county_fips"].str.zfill(5)
    return (df.groupby("county_fips", as_index=False)
              .size().rename(columns={"size": "count"}))

def safe_ratio(numer, denom):
    x = pd.to_numeric(numer, errors="coerce")
    y = pd.to_numeric(denom, errors="coerce")
    return np.where((y > 0) & np.isfinite(y), x / y, np.nan)

def robust_range(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return (0.0, 1.0)
    lo, hi = float(np.nanquantile(s, 0.05)), float(np.nanquantile(s, 0.95))
    if not np.isfinite(lo): lo = 0.0
    if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
    return (max(0.0, lo), hi)

# ---------------- Build master table ----------------
geo, names = load_ga_geojson()

np_counts   = load_points(NP_FILE).rename(columns={"count": "np_count"})
phys_counts = load_points(PHYS_FILE).rename(columns={"count": "doc_count"})

tbl = names.merge(np_counts, on="county_fips", how="left") \
           .merge(phys_counts, on="county_fips", how="left")
tbl["np_count"]  = tbl["np_count"].fillna(0).astype(int)
tbl["doc_count"] = tbl["doc_count"].fillna(0).astype(int)
tbl["doc_np_ratio"] = safe_ratio(tbl["doc_count"], tbl["np_count"])

# ---- demographics (prefer data_work/demographics.csv, else fall back to censuscountiesclean.csv) ----
if DEM_FILE.exists():
    demo = pd.read_csv(DEM_FILE, dtype=str)
    demo["county_fips"] = demo["county_fips"].str.zfill(5)
    for c in ["tot_pop","white_pct","black_pct","asian_pct","aian_pct","nhpi_pct","two_plus_pct","hispanic_pct"]:
        if c in demo.columns:
            demo[c] = pd.to_numeric(demo[c], errors="coerce")
    tbl = tbl.merge(
        demo[["county_fips","tot_pop","white_pct","black_pct","asian_pct","aian_pct","nhpi_pct","two_plus_pct","hispanic_pct"]],
        on="county_fips", how="left"
    )
else:
    if TOTPOP_FALLB.exists():
        cc = pd.read_csv(TOTPOP_FALLB, dtype=str)
        rename_map = {
            "FIPS":"county_fips",
            "TOT_POP":"tot_pop",
            "WA_TOTAL":"white_pct",
            "BA_TOTAL":"black_pct",
            "AA_TOTAL":"asian_pct",
            "IA_TOTAL":"aian_pct",
            "NA_TOTAL":"nhpi_pct",
            "TOM_TOTAL":"two_plus_pct",
            "H_TOTAL":"hispanic_pct",
        }
        keep = {src: dst for src, dst in rename_map.items() if src in cc.columns}
        cc = cc[list(keep)].rename(columns=keep)
        cc["county_fips"] = cc["county_fips"].astype(str).str.zfill(5)
        for c in cc.columns:
            if c != "county_fips":
                cc[c] = pd.to_numeric(cc[c], errors="coerce")
        cc = cc[cc["county_fips"].str.startswith("13")]
        tbl = tbl.merge(cc, on="county_fips", how="left")

# NP density per 10k when tot_pop present
if "tot_pop" in tbl.columns:
    tot = pd.to_numeric(tbl["tot_pop"], errors="coerce")
    tbl["np_density_10k"] = np.where((tot > 0) & np.isfinite(tot), tbl["np_count"] / tot * 10000.0, np.nan)
else:
    tbl["np_density_10k"] = np.nan

# ---- county type + income (optional) ----
if CTYTYPE_FILE.exists():
    ct = pd.read_csv(CTYTYPE_FILE, dtype=str)
    ct["county_fips"] = ct["county_fips"].str.zfill(5)
    cols = ["county_fips"]
    if "county_type" in ct.columns:
        cols.append("county_type")
    ct = ct[cols].drop_duplicates("county_fips")
    tbl = tbl.merge(ct, on="county_fips", how="left")
else:
    tbl["county_type"] = "Unknown"

if INCOME_FILE.exists():
    inc = pd.read_csv(INCOME_FILE, dtype=str)
    if "county_fips" in inc.columns:
        inc["county_fips"] = inc["county_fips"].astype(str).str.zfill(5)
        if "median_income" in inc.columns:
            inc["median_income"] = pd.to_numeric(inc["median_income"], errors="coerce")
            tbl = tbl.merge(inc[["county_fips", "median_income"]], on="county_fips", how="left")

tbl["county_type"] = tbl["county_type"].fillna("Unknown")

# hover (name, NPs, ratio, type)
ratio_txt = pd.Series(tbl["doc_np_ratio"]).round(2).astype(str)
tbl["hover_text"] = (
    tbl["county_name"] + "<br>"
    + "NPs: " + tbl["np_count"].astype(str) + "<br>"
    + "Doctor:NP ratio: " + ratio_txt + "<br>"
    + "Type: " + tbl["county_type"]
)

# metric choices
metric_options = [{"label": "NP count", "value": "np_count"}]
if tbl["np_density_10k"].notna().any():
    metric_options.append({"label": "NP density (per 10k)", "value": "np_density_10k"})
metric_options.append({"label": "Doctor:NP ratio", "value": "doc_np_ratio"})

DEFAULT_FIPS = tbl["county_fips"].iloc[0]

# ---------------- Figure builders ----------------
def make_map(df: pd.DataFrame, geojson, metric: str) -> go.Figure:
    plot = df.copy()
    if metric not in plot.columns:
        plot[metric] = 0
    plot[metric] = pd.to_numeric(plot[metric], errors="coerce").fillna(0)
    cmin, cmax = robust_range(plot[metric])
    palette = "Blues" if metric in ("np_count", "np_density_10k") else "OrRd"

    fig = px.choropleth(
        plot,
        geojson=geojson,
        locations="county_fips",
        featureidkey="id",
        color=metric,
        color_continuous_scale=palette,
        hover_name="county_name",
        hover_data={}
    )
    # always show outlines
    fig.update_traces(marker_line_width=0.6, marker_line_color="#777", hovertemplate=plot["hover_text"])
    fig.update_coloraxes(cmin=cmin, cmax=cmax)
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        height=600, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def make_pie(row: pd.Series) -> go.Figure | None:
    needed = ["white_pct","black_pct","asian_pct","aian_pct","nhpi_pct","two_plus_pct","hispanic_pct"]
    if not all(k in row.index for k in needed) or pd.isna(row.get("white_pct")):
        return None

    labels = ["White","Black","Asian","AIAN","NHPI","Two+","Hispanic"]
    vals = [float(row.get(k, 0) or 0) for k in needed]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=vals, hole=0.36,
        sort=False,
        textinfo="percent",
        textposition="inside",
        insidetextorientation="radial",
        marker=dict(line=dict(color="white", width=1)),
        hovertemplate="%{label}: %{value:.2f}%<extra></extra>"
    )])
    # lock size + keep legend slightly lower
    fig.update_layout(
        autosize=False, width=420, height=320,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=-0.22, x=0.0, xanchor="left")
    )
    return fig

# ---------------- Dash app ----------------
app = Dash(__name__)
app.title = "Georgia NP Dashboard"
server = app.server

app.layout = html.Div(
    style={"padding": "16px", "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"},
    children=[
        html.H1("Georgia Nurse Practitioners — County View", style={"margin": "0 0 12px 0"}),

        # Color-by radio buttons
        html.Div([
            html.Strong("Color by:", style={"display": "block", "marginBottom": "4px"}),
            dcc.RadioItems(
                id="metric",
                options=metric_options,
                value=metric_options[0]["value"],
                inline=True
            ),
        ], style={"marginBottom": "8px"}),

        # Vertical stack: label, dropdown, title link
        html.Div([
            html.Strong("County type filter:", style={"display": "block", "marginBottom": "4px"}),
            dcc.Dropdown(
                id="type_filter",
                options=[{"label": "All", "value": "All"}] +
                        [{"label": t, "value": t} for t in sorted(tbl["county_type"].dropna().unique())],
                value="All", clearable=False, style={"width": "240px"}
            ),
            html.A(
                "NP Workforce by County (Count)",
                id="map_title",
                href="#",
                style={"display": "block", "marginTop": "8px", "fontWeight": "600", "color": "#1f77b4"}
            ),
        ], style={"display": "flex", "flexDirection": "column", "alignItems": "flex-start",
                  "gap": "6px", "marginBottom": "8px"}),

        html.Div([
            # Map
            html.Div([
                dcc.Graph(id="ga_map", style={"height": "600px", "width": "100%"})
            ], style={"flex": "2", "minWidth": "540px", "paddingRight": "18px"}),

            # Details panel
            html.Div([
                html.H3(id="detail_title"),
                html.Div([
                    html.Div([html.Div("NPs", style={"fontSize":"12px","color":"#999"}),
                              html.Div(id="d_np", style={"fontSize":"28px"})], style={"flex":"1"}),
                    html.Div([html.Div("Doctors", style={"fontSize":"12px","color":"#999"}),
                              html.Div(id="d_doc", style={"fontSize":"28px"})], style={"flex":"1"}),
                    html.Div([html.Div("Doctor:NP", style={"fontSize":"12px","color":"#999"}),
                              html.Div(id="d_ratio", style={"fontSize":"28px"})], style={"flex":"1"}),
                ], style={"display":"flex","gap":"16px","marginBottom":"8px"}),

                html.Div([
                    html.Div([html.Div("NP density (per 10k)", style={"fontSize":"12px","color":"#999"}),
                              html.Div(id="d_density")], style={"flex":"1"}),
                    html.Div([html.Div("Median household income", style={"fontSize":"12px","color":"#999"}),
                              html.Div(id="d_income")], style={"flex":"1"}),
                ], style={"display":"flex","gap":"16px","marginBottom":"8px"}),

                html.Div([html.Div("County type", style={"fontSize":"12px","color":"#999"}),
                          html.Div(id="d_type")], style={"marginBottom":"12px"}),

                html.H4("Racial / Ethnic Composition (%)"),
                dcc.Graph(id="pie", style={"height":"340px", "width":"420px"}),
            ], style={"flex":"1", "minWidth":"360px"})
        ], style={"display": "flex", "gap": "12px"}),

        dcc.Store(id="selected_fips", data=DEFAULT_FIPS),
    ]
)

# ---------------- Callbacks ----------------
@app.callback(
    Output("ga_map", "figure"),
    Output("map_title", "children"),
    Input("metric", "value"),
    Input("type_filter", "value"),
)
def update_map(metric, type_filter):
    df = tbl.copy()
    if type_filter and type_filter != "All":
        df = df[df["county_type"].fillna("Unknown") == type_filter]

    fig = make_map(df, geo, metric)

    title_map = {
        "np_count": "NP Workforce by County (Count)",
        "np_density_10k": "NP Workforce by County (per 10k)",
        "doc_np_ratio": "Doctor:NP Ratio by County",
    }
    return fig, title_map.get(metric, "NP Workforce by County")

@app.callback(
    Output("selected_fips", "data"),
    Input("ga_map", "clickData"),
    State("selected_fips", "data"),
    prevent_initial_call=True
)
def pick_county(clickData, cur):
    if clickData and "points" in clickData and clickData["points"]:
        loc = clickData["points"][0].get("location")
        if loc:
            return str(loc).zfill(5)
    return cur

@app.callback(
    Output("detail_title", "children"),
    Output("d_np", "children"),
    Output("d_doc", "children"),
    Output("d_ratio", "children"),
    Output("d_density", "children"),
    Output("d_income", "children"),
    Output("d_type", "children"),
    Output("pie", "figure"),
    Input("selected_fips", "data")
)
def update_details(fips):
    row = tbl.loc[tbl["county_fips"] == fips].squeeze()
    title   = f"{row['county_name']} County"
    d_np    = f"{int(row['np_count']):,}"
    d_doc   = f"{int(row['doc_count']):,}"
    d_ratio = f"{row['doc_np_ratio']:.2f}" if np.isfinite(row["doc_np_ratio"]) else "N/A"
    d_dens  = f"{row['np_density_10k']:.2f}" if pd.notna(row.get("np_density_10k", np.nan)) else "N/A"
    d_inc   = f"${int(row['median_income']):,}" if pd.notna(row.get("median_income", np.nan)) else "N/A"
    d_type  = str(row.get("county_type", "Unknown"))

    pie_fig = make_pie(row)
    if pie_fig is None:
        pie_fig = go.Figure(layout=dict(
            autosize=False, width=420, height=320,
            annotations=[dict(text="No demographics available", x=0.5, y=0.5, showarrow=False)],
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", y=-0.22, x=0.0, xanchor="left")
        ))
    return title, d_np, d_doc, d_ratio, d_dens, d_inc, d_type, pie_fig

if __name__ == "__main__":
    app.run(debug=True)
