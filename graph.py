from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px

# --- paths (same structure as app_dash.py) ---
ROOT = Path(__file__).resolve().parent
NP_FILE       = ROOT / "data_work" / "np_geocoded.csv"
DEM_FILE      = ROOT / "data_work" / "demographics.csv"
TOTPOP_FALLB  = ROOT / "data_raw" / "censuscountiesclean.csv"
INCOME_FILE = ROOT / "data_work" / "income_by_fips.csv"

def load_np_counts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    # keep only matched / zip_fallback like in your app
    df = df[df["status"].isin(["matched", "zip_fallback"])].copy()
    df["county_fips"] = df["county_fips"].str.zfill(5)
    counts = (
        df.groupby("county_fips", as_index=False)
          .size()
          .rename(columns={"size": "np_count"})
    )
    return counts

# ----- load NP counts -----
np_counts = load_np_counts(NP_FILE)

# ----- load demographics (prefer demographics.csv, else fall back to censuscountiesclean.csv) -----
if DEM_FILE.exists():
    demo = pd.read_csv(DEM_FILE, dtype=str)
    demo["county_fips"] = demo["county_fips"].str.zfill(5)
    # assume these are already percentages 0–100
    for c in ["tot_pop", "white_pct"]:
        if c in demo.columns:
            demo[c] = pd.to_numeric(demo[c], errors="coerce")
    demo = demo[["county_fips", "tot_pop", "white_pct"]]
else:
    # fallback to censuscountiesclean.csv
    cc = pd.read_csv(TOTPOP_FALLB, dtype=str)
    # expected columns: FIPS, TOT_POP, WA_TOTAL (white %)
    cc["county_fips"] = cc["FIPS"].astype(str).str.zfill(5)
    cc["tot_pop"]     = pd.to_numeric(cc["TOT_POP"], errors="coerce")
    cc["white_pct"]   = pd.to_numeric(cc["WA_TOTAL"], errors="coerce")
    # keep only GA counties
    cc = cc[cc["county_fips"].str.startswith("13")]
    demo = cc[["county_fips", "tot_pop", "white_pct"]]

# ----- merge NP counts + demographics -----
tbl = np_counts.merge(demo, on="county_fips", how="left")

# ----- compute NP density per 10k -----
tbl["np_count"] = pd.to_numeric(tbl["np_count"], errors="coerce").fillna(0)
tbl["tot_pop"]  = pd.to_numeric(tbl["tot_pop"],  errors="coerce")

tbl["np_density_10k"] = np.where(
    (tbl["tot_pop"] > 0) & np.isfinite(tbl["tot_pop"]),
    tbl["np_count"] / tbl["tot_pop"] * 10000.0,
    np.nan
)

# ----- compute % non-white -----
tbl["white_pct"]   = pd.to_numeric(tbl["white_pct"], errors="coerce")
tbl["nonwhite_pct"] = 100 - tbl["white_pct"]

# drop rows with missing pieces
tbl_clean = tbl.dropna(subset=["np_density_10k", "nonwhite_pct"]).copy()

# ----- print correlation -----
corr = tbl_clean["nonwhite_pct"].corr(tbl_clean["np_density_10k"])
print(f"Correlation between % non-white and NP density per 10k: {corr:.3f}")

# ----- load income and merge -----
if INCOME_FILE.exists():
    inc = pd.read_csv(INCOME_FILE, dtype=str)
    # expect columns: county_fips, median_income
    inc["county_fips"] = inc["county_fips"].astype(str).str.zfill(5)
    inc["median_income"] = pd.to_numeric(inc["median_income"], errors="coerce")
    tbl = tbl.merge(inc[["county_fips", "median_income"]], on="county_fips", how="left")
else:
    tbl["median_income"] = np.nan

# ----- make scatterplot -----
fig = px.scatter(
    tbl_clean,
    x="nonwhite_pct",
    y="np_density_10k",
    labels={
        "nonwhite_pct": "% non-white population",
        "np_density_10k": "NP density (per 10,000 residents)"
    },
    title="NP Density vs. % Non-White Population (Georgia Counties)",
    trendline="ols"  # regression line for visual correlation
)

fig.show()

# ----- income vs NP density -----
tbl_inc = tbl.dropna(subset=["np_density_10k", "median_income"]).copy()

corr_inc = tbl_inc["median_income"].corr(tbl_inc["np_density_10k"])
print(f"Correlation between median income and NP density per 10k: {corr_inc:.3f}")

fig_inc = px.scatter(
    tbl_inc,
    x="median_income",
    y="np_density_10k",
    labels={
        "median_income": "Median household income ($)",
        "np_density_10k": "NP density (per 10,000 residents)"
    },
    title="NP Density vs. Median Household Income (Georgia Counties)",
    trendline="ols"
)

fig_inc.show()
