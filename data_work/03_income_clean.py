# data_work/03_income_clean.py
from __future__ import annotations
from pathlib import Path
import os
import re
import sys
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data_raw" / "HDPulse_data_export.csv"
OUT  = ROOT / "data_work" / "income_by_fips.csv"

# Optional: force a specific column by name if autodetect struggles.
# Example: os.environ["FORCE_INCOME_COL"] = "Unnamed: 3"
FORCE_INCOME_COL = os.environ.get("FORCE_INCOME_COL", "").strip() or None

def load_ga_name_fips():
    """GA county name -> FIPS table from Plotly GeoJSON."""
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    geo = requests.get(url, timeout=30).json()
    feats = [f for f in geo.get("features", []) if str(f.get("id","")).startswith("13")]
    rows = []
    for f in feats:
        fips = str(f.get("id","")).zfill(5)
        nm   = f.get("properties",{}).get("NAME","")
        rows.append({"county_fips": fips, "county_name": nm})
    df = pd.DataFrame(rows)
    df["k"] = (df["county_name"].str.strip().str.lower()
               .str.replace(r"\s+county$", "", regex=True))
    return df[["k","county_fips","county_name"]]

def looks_like_fips(series: pd.Series) -> bool:
    """True if values look like GA county codes (13001..13321-ish)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0: 
        return False
    frac_int   = (np.mod(s, 1) == 0).mean()
    frac_ga    = ((s >= 13001) & (s <= 13321)).mean()
    return (frac_int > 0.9) and (frac_ga > 0.8)

def coerce_income_col(s: pd.Series) -> pd.Series:
    """Strip $, commas; to float; auto-scale 'in thousands' if median < 1000."""
    t = s.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
    t = t.str.replace(r"\*", "", regex=True).str.replace(r"\s*\[.*?\]\s*", "", regex=True)
    vals = pd.to_numeric(t, errors="coerce")
    valid = vals.dropna()
    if len(valid) >= 5 and float(valid.median()) < 1000:
        vals = vals * 1000.0
    return vals

def pick_county_col(df: pd.DataFrame) -> str | None:
    """Choose the column that contains county names."""
    # Prefer a column whose cells end with 'County'
    best, best_score = None, -1
    for c in df.columns:
        s = df[c].astype(str)
        score = s.str.contains(r"[A-Za-z]").mean() - s.str.fullmatch(r"\d+(\.\d+)?").mean()
        # bonus if many rows end with 'County'
        score += 0.5 * s.str.strip().str.endswith("County").mean()
        if score > best_score:
            best, best_score = c, score
    return best

def pick_income_col(df: pd.DataFrame, county_col: str) -> str | None:
    """Pick the most plausible income column, ignoring FIPS-like columns."""
    if FORCE_INCOME_COL and FORCE_INCOME_COL in df.columns:
        return FORCE_INCOME_COL

    candidates = [c for c in df.columns if c != county_col]
    best, best_score = None, -1.0

    for c in candidates:
        raw = df[c]
        # reject clearly text columns
        if raw.astype(str).str.fullmatch(r"\d+(\.\d+)?").mean() < 0.3 and \
           raw.astype(str).str.contains(r"[A-Za-z]").mean() > 0.2:
            continue

        vals = coerce_income_col(raw)
        if looks_like_fips(vals):
            continue  # it's FIPS, not income

        valid = vals.dropna()
        if len(valid) < 20:
            continue

        # plausibility score: how many are in $10k..$200k, plus dispersion
        in_range = ((valid >= 10000) & (valid <= 200000)).mean()
        spread   = float(valid.quantile(0.9) - valid.quantile(0.1))
        score    = in_range + (spread / 100000.0)  # normalize spread a bit

        if score > best_score:
            best, best_score = c, score

    return best

def main():
    if not RAW.exists():
        print(f"❗ Missing file: {RAW}")
        sys.exit(1)

    raw = pd.read_csv(RAW, dtype=str, encoding_errors="ignore").dropna(how="all")

    county_col = pick_county_col(raw)
    income_col = pick_income_col(raw, county_col) if county_col else None

    print("Detected columns:", {"county_col": county_col, "income_col": income_col})
    if not county_col or not income_col:
        print("❗ Could not confidently detect columns. You can set FORCE_INCOME_COL env var to the correct column name.")
        sys.exit(1)

    use = raw[[county_col, income_col]].rename(columns={county_col: "county_raw", income_col: "income_raw"}).copy()
    use = use[use["county_raw"].astype(str).str.contains(r"[A-Za-z]", na=False)]

    # Normalize county key
    use["k"] = (use["county_raw"].astype(str)
                .str.strip().str.lower()
                .str.replace(r"\s+county$", "", regex=True))

    # Clean income
    use["median_income"] = coerce_income_col(use["income_raw"])

    # Map to FIPS via GA names
    name_map = load_ga_name_fips()
    out = use.merge(name_map, on="k", how="inner")

    # Keep one row per county, last non-null
    out = (out.dropna(subset=["median_income"])
              .drop_duplicates(subset=["county_fips"], keep="last"))

    # Clip absurd values
    out["median_income"] = out["median_income"].clip(lower=10000, upper=200000)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out[["county_fips","median_income"]].to_csv(OUT, index=False)
    print(f"✅ Wrote {OUT} with {len(out)} rows.")
    print(out.head(8).to_string(index=False))

    mn, mx, med = out["median_income"].min(), out["median_income"].max(), out["median_income"].median()
    print(f"Range: ${mn:,.0f} – ${mx:,.0f} | median=${med:,.0f}")

if __name__ == "__main__":
    main()
