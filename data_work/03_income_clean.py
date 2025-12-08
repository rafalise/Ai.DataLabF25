# data_work/03_income_clean.py
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw" / "HDPulse_data_export.csv"
OUT = ROOT / "data_work" / "income_by_fips.csv"

def load_ga_fips_table() -> pd.DataFrame:
    """Build (county_fips, k) where k is normalized county name (lowercase, no ' county')."""
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    geo = requests.get(url, timeout=30).json()
    rows = []
    for f in geo.get("features", []):
        fid = str(f.get("id",""))
        if fid.startswith("13"):
            nm = f.get("properties",{}).get("NAME","")
            rows.append({"county_fips": fid, "county_name": nm})
    df = pd.DataFrame(rows)
    df["k"] = (
        df["county_name"].astype(str).str.strip().str.lower()
          .str.replace(r"\s+county$", "", regex=True)
    )
    return df[["county_fips","k"]]

def money_to_int(x):
    if pd.isna(x): 
        return np.nan
    s = re.sub(r"[^\d]", "", str(x))
    return int(s) if s else np.nan

def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Missing: {RAW}")

    df = pd.read_csv(RAW, dtype=str)
    print("HDPulse columns:", list(df.columns))

    # Expect first column = county names (header like 'Income (Median family income) for Georgia by County')
    county_col = df.columns[0]
    # Find the first 'Unnamed:' column that looks numeric
    cand_value_cols = [c for c in df.columns[1:] if c.lower().startswith("unnamed")]
    if not cand_value_cols:
        # if no 'Unnamed', take the second column as a fallback
        cand_value_cols = [df.columns[1]] if len(df.columns) > 1 else []

    def numeric_frac(col):
        s = pd.to_numeric(
            df[col].astype(str).str.replace(r"[^\d\.]", "", regex=True),
            errors="coerce"
        )
        return (s.notna() & (s>=0)).mean()

    if not cand_value_cols:
        raise ValueError("Could not find a numeric 'Unnamed' column for income values.")
    value_col = max(cand_value_cols, key=numeric_frac)
    print("Chosen value column:", value_col)

    use = df[[county_col, value_col]].copy()
    use.columns = ["county_label", "median_income_raw"]

    # Drop empty / header-like rows
    use = use[use["county_label"].notna() & use["county_label"].astype(str).str.strip().ne("")].copy()

    # Normalize county names -> key k
    use["k"] = (
        use["county_label"].astype(str)
            .str.strip().str.lower()
            .str.replace(r"\s+county$", "", regex=True)
            .str.replace(r",?\s*georgia$", "", regex=True)
            .str.replace(r",?\s*ga$", "", regex=True)
    )
    # Remove obvious aggregate rows
    use = use[~use["k"].isin(["georgia","state of georgia","total"])]

    # Parse currency-like values to int
    use["median_income"] = use["median_income_raw"].apply(money_to_int)

    # Map to GA FIPS
    ga = load_ga_fips_table()
    out = use.merge(ga, on="k", how="left")
    out = out[out["county_fips"].notna()].copy()
    out = out[["county_fips","median_income"]].dropna(subset=["median_income"]).drop_duplicates("county_fips")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"✅ Wrote {OUT} with {len(out)} rows.")

if __name__ == "__main__":
    main()
