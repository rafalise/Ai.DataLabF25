# data_work/02_build_demographics.py
# Purpose: Convert your censuscounties.csv into a clean GA county file
# with FIPS, population, and race/ethnicity percentages your app can use.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_IN   = ROOT / "data_raw" / "censuscounties.csv"          # your original file
CLEAN_OUT = ROOT / "data_raw" / "ga_county_demographics.csv" # what app_main.py expects

def main():
    # 1) Load
    df = pd.read_csv(RAW_IN)

    # 2) (Matches your Colab) Keep YEAR == 6 by removing 1..5
    # If your file doesn't have YEAR, comment the next 2 lines.
    if "YEAR" in df.columns:
        df = df[~df["YEAR"].isin([1,2,3,4,5])].copy()

    # 3) Keep only columns we need (adjust names if your CSV differs)
    keep_cols = [
        "COUNTY","CTYNAME","TOT_POP",
        "WA_MALE","WA_FEMALE",           # White Alone
        "BA_MALE","BA_FEMALE",           # Black/African American
        "IA_MALE","IA_FEMALE",           # American Indian/Alaska Native
        "AA_MALE","AA_FEMALE",           # Asian
        "NA_MALE","NA_FEMALE",           # Native Hawaiian/Other Pacific Islander
        "TOM_MALE","TOM_FEMALE",         # Two or more races
        "H_MALE","H_FEMALE",             # Hispanic (ethnicity)
    ]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        print("⚠️ Missing columns in your CSV:", missing)
        # continue with available columns

    cols_present = [c for c in keep_cols if c in df.columns]
    df = df[["COUNTY","CTYNAME","TOT_POP"] + [c for c in cols_present if c not in ("COUNTY","CTYNAME","TOT_POP")]].copy()

    # 4) Ensure numeric
    for c in df.columns:
        if c not in ("COUNTY","CTYNAME"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 5) Sum male+female into *_TOTAL (only where both exist)
    def total_if_available(prefix):
        m = f"{prefix}_MALE"; f = f"{prefix}_FEMALE"; t = f"{prefix}_TOTAL"
        if m in df.columns and f in df.columns:
            df[t] = df[m].fillna(0) + df[f].fillna(0)

    for prefix in ["WA","BA","IA","AA","NA","TOM","H"]:
        total_if_available(prefix)

    # 6) Collapse to one row per county (safety; your file might already be county-level)
    totals_cols = [c for c in df.columns if c.endswith("_TOTAL")]
    out = (df.groupby(["COUNTY","CTYNAME"], as_index=False)
             .agg({**{"TOT_POP":"sum"}, **{c:"sum" for c in totals_cols}}))

    # 7) Build 5-digit county FIPS for Georgia (state FIPS '13' + 3-digit county code)
    out["county_fips"] = out["COUNTY"].apply(lambda x: f"13{int(x):03d}")

    # 8) Percentages (0–100). We’ll compute the three you want for the dashboard.
    # Note: 'H_TOTAL' is an ethnicity (can overlap with race in some datasets).
    # We treat it as its own % because your research question uses it that way.
    out["population"]   = out["TOT_POP"].clip(lower=0)
    safe = out["population"].replace({0: pd.NA})

    def pct(col):
        return (out[col] / safe) * 100 if col in out.columns else pd.NA

    out["pct_white"]    = pct("WA_TOTAL")
    out["pct_black"]    = pct("BA_TOTAL")
    out["pct_hispanic"] = pct("H_TOTAL")

    # 9) Keep the columns your app expects + a readable name
    final_cols = ["county_fips","CTYNAME","population","pct_white","pct_black","pct_hispanic"]
    out = out[final_cols].rename(columns={"CTYNAME":"county_name"})

    # 10) Round percentages for nicer display
    for c in ["pct_white","pct_black","pct_hispanic"]:
        out[c] = out[c].round(2)

    # 11) Save to the path your app already looks for
    CLEAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLEAN_OUT, index=False)
    print(f"✅ Wrote {CLEAN_OUT} with {len(out)} rows")

if __name__ == "__main__":
    main()
