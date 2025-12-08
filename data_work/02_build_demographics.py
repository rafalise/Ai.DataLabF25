# data_work/02_build_demographics.py
# Purpose: download Georgia county demographics from the Census API,
#          compute % by group, and save a clean CSV for your dashboard.

from pathlib import Path
import os
import requests
import pandas as pd

# ---- basic paths ----
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_work" / "ga_demographics.csv"

# ---- choose ACS 5-year year (use 2023; change to 2022 if needed) ----
YEAR = "2023"
BASE = f"https://api.census.gov/data/{YEAR}/acs/acs5"

# ---- which variables to fetch (counts) ----
# total pop = B03002_001E
# White alone = B02001_002E
# Black or African American alone = B02001_003E
# Asian alone = B02001_005E
# Hispanic or Latino = B03002_012E
VARS = {
    "total":   "B03002_001E",
    "white":   "B02001_002E",
    "black":   "B02001_003E",
    "asian":   "B02001_005E",
    "hispanic":"B03002_012E",
}

def main():
    # Build request
    get_fields = ["NAME"] + list(VARS.values())
    params = {
        "get": ",".join(get_fields),
        "for": "county:*",      # all counties
        "in": "state:13",       # GA = 13
    }

    # Optional: Census API key (fewer rate limits). If you have it, export first:
    #   export CENSUS_API_KEY="your_key_here"
    key = os.getenv("CENSUS_API_KEY")
    if key:
        params["key"] = key

    # Call API
    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Convert to DataFrame
    cols = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=cols)

    # Rename columns to friendly names
    rename_map = {code: name for name, code in VARS.items()}
    df = df.rename(columns=rename_map)

    # Make numbers numeric
    for c in VARS.keys():
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build county FIPS + nice county name
    df["county_fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    df["county_name"] = df["NAME"].str.replace(", Georgia", "", regex=False)

    # Compute percentages
    for c in ["white", "black", "asian", "hispanic"]:
        df[c + "_pct"] = (df[c] / df["total"] * 100).round(2)

    # Keep tidy columns
    keep = [
        "county_fips", "county_name", "total",
        "white_pct", "black_pct", "asian_pct", "hispanic_pct"
    ]
    tidy = df[keep].sort_values("county_fips").reset_index(drop=True)

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(OUT, index=False)
    print(f"✅ Saved: {OUT} (rows={len(tidy)})")

if __name__ == "__main__":
    main()
