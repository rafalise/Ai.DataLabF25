# data_work/03_income_clean.py
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw" / "HDPulse_data_export.csv"   # ← moved here
OUT = ROOT / "data_work" / "ga_income.csv"

def main():
    if not RAW.exists():
        print(f"❌ Can’t find: {RAW}")
        # show nearby files to help debug
        data_raw = RAW.parent
        if data_raw.exists():
            print("Files in data_raw/:", [p.name for p in data_raw.iterdir()])
        sys.exit(1)

    # adjust skiprows if your file has 4 header lines; keep as-is if that’s correct
    df = pd.read_csv(RAW, skiprows=4)

    df = df[df["FIPS"] >= 13001].copy()
    df["county_fips"] = df["FIPS"].astype(int).astype(str).str.zfill(5)
    df["median_income"] = (
        df["Value (Dollars)"].astype(str).str.replace(",", "", regex=False).astype(int)
    )

    keep_cols = ["county_fips", "median_income"]
    if "County" in df.columns:
        df = df.rename(columns={"County": "county_name"})
        keep_cols.append("county_name")

    tidy = df[keep_cols].drop_duplicates("county_fips")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(OUT, index=False)
    print(f"✅ Saved: {OUT}  (rows={len(tidy)})")

if __name__ == "__main__":
    main()

