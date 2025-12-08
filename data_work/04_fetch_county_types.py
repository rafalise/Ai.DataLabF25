# data_work/04_fetch_county_types.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parents[1]
RAW_DIR   = ROOT / "data_raw"
OUT_FILE  = ROOT / "data_work" / "county_type_by_fips.csv"

def _norm_fips(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d{5})", expand=False).str.zfill(5)

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowmap = {c.lower(): c for c in df.columns}
    for want in candidates:
        if want in df.columns: return want
        if want.lower() in lowmap: return lowmap[want.lower()]
    return None

def _load_first_matching(patterns: list[str]) -> pd.DataFrame | None:
    for pat in patterns:
        for p in RAW_DIR.glob(pat):
            try:
                if p.suffix.lower() in [".xls", ".xlsx"]:
                    return pd.read_excel(p, dtype=str)
                return pd.read_csv(p, dtype=str, low_memory=False)
            except Exception:
                pass
    return None

def read_nchs() -> pd.DataFrame | None:
    df = _load_first_matching(["*nchs*.xls*", "*nchs*.csv*"])
    if df is None: return None
    fips = _find_col(df, ["FIPS", "FIPS Code", "county_fips", "Geo_FIPS"])
    code = _find_col(df, ["NCHS", "2013 Code", "NCHS Code", "NCHS_2013", "Code",
                          "NCHS Urban-Rural Classification", "Description", "2013 Description"])
    if fips is None or code is None: return None
    out = pd.DataFrame({
        "county_fips": _norm_fips(df[fips]),
        "raw_code": df[code].astype(str).str.strip()
    }).dropna(subset=["county_fips"]).drop_duplicates("county_fips")
    num = pd.to_numeric(out["raw_code"], errors="coerce")
    if num.notna().any():
        out["county_type"] = "Unknown"
        out.loc[num.between(1,2), "county_type"] = "Urban"
        out.loc[num.between(3,4), "county_type"] = "Suburban"
        out.loc[num.between(5,6), "county_type"] = "Rural"
    else:
        txt = out["raw_code"].str.lower()
        out["county_type"] = "Unknown"
        out.loc[txt.str.contains(r"large|central|large metro", na=False), "county_type"] = "Urban"
        out.loc[txt.str.contains(r"medium metro|small metro", na=False), "county_type"] = "Suburban"
        out.loc[txt.str.contains(r"micropolitan|noncore|non-core", na=False), "county_type"] = "Rural"
    out["source"] = "NCHS2013"
    return out[["county_fips","county_type","source","raw_code"]]

def read_rucc() -> pd.DataFrame | None:
    df = _load_first_matching(["*rucc*.xls*", "*rucc*.csv*"])
    if df is None: return None
    fips = _find_col(df, ["FIPS", "county_fips", "FIPS code"])
    rucc = _find_col(df, ["RUCC_2013", "RUCC 2013", "RUCC", "RUCC_2023"])
    if fips is None or rucc is None: return None
    out = pd.DataFrame({
        "county_fips": _norm_fips(df[fips]),
        "raw_code": pd.to_numeric(df[rucc], errors="coerce")
    }).dropna(subset=["county_fips"]).drop_duplicates("county_fips")
    out["county_type"] = "Unknown"
    out.loc[out["raw_code"].between(1,3), "county_type"] = "Urban"
    out.loc[out["raw_code"].between(4,6), "county_type"] = "Suburban"
    out.loc[out["raw_code"].between(7,9), "county_type"] = "Rural"
    out["source"] = "RUCC"
    return out[["county_fips","county_type","source","raw_code"]]

def read_income_from_hdpulse() -> pd.DataFrame | None:
    p = RAW_DIR / "HDPulse_data_export.csv"
    if not p.exists(): return None
    df = pd.read_csv(p, dtype=str, low_memory=False)
    fips = _find_col(df, ["FIPS","FIPS Code","CountyFIPS","county_fips","Geo_FIPS"])
    inc  = _find_col(df, ["Median household income","Median_Household_Income",
                          "Median Income","Median_HH_Income","median_income"])
    if fips is None or inc is None: return None
    out = pd.DataFrame({
        "county_fips": _norm_fips(df[fips]),
        "median_income": pd.to_numeric(
            df[inc].astype(str).str.replace(r"[^\d.]", "", regex=True),
            errors="coerce"
        )
    }).dropna(subset=["county_fips"]).drop_duplicates("county_fips")
    return out

def fallback_from_censuscounties() -> pd.DataFrame | None:
    """Use data_raw/censuscountiesclean.csv (FIPS, TOT_POP) to make a provisional county_type."""
    p = RAW_DIR / "censuscountiesclean.csv"
    if not p.exists(): return None
    df = pd.read_csv(p, dtype=str)
    fips = _find_col(df, ["FIPS","county_fips"])
    tot  = _find_col(df, ["TOT_POP","tot_pop","TotalPop"])
    if fips is None or tot is None: return None
    tmp = pd.DataFrame({
        "county_fips": _norm_fips(df[fips]),
        "tot_pop": pd.to_numeric(df[tot], errors="coerce")
    }).dropna(subset=["county_fips"]).drop_duplicates("county_fips")
    # GA-only rows already (FIPS starts with 13) — if not, filter:
    tmp = tmp[tmp["county_fips"].str.startswith("13")]
    # Quantile thresholds: bottom 30% rural, mid 50% suburban, top 20% urban
    q30, q80 = np.nanquantile(tmp["tot_pop"], [0.30, 0.80])
    tmp["county_type"] = np.select(
        [
            tmp["tot_pop"] <= q30,
            (tmp["tot_pop"] > q30) & (tmp["tot_pop"] <= q80),
            tmp["tot_pop"] > q80
        ],
        ["Rural","Suburban","Urban"],
        default="Unknown"
    )
    tmp["source"] = "Fallback_PopQuantiles"
    tmp["raw_code"] = tmp["tot_pop"]
    return tmp[["county_fips","county_type","source","raw_code"]]

def main():
    nchs = read_nchs()
    rucc = read_rucc()

    if nchs is not None:
        combo = nchs
        if rucc is not None:
            missing = rucc[~rucc["county_fips"].isin(combo["county_fips"])]
            combo = pd.concat([combo, missing], ignore_index=True)
    elif rucc is not None:
        combo = rucc
    else:
        # Fallback using your censuscountiesclean.csv
        fb = fallback_from_censuscounties()
        if fb is None:
            print("❗ No NCHS/RUCC file found and no censuscountiesclean.csv fallback available.")
            print("   Put one of these in data_raw/:")
            print("     • NCHS 2013 (Excel/CSV) or • RUCC 2013/2023 (Excel/CSV)")
            print("   OR provide data_raw/censuscountiesclean.csv with FIPS & TOT_POP.")
            return
        combo = fb

    income = read_income_from_hdpulse()
    if income is not None:
        combo = combo.merge(income, on="county_fips", how="left")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combo.to_csv(OUT_FILE, index=False)
    print(f"✅ Wrote {OUT_FILE} with {len(combo)} rows.")
    print("   Columns:", list(combo.columns))

if __name__ == "__main__":
    main()
