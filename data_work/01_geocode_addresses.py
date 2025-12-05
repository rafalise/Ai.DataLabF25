# data_work/01_geocode_addresses.py
# Day 1: Clean addresses and geocode them to COUNTY FIPS using the
# free U.S. Census Geocoder via censusgeocode (Option A with pinned deps).

from __future__ import annotations

from pathlib import Path
from typing import Optional
import time
import re
import os

import pandas as pd
from censusgeocode import CensusGeocode
from tqdm import tqdm


# -------------------------- Paths / Inputs --------------------------

ROOT = Path(__file__).resolve().parents[1]   # project root (one up from data_work/)
RAW = ROOT / "data_raw"                      # originals live here
OUT = ROOT / "data_work"                     # intermediate outputs + caches
OUT.mkdir(parents=True, exist_ok=True)

# Raw files (adjust names if yours differ)
NP_CSV = RAW / "Georgia_NPs_AddressesNPIs_new(in).csv"
PROTO_XLSX = RAW / "ProtocolAgreements.xlsx"


# -------------------------- Helpers --------------------------

def clean_str(x) -> str:
    """Return a normalized string (no NaN, no weird spaces)."""
    if pd.isna(x):
        return ""
    s = str(x).strip().replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Column-wise clean (avoids applymap deprecation)."""
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].map(clean_str)
    return df


def build_oneline(addr1, addr2, city, state, zipcode) -> str:
    """Compose '123 MAIN ST STE 100, ATLANTA, GA, 30303'."""
    parts: list[str] = []
    for p in (addr1, addr2):
        p = clean_str(p)
        # skip PO Boxes—Census county placement is unreliable
        if p and p.upper().startswith(("PO BOX", "P.O. BOX")):
            continue
        if p:
            parts.append(p)
    city = clean_str(city)
    if city:
        parts.append(city)
    state = clean_str(state) or "GA"  # default GA for this project
    parts.append(state)
    z = clean_str(zipcode)
    if z:
        m = re.search(r"\d{5}", z)    # keep 5-digit ZIP if present
        if m:
            parts.append(m.group(0))
    return ", ".join(parts)


def best_id(row: pd.Series, *fields: str) -> Optional[str]:
    """Pick the first non-empty identifier among given fields (e.g., NPI)."""
    for f in fields:
        v = clean_str(row.get(f, ""))
        if v:
            return v
    return None


def cg_first_county(match: dict) -> tuple[Optional[str], Optional[str], Optional[float], Optional[float]]:
    """Extract county FIPS + county name + lat/lon from a censusgeocode match."""
    if not match:
        return None, None, None, None

    geos = match.get("geographies", {}) if isinstance(match, dict) else {}
    counties = geos.get("Counties") or []
    if counties:
        c = counties[0]
        geoid = c.get("GEOID")      # 5-digit county FIPS = state(2) + county(3)
        name = c.get("NAME")
    else:
        # Fallback: sometimes only Census Tracts exist; still contains STATE+COUNTY
        tracts = geos.get("Census Tracts") or []
        if tracts:
            t = tracts[0]
            geoid = (t.get("STATE") or "") + (t.get("COUNTY") or "")
            name = None
        else:
            return None, None, None, None

    coords = match.get("coordinates", {})
    lon = coords.get("x")
    lat = coords.get("y")
    return geoid, name, lat, lon


# --------- Simple CSV cache so you can stop/resume geocoding ----------

def load_cache(path: Path) -> pd.DataFrame:
    if path and path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame(columns=["oneline", "county_fips", "county_name", "lat", "lon", "status"])


def append_cache(path: Path, chunk: pd.DataFrame) -> None:
    header = not path.exists() or path.stat().st_size == 0
    chunk.to_csv(path, mode="a", header=header, index=False)


# -------------------------- Geocoding --------------------------

def geocode_addresses(
    address_series: pd.Series,
    sleep_sec: float = 0.08,                    # polite but a bit faster than 0.15
    cache_path: Optional[Path] = None,
    save_every: int = 250
) -> pd.DataFrame:
    """
    Geocode each UNIQUE address via censusgeocode, with caching.
    Returns a DataFrame: oneline, county_fips, county_name, lat, lon, status
    """
    cg = CensusGeocode()  # using pinned compatible versions (Option A)

    unique = sorted(set(a for a in address_series if isinstance(a, str) and a.strip()))
    cached = load_cache(cache_path) if cache_path else load_cache(Path())  # empty frame if None
    done = set(cached["oneline"].tolist())

    to_do = [a for a in unique if a not in done]
    rows: list[dict] = []

    for i, addr in enumerate(tqdm(to_do, desc="Geocoding")):
        status = "no_match"
        county_fips = county_name = None
        lat = lon = None
        try:
            matches = cg.onelineaddress(addr)
            if isinstance(matches, dict) and "result" in matches:
                matches = matches.get("result", {}).get("addressMatches", [])
            if matches:
                county_fips, county_name, lat, lon = cg_first_county(matches[0])
                status = "matched" if county_fips else "no_county"
        except Exception as e:
            status = f"error:{type(e).__name__}"

        rows.append({
            "oneline": addr,
            "county_fips": county_fips,
            "county_name": county_name,
            "lat": lat,
            "lon": lon,
            "status": status
        })

        # Periodically flush rows to cache so you never lose progress
        if cache_path and (i + 1) % save_every == 0:
            chunk = pd.DataFrame(rows)
            append_cache(cache_path, chunk)
            cached = pd.concat([cached, chunk], ignore_index=True)
            rows = []
        time.sleep(sleep_sec)

    # flush any remaining rows
    if rows:
        chunk = pd.DataFrame(rows)
        if cache_path:
            append_cache(cache_path, chunk)
            cached = pd.concat([cached, chunk], ignore_index=True)
        else:
            cached = pd.concat([cached, chunk], ignore_index=True)

    # if we used a cache file, ensure we return the full cached data
    if cache_path:
        return load_cache(cache_path)

    return cached


# -------------------------- Main pipeline --------------------------

def main():
    # -------- NPs: read, clean, build one-line addresses --------
    print("Reading NP file…")
    np_df = pd.read_csv(NP_CSV, dtype=str)
    for c in np_df.columns:
        np_df[c] = np_df[c].map(clean_str)
    print("NP columns:", list(np_df.columns))

    # Your actual headers (from your printout)
    addr1_col = "Street1"
    addr2_col = "Street2"
    city_col  = "City"
    state_col = "State"
    zip_col   = "ZIP"
    npi_col   = "NPI"

    # Safety check
    required = [addr1_col, city_col, state_col, zip_col]
    if not all(col in np_df.columns for col in required):
        raise SystemExit(
            f"Missing required NP columns. Expected: {required}. Got: {list(np_df.columns)}"
        )

    np_df["oneline"] = np_df.apply(
        lambda r: build_oneline(r.get(addr1_col), r.get(addr2_col), r.get(city_col), r.get(state_col), r.get(zip_col)),
        axis=1
    )
    np_df["np_id"] = np_df.apply(lambda r: best_id(r, npi_col), axis=1)

    np_addrs = (np_df[["np_id", "oneline"]]
                .dropna(subset=["oneline"])
                .drop_duplicates())
    print(f"NP addresses to geocode: {len(np_addrs)}")

    # -------- Physicians: read, clean, get protocol addresses --------
    print("Reading ProtocolAgreements…")
    proto = pd.read_excel(PROTO_XLSX, sheet_name=0, dtype=str)
    for c in proto.columns:
        proto[c] = proto[c].map(clean_str)

    # Auto-detect common headers (handles 'Protcol Address' typo)
    phy_col  = next((c for c in proto.columns if c.lower().startswith("phy")), None)  # e.g., "PHY#"
    addr_col = next((c for c in proto.columns if "protcol address" in c.lower() or "protocol address" in c.lower()), None)

    if not addr_col:
        raise SystemExit("Could not find 'Protocol Address' column in ProtocolAgreements.xlsx")

    phys_df = (proto[[phy_col, addr_col]]
               .rename(columns={phy_col: "phy_id", addr_col: "oneline"})
               .dropna()
               .drop_duplicates())
    print(f"Physician addresses to geocode: {len(phys_df)}")

    # -------- Geocode with cache & resume --------
    np_cache   = OUT / "np_geocode_cache.csv"
    phys_cache = OUT / "phys_geocode_cache.csv"

    np_geo_map   = geocode_addresses(np_addrs["oneline"],  sleep_sec=0.08, cache_path=np_cache,   save_every=250)
    phys_geo_map = geocode_addresses(phys_df["oneline"],   sleep_sec=0.08, cache_path=phys_cache, save_every=250)

    # -------- Join county info back to each row --------
    np_out   = np_addrs.merge(np_geo_map, on="oneline", how="left")
    phys_out = phys_df.merge(phys_geo_map, on="oneline", how="left")

    # -------- Save results for Day 2 --------
    np_out_path   = OUT / "np_geocoded.csv"
    phys_out_path = OUT / "phys_geocoded.csv"

    np_out.to_csv(np_out_path, index=False)
    phys_out.to_csv(phys_out_path, index=False)

    print(f"Saved: {np_out_path}   ({len(np_out)} rows)")
    print(f"Saved: {phys_out_path} ({len(phys_out)} rows)")

    # -------- Tiny sanity summaries --------
    print("\nNP geocode status:")
    print(np_out["status"].value_counts(dropna=False).head())

    print("\nPhys geocode status:")
    print(phys_out["status"].value_counts(dropna=False).head())


if __name__ == "__main__":
    main()
