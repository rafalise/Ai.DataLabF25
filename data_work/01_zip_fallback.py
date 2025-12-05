# data_work/01_zip_fallback.py
from pathlib import Path
import re, json, time, requests
import pandas as pd
import pgeocode

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data_work"
CACHE = WORK / "zip_fallback_cache.json"

FCC_URL = "https://geo.fcc.gov/api/census/block/find"  # returns County FIPS from lat/lon

def extract_zip5(s: str) -> str | None:
    m = re.search(r"(\d{5})(?:-\d{4})?\b", str(s))
    return m.group(1) if m else None

def load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return {}
    return {}

def save_cache(cache: dict):
    CACHE.write_text(json.dumps(cache))

def zip_to_centroid(zip5: str, nomi) -> tuple[float|None, float|None]:
    rec = nomi.query_postal_code(zip5)
    if rec is None or pd.isna(rec.latitude) or pd.isna(rec.longitude):
        return None, None
    # pgeocode returns latitude, longitude (not the other way around)
    return float(rec.latitude), float(rec.longitude)

def fcc_county(lat: float, lon: float, timeout=8) -> tuple[str|None, str|None]:
    """Return (county_fips, county_name) using FCC API; only accept Georgia (FIPS starts '13')."""
    try:
        r = requests.get(FCC_URL, params={"latitude": lat, "longitude": lon, "format": "json"}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        county = data.get("County") or {}
        fips = county.get("FIPS")
        name = county.get("name")
        # keep Georgia only
        if fips and fips.startswith("13"):
            return fips, name
        return None, None
    except Exception:
        return None, None

def apply_zip_fallback(in_csv: str, out_csv: str):
    path_in  = WORK / in_csv
    path_out = WORK / out_csv
    df = pd.read_csv(path_in, dtype=str)

    # rows that still aren’t matched
    miss_mask = ~(df["status"].fillna("").eq("matched"))
    miss = df[miss_mask].copy()
    if miss.empty:
        print(f"{in_csv}: nothing to fallback; writing unchanged copy.")
        df.to_csv(path_out, index=False)
        return

    # pull ZIP5s
    miss["zip5"] = miss["oneline"].map(extract_zip5)
    miss = miss[miss["zip5"].notna()].copy()
    if miss.empty:
        print(f"{in_csv}: no ZIPs found in no_match rows; writing unchanged copy.")
        df.to_csv(path_out, index=False)
        return

    # de-dup ZIPs, use cache to avoid repeat calls
    nomi = pgeocode.Nominatim("us")
    cache = load_cache()
    need = [z for z in miss["zip5"].unique().tolist() if z not in cache]

    print(f"{in_csv}: unique ZIPs to resolve: {len(need)} (cached: {len(cache)})")

    for i, z in enumerate(need, 1):
        lat, lon = zip_to_centroid(z, nomi)
        fips = name = None
        if lat is not None and lon is not None:
            fips, name = fcc_county(lat, lon)
        # store (fips, name, lat, lon) – lat/lon are ZIP centroids
        cache[z] = [fips, name, lat, lon]
        if i % 50 == 0:
            save_cache(cache)
            print(f"  saved checkpoint at {i}/{len(need)}")
        time.sleep(0.05)  # be polite

    save_cache(cache)

    # map back
    vals = miss["zip5"].map(cache.get)
    miss["county_fips_fallback"] = vals.map(lambda v: v[0] if v else None)
    miss["county_name_fallback"] = vals.map(lambda v: v[1] if v else None)
    miss["lat_fallback"] = vals.map(lambda v: v[2] if v else None)
    miss["lon_fallback"] = vals.map(lambda v: v[3] if v else None)

    # apply only when we actually got a GA county fips
    has_ga = miss["county_fips_fallback"].notna()
    for col, fb_col in [
        ("county_fips", "county_fips_fallback"),
        ("county_name", "county_name_fallback"),
        ("lat", "lat_fallback"),
        ("lon", "lon_fallback"),
    ]:
        df.loc[miss.index[has_ga], col] = miss.loc[has_ga, fb_col]

    df.loc[miss.index[has_ga], "status"] = "zip_fallback"

    df.to_csv(path_out, index=False)
    print(f"✅ Wrote {out_csv} with ZIP fallback applied to {has_ga.sum()} rows.")

def main():
    apply_zip_fallback("np_geocoded.csv",   "np_geocoded_zip.csv")
    apply_zip_fallback("phys_geocoded.csv", "phys_geocoded_zip.csv")

if __name__ == "__main__":
    main()
