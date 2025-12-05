from pathlib import Path
import re, time, sys, json
import pandas as pd
from censusgeocode import CensusGeocode
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data_work"
CHECKPOINTS = WORK / "retry_checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)

def log(msg): print(msg, flush=True)

# ---------------- cleaning helpers ----------------
def clean_spaces(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)  # CirCanton -> Cir Canton
    return s

def strip_leading_org(s: str) -> str:
    s = clean_spaces(s)
    m = re.search(r"\d", s)
    return s[m.start():] if m else s

def strip_suite(s: str) -> str:
    return re.sub(r"\b(?:STE|SUITE|UNIT|APT|#)\s*\w+\b", "", s, flags=re.I)

def expand_abbrev(s: str) -> str:
    s = re.sub(r"\bPKWY\b", "PARKWAY", s, flags=re.I)
    s = re.sub(r"\bAVE\b",  "AVENUE",  s, flags=re.I)
    s = re.sub(r"\bRD\.\b", "RD",      s, flags=re.I)
    s = re.sub(r"\bST\.\b", "ST",      s, flags=re.I)
    return s

def ensure_commas(s: str) -> str:
    s = clean_spaces(s)
    s = re.sub(r"\sGA,", ", GA,", s)              # 'Athens GA, 30606' -> ', GA,'
    s = re.sub(r"\sGA\s(\d{5})", r", GA, \1", s)  # 'Atlanta GA 30309' -> ', GA, 30309'
    return s

def canonicalize_address(s: str) -> str:
    s = clean_spaces(s)
    s = strip_leading_org(s)
    s = strip_suite(s)
    s = expand_abbrev(s)
    s = ensure_commas(s)
    return clean_spaces(s)

# ---------------- geocoding (unique addresses) ----------------
def cg_first_county(match: dict):
    geos = match.get("geographies", {}) if isinstance(match, dict) else {}
    counties = geos.get("Counties") or []
    if not counties:
        return None, None, None, None
    c = counties[0]
    coords = match.get("coordinates", {})
    return c.get("GEOID"), c.get("NAME"), coords.get("y"), coords.get("x")

def load_checkpoint(file_key: str) -> dict:
    path = CHECKPOINTS / f"{file_key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}

def save_checkpoint(file_key: str, cache: dict):
    path = CHECKPOINTS / f"{file_key}.json"
    path.write_text(json.dumps(cache))

def geocode_unique(addresses: list[str], file_key: str, sleep_sec=0.08, timeout=15, max_retries=2) -> dict:
    """
    Returns: dict[address] = (county_fips, county_name, lat, lon, status)
    Uses a JSON checkpoint so you can stop/restart safely.
    """
    cache = load_checkpoint(file_key)
    cg = CensusGeocode()

    for addr in tqdm(addresses, desc=f"Geocode unique ({file_key})"):
        if addr in cache:
            continue
        status = "no_match"; fips = name = lat = lon = None

        for attempt in range(max_retries + 1):
            try:
                res = cg.onelineaddress(addr, timeout=timeout)  # timeout helps avoid long hangs
                if isinstance(res, dict):
                    res = res.get("result", {}).get("addressMatches", [])
                if res:
                    fips, name, lat, lon = cg_first_county(res[0])
                    status = "matched" if fips else "no_match"
                break
            except Exception as e:
                status = f"error:{type(e).__name__}"
                if attempt < max_retries:
                    time.sleep(1.0)  # brief backoff then retry
                    continue
            # if exception on last attempt, we keep status as error:...
        cache[addr] = [fips, name, lat, lon, status]

        # periodic checkpoint
        if len(cache) % 200 == 0:
            save_checkpoint(file_key, cache)
        time.sleep(sleep_sec)

    # final checkpoint save
    save_checkpoint(file_key, cache)
    return cache

def retry_file(in_name: str, out_name: str) -> int:
    in_path  = WORK / in_name
    out_path = WORK / out_name
    if not in_path.exists():
        log(f"❌ Missing: {in_path}")
        return 1

    df = pd.read_csv(in_path, dtype=str)
    if "status" not in df.columns or "oneline" not in df.columns:
        log(f"❌ {in_name} must have columns: oneline, status")
        return 1

    log(f"\n=== {in_name} ===")
    log(f"Rows (before): {len(df)}")
    log(f"Status (before):\n{df['status'].value_counts(dropna=False)}")

    # work only on rows not matched
    miss_mask = ~(df["status"].fillna("").eq("matched"))
    miss = df[miss_mask].copy()
    log(f"Rows to retry: {len(miss)}")

    if len(miss) == 0:
        df.to_csv(out_path, index=False)
        log(f"No work needed. Wrote unchanged copy to {out_path}")
        return 0

    # build cleaned address for retry
    miss["oneline_retry"] = miss["oneline"].map(canonicalize_address)

    # STEP 1: geocode each unique retry address ONCE
    uniq = pd.Series(miss["oneline_retry"].unique(), dtype=str).tolist()
    addr_map = geocode_unique(uniq, file_key=in_name.replace(".csv",""))

    # STEP 2: map results back to each row (no merges => no cartesian blowup)
    vals = miss["oneline_retry"].map(addr_map)

    # unzip the tuples into columns safely
    miss["county_fips_new"] = vals.map(lambda v: v[0] if isinstance(v, (list, tuple)) else None)
    miss["county_name_new"] = vals.map(lambda v: v[1] if isinstance(v, (list, tuple)) else None)
    miss["lat_new"]         = vals.map(lambda v: v[2] if isinstance(v, (list, tuple)) else None)
    miss["lon_new"]         = vals.map(lambda v: v[3] if isinstance(v, (list, tuple)) else None)
    miss["status_new"]      = vals.map(lambda v: v[4] if isinstance(v, (list, tuple)) else "no_match")

    # STEP 3: prefer new values when status_new == matched
    for col in ["county_fips","county_name","lat","lon","status"]:
        miss[col] = miss[f"{col}_new"].where(miss["status_new"].eq("matched"), miss[col])

    # STEP 4: stitch back together with rows already matched
    done = pd.concat([df[~miss_mask], miss[df.columns]], ignore_index=True)

    log(f"Rows (after): {len(done)}  [should equal input rows]")
    log(f"Status (after):\n{done['status'].value_counts(dropna=False)}")
    done.to_csv(out_path, index=False)
    log(f"✅ Saved improved: {out_path}")
    return 0

def main():
    rc1 = retry_file("np_geocoded.csv",   "np_geocoded_improved.csv")
    rc2 = retry_file("phys_geocoded.csv", "phys_geocoded_improved.csv")
    sys.exit(rc1 or rc2)

if __name__ == "__main__":
    main()
