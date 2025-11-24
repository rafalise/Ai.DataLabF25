from pathlib import Path
import pandas as pd
import re

IN_XLSX  = Path("/Users/kultumlhabaik/Documents/Ai.DataLabF25/data/ProtocolAgreements_combined.xlsx")
OUT_XLSX = IN_XLSX.with_name("ProtocolAgreements_combined_clean.xlsx")
SHEET    = "AllData"    # change if needed

# Your canonical column names
CANON_HEADERS = ["APRN Name","RN#","Delegating Physicia","PHY#","Protocol #","Protcol Address","Effective"]
CANON_NORM = {h.lower() for h in CANON_HEADERS}

# Patterns that represent header/title tokens (not real data)
HEADER_PATTERNS = [
    r"aprn\s*name[s]?",
    r"rn\s*#|rn\s*number",
    r"delegating\s*physici.*",
    r"phy\s*#|physician\s*#|phys\s*#?",
    r"protocol\s*#|prot\s*#|protocol\s*number",
    r"prot(col)?\s*address|protocol\s*address",
    r"effective(\s*date)?|eff(\.)?\s*date|effective",
]
HEADER_REGEXES = [re.compile(rf"^(?:{p})$", re.IGNORECASE) for p in HEADER_PATTERNS]

def norm(x):
    if pd.isna(x): return ""
    return re.sub(r"\s+"," ",str(x).replace("\u00a0"," ")).strip().lower()

def is_header_token(s: str) -> bool:
    n = norm(s)
    return (n in CANON_NORM) or any(rx.fullmatch(n) for rx in HEADER_REGEXES)

def looks_like_real_data(s: str) -> bool:
    s_raw = str(s)
    # names like "Abbey, Ophelia" or "Abajobir, Jaleny T."
    if re.search(r"[A-Za-z],\s*[A-Za-z]", s_raw):
        return True
    # addresses often have digits + street words
    if re.search(r"\d{2,} .+", s_raw):
        return True
    # IDs with letters+digits
    if re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", s_raw):
        return True
    # anything long-ish that's not exactly a header token
    if len(s_raw.strip()) >= 6 and not is_header_token(s_raw):
        return True
    return False

df = pd.read_excel(IN_XLSX, sheet_name=SHEET)

# Build a normalized version of the column names (to detect full header rows)
cols_norm = [norm(c) for c in df.columns]

def row_is_headerish(row) -> bool:
    # non-empty values in the row
    raw_vals = [v for v in row if (pd.notna(v) and str(v).strip())]
    if not raw_vals:
        return False
    vals_norm = [norm(v) for v in raw_vals]

    # Case 1: row equals the actual header row (normalized)
    if len(vals_norm) == len(cols_norm) and vals_norm == cols_norm:
        return True

    # Case 2: row has 1–3 non-empty cells and ALL are header tokens
    if len(vals_norm) <= 3 and all(is_header_token(v) for v in vals_norm):
        return True

    # Case 3: all non-empty cells are header tokens AND nothing looks like real data
    if all(is_header_token(v) for v in vals_norm) and not any(looks_like_real_data(v) for v in raw_vals):
        return True

    return False

mask = df.apply(row_is_headerish, axis=1)
before = len(df)
cleaned = df.loc[~mask].copy()
after = len(cleaned)

# Drop rows that became fully empty
cleaned = cleaned.replace("", pd.NA).dropna(how="all")

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
    cleaned.to_excel(w, sheet_name=SHEET, index=False)

print(f"Removed {before - after} header/title rows. Saved → {OUT_XLSX}")
