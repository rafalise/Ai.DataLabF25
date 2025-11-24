from pathlib import Path
import pandas as pd
import camelot
import re

PDF_PATH = Path("/Users/kultumlhabaik/Documents/Ai.DataLabF25/data/ProtocolAgreements.pdf")
PAGES = "all"   # try "1-5" first if you want to test
OUT_XLSX = PDF_PATH.with_name("ProtocolAgreements_combined.xlsx")

def promote_header_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # if columns are 0..N-1 and first row looks header-ish, promote
    if list(df.columns) == list(range(len(df.columns))):
        first = df.iloc[0].astype(str)
        if (first.str.strip() != "").sum() >= max(1, int(0.6 * len(first))):
            df = df.copy()
            df.columns = [re.sub(r"\s+", " ", s).strip() for s in first]
            df = df.iloc[1:].reset_index(drop=True)
    return df

def basic_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    # remove embedded newlines / NBSPs, drop fully empty rows/columns
    df = df.applymap(lambda x: str(x).replace("\n", "").replace("\u00a0", " ") if isinstance(x, str) else x)
    df = df.replace("", pd.NA).dropna(how="all").dropna(axis=1, how="all")
    return df

def fix_hyphen_splits(df: pd.DataFrame) -> pd.DataFrame:
    # join GAA- + NP001155 across columns and across the next row (if next row is otherwise empty)
    if df is None or df.empty:
        return df
    df = df.copy()

    # across columns
    r, c = df.shape
    for i in range(r):
        for j in range(c - 1):
            left = df.iat[i, j]
            right = df.iat[i, j + 1]
            if isinstance(left, str) and left.endswith("-") and isinstance(right, str) and right.strip():
                df.iat[i, j] = left + right
                df.iat[i, j + 1] = ""

    # across rows (only when the next row is otherwise empty in other cols)
    df = df.replace("", pd.NA)
    rows_to_drop = set()
    for col_i in range(df.shape[1]):
        for i in range(df.shape[0] - 1):
            cur = df.iat[i, col_i]
            nxt = df.iat[i + 1, col_i]
            if isinstance(cur, str) and cur.endswith("-") and isinstance(nxt, str) and nxt.strip():
                other = df.drop(df.columns[col_i], axis=1).iloc[i + 1]
                if not other.dropna().astype(str).str.strip().any():
                    df.iat[i, col_i] = cur + nxt
                    rows_to_drop.add(i + 1)
    if rows_to_drop:
        df = df.drop(index=sorted(rows_to_drop)).reset_index(drop=True)

    # drop empty cols again
    df = df.dropna(axis=1, how="all").fillna("")
    return df

print(f"Reading {PDF_PATH.name} (pages={PAGES}) …")
tables = camelot.read_pdf(str(PDF_PATH), pages=PAGES, flavor="stream", strip_text="\n\t ")
if tables.n == 0:
    print("No tables with stream — trying lattice …")
    tables = camelot.read_pdf(str(PDF_PATH), pages=PAGES, flavor="lattice", strip_text="\n\t ")

frames = []
for t in tables:
    df = t.df
    df = promote_header_if_needed(df)
    df = basic_cleanup(df)
    df = fix_hyphen_splits(df)
    if not df.empty:
        frames.append(df)

combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
    combined.to_excel(w, sheet_name="AllData", index=False)

print(f"✅ Saved raw combined sheet → {OUT_XLSX}")