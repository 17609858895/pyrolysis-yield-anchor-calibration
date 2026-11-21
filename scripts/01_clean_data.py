"""Step 01 - clean the Zenodo 13645157 v2 pyrolysis table (raw file is never modified).

Outputs
-------
data_clean/pyrolysis_clean.csv      analysis table with group identifiers
tables/T01_cleaning_log.csv         every cell that was changed and why
tables/T01_missingness.csv          missing share per column
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "SI_Data.xlsx"
OUT = ROOT / "data_clean"
TAB = ROOT / "tables"
OUT.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

RENAME = {
    "Feed Stock Type": "feedstock",
    "Carbon": "C", "Hydrogen": "H", "Nitrogen": "N", "Oxygen": "O",
    "Raw Material Supply (g)": "mass_g",
    "Temperature (C )": "T",
    "Residence Time (min)": "RT",
    "Gas Flow Rate (L/min)": "GFR",
    "Heating Rate (C/min)": "HR",
    "Moisture (%)": "M", "VM": "VM", "Ash": "Ash", "FC": "FC",
    "Particle Size (mm)": "PS",
    "Biochar Yield (%)": "char", "Bio Oil Yield (%)": "oil",
    "References 1": "ref1", "References 2": "ref2",
}
NUM = ["C", "H", "N", "O", "mass_g", "T", "RT", "GFR", "HR", "M", "VM", "Ash", "FC", "PS", "char", "oil"]

raw = pd.read_excel(RAW)
df = raw.rename(columns=RENAME).copy()
df.insert(0, "excel_row", np.arange(2, len(df) + 2))
log = []


def fix(value, col, row):
    """Repair wrapped / spaced numerals; ambiguous strings become NaN (logged)."""
    if not isinstance(value, str):
        return value
    s = value.replace("\n", "").replace(" ", "")
    s2 = re.sub(r"\.\.+", ".", s)
    if s2.count(".") > 1:           # e.g. '0.0.5' cannot be resolved without the source paper
        log.append((row, col, repr(value), np.nan, "ambiguous -> NaN"))
        return np.nan
    try:
        out = float(s2)
    except ValueError:
        log.append((row, col, repr(value), np.nan, "unparseable -> NaN"))
        return np.nan
    log.append((row, col, repr(value), out, "wrapped/space numeral joined"))
    return out


for c in NUM:
    df[c] = [fix(v, c, r) for v, r in zip(df[c], df["excel_row"])]
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["feedstock"] = df["feedstock"].str.strip().replace({"Switch Grass": "Switchgrass"})

# exact duplicates over all original columns (after repair) -> keep first
dup = df.duplicated(subset=[c for c in df.columns if c != "excel_row"], keep="first")
for r in df.loc[dup, "excel_row"]:
    log.append((r, "*row*", "exact duplicate", "dropped", "exact duplicate of an earlier row"))
df = df.loc[~dup].reset_index(drop=True)

# physically impossible targets
bad = (df["char"].fillna(0) + df["oil"].fillna(0)) > 100.5
for r in df.loc[bad, "excel_row"]:
    log.append((r, "char+oil", ">100.5", "flag", "mass balance violated; kept but flagged"))
df["mb_violation"] = bad.astype(int)

# ---- group identifiers -----------------------------------------------------
# study  : primary reference (ref1), falling back to ref2 when ref1 is blank
# sample : a physical feedstock sample = identical feedstock + ultimate/proximate fingerprint
df["study"] = df["ref1"].fillna(df["ref2"]).astype(int)
finger = ["feedstock", "C", "H", "N", "O", "M", "VM", "Ash", "FC"]
df["sample"] = df[finger].astype(str).agg("|".join, axis=1).factorize()[0]

# source group: connected components of the study<->sample bipartite graph, so that neither a
# study nor a physical feedstock sample is ever split across train and test.
parent = {}


def find(a):
    parent.setdefault(a, a)
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


for s, k in zip(df["study"], df["sample"]):
    union(("study", s), ("sample", k))
df["group"] = pd.factorize([find(("study", s)) for s in df["study"]])[0]

# a series = same sample, same operating set-up, only temperature varies (yield-temperature curve)
series_keys = ["sample", "study", "mass_g", "RT", "GFR", "HR", "PS"]
df["series"] = df[series_keys].astype(str).agg("|".join, axis=1).factorize()[0]

df.to_csv(OUT / "pyrolysis_clean.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(log, columns=["excel_row", "column", "raw", "clean", "action"]).to_csv(
    TAB / "T01_cleaning_log.csv", index=False, encoding="utf-8-sig")
miss = df[NUM].isna().mean().rename("missing_share").to_frame()
miss["n_present"] = df[NUM].notna().sum()
miss.to_csv(TAB / "T01_missingness.csv", encoding="utf-8-sig")

print("rows", len(df), "| feedstocks", df.feedstock.nunique(), "| studies", df.study.nunique(),
      "| samples", df["sample"].nunique(), "| groups", df.group.nunique(), "| series", df.series.nunique())
print("char n", df.char.notna().sum(), "oil n", df.oil.notna().sum(),
      "both", (df.char.notna() & df.oil.notna()).sum())
print("mass-balance violations", int(bad.sum()))
print(miss.round(3).to_string())
print("largest groups:", df.group.value_counts().head(8).to_dict())
print("studies per sample >1:", (df.groupby("sample").study.nunique() > 1).sum())
print("series length distribution:\n", df.groupby("series").size().describe().round(2).to_string())
print("series with >=3 distinct T:", (df.groupby("series")["T"].nunique() >= 3).sum())
print(df[NUM].describe().T.round(2).to_string())
