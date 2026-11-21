"""Shared utilities: data loading, feature sets, leakage-safe splitters, metrics."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_clean" / "pyrolysis_clean.csv"
RES = ROOT / "results"
TAB = ROOT / "tables"
FIG = ROOT / "figures"
for _p in (RES, TAB, FIG):
    _p.mkdir(exist_ok=True)

FEED = ["C", "H", "N", "O", "M", "VM", "Ash", "FC"]          # feedstock descriptors
OPER = ["T", "RT", "HR", "PS", "GFR", "mass_g"]                # operating variables
FEATS = FEED + OPER
TARGETS = ["char", "oil"]
LABEL = {
    "C": "C (wt%)", "H": "H (wt%)", "N": "N (wt%)", "O": "O (wt%)", "M": "Moisture (wt%)",
    "VM": "Volatile matter (wt%)", "Ash": "Ash (wt%)", "FC": "Fixed carbon (wt%)",
    "T": "Temperature (°C)", "RT": "Residence time (min)", "HR": "Heating rate (°C min$^{-1}$)",
    "PS": "Particle size (mm)", "GFR": "Gas flow (L min$^{-1}$)", "mass_g": "Sample mass (g)",
    "char": "Biochar yield (wt%)", "oil": "Bio-oil yield (wt%)",
}


def load():
    return pd.read_csv(DATA)


def random_folds(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(idx, k)]


def group_folds(groups, k=5, seed=0):
    """Size-balanced grouped folds: every group lands in exactly one test fold."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]     # big groups first, ties shuffled
    load_ = np.zeros(k)
    assign = {}
    for j in order:
        f = int(np.argmin(load_ + rng.uniform(0, 1e-6, k)))
        assign[uniq[j]] = f
        load_[f] += counts[j]
    fold_of = np.array([assign[g] for g in groups])
    return [np.where(fold_of == f)[0] for f in range(k)]


def splits(df_t, protocol, k=5, seed=0):
    """Yield (train_idx, test_idx) positional indices for a target-specific frame."""
    n = len(df_t)
    if protocol == "random":
        folds = random_folds(n, k, seed)
    elif protocol == "group":            # source-disjoint: studies and physical samples never split
        folds = group_folds(df_t["group"].values, k, seed)
    elif protocol == "feedstock":        # unseen feedstock type
        folds = group_folds(df_t["feedstock"].values, k, seed)
    else:
        raise ValueError(protocol)
    allidx = np.arange(n)
    for f in folds:
        yield np.setdiff1d(allidx, f), f


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"R2": r2_score(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": mean_absolute_error(y, p)}
