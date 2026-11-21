"""Step 08 - conventional stacking ensemble under random vs source-disjoint validation.

Base learners: CatBoost, XGBoost, LightGBM, ExtraTrees, SVR, Ridge.
Meta learner : non-negative least squares on inner out-of-fold predictions (weights sum is free).
The inner folds follow the same protocol as the outer split (random inside random, grouped inside
grouped), so no outer-test information reaches the meta learner.
Appends rows 'Stacking' and 'Blend(mean)' to tables/T02_benchmark_metrics.csv -> T08_stacking_metrics.csv
"""
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import clone
from common import load, FEATS, TARGETS, splits, group_folds, random_folds, metrics, TAB, RES
import importlib

warnings.filterwarnings("ignore")
bench = importlib.import_module("02_benchmark_20")
BASE = ["CatBoost", "XGBoost", "LightGBM", "ET", "SVR", "Ridge"]


def main(seeds=(0, 1, 2), k=5):
    df = load()
    rows, wrows = [], []
    for target in TARGETS:
        d = df[df[target].notna()].reset_index(drop=True)
        X, y, g = d[FEATS].values, d[target].values, d["group"].values
        for protocol in ("random", "group"):
            for seed in seeds:
                models = {n: m for n, m in bench.build_models(seed).items() if n in BASE}
                stack, blend = np.full(len(d), np.nan), np.full(len(d), np.nan)
                for tr, te in splits(d, protocol, k, seed):
                    inner = (group_folds(g[tr], 4, seed) if protocol == "group" else random_folds(len(tr), 4, seed))
                    Z = np.zeros((len(tr), len(BASE)))
                    for f in inner:
                        trn = np.setdiff1d(np.arange(len(tr)), f)
                        for j, n in enumerate(BASE):
                            Z[f, j] = clone(models[n]).fit(X[tr][trn], y[tr][trn]).predict(X[tr][f])
                    w, _ = nnls(np.column_stack([Z, np.ones(len(tr))]), y[tr])
                    P = np.column_stack([clone(models[n]).fit(X[tr], y[tr]).predict(X[te]) for n in BASE])
                    stack[te] = P @ w[:-1] + w[-1]
                    blend[te] = P.mean(1)
                    wrows.append({"target": target, "protocol": protocol, "seed": seed,
                                  **dict(zip(BASE, w[:-1])), "intercept": w[-1]})
                for nm, p in (("Stacking", stack), ("Blend(mean)", blend)):
                    rows.append({"target": target, "protocol": protocol, "model": nm, "seed": seed, **metrics(y, p)})
                print(target, protocol, seed, {r["model"]: round(r["R2"], 3) for r in rows[-2:]}, flush=True)
    out = pd.DataFrame(rows)
    agg = out.groupby(["target", "protocol", "model"]).agg(R2=("R2", "mean"), RMSE=("RMSE", "mean"), MAE=("MAE", "mean"),
                                                           R2_sd_seeds=("R2", "std")).reset_index()
    agg.to_csv(TAB / "T08_stacking_metrics.csv", index=False)
    pd.DataFrame(wrows).groupby(["target", "protocol"]).mean(numeric_only=True).drop(columns="seed").round(3).to_csv(
        TAB / "T08_stacking_weights.csv")
    print(agg.round(3).to_string())
    print(pd.read_csv(TAB / "T08_stacking_weights.csv").to_string())


if __name__ == "__main__":
    main()
