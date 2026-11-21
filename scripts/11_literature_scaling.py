"""Step 11 - how much literature is one anchor run worth?

For every outer source-disjoint fold the training sources are subsampled (nested subsets of the
source groups: 10, 20, 35, 50, 70, 100 %).  The literature model (CatBoost) is refitted on each subset
and evaluated on the same held-out campaigns and the same k-shot problems:

  CB          no anchor                              (k = 0)
  CB+BLUP     shrunken anchor offset                 (k = 1, 2, 3, 5)
  AnchorMean  mean of the anchor yields, no model    (k = 1, 2, 3, 5)

The anchor draws are identical for every training-set size, so the comparison along the literature axis
is paired.  Output: results/litscale_s<seed>.csv (pooled R2 / MAE per seed, fraction, target, method, k).

usage: python 11_literature_scaling.py <seed>
"""
import sys
import time
import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from common import load, FEATS, splits, RES

k04 = importlib.import_module("04_kshot_eval")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
FRACS = (0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
K_LIST, REPS, TARGETS = (0, 1, 2, 3, 5), 3, ("char", "oil")


def main():
    df = load()
    df["camp"] = (df["study"].astype(str) + "_" + df["sample"].astype(str)).factorize()[0]
    X = df[FEATS].values
    Y = {t: df[t].values for t in TARGETS}
    camp, grp, study = df["camp"].values, df["group"].values, df["study"].values
    rng = np.random.default_rng(5000 + seed)
    frames, size_rows = [], []
    for fold, (tr, te) in enumerate(splits(df, "group", 5, seed)):
        t0 = time.time()
        order = rng.permutation(np.unique(grp[tr]))
        # ---- k-shot problems on the held-out campaigns (shared by every training-set size)
        A_rows, A_pid, Q_rows, Q_pid, meta = [], [], [], [], []
        for c in np.unique(camp[te]):
            rows = te[camp[te] == c]
            for rep in range(REPS):
                perm = rng.permutation(rows)
                for k in K_LIST:
                    if k > 0 and len(rows) <= k:
                        continue
                    p = len(meta)
                    meta.append((c, rep, k))
                    A_rows += list(perm[:k]); A_pid += [p] * k
                    Q_rows += list(perm[k:]); Q_pid += [p] * (len(perm) - k)
        A_rows, A_pid = np.array(A_rows, int), np.array(A_pid, int)
        Q_rows, Q_pid = np.array(Q_rows, int), np.array(Q_pid, int)
        meta = np.array(meta)
        n_prob = len(meta)

        def anchor_stat(values, shrink=None):
            s = pd.DataFrame({"p": A_pid, "v": values}).dropna().groupby("p")["v"].agg(["mean", "count"])
            mean = s["mean"].reindex(range(n_prob)).values
            cnt = s["count"].reindex(range(n_prob)).fillna(0).values
            if shrink is not None:
                mean = mean * cnt / (cnt + shrink)
            return mean[Q_pid]

        for frac in FRACS:
            keep = order[:max(5, int(round(frac * len(order))))]
            sub = tr[np.isin(grp[tr], keep)]
            for t in TARGETS:
                ytr = Y[t][sub]
                ok = ~np.isnan(ytr)
                if len(np.unique(grp[sub][ok])) < 4:      # too few labelled sources for the inner grouped folds
                    continue
                f = k04.cb_model(seed).fit(X[sub][ok], ytr[ok]).predict(X)
                oof = k04.inner_oof(X[sub], ytr, grp[sub], seed)
                kappa, _, _ = k04.variance_components(ytr - oof, camp[sub])
                size_rows.append({"seed": seed, "fold": fold, "frac": frac, "target": t, "n_rows": int(ok.sum()),
                                  "n_studies": len(np.unique(study[sub][ok])), "n_campaigns": len(np.unique(camp[sub][ok])),
                                  "kappa": kappa})
                ya = Y[t][A_rows]
                base = f[Q_rows]
                am = anchor_stat(ya)
                preds = {"CB": base,
                         "CB+BLUP": base + np.nan_to_num(anchor_stat(ya - f[A_rows], kappa)),
                         "AnchorMean": np.where(np.isnan(am), np.nanmean(ytr), am)}
                for nm, p in preds.items():
                    frames.append(pd.DataFrame({"seed": seed, "fold": fold, "frac": frac, "target": t, "method": nm,
                                                "camp": meta[Q_pid, 0], "rep": meta[Q_pid, 1], "k": meta[Q_pid, 2],
                                                "y": Y[t][Q_rows], "pred": p}))
        print(f"seed {seed} fold {fold}: {time.time() - t0:.0f}s", flush=True)

    out = pd.concat(frames, ignore_index=True).dropna(subset=["y"])
    out = out[((out.method == "CB") & (out.k == 0)) | ((out.method != "CB") & (out.k > 0))]
    summ = (out.groupby(["seed", "frac", "target", "method", "k"])
            .apply(lambda g: pd.Series({"R2": r2_score(g.y, g.pred), "MAE": mean_absolute_error(g.y, g.pred), "n": len(g)}))
            .reset_index())
    size = pd.DataFrame(size_rows).groupby(["seed", "frac", "target"])[["n_rows", "n_studies", "n_campaigns"]].mean().reset_index()
    summ = summ.merge(size, on=["seed", "frac", "target"])
    summ.to_csv(RES / f"litscale_s{seed}.csv", index=False)
    # campaign-level errors for paired statistics
    camp_err = (out.assign(ae=lambda d: (d.y - d.pred).abs())
                .groupby(["seed", "frac", "target", "method", "k", "camp"])["ae"].mean().reset_index())
    camp_err.to_parquet(RES / f"litscale_camp_s{seed}.parquet")
    for t in TARGETS:
        s = summ[summ.target == t]
        print(t, "MAE (rows: fraction of training sources; columns: method, k)")
        print(s.pivot(index="frac", columns=["method", "k"], values="MAE").round(2).to_string())


if __name__ == "__main__":
    main()
