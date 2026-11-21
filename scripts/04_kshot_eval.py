"""Step 04 - source-disjoint evaluation with k anchor experiments per held-out campaign.

Outer split : grouped 5-fold on `group` (no study and no physical feedstock sample is shared
              between training and test).
Campaign    : one feedstock sample in one study (the unit a laboratory would calibrate).
k-shot      : k randomly chosen runs of a held-out campaign are revealed (anchors); every method
              predicts the remaining runs.  k = 0 is the zero-shot (literature-only) setting.

Methods
  AnchorMean    mean yield of the anchors (no model at all; the naive reference)
  CB            CatBoost, zero-shot
  CB+offset     CatBoost + additive anchor offset (mean anchor residual)
  CB+BLUP       CatBoost + shrunken offset n/(n+kappa); kappa from grouped inner-OOF variance components
  CB-ME         mixed-effects CatBoost (MERF-style): refit on targets with campaign BLUPs removed, + BLUP offset
  AnchorBoost   CatBoost trained on within-campaign (query, anchor) pairs; anchor yields are inputs
  TS            thermal-survival network, no latent effects; TS+offset adds an additive anchor offset
  NN-RE         unconstrained network with additive campaign offsets (structure ablation)
  HTS           hierarchical thermal-survival network; latents calibrated by MAP on the anchors
  HTS[...]      random-effect subsets (ablation)
  HTS-Boost     same hierarchical survival structure, link functions learned by Gauss-Newton tree boosting
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from common import load, FEATS, OPER, splits, group_folds, RES
from hts import HTS
from hts_boost import HTSBoost

warnings.filterwarnings("ignore")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TAG = sys.argv[2] if len(sys.argv) > 2 else "main"
K_LIST = (0, 1, 2, 3, 5)
REPS = 3
T_COL = FEATS.index("T")
LOG_COLS = [FEATS.index(c) for c in ["RT", "HR", "PS", "GFR", "mass_g"]]
OPER_IDX = [FEATS.index(c) for c in OPER]
TARGETS = ("char", "oil")


def cb_model(s, it=1000):
    return CatBoostRegressor(iterations=it, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                             random_seed=s, verbose=0, thread_count=6, allow_writing_files=False)


def pair_feats(Xq, Xa, yca, yoa):
    return np.column_stack([Xq, Xa[:, OPER_IDX], Xq[:, T_COL] - Xa[:, T_COL], yca, yoa])


def train_pairs(camp, rng, max_anchor=12):
    qi, ai = [], []
    for c in np.unique(camp):
        rows = np.where(camp == c)[0]
        if len(rows) < 2:
            continue
        for q in rows:
            others = rows[rows != q]
            if len(others) > max_anchor:
                others = rng.choice(others, max_anchor, replace=False)
            qi += [q] * len(others)
            ai += list(others)
    return np.array(qi), np.array(ai)


def variance_components(res, camp):
    d = pd.DataFrame({"r": res, "c": camp}).dropna()
    g = d.groupby("c")["r"]
    n_i = g.size()
    s2w = ((d["r"] - g.transform("mean")) ** 2).sum() / max(len(d) - len(n_i), 1)
    n0 = (len(d) - (n_i ** 2).sum() / len(d)) / max(len(n_i) - 1, 1)
    msb = (n_i * (g.mean() - d["r"].mean()) ** 2).sum() / max(len(n_i) - 1, 1)
    s2b = max((msb - s2w) / n0, 1e-6)
    return s2w / s2b, s2w, s2b


def inner_oof(Xtr, ytr, groups, s):
    """Grouped inner out-of-fold CatBoost predictions on the training fold."""
    oof = np.full(len(ytr), np.nan)
    sub = np.where(~np.isnan(ytr))[0]
    for f in group_folds(groups[sub], 4, s):
        trn = np.setdiff1d(np.arange(len(sub)), f)
        oof[sub[f]] = cb_model(s, 500).fit(Xtr[sub][trn], ytr[sub][trn]).predict(Xtr[sub][f])
    return oof


def blup(res, camp, kappa):
    d = pd.DataFrame({"r": res, "c": camp})
    g = d.groupby("c")["r"]
    n, m = g.transform("count"), g.transform("mean")
    return (m * n / (n + kappa)).fillna(0.0).values


def main():
    df = load()
    df["camp"] = (df["study"].astype(str) + "_" + df["sample"].astype(str)).factorize()[0]
    X = df[FEATS].values
    Y = {t: df[t].values for t in TARGETS}
    camp, grp, erow = df["camp"].values, df["group"].values, df["excel_row"].values
    rng = np.random.default_rng(1000 + seed)
    frames, vc_rows = [], []
    for fold, (tr, te) in enumerate(splits(df, "group", 5, seed)):
        t0 = time.time()
        Xtr = X[tr]
        f_all, fme_all, kappa = {}, {}, {}
        for t in TARGETS:
            ytr = Y[t][tr]
            ok = ~np.isnan(ytr)
            f_all[t] = cb_model(seed).fit(Xtr[ok], ytr[ok]).predict(X)
            oof = inner_oof(Xtr, ytr, grp[tr], seed)
            kappa[t], s2w, s2b = variance_components(ytr - oof, camp[tr])
            vc_rows.append({"seed": seed, "fold": fold, "target": t, "kappa": kappa[t],
                            "s2_within": s2w, "s2_between": s2b})
            # mixed-effects CatBoost: remove campaign BLUPs (from OOF residuals), refit, iterate once
            b = blup(ytr - oof, camp[tr], kappa[t])
            oof2 = inner_oof(Xtr, ytr - b, grp[tr], seed)
            b = blup(ytr - oof2, camp[tr], kappa[t])
            fme_all[t] = cb_model(seed).fit(Xtr[ok], (ytr - b)[ok]).predict(X)
        # AnchorBoost
        qi, ai = train_pairs(camp[tr], rng)
        P = pair_feats(Xtr[qi], Xtr[ai], Y["char"][tr][ai], Y["oil"][tr][ai])
        ab = {}
        for t in TARGETS:
            yq = Y[t][tr][qi]
            ok = ~np.isnan(yq)
            ab[t] = cb_model(seed, 800).fit(P[ok], yq[ok])
        # networks
        nets = {
            "TS": HTS(re=(), seed=seed, n_ens=3),
            "NN-RE": HTS(structure="plain", seed=seed, n_ens=3),
            "HTS": HTS(seed=seed, n_ens=5),
            "HTS[pi,phi]": HTS(re=("pi", "phi"), seed=seed, n_ens=3),
            "HTS[mu]": HTS(re=("mu_d", "mu_k"), seed=seed, n_ens=3),
            "HTS-Boost": HTSBoost(seed=seed, n_ens=3),
        }
        for m in nets.values():
            m.fit(Xtr, Y["char"][tr], Y["oil"][tr], camp[tr], T_COL, LOG_COLS)
        print(f"seed {seed} fold {fold}: models fitted in {time.time() - t0:.0f}s", flush=True)

        # ---------- enumerate k-shot problems ------------------------------
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
        n_prob = len(meta)
        meta = np.array(meta)
        preds = {}

        def anchor_stat(values, shrink=None):
            """Per-problem mean of anchor values mapped to query rows (0 when no labelled anchor)."""
            s = pd.DataFrame({"p": A_pid, "v": values}).dropna().groupby("p")["v"].agg(["mean", "count"])
            mean = s["mean"].reindex(range(n_prob)).values
            cnt = s["count"].reindex(range(n_prob)).fillna(0).values
            if shrink is not None:
                mean = mean * cnt / (cnt + shrink)
            return mean[Q_pid], cnt[Q_pid]

        for t in TARGETS:
            ya = Y[t][A_rows]
            base = f_all[t][Q_rows]
            preds[("CB", t)] = base
            off, cnt = anchor_stat(ya - f_all[t][A_rows])
            preds[("CB+offset", t)] = base + np.nan_to_num(off)
            off_s, _ = anchor_stat(ya - f_all[t][A_rows], kappa[t])
            preds[("CB+BLUP", t)] = base + np.nan_to_num(off_s)
            off_me, _ = anchor_stat(ya - fme_all[t][A_rows], kappa[t])
            preds[("CB-ME", t)] = fme_all[t][Q_rows] + np.nan_to_num(off_me)
            am, _ = anchor_stat(ya)
            preds[("AnchorMean", t)] = np.where(np.isnan(am), np.nanmean(Y[t][tr]), am)
            # AnchorBoost: all (query, anchor) pairs within a problem, averaged over anchors
            a_by_p = pd.Series(A_rows).groupby(A_pid).apply(list).to_dict()
            qq, aa, qpos = [], [], []
            for i, (q, p) in enumerate(zip(Q_rows, Q_pid)):
                for a in a_by_p.get(p, []):
                    qq.append(q); aa.append(a); qpos.append(i)
            pr = base.copy()
            if qq:
                qq, aa, qpos = np.array(qq), np.array(aa), np.array(qpos)
                pv = ab[t].predict(pair_feats(X[qq], X[aa], Y["char"][aa], Y["oil"][aa]))
                mean_pv = pd.Series(pv).groupby(qpos).mean()
                pr[mean_pv.index.values] = mean_pv.values
            preds[("AnchorBoost", t)] = pr
        for nm, m in nets.items():
            U = m.calibrate_batch(X[A_rows], Y["char"][A_rows], Y["oil"][A_rows], A_pid, n_prob)
            pc, po = m.predict(X[Q_rows], [u[Q_pid] for u in U] if m.n_re else None)
            preds[(nm, "char")], preds[(nm, "oil")] = pc, po
            if nm == "TS":
                zc, zo = m.predict(X[A_rows]) if len(A_rows) else (np.array([]), np.array([]))
                for t, zq, za in (("char", pc, zc), ("oil", po, zo)):
                    off, _ = anchor_stat(Y[t][A_rows] - za)
                    preds[("TS+offset", t)] = zq + np.nan_to_num(off)
        for (nm, t), p in preds.items():
            frames.append(pd.DataFrame({"seed": seed, "fold": fold, "camp": meta[Q_pid, 0], "rep": meta[Q_pid, 1],
                                        "k": meta[Q_pid, 2], "method": nm, "target": t,
                                        "excel_row": erow[Q_rows], "T": X[Q_rows, T_COL],
                                        "y": Y[t][Q_rows], "pred": p}))
        print(f"seed {seed} fold {fold}: done in {time.time() - t0:.0f}s ({n_prob} problems)", flush=True)

    out = pd.concat(frames, ignore_index=True).dropna(subset=["y"])
    out.to_parquet(RES / f"kshot_{TAG}_s{seed}.parquet")
    pd.DataFrame(vc_rows).to_csv(RES / f"kshot_varcomp_{TAG}_s{seed}.csv", index=False)
    summ = (out.groupby(["target", "method", "k"])
            .apply(lambda g: pd.Series({"R2": r2_score(g.y, g.pred), "MAE": mean_absolute_error(g.y, g.pred),
                                        "RMSE": np.sqrt(((g.y - g.pred) ** 2).mean()), "n": len(g)}))
            .reset_index())
    summ.to_csv(RES / f"kshot_summary_{TAG}_s{seed}.csv", index=False)
    for t in TARGETS:
        print(t, "R2")
        print(summ[summ.target == t].pivot(index="method", columns="k", values="R2").round(3).to_string())
        print(t, "MAE")
        print(summ[summ.target == t].pivot(index="method", columns="k", values="MAE").round(2).to_string())


if __name__ == "__main__":
    main()
