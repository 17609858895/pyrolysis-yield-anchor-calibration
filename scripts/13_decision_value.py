"""Step 13 - decision value of anchor calibration on temperature series of held-out laboratories.

Unit        : a temperature series = runs of one held-out campaign that share every operating variable
              except temperature, with >= 4 distinct temperature levels (replicate runs at one level are
              averaged).  The series is what a laboratory scans when it looks for an operating temperature.
Outer split : source-disjoint 5-fold on `group`, as in 04_kshot_eval.py; all models are fitted on the
              training fold only.
Models      : hierarchical thermal-survival network (HTS), mixed-effects CatBoost (CB-ME) and their
              equal-weight average (HYB), each calibrated with the same anchors.

Part A  random anchors.  k = 0, 1, 2, 3 levels of a series are revealed (3 draws).  The calibrated models predict
        every level of the series and a temperature grid (250-900 degC) at the conditions of the series.  The grid
        curve is inverted for the temperature that gives a target biochar yield (the observed yield of a level
        that was not an anchor).
Part B  sequential search for the bio-oil optimum.  The model proposes the next level, the run is revealed, the
        model is recalibrated; up to 6 runs.

Outputs : results/decision_levels_s<seed>.parquet   one row per problem x level x method x product
          results/decision_curves_s<seed>.parquet   grid curves (first anchor draw only)
          results/decision_seq_s<seed>.csv          order in which levels were run in Part B
Usage   : python 13_decision_value.py <seed>
"""
import sys
import time
import importlib
import warnings
import numpy as np
import pandas as pd
from common import load, FEATS, splits, RES
from hts import HTS

warnings.filterwarnings("ignore")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
sys.argv = [sys.argv[0], str(seed), "decision"]
base = importlib.import_module("04_kshot_eval")
T_COL, LOG_COLS, TARGETS = base.T_COL, base.LOG_COLS, base.TARGETS
K_LIST, REPS, MIN_LEVELS, MAX_RUNS = (0, 1, 2, 3), 3, 4, 6
GRID = np.arange(250.0, 901.0, 10.0)
METHODS = ("HTS", "CB-ME", "HYB")


def level_table(df):
    """One row per (series, temperature level); replicate runs are averaged."""
    agg = {c: "first" for c in FEATS if c != "T"}
    agg.update({"camp": "first", "group": "first", "study": "first", "feedstock": "first",
                "char": "mean", "oil": "mean", "excel_row": "size"})
    lev = df.groupby(["series", "T"], as_index=False).agg(agg).rename(columns={"excel_row": "n_runs"})
    n = lev.groupby("series")["T"].transform("size")
    return lev[n >= MIN_LEVELS].reset_index(drop=True)


def main():
    df = load()
    df["camp"] = (df["study"].astype(str) + "_" + df["sample"].astype(str)).factorize()[0]
    X = df[FEATS].values
    Y = {t: df[t].values for t in TARGETS}
    camp, grp = df["camp"].values, df["group"].values
    lev = level_table(df)
    rng = np.random.default_rng(7000 + seed)
    out_lev, out_cur, out_seq = [], [], []
    for fold, (tr, te) in enumerate(splits(df, "group", 5, seed)):
        t0 = time.time()
        Xtr = X[tr]
        cbme, kappa = {}, {}
        for t in TARGETS:
            ytr = Y[t][tr]
            ok = ~np.isnan(ytr)
            oof = base.inner_oof(Xtr, ytr, grp[tr], seed)
            kappa[t] = base.variance_components(ytr - oof, camp[tr])[0]
            b = base.blup(ytr - oof, camp[tr], kappa[t])
            oof2 = base.inner_oof(Xtr, ytr - b, grp[tr], seed)
            b = base.blup(ytr - oof2, camp[tr], kappa[t])
            cbme[t] = base.cb_model(seed).fit(Xtr[ok], (ytr - b)[ok])
        hts = HTS(seed=seed, n_ens=5).fit(Xtr, Y["char"][tr], Y["oil"][tr], camp[tr], T_COL, LOG_COLS)

        L = lev[lev["group"].isin(np.unique(grp[te]))].reset_index(drop=True)
        XL = L[FEATS].values
        yL = {t: L[t].values for t in TARGETS}
        sid = L["series"].values
        cb_at_levels = {t: cbme[t].predict(XL) for t in TARGETS}

        def predict(A_idx, A_pid, n_prob, Xq, q_pid):
            """Calibrate with anchors (level rows A_idx of problems A_pid) and predict rows Xq of problems q_pid."""
            A_idx, A_pid = np.asarray(A_idx, int), np.asarray(A_pid, int)
            U = hts.calibrate_batch(XL[A_idx], yL["char"][A_idx], yL["oil"][A_idx], A_pid, n_prob)
            pc, po = hts.predict(Xq, [u[q_pid] for u in U])
            res = {"HTS": {"char": pc, "oil": po}, "CB-ME": {}, "HYB": {}}
            for t in TARGETS:
                off = np.zeros(n_prob)
                if len(A_idx):
                    s = (pd.DataFrame({"p": A_pid, "v": yL[t][A_idx] - cb_at_levels[t][A_idx]}).dropna()
                         .groupby("p")["v"].agg(["mean", "count"]))
                    mean = s["mean"].reindex(range(n_prob)).values
                    cnt = s["count"].reindex(range(n_prob)).fillna(0).values
                    off = np.nan_to_num(mean * cnt / (cnt + kappa[t]))
                res["CB-ME"][t] = cbme[t].predict(Xq) + off[q_pid]
                res["HYB"][t] = 0.5 * (res["HTS"][t] + res["CB-ME"][t])
            return res

        # ---------- Part A: random anchors ---------------------------------
        meta, A_idx, A_pid, Q_idx, Q_pid, Q_anchor = [], [], [], [], [], []
        for s in np.unique(sid):
            rows = np.where(sid == s)[0]
            for rep in range(REPS):
                perm = rng.permutation(rows)
                for k in K_LIST:
                    p = len(meta)
                    meta.append((s, rep, k))
                    A_idx += list(perm[:k]); A_pid += [p] * k
                    Q_idx += list(rows); Q_pid += [p] * len(rows)
                    Q_anchor += list(np.isin(rows, perm[:k]))
        meta = np.array(meta)
        n_prob = len(meta)
        Q_idx, Q_pid, Q_anchor = np.array(Q_idx), np.array(Q_pid), np.array(Q_anchor)
        A_idx_a, A_pid_a = np.array(A_idx, int), np.array(A_pid, int)
        pr = predict(A_idx, A_pid, n_prob, XL[Q_idx], Q_pid)

        # grid curves at the conditions of each series
        first = pd.Series(np.arange(len(L))).groupby(sid).first()
        G_rows = np.repeat(first.reindex(meta[:, 0]).values, len(GRID))
        G_pid = np.repeat(np.arange(n_prob), len(GRID))
        XG = XL[G_rows].copy()
        XG[:, T_COL] = np.tile(GRID, n_prob)
        pg = predict(A_idx, A_pid, n_prob, XG, G_pid)
        curve = {m: pg[m]["char"].reshape(n_prob, len(GRID)) for m in METHODS}

        # inverse design: temperature for a target biochar yield (= observed yield of the level)
        T_inv = {}
        tgt = yL["char"][Q_idx]
        for m in METHODS:
            j = np.abs(curve[m][Q_pid] - tgt[:, None]).argmin(1)
            T_inv[m] = np.where(np.isnan(tgt), np.nan, GRID[j])
        # no-model reference: straight line through the anchors (k >= 2), the anchor temperature itself (k = 1)
        T_ref = np.full(len(Q_idx), np.nan)
        for p in range(n_prob):
            a = A_idx_a[A_pid_a == p]
            a = a[~np.isnan(yL["char"][a])]
            q = np.where(Q_pid == p)[0]
            if len(a) == 1:
                T_ref[q] = XL[a[0], T_COL]
            elif len(a) >= 2:
                slope, icpt = np.polyfit(XL[a, T_COL], yL["char"][a], 1)
                if slope < -1e-4:
                    T_ref[q] = np.clip((tgt[q] - icpt) / slope, GRID[0], GRID[-1])
                else:        # non-decreasing anchors: fall back to the anchor with the closest yield
                    T_ref[q] = XL[a, T_COL][np.abs(yL["char"][a][None, :] - tgt[q][:, None]).argmin(1)]
        for m in METHODS:
            for t in TARGETS:
                out_lev.append(pd.DataFrame({
                    "seed": seed, "fold": fold, "series": meta[Q_pid, 0], "rep": meta[Q_pid, 1], "k": meta[Q_pid, 2],
                    "method": m, "target": t, "T": XL[Q_idx, T_COL], "is_anchor": Q_anchor,
                    "y": yL[t][Q_idx], "pred": pr[m][t],
                    "T_inv": T_inv[m] if t == "char" else np.nan, "T_inv_ref": T_ref if t == "char" else np.nan}))
        keep = np.isin(G_pid, np.where(meta[:, 1] == 0)[0])
        for m in METHODS:
            out_cur.append(pd.DataFrame({
                "seed": seed, "series": meta[G_pid[keep], 0], "k": meta[G_pid[keep], 2], "method": m,
                "T": XG[keep, T_COL], "char": pg[m]["char"][keep].astype(np.float32),
                "oil": pg[m]["oil"][keep].astype(np.float32)}))

        # ---------- Part B: sequential search for the bio-oil optimum -----------
        ok_series = [s for s in np.unique(sid) if not np.isnan(yL["oil"][sid == s]).any()]
        rows_of = {s: np.where(sid == s)[0] for s in ok_series}
        pid_of = {s: i for i, s in enumerate(ok_series)}
        q_idx = np.concatenate([rows_of[s] for s in ok_series]) if ok_series else np.array([], int)
        q_pid = np.concatenate([[pid_of[s]] * len(rows_of[s]) for s in ok_series]) if ok_series else np.array([], int)
        chosen = {s: [] for s in ok_series}
        for step in range(MAX_RUNS):
            a_idx = [r for s in ok_series for r in chosen[s]]
            a_pid = [pid_of[s] for s in ok_series for _ in chosen[s]]
            po = predict(a_idx, a_pid, len(ok_series), XL[q_idx], q_pid)["HYB"]["oil"]
            for s in ok_series:
                rows = rows_of[s]
                left = np.setdiff1d(rows, chosen[s])
                if not len(left):
                    continue
                pv = po[np.isin(q_idx, left) & (q_pid == pid_of[s])]
                pick = left[int(np.argmax(pv))]
                chosen[s].append(pick)
                out_seq.append({"seed": seed, "fold": fold, "series": s, "step": step + 1,
                                "T": XL[pick, T_COL], "oil": yL["oil"][pick], "pred": float(pv.max())})
        print(f"seed {seed} fold {fold}: {n_prob} problems, {len(ok_series)} bio-oil series, {time.time() - t0:.0f}s",
              flush=True)

    pd.concat(out_lev, ignore_index=True).to_parquet(RES / f"decision_levels_s{seed}.parquet")
    pd.concat(out_cur, ignore_index=True).to_parquet(RES / f"decision_curves_s{seed}.parquet")
    pd.DataFrame(out_seq).to_csv(RES / f"decision_seq_s{seed}.csv", index=False)
    print("saved")


if __name__ == "__main__":
    main()
