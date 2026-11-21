"""Step 13b - decision metrics from 13_decision_value.py (temperature series of held-out laboratories).

Decisions (anchor levels are known to the laboratory, every other level is predicted)
  optimum      which tested temperature gives the most bio-oil; loss = bio-oil forgone (wt%).
  maximum      highest bio-oil yield among the levels that were not anchors: predicted vs observed.
  go/no-go     does that maximum reach a threshold (20-60 wt%)?
  reliability  share of series whose untested levels are predicted within 5 wt% on average.
  inverse      temperature that gives a target biochar yield: the calibrated curve is inverted on a 250-900 degC
               grid; the target is the observed yield of a level that was not an anchor.
  constrained  most bio-oil subject to biochar >= c0; success = the chosen level is truly feasible and within
               2 wt% bio-oil of the best feasible level.
  search       best bio-oil yield found after n runs when the model chooses the next temperature.

Information sources: literature only (k = 0), anchors only (no model: best anchor, anchor mean or maximum,
straight line through the anchors), literature + anchors (calibrated hybrid).  Rules without data: fixed 500 degC,
random level, majority guess, ascending sweep.

Outputs: tables/T16_decision_metrics.csv, T17_sequential_search.csv, T18_anchor_query_temperature.csv,
         T19_decision_matrix.csv; results/decision_problem_oil.parquet, decision_inverse.parquet,
         decision_search.parquet
"""
import glob
from math import comb
import numpy as np
import pandas as pd
from common import RES, TAB

rng = np.random.default_rng(0)
KEY = ["seed", "series", "rep", "k", "method"]
TBINS, TLAB = [0, 400, 500, 600, 2000], ["≤400", "400–500", "500–600", ">600"]
C0_LIST, C0_MAIN, TOL_OIL = (25, 30, 35), 30, 2.0


def boot(d, col, n=2000):
    """Mean over series (series means first) with a series-bootstrap 95 % interval."""
    s = d.groupby("series")[col].mean().dropna().values
    b = s[rng.integers(0, len(s), (n, len(s)))].mean(1)
    return s.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(s)


def oil_problem(g):
    y, p, a, T = g["y"].values, g["pred"].values, g["is_anchor"].values, g["T"].values
    v = np.where(a, y, p)                       # anchors are known, the rest is predicted
    i, j, r = int(np.argmax(v)), int(np.argmax(y)), int(np.argmin(np.abs(T - 500)))
    out = {"n_levels": len(y), "T_best": T[j], "T_pick": T[i], "dT": abs(T[i] - T[j]), "loss": y[j] - y[i],
           "hit": float(i == j), "loss_model": y[j] - y[int(np.argmax(p))], "y_best": y[j],
           "loss_random": y[j] - y.mean(), "dT_random": np.abs(T - T[j]).mean(),
           "loss_rule": y[j] - y[r], "dT_rule": abs(T[r] - T[j]),
           "loss_anchor": np.nan, "dT_anchor": np.nan, "anchor_mean": np.nan, "anchor_max": np.nan, "mae_anchor_mean": np.nan}
    q = ~a
    if a.any():
        ia = np.where(a)[0][int(np.argmax(y[a]))]
        out.update(loss_anchor=y[j] - y[ia], dT_anchor=abs(T[ia] - T[j]), anchor_mean=y[a].mean(), anchor_max=y[a].max(),
                   mae_anchor_mean=np.abs(y[q] - y[a].mean()).mean())
    out.update(max_obs=y[q].max(), max_pred=p[q].max(), mae_levels=np.abs(y[q] - p[q]).mean())
    return pd.Series(out)


def constrained(g, c0):
    yc, yo, pc, po = g["y_char"].values, g["y_oil"].values, g["pred_char"].values, g["pred_oil"].values
    a, T = g["is_anchor"].values, g["T"].values
    feas = yc >= c0
    nan = {"success": np.nan, "violate": np.nan, "success_rule": np.nan, "success_anchor": np.nan}
    if not feas.any() or feas.all():            # the constraint does not discriminate between the tested levels
        return pd.Series(nan)
    vc, vo = np.where(a, yc, pc), np.where(a, yo, po)
    f = vc >= c0
    i = int(np.argmax(np.where(f, vo, -1e9))) if f.any() else int(np.argmax(vc))
    oracle = yo[feas].max()
    r = int(np.argmin(np.abs(T - 500)))
    fa = a & feas
    return pd.Series({"success": float(feas[i] and oracle - yo[i] <= TOL_OIL), "violate": float(not feas[i]),
                      "success_rule": float(feas[r] and oracle - yo[r] <= TOL_OIL),
                      "success_anchor": float(fa.any() and oracle - yo[fa].max() <= TOL_OIL) if a.any() else np.nan})


def main():
    lev = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(RES / "decision_levels_s*.parquet")))], ignore_index=True)
    seq = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / "decision_seq_s*.csv")))], ignore_index=True)
    rows = []

    def add(analysis, target, source, k, metric, d, col, threshold=np.nan):
        m, lo, hi, n = boot(d, col)
        rows.append(dict(analysis=analysis, target=target, source=source, k=k, metric=metric, threshold=threshold,
                         value=m, lo=lo, hi=hi, n_series=n))

    # ---------------- bio-oil: optimum, attainable maximum, go/no-go, reliability -----------------
    oil = lev[(lev.target == "oil") & lev.y.notna()]
    P = oil.groupby(KEY)[["y", "pred", "is_anchor", "T"]].apply(oil_problem).reset_index()
    P["err_max"] = (P.max_pred - P.max_obs).abs()
    P["bias_max"] = P.max_pred - P.max_obs
    P["err_anchor_max"] = (P.anchor_max - P.max_obs).abs()
    P["err_anchor_mean"] = (P.anchor_mean - P.max_obs).abs()
    P["in5"] = (P.mae_levels <= 5).astype(float)
    P["in5_anchor"] = np.where(P.k > 0, (P.mae_anchor_mean <= 5).astype(float), np.nan)
    P.to_parquet(RES / "decision_problem_oil.parquet")
    for (m, k), d in P.groupby(["method", "k"]):
        add("optimum", "oil", m, k, "temperature error (degC)", d, "dT")
        add("optimum", "oil", m, k, "bio-oil forgone (wt%)", d, "loss")
        add("optimum", "oil", m, k, "best level chosen (share)", d, "hit")
        add("optimum", "oil", m, k, "bio-oil forgone, model ranking only (wt%)", d, "loss_model")
        add("maximum", "oil", m, k, "absolute error (wt%)", d, "err_max")
        add("maximum", "oil", m, k, "bias (wt%)", d, "bias_max")
        add("reliability", "oil", m, k, "MAE over untested levels (wt%)", d, "mae_levels")
        add("reliability", "oil", m, k, "series within 5 wt% (share)", d, "in5")
        for thr in range(20, 61, 5):
            add("go/no-go", "oil", m, k, "correct (share)", d.assign(ok=((d.max_pred >= thr) == (d.max_obs >= thr)).astype(float)),
                "ok", thr)
        if m != "HYB":
            continue                             # references without a model, on the same problems
        add("optimum", "oil", "fixed 500 degC", k, "bio-oil forgone (wt%)", d, "loss_rule")
        add("optimum", "oil", "fixed 500 degC", k, "temperature error (degC)", d, "dT_rule")
        add("optimum", "oil", "random level", k, "bio-oil forgone (wt%)", d, "loss_random")
        add("optimum", "oil", "random level", k, "temperature error (degC)", d, "dT_random")
        for thr in range(20, 61, 5):
            prev = (d.max_obs >= thr).groupby(d.series).mean().mean()
            rows.append(dict(analysis="go/no-go", target="oil", source="majority guess", k=k, metric="correct (share)",
                             threshold=thr, value=max(prev, 1 - prev), lo=np.nan, hi=np.nan, n_series=d.series.nunique()))
        if k == 0:
            continue
        add("optimum", "oil", "anchors only", k, "bio-oil forgone (wt%)", d, "loss_anchor")
        add("optimum", "oil", "anchors only", k, "temperature error (degC)", d, "dT_anchor")
        add("maximum", "oil", "anchors only", k, "absolute error (wt%)", d, "err_anchor_max")
        add("maximum", "oil", "anchor mean", k, "absolute error (wt%)", d, "err_anchor_mean")
        add("reliability", "oil", "anchors only", k, "MAE over untested levels (wt%)", d, "mae_anchor_mean")
        add("reliability", "oil", "anchors only", k, "series within 5 wt% (share)", d, "in5_anchor")
        for thr in range(20, 61, 5):
            add("go/no-go", "oil", "anchors only", k, "correct (share)",
                d.assign(ok=((d.anchor_max >= thr) == (d.max_obs >= thr)).astype(float)), "ok", thr)

    # ---------------- biochar: inverse design ---------------------------------
    ch = lev[(lev.target == "char") & lev.y.notna() & ~lev.is_anchor].copy()
    ch["err"] = (ch.T_inv - ch["T"]).abs()
    ch["err_ref"] = (ch.T_inv_ref - ch["T"]).abs()
    ch["in50"] = (ch.err <= 50).astype(float)
    ch["in50_ref"] = np.where(ch.k > 0, (ch.err_ref <= 50).astype(float), np.nan)
    ch["mae"] = (ch.pred - ch.y).abs()
    ch[KEY + ["T", "y", "pred", "T_inv", "T_inv_ref", "err", "err_ref"]].to_parquet(RES / "decision_inverse.parquet")
    for (m, k), d in ch.groupby(["method", "k"]):
        add("inverse", "char", m, k, "temperature error (degC)", d, "err")
        add("inverse", "char", m, k, "within 50 degC (share)", d, "in50")
        add("inverse", "char", m, k, "MAE over untested levels (wt%)", d, "mae")
        rows.append(dict(analysis="inverse", target="char", source=m, k=k, metric="median temperature error (degC)",
                         threshold=np.nan, value=d.err.median(), lo=np.nan, hi=np.nan, n_series=d.series.nunique()))
        if m == "HYB" and k > 0:
            add("inverse", "char", "anchors only", k, "temperature error (degC)", d, "err_ref")
            add("inverse", "char", "anchors only", k, "within 50 degC (share)", d, "in50_ref")
            rows.append(dict(analysis="inverse", target="char", source="anchors only", k=k, metric="median temperature error (degC)",
                             threshold=np.nan, value=d.err_ref.median(), lo=np.nan, hi=np.nan, n_series=d.series.nunique()))

    # ---------------- both products: constrained optimum ---------------------
    both = lev.pivot_table(index=KEY + ["T", "is_anchor"], columns="target", values=["y", "pred"]).dropna()
    both.columns = [f"{a}_{b}" for a, b in both.columns]
    both = both.reset_index()
    cols = ["y_char", "y_oil", "pred_char", "pred_oil", "is_anchor", "T"]
    for c0 in C0_LIST:
        C = both.groupby(KEY)[cols].apply(constrained, c0=c0).reset_index().dropna(subset=["success"])
        for (m, k), d in C.groupby(["method", "k"]):
            add("constrained", "both", m, k, "success (share)", d, "success", c0)
            add("constrained", "both", m, k, "constraint violated (share)", d, "violate", c0)
            if m == "HYB":
                add("constrained", "both", "fixed 500 degC", k, "success (share)", d, "success_rule", c0)
                if k > 0:
                    add("constrained", "both", "anchors only", k, "success (share)", d, "success_anchor", c0)
    T16 = pd.DataFrame(rows)
    T16.to_csv(TAB / "T16_decision_metrics.csv", index=False)

    # ---------------- anchor temperature x query temperature (one anchor, hybrid) -------------
    h = lev[(lev.method == "HYB") & (lev.k == 1) & lev.y.notna()].assign(ae=lambda x: (x.pred - x.y).abs())
    pk = ["seed", "series", "rep", "target"]
    q = h[~h.is_anchor].merge(h[h.is_anchor][pk + ["T"]].rename(columns={"T": "T_anchor"}), on=pk)
    q["anchor_bin"] = pd.cut(q.T_anchor, TBINS, labels=TLAB)
    q["query_bin"] = pd.cut(q["T"], TBINS, labels=TLAB)
    T18 = (q.groupby(["target", "anchor_bin", "query_bin"], observed=True)
           .agg(MAE=("ae", "mean"), n_levels=("ae", "size"), n_series=("series", "nunique")).reset_index())
    T18.to_csv(TAB / "T18_anchor_query_temperature.csv", index=False)

    # ---------------- sequential search ---------------------------------------
    truth = oil[(oil.method == "HYB") & (oil.k == 0) & (oil.rep == 0)][["seed", "series", "T", "y", "pred"]]
    S = []
    for (sd, s), g in truth.groupby(["seed", "series"]):
        y, T, p = g["y"].values, g["T"].values, g["pred"].values
        L, best = len(y), y.max()
        order = {"model, recalibrated": seq[(seq.seed == sd) & (seq.series == s)].sort_values("step")["oil"].values,
                 "model, no anchors": y[np.argsort(-p)],
                 "ascending sweep": y[np.argsort(T)],
                 "500 degC first": y[np.argsort(np.abs(T - 500), kind="stable")]}
        ys = np.sort(y)
        for n in range(1, 7):
            for name, o in order.items():
                gap = best - o[:n].max()
                S.append(dict(seed=sd, series=s, strategy=name, n=n, gap=gap, found=float(gap < 1e-9)))
            m = min(n, L)                       # random order: exact expectation of the best of m draws
            e = sum(ys[i] * comb(i, m - 1) for i in range(m - 1, L)) / comb(L, m)
            S.append(dict(seed=sd, series=s, strategy="random order", n=n, gap=best - e,
                          found=min(1.0, m / L) if (ys == best).sum() == 1 else np.nan))
    S = pd.DataFrame(S)
    S.to_parquet(RES / "decision_search.parquet")
    r17 = []
    for (name, n), d in S.groupby(["strategy", "n"]):
        m, lo, hi, ns = boot(d, "gap")
        f = boot(d, "found")
        r17.append(dict(strategy=name, n_runs=n, gap=m, gap_lo=lo, gap_hi=hi, found=f[0], found_lo=f[1], found_hi=f[2], n_series=ns))
    T17 = pd.DataFrame(r17)
    T17.to_csv(TAB / "T17_sequential_search.csv", index=False)

    # ---------------- decision matrix (what each information source answers) -------------------
    def val(analysis, metric, source, k, threshold=None):
        d = T16[(T16.analysis == analysis) & (T16.metric == metric) & (T16.source == source) & (T16.k == k)]
        if threshold is not None:
            d = d[d.threshold == threshold]
        return float(d.value.iloc[0])

    spec = [("Temperature for the most bio-oil", "Bio-oil forgone (wt%)", "lower", 1, "optimum", "bio-oil forgone (wt%)", None),
            ("Attainable bio-oil yield", "Absolute error (wt%)", "lower", 1, "maximum", "absolute error (wt%)", None),
            ("Go/no-go at 40 wt% bio-oil", "Correct calls (%)", "higher", 100, "go/no-go", "correct (share)", 40),
            ("Bio-oil yield of untested runs", "Series within 5 wt% (%)", "higher", 100, "reliability", "series within 5 wt% (share)", None),
            ("Temperature for a biochar target", "Within 50 °C (%)", "higher", 100, "inverse", "within 50 degC (share)", None),
            (f"Most bio-oil with biochar ≥ {C0_MAIN} wt%", "Successful choices (%)", "higher", 100, "constrained", "success (share)", C0_MAIN)]
    cols = [("Literature only", "HYB", 0), ("One anchor only", "anchors only", 1), ("Literature + 1 anchor", "HYB", 1),
            ("Three anchors only", "anchors only", 3), ("Literature + 3 anchors", "HYB", 3)]
    T19 = pd.DataFrame([{"decision": dec, "metric": met, "better": better,
                         **{c: scale * val(an, m16, src, k, thr) for c, src, k in cols}}
                        for dec, met, better, scale, an, m16, thr in spec])
    T19.to_csv(TAB / "T19_decision_matrix.csv", index=False)

    pd.set_option("display.width", 250)
    for a in ("optimum", "maximum", "reliability", "inverse", "constrained"):
        print(f"\n== {a} ==")
        print(T16[T16.analysis == a].pivot_table(index=["metric", "threshold", "source"], columns="k", values="value", dropna=False)
              .dropna(how="all").round(3).to_string())
    print("\n== decision matrix ==")
    print(T19.round(2).to_string())
    print("\n== anchor temperature x query temperature, MAE (wt%) ==")
    print(T18.pivot_table(index=["target", "anchor_bin"], columns="query_bin", values="MAE", observed=True).round(2).to_string())
    print("\n== sequential search: gap to the best level (wt%) | share of series whose best level has been run ==")
    print(T17.pivot(index="strategy", columns="n_runs", values="gap").round(2).to_string())
    print(T17.pivot(index="strategy", columns="n_runs", values="found").round(3).to_string())
    print("\nseries: oil", P.series.nunique(), "| char", ch.series.nunique(), "| both products", both.series.nunique())


if __name__ == "__main__":
    main()
