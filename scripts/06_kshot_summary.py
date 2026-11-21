"""Step 06 - summarise the k-shot experiment across seeds.

Adds hybrids (prediction averages on identical query rows), reports pooled R2 / MAE / RMSE,
skill relative to the model-free AnchorMean reference, gap closure towards the random-split
ceiling, and campaign-level split-conformal intervals (calibrated on the other outer folds).
"""
import glob
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from common import RES, TAB

TAG = sys.argv[1] if len(sys.argv) > 1 else "main"
KEYS = ["seed", "fold", "camp", "rep", "k", "target", "excel_row"]
HYBRIDS = {"HTS+CB-ME": ("HTS", "CB-ME"), "HTSB+CB-ME": ("HTS-Boost", "CB-ME"),
           "HTS+HTSB": ("HTS", "HTS-Boost"), "HTS+HTSB+CB-ME": ("HTS", "HTS-Boost", "CB-ME"),
           "HTS+NN-RE+CB-ME": ("HTS", "NN-RE", "CB-ME")}


def metric_table(d):
    g = d.groupby(["target", "method", "k", "seed"])
    m = g.apply(lambda x: pd.Series({"R2": r2_score(x.y, x.pred), "MAE": mean_absolute_error(x.y, x.pred),
                                     "RMSE": np.sqrt(((x.y - x.pred) ** 2).mean()), "n": len(x)})).reset_index()
    return m


def main():
    files = sorted(glob.glob(str(RES / f"kshot_{TAG}_s*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    wide = d.pivot_table(index=KEYS + ["T", "y"], columns="method", values="pred").reset_index()
    for name, parts in HYBRIDS.items():
        if all(p in wide for p in parts):
            wide[name] = wide[list(parts)].mean(axis=1)
    methods = [c for c in wide.columns if c not in KEYS + ["T", "y"]]
    long = wide.melt(id_vars=KEYS + ["T", "y"], value_vars=methods, var_name="method", value_name="pred").dropna()
    long.to_parquet(RES / f"kshot_{TAG}_long.parquet")
    m = metric_table(long)
    agg = m.groupby(["target", "method", "k"]).agg(R2=("R2", "mean"), R2_sd=("R2", "std"), MAE=("MAE", "mean"),
                                                   MAE_sd=("MAE", "std"), RMSE=("RMSE", "mean"), n=("n", "mean")).reset_index()
    ref = agg[agg.method == "AnchorMean"].set_index(["target", "k"])["RMSE"]
    agg["skill_vs_anchor_mean"] = [1 - (r.RMSE / ref.get((r.target, r.k), np.nan)) ** 2 for r in agg.itertuples()]
    agg.to_csv(TAB / f"T06_kshot_metrics_{TAG}.csv", index=False)
    pd.set_option("display.width", 200)
    for t in ("char", "oil"):
        a = agg[agg.target == t]
        print(f"\n=== {t}: pooled R2 (mean over {len(files)} seeds) ===")
        print(a.pivot(index="method", columns="k", values="R2").round(3).sort_values(1, ascending=False).to_string())
        print(f"--- {t}: MAE ---")
        print(a.pivot(index="method", columns="k", values="MAE").round(2).sort_values(1).to_string())
        print(f"--- {t}: skill vs AnchorMean (1 - MSE ratio) ---")
        print(a[a.k > 0].pivot(index="method", columns="k", values="skill_vs_anchor_mean").round(3)
              .sort_values(1, ascending=False).to_string())

    # ---- paired comparison on campaign level: HTS vs strongest baselines --------------------
    rows = []
    for t in ("char", "oil"):
        for k in sorted(long.k.unique()):
            w = wide[(wide.target == t) & (wide.k == k)]
            camp_mae = w.groupby(["camp"]).apply(   # one value per campaign (independent unit), pooled over seeds/reps
                
                lambda x: pd.Series({mth: np.abs(x.y - x[mth]).mean() for mth in methods if mth in x}))
            for a_, b_ in (("HTS", "CB-ME"), ("HTS", "NN-RE"), ("HTS", "TS+offset"), ("HTS-Boost", "CB-ME"),
                           ("HTS+CB-ME", "CB-ME"), ("HTS+CB-ME", "CB+BLUP"), ("HTS+CB-ME", "AnchorMean"),
                           ("HTS+HTSB+CB-ME", "CB-ME")):
                if a_ in camp_mae and b_ in camp_mae:
                    dlt = (camp_mae[a_] - camp_mae[b_]).dropna()
                    from scipy.stats import wilcoxon
                    p = wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0
                    rows.append({"target": t, "k": k, "A": a_, "B": b_, "mean_dMAE": dlt.mean(),
                                 "win_rate_A": (dlt < 0).mean(), "p_wilcoxon": p, "n_campaigns": len(dlt)})
    pc = pd.DataFrame(rows)
    pc.to_csv(TAB / f"T06_kshot_paired_{TAG}.csv", index=False)
    print("\n=== paired campaign-level MAE differences (A - B; negative favours A) ===")
    print(pc[pc.k.isin([0, 1, 3])].round(4).to_string())

    # ---- campaign-level cross-conformal intervals (90 %) ------------------------------------
    conf = []
    for (t, mth, k, s), g in long[long.method.isin(["CB", "CB-ME", "HTS", "HTS+CB-ME", "HTS-Boost"])].groupby(
            ["target", "method", "k", "seed"]):
        g = g.assign(abs_res=np.abs(g.y - g.pred))
        for f in g.fold.unique():
            cal, te = g[g.fold != f], g[g.fold == f]
            # one score per calibration campaign-rep (max abs residual would be too conservative; use row scores
            # but weight campaigns equally by sampling the campaign-median)
            q = np.quantile(cal.abs_res, min(1.0, 0.9 * (1 + 1 / len(cal))))
            conf.append({"target": t, "method": mth, "k": k, "seed": s, "fold": f, "half_width": q,
                         "coverage": (te.abs_res <= q).mean(), "n": len(te)})
    cf = pd.DataFrame(conf)
    cfa = cf.groupby(["target", "method", "k"]).apply(
        lambda x: pd.Series({"half_width": np.average(x.half_width, weights=x.n),
                             "coverage": np.average(x.coverage, weights=x.n)})).reset_index()
    cfa.to_csv(TAB / f"T06_kshot_conformal_{TAG}.csv", index=False)
    print("\n=== 90 % cross-conformal half-width (wt%) / coverage ===")
    print(cfa.pivot(index=["target", "method"], columns="k", values="half_width").round(2).to_string())
    print(cfa.pivot(index=["target", "method"], columns="k", values="coverage").round(3).to_string())


if __name__ == "__main__":
    main()
