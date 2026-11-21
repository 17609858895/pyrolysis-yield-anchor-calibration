"""Summarise literature-size experiments and paired errors for the same campaigns."""
import pandas as pd
from scipy.stats import wilcoxon

from common import RES, TAB


def main():
    scores = pd.concat([pd.read_csv(RES / f"litscale_s{seed}.csv") for seed in (3, 4, 5)])
    summary = scores.groupby(["target", "method", "k", "frac"]).agg(
        MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), R2=("R2", "mean"),
        R2_sd=("R2", "std"), n_studies=("n_studies", "mean"),
        n_campaigns=("n_campaigns", "mean"), n_rows=("n_rows", "mean"),
    ).reset_index()
    summary.to_csv(TAB / "T12_literature_scaling.csv", index=False)
    campaign_scores = pd.concat([pd.read_parquet(RES / f"litscale_camp_s{seed}.parquet") for seed in (3, 4, 5)])
    means = campaign_scores.groupby(["target", "method", "k", "frac", "camp"])["ae"].mean()
    comparisons = [
        ("1 anchor @10% lit  vs  zero-shot @100% lit", ("CB+BLUP", 1, 0.1), ("CB", 0, 1.0)),
        ("zero-shot @100% lit vs zero-shot @10% lit", ("CB", 0, 1.0), ("CB", 0, 0.1)),
        ("1 anchor @100% vs 1 anchor @10%", ("CB+BLUP", 1, 1.0), ("CB+BLUP", 1, 0.1)),
    ]
    rows = []
    for target in ("char", "oil"):
        for name, a, b in comparisons:
            d = (means.loc[(target, *a)] - means.loc[(target, *b)]).dropna()
            rows.append(dict(target=target, comparison=name, n_campaigns=len(d),
                             mean_diff=d.mean(), median_diff=d.median(),
                             win_pct=100 * (d < 0).mean(), p=wilcoxon(d).pvalue))
    pd.DataFrame(rows).to_csv(TAB / "T12_literature_scaling_paired.csv", index=False)


if __name__ == "__main__":
    main()
