"""Step 12 - do the reported inputs identify the source study?  (the "laboratory fingerprint")

1. nearest neighbour in standardised input space: share of runs whose nearest other run comes from the
   same study / the same campaign (all runs, and excluding neighbours with identical inputs);
2. a random-forest classifier predicting the study label from the inputs (stratified 5-fold, studies
   with >= 5 runs), for all 14 inputs, feedstock descriptors only and operating variables only;
3. exposure under a random record-level split: share of test runs whose campaign / study also occurs
   in the training folds;
4. a t-SNE embedding of the inputs for display.

Outputs: tables/T13_lab_fingerprint.csv, results/fingerprint_tsne.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from common import load, FEATS, FEED, OPER, TAB, RES, random_folds

LOGS = ["RT", "HR", "PS", "GFR", "mass_g"]


def design(df, cols):
    Z = df[cols].copy()
    for c in cols:
        if c in LOGS:
            Z[c] = np.log10(Z[c].clip(lower=1e-3))
    Z = Z.fillna(Z.median())
    return ((Z - Z.mean()) / Z.std().replace(0, 1)).values


def main():
    df = load()
    df["camp"] = (df["study"].astype(str) + "_" + df["sample"].astype(str)).factorize()[0]
    study, camp = df["study"].values, df["camp"].values
    rows = []
    Z = design(df, FEATS)

    # 1 ---- nearest neighbour
    dist, idx = NearestNeighbors(n_neighbors=len(Z)).fit(Z).kneighbors(Z)
    # With ties at zero distance the query can occupy any tied position.
    # Select the first genuinely different row before comparing its source.
    query = np.arange(len(Z))
    first = idx[query, np.argmax(idx != query[:, None], axis=1)]
    rows.append(("nn_same_study_pct", 100 * (study[first] == study).mean()))
    rows.append(("nn_same_campaign_pct", 100 * (camp[first] == camp).mean()))
    # nearest neighbour whose inputs differ (drops replicate runs at identical reported conditions)
    nz = np.array([idx[i, np.argmax(dist[i] > 1e-9)] for i in range(len(Z))])
    rows.append(("nn_distinct_inputs_same_study_pct", 100 * (study[nz] == study).mean()))
    rows.append(("nn_distinct_inputs_same_campaign_pct", 100 * (camp[nz] == camp).mean()))
    # chance level: probability that a random other run comes from the same study
    n_s = pd.Series(study).value_counts()
    rows.append(("chance_same_study_pct", 100 * ((n_s * (n_s - 1)).sum() / (len(Z) * (len(Z) - 1)))))

    # 2 ---- study classifier
    keep = pd.Series(study).map(n_s).values >= 5
    y = pd.factorize(study[keep])[0]
    rows.append(("clf_n_runs", int(keep.sum())))
    rows.append(("clf_n_studies", int(len(np.unique(y)))))
    rows.append(("clf_majority_baseline_pct", 100 * np.bincount(y).max() / len(y)))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, cols in (("all14", FEATS), ("feedstock8", FEED), ("operating6", OPER),
                       ("operating5_noT", [c for c in OPER if c != "T"])):
        Zs = design(df, cols)[keep]
        p = cross_val_predict(RandomForestClassifier(400, random_state=0, n_jobs=8), Zs, y, cv=cv)
        rows.append((f"clf_accuracy_{name}_pct", 100 * (p == y).mean()))

    # 3 ---- exposure under a record-level random split
    for t in ("char", "oil"):
        d = df[df[t].notna()].reset_index(drop=True)
        sc, ss = [], []
        for s in range(3):
            for f in random_folds(len(d), 5, s):
                tr = np.setdiff1d(np.arange(len(d)), f)
                sc.append(np.isin(d.camp.values[f], d.camp.values[tr]).mean())
                ss.append(np.isin(d.study.values[f], d.study.values[tr]).mean())
        rows.append((f"random_split_{t}_test_runs_with_campaign_in_train_pct", 100 * np.mean(sc)))
        rows.append((f"random_split_{t}_test_runs_with_study_in_train_pct", 100 * np.mean(ss)))

    # 4 ---- embedding for display
    emb = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z)
    out = df[["excel_row", "study", "camp", "feedstock", "char", "oil"]].copy()
    out["x"], out["y"] = emb[:, 0], emb[:, 1]
    out["n_study"] = pd.Series(study).map(n_s).values
    out.to_csv(RES / "fingerprint_tsne.csv", index=False)

    t13 = pd.DataFrame(rows, columns=["metric", "value"])
    t13.to_csv(TAB / "T13_lab_fingerprint.csv", index=False)
    print(t13.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
