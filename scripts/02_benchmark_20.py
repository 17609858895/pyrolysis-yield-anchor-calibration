"""Step 02 - 21-algorithm benchmark under random vs source-disjoint (grouped) cross-validation.

All preprocessing (imputation, scaling) is fitted inside each training fold.
Outputs: results/bench_oof_{target}_{protocol}.csv, tables/T02_benchmark_metrics.csv
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from common import load, FEATS, TARGETS, splits, metrics, RES, TAB

warnings.filterwarnings("ignore")


def lin(est):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), est)


def tree(est):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), est)


def build_models(seed=0):
    return {
        "OLS": lin(LinearRegression()),
        "Ridge": lin(Ridge(alpha=10.0)),
        "Lasso": lin(Lasso(alpha=0.05, max_iter=20000)),
        "ElasticNet": lin(ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000)),
        "BayesRidge": lin(BayesianRidge()),
        "Huber": lin(HuberRegressor(max_iter=2000)),
        "PLS": lin(PLSRegression(n_components=6)),
        "kNN": lin(KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "SVR": lin(SVR(C=30.0, epsilon=0.5, gamma="scale")),
        "KRR": lin(KernelRidge(kernel="rbf", alpha=0.1, gamma=0.03)),
        "GPR": lin(GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(1) * 3.0, nu=2.5) + WhiteKernel(0.1),
            normalize_y=True, random_state=seed)),
        "MLP": lin(MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=2e-3,
                                max_iter=1500, early_stopping=True, n_iter_no_change=40, random_state=seed)),
        "DT": tree(DecisionTreeRegressor(min_samples_leaf=5, random_state=seed)),
        "RF": tree(RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.5,
                                         n_jobs=-1, random_state=seed)),
        "ET": tree(ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.7,
                                       n_jobs=-1, random_state=seed)),
        "AdaBoost": tree(AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=300,
                                           learning_rate=0.5, random_state=seed)),
        "GBR": tree(GradientBoostingRegressor(n_estimators=600, learning_rate=0.04, max_depth=4,
                                              subsample=0.8, min_samples_leaf=3, random_state=seed)),
        "HistGB": HistGradientBoostingRegressor(max_iter=600, learning_rate=0.04, max_leaf_nodes=15,
                                                min_samples_leaf=8, l2_regularization=1.0, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=5, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0,
                                n_jobs=8, random_state=seed, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=15, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, min_child_samples=8,
                                  reg_lambda=2.0, n_jobs=8, random_state=seed, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=1200, learning_rate=0.04, depth=6, l2_leaf_reg=3.0,
                                      random_seed=seed, verbose=0, thread_count=8, allow_writing_files=False),
    }


def run(protocols=("random", "group"), k=5, seeds=(0, 1, 2)):
    df = load()
    rows = []
    for target in TARGETS:
        d = df[df[target].notna()].reset_index(drop=True)
        X, y = d[FEATS].values, d[target].values
        for protocol in protocols:
            oof = pd.DataFrame({"excel_row": d.excel_row, "y": y})
            for name, proto in build_models().items():
                t0 = time.time()
                per_seed = []
                for seed in seeds:
                    pred = np.full(len(d), np.nan)
                    fold_r2 = []
                    for tr, te in splits(d, protocol, k, seed):
                        m = clone(proto)
                        m.fit(X[tr], y[tr])
                        pred[te] = np.ravel(m.predict(X[te]))
                        fold_r2.append(metrics(y[te], pred[te])["R2"])
                    mt = metrics(y, pred)
                    mt.update(fold_R2_mean=np.mean(fold_r2), fold_R2_sd=np.std(fold_r2), seed=seed)
                    per_seed.append(mt)
                    oof[f"{name}_s{seed}"] = pred
                agg = pd.DataFrame(per_seed).mean(numeric_only=True)
                rows.append({"target": target, "protocol": protocol, "model": name,
                             "R2": agg.R2, "RMSE": agg.RMSE, "MAE": agg.MAE,
                             "R2_sd_seeds": pd.DataFrame(per_seed).R2.std(),
                             "fold_R2_mean": agg.fold_R2_mean, "fold_R2_sd": agg.fold_R2_sd,
                             "sec": time.time() - t0})
                print(f"{target:5s} {protocol:6s} {name:10s} R2={agg.R2:6.3f} RMSE={agg.RMSE:6.2f} "
                      f"MAE={agg.MAE:5.2f}  ({time.time() - t0:5.1f}s)", flush=True)
            oof.to_csv(RES / f"bench_oof_{target}_{protocol}.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "T02_benchmark_metrics.csv", index=False)
    piv = out.pivot_table(index="model", columns=["target", "protocol"], values="R2").round(3)
    print(piv.sort_values(("char", "group"), ascending=False).to_string())


if __name__ == "__main__":
    run()
