"""Compare zero and one anchor on the same query predictions, with source-group resampling."""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METHODS = ['CB+BLUP', 'HTS+CB-ME']
PAIR = ['method', 'target', 'seed', 'fold', 'camp', 'rep', 'excel_row']


def compare():
    frame = pd.read_parquet(
        ROOT / 'results' / 'kshot_confirm_long.parquet',
        columns=PAIR + ['k', 'y', 'pred'],
        filters=[('method', 'in', METHODS), ('k', 'in', [0, 1])],
    )
    frame['error'] = (frame['pred'] - frame['y']).abs()
    base = frame.loc[frame.k == 0, PAIR + ['error']]
    anchor = frame.loc[frame.k == 1, PAIR + ['error']]
    matched = anchor.merge(base, how='left', on=PAIR, validate='one_to_one',
                           suffixes=('_one', '_zero'))
    assert len(matched) == len(anchor) and matched.error_zero.notna().all()
    runs = pd.read_csv(ROOT / 'data_clean' / 'pyrolysis_clean.csv', usecols=['excel_row', 'group'])
    matched = matched.merge(runs, how='left', on='excel_row', validate='many_to_one')
    assert matched['group'].notna().all()
    out = []
    for (target, method), scores in matched.groupby(['target', 'method'], sort=True):
        # Equal weight per query occurrence, preserving the original scoring convention.
        # Resample the same set of source groups for both methods for a given product.
        group = scores.groupby('group').agg(total_one=('error_one', 'sum'),
                                             total_zero=('error_zero', 'sum'),
                                             n=('excel_row', 'size'))
        rng = np.random.default_rng(21)
        sample = rng.integers(len(group), size=(10000, len(group)))
        delta = group.total_one.to_numpy() - group.total_zero.to_numpy()
        draw = delta[sample].sum(axis=1) / group.n.to_numpy()[sample].sum(axis=1)
        lo, hi = np.quantile(draw, [0.025, 0.975])
        out.append(dict(product=target, method=method, groups=len(group), campaigns=scores.camp.nunique(),
                        runs=scores.excel_row.nunique(), predictions=len(scores),
                        zero=scores.error_zero.mean(), one=scores.error_one.mean(),
                        difference=delta.sum()/len(scores), lower=lo, upper=hi))
    return pd.DataFrame(out)


if __name__ == '__main__':
    result = compare()
    result.to_csv(ROOT / 'tables' / 'T23_matched_anchor_comparison.csv', index=False)
    print(result.round(3).to_string(index=False))
