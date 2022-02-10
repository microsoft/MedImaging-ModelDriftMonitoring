import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from model_drift.helpers import align_frames


def calc_stats(other_df, standardize_dates):
    standardize_ix = pd.date_range(*standardize_dates)
    stats = other_df.dropna(axis=1).reindex(standardize_ix)
    stats = stats.agg(["mean", "std"])
    return stats


def standardize(other_df, std_dates=None, std_stats=None, clip=None) -> pd.DataFrame:
    if std_stats is None:
        std_stats = calc_stats(other_df, std_dates)
    otherstd = other_df.copy()

    std_stats = std_stats[otherstd.columns].copy()

    # cannot divide by zero
    std_stats.loc["std", std_stats.loc['std'] == 0] = 1
    otherstd = (otherstd - std_stats.loc['mean']) / (std_stats.loc["std"])

    if clip is not None:
        otherstd = otherstd.clip(-1 * clip, clip)

    return otherstd


def calculate_weights(yp, otherstd) -> pd.DataFrame:
    all_corr_df = correlate_performance(yp.rename('auroc'), otherstd)
    all_ig_df = mutual_info_performance(yp.rename('auroc'), otherstd, bins=25)
    m_ = all_ig_df.to_frame().join(all_corr_df.apply(lambda x: max(0, x)).rename('corr')).join(
        all_corr_df.abs().rename('abs(corr)'))
    m_ = m_.join(m_.mean(axis=1).rename('mean[abs(corr),info_gain]'))
    m_ = m_.assign(no_weights=1)
    m_ = m_.fillna(0)

    return m_


def correlate_performance(perf_dataframe, other_dataframe, **kwargs):
    X, Y = align_frames(perf_dataframe, other_dataframe, **kwargs)
    return X.corrwith(Y).rename("correlation")


def mutual_info_performance(perf_dataframe, other_dataframe, bins=10, **kwargs):
    X, Y = align_frames(perf_dataframe, other_dataframe, **kwargs)
    Y, bins = pd.cut(Y, bins=bins, retbins=True)
    info_gain = mutual_info_classif(X.values, Y.cat.codes)
    return pd.Series(info_gain, index=X.columns.tolist(), name="info_gain")


def w_avg(df, weights):
    cols = weights.index.intersection(df.columns)
    weights = np.array([weights[c] for c in cols])
    weights = weights / weights.sum()
    tmp = df[cols].copy()
    for c, w in zip(cols, weights):
        tmp[c] = tmp[c] * w
    return tmp.sum(axis=1, skipna=False)


def calculate_mmc(metrics_df, weights, std_stats, clip=10):
    metrics_std = standardize(metrics_df, std_stats=std_stats, clip=clip)
    return -w_avg(metrics_std, weights=weights)
