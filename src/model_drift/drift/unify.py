import numpy as np
import pandas as pd
import six
from sklearn.feature_selection import mutual_info_classif

from model_drift.helpers import prepare_output_csv, filter_columns, \
    align_frames


def calc_stats(other_df, standardize_dates):
    standardize_ix = pd.date_range(*standardize_dates)
    stats = other_df.dropna(axis=1).reindex(standardize_ix)
    stats = stats.agg(["mean", "std"])
    return stats


def standardize(other_df, std_dates=None, std_stats=None, clip=None) -> pd.DataFrame:
    if std_stats is None:
        std_stats = calc_stats(other_df, std_dates)
    otherstd = other_df.copy()

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


def load_weights(fn):
    return pd.read_csv(fn, index_col=[0,1,2]).iloc[:,0]

def load_stats(fn):
    pd.read_csv(fn, index_col=[0], header=[0,1,2])

class DriftUnifier(object):

    def __init__(self, which="mean", performance_col=("performance", "micro avg", "auroc"), stat=('distance')
                 ):

        self.which = which
        self.performance_col = performance_col
        self.stat = stat



    def unify(self, result_csv, std_dates=None, std_stats=None, clip=10, metric_weights=None, include=None,
              exclude=None):
        if std_stats is None and std_dates is None:
            raise ValueError("Must pass standardization dates or weights.")

        if metric_weights is None:
            pass  # do all ones

        error_df, combined_df = prepare_output_csv(result_csv, self.which)
        perf_df = combined_df[self.performance_col]
        other_df = filter_columns(combined_df, exclude=['performance', 'count'])
        other_df = filter_columns(combined_df, include=self.stat)
        other_df = filter_columns(combined_df, include=include, exclude=exclude)
        other_df = standardize(other_df, std_dates=std_dates, std_stats=std_stats, clip=clip)

        if metric_weights is None:
            metric_weights = {c: 1 for c in other_df}

        return -w_avg(other_df, weights=metric_weights)


def correlate_performance(perf_dataframe, other_dataframe, **kwargs):
    X, Y = align_frames(perf_dataframe, other_dataframe, **kwargs)
    return X.corrwith(Y).rename("correlation")


def mutual_info_performance(perf_dataframe, other_dataframe, bins=10, **kwargs):
    X, Y = align_frames(perf_dataframe, other_dataframe, **kwargs)
    Y, bins = pd.cut(Y, bins=bins, retbins=True)
    info_gain = mutual_info_classif(X.values, Y.cat.codes)
    return pd.Series(info_gain, index=X.columns.tolist(), name="info_gain")


def w_avg(df, weights):
    cols = df.columns
    cols = [c for c in weights if c in cols]
    weights = np.array([weights[c] for c in cols])
    weights = weights / weights.sum()
    tmp = df[cols].copy()
    for c, w in zip(cols, weights):
        tmp[c] = tmp[c] * w
    return tmp.sum(axis=1, skipna=False)
