import pandas as pd
import numpy as np
from model_drift.data.utils import nested2series
import tqdm
from collections import defaultdict
from model_drift.data.utils import nested2series, rolling_window_dt_apply
from .collection import DriftCollectionCalculator

# from .base import DriftStatBase

tqdm_func = tqdm.tqdm


def sample_frame(df, day, window='30D'):
    day_dt = pd.to_datetime(day)
    delta = pd.tseries.frequencies.to_offset(window)
    return df.loc[str(day_dt - delta):str(day_dt)]


class TabularDriftCalculator(object):
    # TODO: Handle NaNs and Non-numerics
    def __init__(self, df_ref,):
        self.ref = df_ref
        self.drift_metric_dict = defaultdict(set)
        self._metric_collections = {}

    def auto_add_drift_calculators(self):
        pass

    def add_drift_stat(self, col, drift_cls, **drift_kwargs):
        item = (drift_cls, tuple(sorted(drift_kwargs.items())))
        self.drift_metric_dict[col].add(item)

    def prepare(self):
        for col, drift_metric_set in self.drift_metric_dict.items():
            ref = self.ref[self.col_to_col(col)]
            self._metric_collections[col] = DriftCollectionCalculator([
                drift_cls(ref, **dict(kwargs))
                for drift_cls, kwargs in drift_metric_set
            ])

    def col_to_col(self, col):
        if isinstance(col, tuple) and col not in self.ref:
            return list(col)
        return col

    def _predict_col(self, col, sample):
        return self._metric_collections[col].predict(sample[self.col_to_col(col)])

    def _predict(self, sample, cols=None, include_count=True):
        # ASSERT PREPARED
        if cols is not None:
            cols = [c for c in self._metric_collections.keys() if c in cols]
        else:
            cols = self._metric_collections.keys()
        out = {col: self._predict_col(col, sample) for col in cols}
        if include_count:
            out['count'] = len(sample)
        return out

    def predict(self, sample, include_count=True, sampler=None, n_samples=1, stratify=None, agg=('mean', 'std'), min_periods=None):

        if sampler is None:
            return self._predict(sample, include_count=include_count)

        index = np.array(range(len(sample)))
        sample_ix = list(sampler.sample_iterator(index, n_samples=n_samples, stratify=stratify))
        samples = {i: nested2series(self._predict(sample.iloc[ix])) for i, ix in enumerate(sample_ix)}

        if n_samples == 1:
            return samples[0]

        obs = nested2series(self._predict(sample, include_count=include_count))

        if agg is None:
            samples["obs"] = obs
            return pd.concat(samples, axis=1)

        return pd.concat(samples, axis=1).agg(agg, axis=1).join(obs.rename('obs')).stack()

    def drilldown(self, df, dates, cols=None, window='30D', include_ref=True):

        if cols is None:
            cols = list(set(self.ref.columns).intersection(df))
        out = []
        if include_ref:
            out.append(self.ref[cols].assign(src="_ref"))

        samples = {day: sample_frame(df[cols], day, window=window).assign(src=day) for day in dates}
        stats = pd.concat({date: nested2series(self.predict(sample, cols=cols)) for date, sample in samples.items()},
                          axis=1)
        data = pd.concat(out + list(samples.values())).reset_index()
        return stats, data

    def rolling_window_predict(self, dataframe, sampler=None, n_samples=1, stratify=None, agg=('mean', 'std'), **kwargs):
        kwargs["include_count"] = False
        return rolling_window_dt_apply(dataframe, lambda x: self.predict(x, include_count=True,
                                                                         sampler=sampler, n_samples=n_samples,
                                                                         stratify=stratify, agg=('mean', 'std')),
                                       **kwargs)
