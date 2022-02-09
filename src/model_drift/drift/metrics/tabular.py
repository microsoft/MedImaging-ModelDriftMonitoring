import numpy as np
import pandas as pd
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
    def __init__(self, df_ref, ):
        self.ref = df_ref
        self.drift_metric_dict = defaultdict(set)
        self._metric_collections = {}
        self.name_to_cols = dict()
        self.squeeze = dict()

    def auto_add_drift_calculators(self):
        pass

    def add_drift_stat(self, name, drift_cls, col=None, include_stat_name=True, **drift_kwargs):
        col = col or name
        item = (drift_cls, tuple(sorted(drift_kwargs.items())))
        self.drift_metric_dict[name].add(item)
        self.name_to_cols[name] = col
        self.squeeze[name] = not include_stat_name

    def prepare(self):
        for name, drift_metric_set in self.drift_metric_dict.items():
            col = self.name_to_cols[name]
            ref = self.ref[self.col_to_col(col)]
            if self.squeeze[name] and len(drift_metric_set) < 2:
                drift_cls, kwargs = list(drift_metric_set)[0]
                self._metric_collections[name] = drift_cls(ref, **dict(kwargs))
            else:
                self._metric_collections[name] = DriftCollectionCalculator([
                    drift_cls(ref, **dict(kwargs))
                    for drift_cls, kwargs in drift_metric_set
                ])

    def col_to_col(self, col):
        if isinstance(col, tuple) and col not in self.ref:
            return list(col)
        return col

    def _predict_col(self, name, sample):
        col = self.name_to_cols[name]
        try:
            return self._metric_collections[name].predict(sample[self.col_to_col(col)])
        except:  # noqa
            print(f"Failed on {name}")
            raise

    def _predict(self, sample, names=None, include_count=True):
        # ASSERT PREPARED
        if names is not None:
            names = [c for c in self._metric_collections.keys() if c in names]
        else:
            names = self._metric_collections.keys()
        out = {name: self._predict_col(name, sample) for name in names}
        if include_count:
            out['count'] = len(sample)
        return out

    def predict(self, sample, include_count=True, sampler=None, n_samples=1, stratify=None, agg=('mean', 'std')):

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

    def rolling_window_predict(self, dataframe, sampler=None, n_samples=1, stratify=None, agg=('mean', 'std', 'median'),
                               **kwargs):
        kwargs["include_count"] = False
        return rolling_window_dt_apply(dataframe, lambda x: self.predict(x, include_count=True,
                                                                         sampler=sampler, n_samples=n_samples,
                                                                         stratify=stratify, agg=agg),
                                       **kwargs)
