import pandas as pd
import tqdm
from collections import defaultdict
from model_drift.helpers import nested2series
from .collection import DriftCollectionCalculator
# from .base import DriftStatBase

tqdm_func = tqdm.tqdm


def sample_frame(df, day, window='30D'):
    day_dt = pd.to_datetime(day)
    delta = pd.tseries.frequencies.to_offset(window)
    return df.loc[str(day_dt - delta):str(day_dt)]


class TabularDriftCalculator(object):
    ## TODO: Handle NaNs and Non-numerics
    def __init__(self, df_ref):
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
            ref = self.ref[col]
            self._metric_collections[col] = DriftCollectionCalculator([
                drift_cls(ref, **dict(kwargs))
                for drift_cls, kwargs in drift_metric_set
            ])

    def predict(self, sample, cols=None, include_count=True):
        # ASSERT PREPARED
        if cols is not None:
            cols = [c for c in self._metric_collections.keys() if c in cols]
        else:
            cols = self._metric_collections.keys()
        out = {col: self._metric_collections[col](sample[col]) for col in cols}
        if include_count:
            out['count'] = len(sample)
        return out

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

    def rolling_window_predict(self, dataframe, window="30D", stride="D", min_periods=1):
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            raise ValueError()

        tmp_index = pd.date_range(
            dataframe.index.min(), dataframe.index.max(), freq=stride
        )

        cols = list(dataframe)
        delta = pd.tseries.frequencies.to_offset(window)
        fdelta = pd.tseries.frequencies.to_offset(window) * 0
        bdelta = delta - fdelta

        def _apply(i):
            x = dataframe.loc[str(i - bdelta):str(i + fdelta)]
            if len(x) < min_periods:
                return None
            preds = self.predict(x, include_count=False)
            preds["count"] = len(x)
            return nested2series(preds)

        out = {}
        for i in tqdm_func(tmp_index):
            out[i] = _apply(i)
        return pd.concat(out, axis=0).unstack(level=0).T
