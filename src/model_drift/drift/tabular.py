#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
import yaml

from model_drift.data.utils import nested2series, rolling_window_dt_apply
from model_drift.drift.base import BaseDriftCalculator
from model_drift.drift.collection import DriftCollectionCalculator
from model_drift.io.serialize import get_dumper, ModelDriftEncoder

tqdm_func = tqdm.tqdm


def sample_frame(df, day, window='30D'):
    day_dt = pd.to_datetime(day)
    delta = pd.tseries.frequencies.to_offset(window)
    return df.loc[str(day_dt - delta):str(day_dt)]


class TabularDriftCalculator(BaseDriftCalculator):

    # TODO: Handle NaNs and Non-numerics
    def __init__(self):

        self.metrics = {}
        self._metric_collections = {}
        self._hist_collections = {}
        self._name_to_cols = {}

    def add_drift_stat(self, name: str, metric: BaseDriftCalculator, col: str = None,
                       include_stat_name: bool = True,
                       group: str = None,
                       drilldown: bool = False):

        if isinstance(col, tuple):
            col = list(col)

        if name not in self.metrics:
            self.metrics[name] = list()

        self.metrics[name].append(
            {"column": col or name, "squeeze": not include_stat_name, "drilldown": drilldown, "metric": metric,
             "group": group})

    def __str__(self) -> str:
        return yaml.dump(self, Dumper=get_dumper())

    def clear_drift_state(self, name):
        if name in self.metrics:
            del self.metrics[name]

        if name in self._metric_collections:
            del self._metric_collections[name]

        if name in self._hist_collections:
            del self._hist_collections[name]

        if name in self._name_to_cols:
            del self._name_to_cols[name]

    def _name2value(self, key):
        return {name: [d[key] for d in mlist if key in d] for name, mlist in self.metrics.items()}

    @property
    def groups(self):
        groups = {}

        for name, groupList in self._name2value('group').items():
            for g in groupList:
                groups.setdefault(g, [])
                if name not in groups[g]:
                    groups[g].append(name)

        return groups

    @staticmethod
    def _prepare_metric_col(ref_, squeeze, metric_lst):
        if squeeze and len(metric_lst) < 2:
            metric = list(metric_lst)[0]
        else:
            metric = DriftCollectionCalculator(metric_lst)
        metric.prepare(ref_)
        return metric

    def prepare(self, ref):
        self._ref = ref
        self._metric_collections = {}

        for name, metric_list in self.metrics.items():
            squeeze_col = False
            drilldowns = []
            metrics = []
            for metric_d in metric_list:
                col, squeeze, drilldown, metric = metric_d['column'], metric_d['squeeze'], metric_d['drilldown'], \
                                                  metric_d['metric']
                squeeze_col = squeeze | squeeze_col
                if name in self._name_to_cols:
                    pass  # add warning
                self._name_to_cols[name] = col

                if drilldown:
                    drilldowns.append(metric)
                else:
                    metrics.append(metric)

            if len(drilldowns):
                self._hist_collections[name] = self._prepare_metric_col(ref[col], squeeze, drilldowns)

            if len(metrics):
                self._metric_collections[name] = self._prepare_metric_col(ref[col], squeeze, metrics)

    def col_to_col(self, col):
        if isinstance(col, tuple) and col not in self.ref:
            return list(col)
        return col

    def _predict_col(self, name, sample, metric):
        col = self._name_to_cols[name]
        try:
            return metric.predict(sample[self.col_to_col(col)])
        except:  # noqa
            print(f"Failed on {name}")
            raise

    def _predict(self, sample, metric_collection, names=None):
        # ASSERT PREPARED
        if names is not None:
            names = [c for c in metric_collection.keys() if c in names]
        else:
            names = metric_collection.keys()
        out = {name: self._predict_col(name, sample, metric_collection[name]) for name in names}
        return out

    def predict(self, sample, include_count=True, sampler=None, n_samples=1, stratify=None,
                agg=('mean', 'std')):

        if sampler is None:
            return self._predict(sample, self._metric_collections)

        indices = np.array(range(len(sample)))
        sample_ix = list(sampler.sample_iterator(indices, n_samples=n_samples, stratify=stratify))
        samples = {i: nested2series(self._predict(sample.iloc[ix], self._metric_collections)) for i, ix in
                   enumerate(sample_ix)}

        if n_samples == 1:
            return samples[0]

        obs = nested2series(self._predict(sample, self._metric_collections))

        if agg is None:
            samples["obs"] = obs
            return pd.concat(samples, axis=1)

        return pd.concat(samples, axis=1).agg(agg, axis=1).join(obs.rename('obs')).stack()

    def drilldown(self, sample, **kwargs):
        return self._predict(sample, self._hist_collections, **kwargs)

    def rolling_window_predict(self, dataframe, sampler=None, n_samples=1, stratify=None, agg=('mean', 'std', 'median'),
                               output_dir="./outputs/",
                               **kwargs):

        output_dir = Path(output_dir)

        with open(output_dir.joinpath("drift_config.yml"), "w") as f:
            print(yaml.dump(self, Dumper=get_dumper()), file=f)

        with open(output_dir.joinpath("groups.json"), "w") as f:
            print(json.dumps(self.groups), file=f)

        history_path = output_dir.joinpath("history")
        drilldowns = self.drilldown(self._ref)

        history_path.mkdir(parents=True, exist_ok=True)
        with open(history_path.joinpath("ref.json"), "w") as f:
            print(json.dumps({
                "info": {'nsamples': len(self._ref)},
                "drilldowns": drilldowns
            }, indent=1, cls=ModelDriftEncoder), file=f)

        return rolling_window_dt_apply(dataframe,
                                       lambda window: self.predict(window, sampler=sampler, n_samples=n_samples,
                                                                   stratify=stratify, agg=agg),
                                       drilldown_func=lambda window: self.drilldown(window),
                                       output_dir=str(history_path),
                                       **kwargs)
