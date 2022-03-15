#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pandas as pd

from model_drift.data.utils import nested2series


class BaseDriftCalculator(object):
    def __init__(self, ref):
        self.ref = ref

    def _predict(self, sample):
        raise NotImplementedError()

    def predict(self, sample, sampler=None, n_samples=1, stratify=None, agg=('mean', 'std')):

        if sampler is None:
            return self._predict(sample)

        samples = sampler.sample_iterator(sample, n_samples=n_samples, stratify=stratify)
        samples = {i: nested2series(self.predict(s)) for i, s in enumerate(samples)}

        if n_samples == 1:
            return samples[0]

        obs = nested2series(self._predict(sample))

        if agg is None:
            samples["obs"] = obs
            return pd.concat(samples, axis=1)

        return pd.concat(samples, axis=1).agg(agg, axis=1).join(obs.rename('obs')).stack()
