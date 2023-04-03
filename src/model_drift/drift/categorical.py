#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from collections import Counter

import numpy as np
from scipy.stats import chi2_contingency, chi2

from model_drift.drift.base import BaseDriftCalculator


def merge_freqs(ref_counts, sample):
    sample_counts = Counter(sample)
    keys = set().union(ref_counts, sample_counts)
    exp = np.array([ref_counts.get(k, 0) for k in keys])
    obs = np.array([sample_counts.get(k, 0) for k in keys])
    return exp, keys, obs


class ChiSqDriftCalculator(BaseDriftCalculator):
    name = "chi2"

    def __init__(self, q_val=0.1, correction=True, lambda_=None, use_freq=False, include_critical_values=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.q_val = q_val
        self.correction = correction
        self.lambda_ = lambda_
        self.use_freq = use_freq
        self.include_critical_values = include_critical_values

    def convert(self, arg):
        return arg.apply(str)

    def prepare(self, ref, **kwargs):
        self._ref_counts = Counter(ref)
        super().prepare(ref)

    def _predict(self, sample):
        exp, keys, obs = merge_freqs(self._ref_counts, sample)

        if self.use_freq:
            exp = exp / exp.sum()
            obs = obs / obs.sum()

        out = {}
        out['distance'], out['pval'], dof, _ = chi2_contingency(np.vstack([exp, obs]),
                                                                correction=self.correction,
                                                                lambda_=self.lambda_)
        if self.include_critical_values:
            out['critical_value'] = chi2.ppf(1 - self.q_val, dof)
            out['critical_diff'] = out['distance'] - out['critical_value']

        return out
