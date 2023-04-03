#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.special import kolmogi
from scipy.stats import ks_2samp

from model_drift.drift.base import BaseDriftCalculator


class NumericBaseDriftCalculator(BaseDriftCalculator):
    def convert(self, arg):
        return pd.to_numeric(arg, errors="coerce")


class KSDriftCalculator(NumericBaseDriftCalculator):
    name = "ks"

    def __init__(self, q_val=0.1, alternative='two-sided', mode='asymp', average='macro', include_critical_value=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.q_val = q_val
        self.alternative = alternative
        self.mode = mode
        self.average = average
        self.include_critical_value = include_critical_value

    def _predict(self, sample):
        nref = len(self._ref)
        nobs = len(sample)
        out = {}
        try:
            out["distance"], out['pval'] = ks_2samp(self._ref, sample, alternative=self.alternative,
                                                    mode=self.mode)
        except TypeError:
            out["distance"], out['pval'] = float("NaN"), float("NaN")

        if self.include_critical_value:
            out['critical_value'] = self.calc_critical_value(nref, nobs, self.q_val)
            out['critical_diff'] = out["distance"] - out['critical_value']

        return out

    @staticmethod
    def calc_critical_value(n1, n2, q=.01):
        return kolmogi(q) * np.sqrt((n1 + n2) / (n1 * n2))


class BasicDriftCalculator(NumericBaseDriftCalculator):
    name = "stats"

    def convert(self, arg):
        return pd.to_numeric(arg, errors="coerce")

    def _predict(self, sample):
        sample = pd.to_numeric(sample, errors="coerce")
        return {
            "mean": np.mean(sample),
            "std": np.std(sample),
            "median": np.median(sample)
        }
