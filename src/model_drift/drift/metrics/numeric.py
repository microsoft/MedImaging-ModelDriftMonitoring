#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.special import kolmogi
from scipy.stats import ks_2samp

from .base import BaseDriftCalculator


class KSDriftCalculator(BaseDriftCalculator):
    name = "ks"

    def __init__(self, ref, q_val=0.1, alternative='two-sided', mode='asymp', average='macro'):
        super().__init__(ref)
        self.q_val = q_val
        self.alternative = alternative
        self.mode = mode
        self.average = average

    def _predict(self, sample):
        # sample = np.stack(sample.values

        nref = len(self.ref)
        nobs = len(sample)

        sample = pd.to_numeric(sample, errors='coerce').astype(float)
        out = {}
        try:
            out["distance"], out['pval'] = ks_2samp(self.ref, sample, alternative=self.alternative,
                                                    mode=self.mode)
        except TypeError:
            out["distance"], out['pval'] = float("NaN"), float("NaN")

        out['critical_value'] = self.calc_critical_value(nref, nobs, self.q_val)
        out['critical_diff'] = out["distance"] - out['critical_value']

        # outs = pd.DataFrame(outs)

        return out

    @staticmethod
    def calc_critical_value(n1, n2, q=.01):
        return kolmogi(q) * np.sqrt((n1 + n2) / (n1 * n2))


class BasicDriftCalculator(BaseDriftCalculator):
    name = "stats"

    def _predict(self, sample):
        sample = pd.to_numeric(sample, errors="coerce")
        return {"mean": np.mean(sample),
                "std": np.std(sample),
                "median": np.median(sample)
                }
