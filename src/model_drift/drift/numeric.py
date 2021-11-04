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
        self.average=average

    def predict(self, sample):
        # sample = np.stack(sample.values

        nref = len(self.ref)
        nobs = len(sample)
        # assert self.ref.shape[samplee[1]
        outs = []
        out = {}
        out["distance"], out['pval'] = ks_2samp(self.ref, sample, alternative=self.alternative,
                                                mode=self.mode)
        out['critical_value'] = self.calc_critical_value(nref, nobs, self.q_val)
        out['critical_diff'] = out["distance"] - out['critical_value']

        # outs = pd.DataFrame(outs)

        return out

    @staticmethod
    def calc_critical_value(n1, n2, q=.01):
        return kolmogi(q) * np.sqrt((n1 + n2) / (n1 * n2))


class BasicDriftCalculator(BaseDriftCalculator):
    name = "stats"

    def predict(self, sample):

        sample = pd.to_numeric(sample, errors="coerce")
        return {"mean": np.mean(sample),
                "std": np.std(sample),
                "median": np.median(sample)
                }
