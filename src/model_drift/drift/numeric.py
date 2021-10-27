import numpy as np
from scipy.special import kolmogi
from scipy.stats import ks_2samp
from .base import DriftStatBase


class KSDriftStat(DriftStatBase):
    name = "ks"

    def __init__(self, ref, q_val=0.1, alternative='two-sided', mode='asymp', ):
        super().__init__(ref)
        self.q_val = q_val
        self.alternative = alternative
        self.mode = mode

    def predict(self, sample):
        out = {}
        nref = len(self.ref)
        nobs = len(sample)
        out["distance"], out['pval'] = ks_2samp(self.ref, sample, alternative=self.alternative,
                                                mode=self.mode)
        out['critical_value'] = self.calc_critical_value(nref, nobs, self.q_val)
        out['critical_diff'] = out["distance"] - out['critical_value']
        return out

    @staticmethod
    def calc_critical_value(n1, n2, q=.01):
        return kolmogi(q) * np.sqrt((n1 + n2) / (n1 * n2))


class BasicDriftStat(DriftStatBase):
    name = "stats"

    def predict(self, sample):
        return {("mean", "ref"): np.mean(self.ref),
                ("std", "ref"): np.std(self.ref),
                ("median", 'ref'): np.median(self.ref),
                ("mean", "sample"): np.mean(sample),
                ("std", "sample"): np.std(sample),
                ("median", 'sample'): np.median(sample)
                }
