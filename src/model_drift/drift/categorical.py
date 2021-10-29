import numpy as np
from collections import Counter
from scipy.stats import chi2_contingency, chi2
from .base import BaseDriftCalculator


class ChiSqDriftCalculator(BaseDriftCalculator):
    name = "chi2"

    def __init__(self, ref, q_val=0.1, correction=True, lambda_=None):
        super().__init__(ref)
        self.q_val = q_val
        self.ref_counts = Counter(self.ref)
        self.correction = correction
        self.lambda_ = lambda_
        self.use_freq = True

    def predict(self, sample):
        sample_counts = Counter(sample)
        keys = set().union(self.ref_counts, sample_counts)
        exp = np.array([self.ref_counts.get(k, 0) for k in keys])
        obs = np.array([sample_counts.get(k, 0) for k in keys])

        if self.use_freq:
            exp = exp/exp.sum()
            obs = obs/obs.sum()

        out = {}
        out['distance'], out['pval'], out['dof'], _ = chi2_contingency(np.vstack([exp, obs]),
                                                                       correction=self.correction,
                                                                       lambda_=self.lambda_)
        out['critical_value'] = chi2.ppf(1 - self.q_val, out['dof'])
        out['critical_diff'] = out['distance'] - out['critical_value']
        return out
