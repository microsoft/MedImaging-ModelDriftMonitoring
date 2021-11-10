import torch
from torchmetrics import AUROC
from torchmetrics.functional import auroc
from .base import BaseDriftCalculator
import pandas as pd
import numpy as np


def macro_auc(scores, labels, skip_missing=True):
    if len(scores) == 0:
        return float('NaN')
    N = labels.shape[1]
    aucs = [0]*N
    for i in range(N):
        try:
            aucs[i] = auroc(torch.tensor(scores[i]), torch.tensor(labels[i]).long()).numpy()
        except Exception as e:
            if "No positive samples in targets" not in str(e):
                raise
            aucs[i] = float('NaN')

    aucs = np.array(aucs)
    c = (~np.isnan(aucs)).sum() if skip_missing else N
    return np.nansum(aucs)/c


def micro_auc(scores, labels):
    return float(auroc(torch.tensor(scores), torch.tensor(labels).long(), average='micro').numpy())


class AUROCCalculator(BaseDriftCalculator):
    name = "auroc"

    def __init__(self, ref=None, label_col=None, score_col=None, average='micro', ignore_nan=True):
        super().__init__(None)
        self.label_col = label_col
        self.score_col = score_col
        self.average = average

    def predict(self, sample):
        labels = sample.iloc[:, 1] if self.label_col is None else sample[self.label_col]
        scores = sample.iloc[:, 0] if self.score_col is None else sample[self.score_col]
        labels = np.stack(labels.values)
        scores = np.stack(scores.values)

        if self.average == "macro":
            return macro_auc(scores, labels)
        return micro_auc(scores, labels)
