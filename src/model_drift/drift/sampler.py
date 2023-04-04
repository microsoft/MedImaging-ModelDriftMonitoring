#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
from sklearn.utils import resample


class Sampler(object):
    def __init__(self, sample_size, replacement=True, random_state=None):
        self.sample_size = sample_size
        self.replacement = replacement
        self.random_state = random_state

    def sample_index(self, index, stratify=None):
        if not self.replacement and len(index) < self.sample_size:
            return np.array(index)
        return resample(index, n_samples=self.sample_size, replace=self.replacement, random_state=self.random_state,
                        stratify=stratify)

    def sample(self, sample, stratify=None):
        return sample[self.sample_index(range(len(sample)), stratify=stratify)]

    def sample_iterator(self, sample, n_samples=1, stratify=None):
        for _ in range(n_samples):
            yield self.sample(sample, stratify=stratify)
