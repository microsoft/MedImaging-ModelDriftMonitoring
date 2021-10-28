import pandas as pd
import tqdm
from model_drift.helpers import nested2series

tqdm_func = tqdm.tqdm

class BaseDriftCalculator(object):
    def __init__(self, ref):
        self.ref = ref

    def predict(self, sample):
        raise NotImplementedError()

    def __call__(self, sample):
        return self.predict(sample)



