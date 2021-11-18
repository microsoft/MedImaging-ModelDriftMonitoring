from .base import BaseDriftCalculator


class DriftCollectionCalculator(BaseDriftCalculator):
    def __init__(self, collection=None):
        self.collection = collection or []

    def _predict(self, sample):
        return {dc.name: dc.predict(sample) for dc in self.collection}
