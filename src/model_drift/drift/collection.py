from .base import DriftStatBase


class StatCollection(DriftStatBase):
    def __init__(self, collection=None):
        self.collection = collection or []

    def predict(self, sample):
        return {dc.name: dc.predict(sample) for dc in self.collection}
