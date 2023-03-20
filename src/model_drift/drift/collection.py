#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from .base import BaseDriftCalculator


class DriftCollectionCalculator(BaseDriftCalculator):
    def __init__(self, collection=None):
        super().__init__()
        self.collection = collection or []
        
    def prepare(self, ref):
        for metric in self.collection:
            metric.prepare(ref)
    
    def _predict(self, sample):
        return {dc.name: dc.predict(sample) for dc in self.collection}
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return str(self.collection)
