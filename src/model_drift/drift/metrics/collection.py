#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from .base import BaseDriftCalculator


class DriftCollectionCalculator(BaseDriftCalculator):
    def __init__(self, collection=None):
        self.collection = collection or []

    def _predict(self, sample):
        return {dc.name: dc.predict(sample) for dc in self.collection}
