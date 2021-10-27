class DriftStatBase(object):
    def __init__(self, ref):
        self.ref = ref

    def predict(self, sample):
        raise NotImplementedError()

    def __call__(self, sample):
        return self.predict(sample)
