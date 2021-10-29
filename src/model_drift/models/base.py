import pytorch_lightning as pl

class VisionModuleBase(pl.LightningModule):

    def __init__(self, labels=None, params=None):
        super().__init__()
        self.labels = labels

    @classmethod
    def add_common_args(cls, parser):
        return parser
