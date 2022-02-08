import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace
from pytorch_lightning.utilities.argparse import from_argparse_args, get_init_arguments_and_types
from typing import Any, List, Tuple, Union


class VisionModuleBase(pl.LightningModule):

    def __init__(self, labels=None, params=None):
        super().__init__()
        self.labels = labels

    @classmethod
    def add_common_args(cls, parser):
        return parser

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
        return from_argparse_args(cls, args, **kwargs)

    @classmethod
    def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
        return get_init_arguments_and_types(cls)
