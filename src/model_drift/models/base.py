#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from argparse import ArgumentParser, Namespace
from typing import Any, List, Tuple, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities.argparse import from_argparse_args, get_init_arguments_and_types


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
