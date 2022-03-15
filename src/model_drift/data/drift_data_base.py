#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import copy

import pandas as pd
import six

from .utils import remap_labels, binarize_label


class ModelDriftData(object):
    LABEL_COL = None
    DATASET_COLS = None
    DATASET_CLASS = None

    def __init__(
            self,
            df,
            label_map=None,
    ) -> None:
        if df is None or isinstance(df, six.string_types):
            df = self.read_csv(df)
        self.df = df
        self.label_map = label_map
        self.is_prepared = False

    def prepare(self):
        if not self.is_prepared:
            self.remap_labels()
            self.label_list_to_khot()
            self.is_prepared = True

    def remap_labels(self, col=None):
        col = col or self.LABEL_COL
        self.df[col] = remap_labels(self.df[col], label_map=self.label_map)

    def merge(self, other_df, **merge_kwargs):
        merge_kwargs.setdefault("how", "left")
        self.df = self.df.merge(other_df, **merge_kwargs)

    def label_list_to_khot(self, col=None):
        col = col or self.LABEL_COL
        if col not in self.df:
            raise KeyError(f"label column {col} not in dataframe")
        binary_labels = binarize_label(self.df[col])
        self.df = self.df.join(binary_labels)

    def __len__(self):
        return len(self.df)

    def _copy_with_df(self, df):
        cpy = copy.copy(self)
        cpy.df = df
        return cpy

    def __copy__(self):
        return type(self)(df=copy.copy(self.df), label_map=copy.copy(self.label_map))

    @staticmethod
    def read_csv(csv_file=None):
        return pd.read_csv(csv_file, low_memory=False)

    @property
    def classes(self):
        return list(self.label_map)

    def to_dataset(self, folder_dir, cls=None, **kwargs):
        if cls is None:
            cls = self.DATASET_CLASS
        return cls(
            folder_dir,
            self.df,
            **kwargs
        )

    @classmethod
    def from_csv(cls, csv_file=None, **init_kwargs):
        return cls(cls.read_csv(csv_file), **init_kwargs)

    def to_csv(self, *args, **kwargs):
        self.df.to_csv(*args, **kwargs)

    @classmethod
    def splits(cls, csv_file=None, studydate_index=False, **init_kwargs):
        raise NotImplementedError()

    def head(self, *args, **kwargs):
        self._copy_with_df(self.head(*args, **kwargs))

    def sample(self, *args, **kwargs):
        self._copy_with_df(self.df.sample(*args, **kwargs))

    def __repr__(self):
        return repr(self.df)

    def save_df(self, filename="./datalist.csv"):
        self.df.to_csv(filename)
