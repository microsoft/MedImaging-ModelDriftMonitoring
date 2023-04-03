#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import itertools
import json
import logging
import os
import random
import sys
from collections.abc import Iterator
from distutils import dir_util
from functools import reduce

import numpy as np
import pandas as pd
import six
import tqdm
from joblib import Parallel


def rolling_dt_apply_with_stride(
        dataframe,
        function,
        window="30D",
        stride="D",
        unique_only=False,
        center=False,
        min_periods=None,
) -> pd.DataFrame:
    """
    Apply a function to rolling windows of a DataFrame with a given stride and window size.
    The resulting DataFrame contains the output of the function applied to each window.


    Args:
        dataframe (pandas.DataFrame): The input DataFrame.
        function (function): The function to apply to each rolling window.
        window (str): The size of the rolling window as a time delta string.
        stride (str): The stride of the rolling window as a time delta string.
        unique_only (bool, optional): Whether to apply the rolling window only to
            unique indices in the input DataFrame. Default is False.
        center (bool, optional): Whether the window should be centered. Default is
            False.
        min_periods (int, optional): The minimum number of observations in a window
            required to have a value. Default is None.

    Returns:
        pandas.DataFrame: The resulting DataFrame, which is the output of applying
        the function to each rolling window of the input DataFrame.

    Raises:
        ValueError: If the `window` or `stride` parameters are not valid time delta
            strings.
    """
    if unique_only:
        tmp_index = dataframe.index.unique()
    else:
        tmp_index = pd.date_range(dataframe.index.min(), dataframe.index.max(), freq=stride)

    try:
        delta = pd.tseries.frequencies.to_offset(window)
        fdelta = (delta / 2) if center else pd.tseries.frequencies.to_offset(window) * 0
        bdelta = delta - fdelta
    except TypeError as e:
        raise ValueError("Centering does not work with all windows and strides") from e

    def _apply(i):
        window = dataframe[i - bdelta: i + fdelta]
        if min_periods is not None and len(window) < min_periods:
            return None
        return window.agg(function)

    return pd.concat({i: _apply(i) for i in tmp_index}, axis=0).unstack()


def copytree(src, dst):
    dir_util.copy_tree(str(src), str(dst))


def modelpath2name(model_path):
    return model_path.replace('/', '-').replace('=', "-")


def print_env():
    print("--- ENV VARIABLES ---")
    for k, v in sorted(os.environ.items()):
        print(f" {k}={v}")
    print("--------------------")


def argsdict2list(d):
    return list(itertools.chain(*[('--' + k, v) for k, v in d.items()]))


def read_jsonl(fn):
    with open(fn, 'r') as f:
        return [json.loads(line) for line in f.readlines()]


def jsonl_files2dataframe(jsonl_files, converter=None, refresh_rate=None, **kwargs):
    if isinstance(jsonl_files, six.string_types):
        jsonl_files = [jsonl_files]

    if converter is None:
        def converter(x): return x

    df = []
    for fn in jsonl_files:
        with open(fn, 'r') as f:
            lines = f.readlines()
            if refresh_rate is not None:
                kwargs['miniters'] = int(len(lines) * refresh_rate)
            for line in tqdm.tqdm(lines, **kwargs):
                df.append(converter(json.loads(line)))
    return pd.json_normalize(df)


def basic_logging(name=None, level=logging.INFO, output_file=None,
                  fmt='[%(asctime)s] %(levelname)s [%(name)s] %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if output_file is not None:
        file_handler = logging.FileHandler(output_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def merge_frames(*dfs, **join_kwargs):
    join_kwargs.setdefault('how', "inner")
    return reduce(lambda df1, df2: df1.join(df2, **join_kwargs), dfs)


def column_xs(df, include=None, exclude=None):
    if isinstance(include, six.string_types):
        include = [include]

    if isinstance(exclude, six.string_types):
        exclude = [exclude]

    cols = df.columns.tolist()
    if include is not None:
        cols = [col for col in cols if any(i in col for i in include)]

    exclude = exclude or []
    cols = [col for col in cols if not any(e in col for e in exclude)]

    return cols


def filter_columns(df, include=None, exclude=None):
    cxs = column_xs(df, include=include, exclude=exclude)
    return df[cxs]


def flatten_index(df, sep='.'):
    def __flatten_index(c):
        if isinstance(c, six.string_types):
            return c
        return sep.join(c)

    _df = df.copy()
    _df.columns = [__flatten_index(col) for col in _df.columns]
    return _df


def align_frames(perf_dataframe, other_dataframe, how='inner', include=None, exclude=None):
    corr_cols = column_xs(other_dataframe, include=include, exclude=exclude)
    if isinstance(perf_dataframe, pd.Series):
        target_col = perf_dataframe.name
    else:
        target_col = list(perf_dataframe.columns)[0]
        perf_dataframe = perf_dataframe[target_col]
    mdf = other_dataframe[corr_cols].join(perf_dataframe, how=how)
    return mdf[corr_cols], mdf[target_col]


def df_standard_scale(idf, nstd=1):
    stats = idf.agg(["mean", "std"])
    return (idf - stats.loc['mean']) / (stats.loc["std"])


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, tqdm_kwargs=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self.tqdm_kwargs = tqdm_kwargs or {}
        self.miniters = tqdm_kwargs.get('miniters', None)
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm.tqdm(disable=not self._use_tqdm, total=self._total,
                       **self.tqdm_kwargs) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def do_refresh(self):
        if self.miniters is None:
            return True
        return (self.n_completed_tasks == self.n_dispatched_tasks or
                (self.n_completed_tasks % self.miniters == 0))

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks

        if self.do_refresh():
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()


def load_vae_preds(vae_pred_file, vae_col='mu'):
    print("loading dataset vae results")
    vae_df = jsonl_files2dataframe(vae_pred_file, desc="reading VAE results", refresh_rate=.1)
    vae_df = pd.concat(
        [
            vae_df,
            pd.DataFrame(vae_df[vae_col].values.tolist(), columns=[f"{vae_col}.{c:0>3}" for c in range(128)])
        ],
        axis=1
    )

    return vae_df


def load_score_preds(label_cols, scores_pred_file, score_col="activation"):
    print("loading dataset predicted probabilities")
    scores_df = jsonl_files2dataframe(scores_pred_file, desc="reading classifier results", refresh_rate=.1)
    scores_df = pd.concat(
        [
            scores_df,
            pd.DataFrame(scores_df[score_col].values.tolist(), columns=[f"{score_col}.{c}" for c in label_cols])
        ],
        axis=1)

    return scores_df


def create_ood_dataframe(outside_data, pct, counts, start_date=None, end_date=None, shuffle=False):
    """
    Create an out-of-distribution (OOD) DataFrame by injecting a percentage of
    outside data into the input DataFrame.

    Args:
        outside_data (pandas.DataFrame): The outside data to inject into the input
            DataFrame.
        pct (float): The percentage of outside data to inject. Must be between 0 and
            1.
        counts (pandas.DataFrame): A DataFrame of daily counts.
        start_date (str, optional): A start date for the daily counts. Default is
            None.
        end_date (str, optional): An end date for the daily counts. Default is None.
        shuffle (bool, optional): Whether to shuffle the rows of the resulting
            DataFrame. Default is False.

    Returns:
        pandas.DataFrame: The resulting out-of-distribution (OOD) DataFrame, which
        contains a percentage of outside data injected into the input DataFrame.

    Raises:
        ValueError: If the percentage is not between 0 and 1.
    """
    if start_date is None:
        start_date = counts.index.min()

    if end_date is None:
        end_date = counts.index.max()

    inject_index = pd.date_range(start_date, end_date, freq='D')
    cl = CycleList(outside_data.index, shuffle=shuffle)
    new_df = {}
    counts = (counts * pct).apply(np.round).reindex(inject_index).fillna(0).astype(int)
    for new_ix, count in counts.items():
        ixes = cl.take(int(count))
        new_df[new_ix] = outside_data.loc[ixes]
    return pd.concat(new_df, axis=0).reset_index(level=1).rename_axis('StudyDate')


def filter_label_by_score(df, q, label_cols, sample_start_date=None, sample_end_date=None, bad=True):
    """
    Filter a DataFrame by selecting samples that fall outside a score prediction
    threshold, which is determined by a quantile value. This function filters the
    DataFrame based on a given quantile threshold for each label column in the input
    DataFrame. For each label column, this function first splits the DataFrame into
    two groups: the samples that have a score of zero (i.e., the negatives), and the
    samples that have a score different than zero (i.e., the positives). It then
    computes the quantile threshold for each group separately. If the `bad` parameter
    is set to `True`, this function selects the samples with a score prediction below
    the quantile threshold for each label column. If `bad` is set to `False`, it
    selects the samples with a score prediction above the quantile threshold for each
    label column.

    Args:
        df (pandas.DataFrame): The input DataFrame containing score predictions and
            activation values.
        q (float): The quantile threshold to use for filtering bad samples. Must be
            between 0 and 1.
        label_cols (list): A list of label columns to filter on. Each label column
            must have a corresponding activation column in the format
            "activation.<label_column>".
        sample_start_date (str, optional): A start date for the sample DataFrame used
            for filtering bad samples. Default is None, which selects all samples in
            the DataFrame.
        sample_end_date (str, optional): An end date for the sample DataFrame used
            for filtering bad samples. Default is None, which selects all samples in
            the DataFrame.
        bad (bool, optional): Whether to consider the bottom or top quantile as bad.
            Default is True, which selects samples with a score prediction below the
            quantile threshold.

    Returns:
        pandas.DataFrame: The filtered DataFrame, containing only the samples that
        fall outside the quantile threshold.

    Raises:
        ValueError: If the label_cols list is empty or the quantile threshold is not
        between 0 and 1.

    """
    # print("Input Len", len(df))
    stuff = df.loc[sample_start_date:sample_end_date].reset_index()
    # print("Sample Len", len(stuff))
    index = set()
    for label_col in label_cols:
        if bad:
            # top of negatives, bottom of positives
            top_df, bottom_df = stuff[stuff[label_col] == 0], stuff[stuff[label_col] != 0]
        else:
            # bottom of negatives, top of positives
            bottom_df, top_df = stuff[stuff[label_col] == 0], stuff[stuff[label_col] != 0]

        lv = bottom_df[f"activation.{label_col}"].quantile(q=q)
        hv = top_df[f"activation.{label_col}"].quantile(q=1 - q)
        bottoms = bottom_df[bottom_df[f"activation.{label_col}"] < lv].index
        tops = top_df[top_df[f"activation.{label_col}"] > hv].index
        index = index.union(bottoms).union(tops)
    return stuff.loc[index]


def filter_midrc(df, midrc_include=None, midrc_exclude=None):
    if midrc_include:
        filti = pd.Series([False] * len(df), index=df.index)
        for col in midrc_include.split(','):
            filti = df[col] | filti
    else:
        filti = pd.Series([True] * len(df), index=df.index)

    filte = pd.Series([True] * len(df), index=df.index)
    if midrc_exclude:
        for col in midrc_exclude.split(','):
            filte = filte & ~df[col]

    return df[filte & filti]


class CycleList(Iterator):
    def __init__(self, lst, shuffle=False):
        self.lst = lst
        self.index = list(range(len(lst)))
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            random.shuffle(self.index)

    def get_curr(self):
        return self.lst[self.index[self.cur]]

    def __next__(self):
        if self.cur >= len(self.lst):
            self.reset()
        v = self.get_curr()
        self.cur += 1
        return v

    def __iter__(self):
        while True:
            yield self.__next__()

    def take(self, n):
        return [next(self) for i in range(n)]


def create_score_based_ood_frame(dataframe, label_cols, counts=None, q=0.25, bottom=True, ood_end_date=None,
                                 ood_start_date=None,
                                 sample_end_date=None, sample_start_date=None):
    """
    Create an out-of-distribution (OOD) DataFrame based on score predictions.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame containing score predictions.
        label_cols (list): A list of label columns to include in the output DataFrame.
        counts (pandas.Series, optional): A pandas Series object representing the counts of the input DataFrame.
            Default is None.
        q (float, optional): The quantile threshold to use for filtering bad samples. Default is 0.25.
        bottom (bool, optional): Whether to consider the bottom or top quantile as bad. Default is True.
        ood_end_date (str, optional): An end date for the OOD DataFrame. Default is None.
        ood_start_date (str, optional): A start date for the OOD DataFrame. Default is None.
        sample_end_date (str, optional): An end date for the sample DataFrame used for filtering bad samples.
            Default is None.
        sample_start_date (str, optional): A start date for the sample DataFrame used for filtering bad samples.
            Default is None.

    Returns:
        pandas.DataFrame: The OOD DataFrame, containing daily counts and score predictions.

    Raises:
        ValueError: If the label_cols list is empty or the quantile threshold is not between 0 and 1."""
    if counts is None:
        counts = dataframe.iloc[:, 0].groupby(dataframe.index.date).count()

    if ood_start_date is None:
        ood_start_date = dataframe.index.min()

    if ood_end_date is None:
        ood_end_date = dataframe.index.max()

    if sample_start_date is None:
        sample_start_date = dataframe.index.min()

    if sample_end_date is None:
        sample_end_date = dataframe.index.max()

    bad_sample_data = filter_label_by_score(dataframe, q, label_cols=label_cols, sample_start_date=sample_start_date,
                                            sample_end_date=sample_end_date, bad=bottom)
    print("len bad_sample_data", len(bad_sample_data))
    bad_sample_data = create_ood_dataframe(bad_sample_data, 1.0, counts, start_date=ood_start_date,
                                           end_date=ood_end_date, shuffle=True)
    return bad_sample_data


def create_data_ood_dataframe(daily_counts, vae_jsonl_file, score_jsonl_file, label_cols, weight=1.0, shuffle=True,
                              start_date=None, end_date=None):
    """
    Create an out-of-distribution (OOD) DataFrame by combining VAE predictions and score predictions.

    Args:
        daily_counts (pandas.DataFrame): A DataFrame of daily counts.
        vae_jsonl_file (str): Path to the JSONL file containing VAE predictions.
        score_jsonl_file (str): Path to the JSONL file containing score predictions.
        label_cols (list): A list of label columns to include in the DataFrame.
        weight (float, optional): A weight parameter used in creating the OOD DataFrame. Default is 1.0.
        shuffle (bool, optional): Whether to shuffle the rows of the DataFrame. Default is True.
        start_date (str, optional): A start date for the daily counts. Default is None.
        end_date (str, optional): An end date for the daily counts. Default is None.

    Returns:
        pandas.DataFrame: The OOD DataFrame, containing daily counts and predictions.

    Raises:
        FileNotFoundError: If either the VAE JSONL file or the score JSONL file does not exist.
        ValueError: If the label_cols list is empty or the weight parameter is less than 0 or greater than or equal to 1
    """
    vae_df = load_vae_preds(vae_jsonl_file)
    scores_df = load_score_preds(label_cols, score_jsonl_file)
    data = scores_df.set_index('index').join(vae_df.set_index('index'))
    if weight < 1.0:
        w = weight / (1 - weight)
    else:
        w = weight
    data = create_ood_dataframe(data, w, daily_counts, start_date=start_date,
                                end_date=end_date,
                                shuffle=shuffle)
    for c in label_cols:
        if c not in data:
            data[c] = 0

    return data


def load_raw_drift_dataframe(vae_pred_file, scores_pred_file, metadata_file, label_cols,
                             metadata_loader=None) -> pd.DataFrame:
    """
        Load and merge raw drift data from VAE predictions, score predictions, and metadata into a single DataFrame.

        Args:
            vae_pred_file (str): Path to the VAE predictions file (jsonl format).
            scores_pred_file (str): Path to the scores predictions file (jsonl format).
            metadata_file (str): Path to the metadata file (CSV format by default).
            label_cols (list of str): List of column names for the label columns in the scores predictions file.
            metadata_loader (callable, optional): Function to load the metadata file.
            If not provided, a default function will be used which reads the file using pandas.read_csv.

        Returns:
            pandas.DataFrame: Merged DataFrame containing raw drift data from VAE predictions, score predictions, and
                              metadata. The DataFrame is indexed by 'StudyDate'.

        """
    if metadata_loader is None:
        def metadata_loader(filename):
            df = pd.read_csv(filename, index_col=0, low_memory=False)
            df.index = pd.to_datetime(df.index)
            return df

    vae_df = load_vae_preds(vae_pred_file)
    scores_df = load_score_preds(label_cols, scores_pred_file)
    metadata_df = metadata_loader(metadata_file)
    data_df = metadata_df
    data_df = data_df.merge(vae_df, left_on="ImageID", right_on="index", how='inner')
    data_df = data_df.merge(scores_df, left_on="ImageID", right_on="index", how='inner')
    data_df["indistro"] = data_df["Frontal"]
    data_df = data_df.set_index('StudyDate')

    return data_df
