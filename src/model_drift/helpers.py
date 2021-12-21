from joblib import Parallel
from tqdm.auto import tqdm
import itertools
import sys

import numpy as np
from functools import reduce

import six
from collections.abc import Iterator
import random
import tqdm
import json
import logging
import os
import pandas as pd
from distutils import dir_util
from sklearn.feature_selection import mutual_info_classif
from . import settings

# from .data.utils import fix_strlst


# def read_padchest(csv_file=None) -> pd.DataFrame:
#     csv_file = csv_file or settings.PADCHEST_FILENAME
#     df = pd.read_csv(csv_file, low_memory=False, index_col=0)
#     df["StudyDate"] = pd.to_datetime(df["StudyDate_DICOM"], format="%Y%m%d")
#     df["PatientBirth"] = pd.to_datetime(df["PatientBirth"], format="%Y")
#     df["Labels"] = fix_strlst(df["Labels"])
#     return df


# def prepare_padchest(df) -> pd.DataFrame:
#     df["StudyDate"] = pd.to_datetime(df["StudyDate_DICOM"], format="%Y%m%d")
#     df["PatientBirth"] = pd.to_datetime(df["PatientBirth"], format="%Y")
#     return df


def rolling_dt_apply_with_stride(
        dataframe,
        function,
        window="30D",
        stride="D",
        unique_only=False,
        center=False,
        min_periods=None,
) -> pd.DataFrame:
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


def get_run_name():
    from azureml.core import Run
    run = Run.get_context()
    return run.display_name


def download_model_azure(model_path, output_dir="./outputs/", local_path_env_var='_LOCAL_MODEL_PATH_'):
    from azureml.core import Run, Model
    run = Run.get_context()
    # Add run context for AML
    ws = run.experiment.workspace
    if local_path_env_var in os.environ:
        model_path = os.getenv(local_path_env_var)
        print(f"Found model path in environment (VAR={local_path_env_var}): {model_path}")
    else:
        model_name = modelpath2name(model_path)
        print(f"Downloading azure registered model: {model_name} ")
        m = Model(ws, model_name)
        os.environ[local_path_env_var] = model_path = m.download(
            exist_ok=True,
            target_dir=os.path.join(output_dir),
        )
        print(f"Download Complete! Path: {model_path}")
    return model_path


def get_azure_logger():
    from azureml.core import Run
    from pytorch_lightning.loggers import MLFlowLogger
    run = Run.get_context()
    mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()

    print("ml flow uri:", mlflow_url)
    mlf_logger = MLFlowLogger(experiment_name=run.experiment.name, tracking_uri=mlflow_url)
    mlf_logger._run_id = run.id
    return mlf_logger


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
                kwargs['miniters'] = int(len(lines)*refresh_rate)
            for line in tqdm.tqdm(lines, **kwargs):
                df.append(converter(json.loads(line)))
    return pd.json_normalize(df)


def basic_logging(level=logging.INFO, output_file=None, fmt='[%(asctime)s] %(levelname)s [%(name)s] %(message)s'):
    logger = logging.getLogger()
    logger.setLevel(level)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if output_file is not None:
        file_handler = logging.FileHandler(output_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("This is the start of logging")


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


def correlate_performance(perf_dataframe, other_dataframe, **kwargs):
    X, Y = align_frames(perf_dataframe, other_dataframe, **kwargs)
    return X.corrwith(Y).rename("correlation")


def mutual_info_performance(perf_dataframe, other_dataframe, bins=10, **kwargs):
    X, Y = align_frames(perf_dataframe, other_dataframe, **kwargs)
    Y, bins = pd.cut(Y, bins=bins, retbins=True)
    info_gain = mutual_info_classif(X.values, Y.cat.codes)
    return pd.Series(info_gain, index=X.columns.tolist(), name="info_gain")


def df_standard_scale(idf, nstd=1):
    stats = idf.agg(["mean", "std"])
    return (idf-stats.loc['mean'])/(stats.loc["std"])


def w_avg(df, weights):
    cols = df.columns
    cols = [c for c in weights if c in cols]
    weights = np.array([weights[c] for c in cols])
    weights = weights/weights.sum()
    tmp = df[cols].copy()
    for c, w in zip(cols, weights):
        tmp[c] = tmp[c]*w
    return tmp.sum(axis=1, skipna=False)


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


class CycleList(Iterator):
    def  __init__(self, lst, shuffle=False):
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