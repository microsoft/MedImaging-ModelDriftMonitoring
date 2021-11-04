import itertools
import sys

import six

import tqdm
import json
import logging
import os
import pandas as pd
from distutils import dir_util
from . import settings
from .data.utils import fix_strlst


def read_padchest(csv_file=None) -> pd.DataFrame:
    csv_file = csv_file or settings.PADCHEST_FILENAME
    df = pd.read_csv(csv_file, low_memory=False, index_col=0)
    df["StudyDate"] = pd.to_datetime(df["StudyDate_DICOM"], format="%Y%m%d")
    df["PatientBirth"] = pd.to_datetime(df["PatientBirth"], format="%Y")
    df["Labels"] = fix_strlst(df["Labels"])
    return df


def prepare_padchest(df) -> pd.DataFrame:
    df["StudyDate"] = pd.to_datetime(df["StudyDate_DICOM"], format="%Y%m%d")
    df["PatientBirth"] = pd.to_datetime(df["PatientBirth"], format="%Y")
    return df


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
    return model_path.replace('/', '-')


def print_env():
    print("--- ENV VARIABLES ---")
    for k, v in sorted(os.environ.items()):
        print(f" {k}={v}")
    print("--------------------")


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


def jsonl_files2dataframe(jsonl_files, converter=None):
    if isinstance(jsonl_files, six.string_types):
        jsonl_files = [jsonl_files]

    if converter is None:
        converter = lambda x: x

    df = []
    for fn in tqdm.tqdm(jsonl_files):
        with open(fn, 'r') as f:
            for line in tqdm.tqdm_notebook(f.readlines()):
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
