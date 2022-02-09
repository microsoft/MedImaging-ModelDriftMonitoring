import os

import pandas as pd
import six
from azureml.core import Experiment

from model_drift.helpers import modelpath2name


def get_run(display_name, experiment):
    for run in experiment.get_runs():
        if run.display_name == display_name:
            return run

    raise KeyError(f"'{display_name}' not found in experiment!")


def run_to_dict(run):
    d = dict(**run.tags)
    d['id'] = run.id
    d['display_name'] = run.display_name
    d['url'] = run.get_portal_url()
    d['run'] = run
    # d["startTimeUtc"] = pd.to_datetime(run.get_details()["startTimeUtc"])
    # d["endTimeUtc"] = pd.to_datetime(run.get_details()["endTimeUtc"])
    return d


def experiment_to_dataframe(experiment, workspace=None):
    if isinstance(experiment, six.string_types):
        if workspace is None:
            raise ValueError("if experiment is string, must provide workspace")
        experiment = Experiment(workspace=workspace, name=experiment)
    df = []
    for run in experiment.get_runs():
        if run.status != "Completed":
            continue
        df.append(run_to_dict(run))
    return pd.DataFrame(df).set_index(['display_name'])  # .sort_values("endTimeUtc", ascending=False)


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
