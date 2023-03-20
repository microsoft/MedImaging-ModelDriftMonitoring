#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path

from environs import Env

env = Env()
env.read_env()  # read .env file, if it exists

TOP_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent

DATA_DIR = env.path("DATA_DIR", default=str(TOP_DIR.joinpath("data")))
MODEL_DIR = env.path("MODEL_DIR", default=str(TOP_DIR.joinpath("models")))
SRC_DIR = env.path("SRC", default=str(TOP_DIR.joinpath("src")))
RESULTS_DIR = env.path("RESULTS_DIR", default=str(TOP_DIR.joinpath("results")))

AZUREML_CONFIG = env.path("AZUREML_CONFIG", default=str(TOP_DIR))

PADCHEST_CSV_FILENAME = "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"

PADCHEST_FILENAME = env.str(
    "PADCHEST_FILENAME",
    default=str(DATA_DIR.joinpath(PADCHEST_CSV_FILENAME)),
)

CHEXPERT_TRAIN_CSV = env.str(
    "CHEXPERT_TRAIN_CSV",
    default=str(DATA_DIR.joinpath("chexpert-train.csv")),
)

CHEXPERT_VALID_CSV = env.str(
    "CHEXPERT_VALID_CSV",
    default=str(DATA_DIR.joinpath("chexpert-valid.csv")),
)

CONDA_ENVIRONMENT_FILE = CHEXPERT_VALID_CSV = env.str(
    "CONDA_ENVIRONMENT_FILE",
    default=str(TOP_DIR.joinpath("environment.yml")),
)

PADCHEST_SPLIT_DATES = env.list("PADCHEST_SPLIT_DATES", "2013-01-01, 2014-01-01")
env.str(
    "CONDA_ENVIRONMENT_FILE",
    default=str(TOP_DIR.joinpath("environment.yml")),
)

PADCHEST_SPLIT_DATES_STR = ','.join(PADCHEST_SPLIT_DATES)

DEFAULT_DRIFT_CONFIG = env.path("DEFAULT_DRIFT_CONFIG", default=str(TOP_DIR.joinpath("src", "model_drift", "drift", "config", "padchest.default.yml" )))
