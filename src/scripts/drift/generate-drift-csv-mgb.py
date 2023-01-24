#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path

library_path = str(Path(__file__).parent.parent.parent)
PYPATH = os.environ.get("PYTHONPATH", "").split(":")
if library_path not in PYPATH:
    PYPATH.append(library_path)
    os.environ["PYTHONPATH"] = ":".join(PYPATH)

from model_drift.data.padchest import PadChest
from model_drift.data.padchest import LABEL_MAP
from model_drift.drift.metrics.sampler import Sampler
from model_drift.drift.metrics.performance import ClassificationReportCalculator
from model_drift.drift.metrics import ChiSqDriftCalculator
from model_drift.drift.metrics.numeric import KSDriftCalculator, BasicDriftCalculator
from model_drift.drift.metrics import TabularDriftCalculator
from model_drift import settings, helpers
import warnings
import pandas as pd
import numpy as np

import argparse

logger = helpers.basic_logging()

parser = argparse.ArgumentParser()

parser.add_argument("--run_azure", type=int, dest="run_azure", help="run in AzureML", default=0)

parser.add_argument("--input_dir", type=str)
parser.add_argument("--vae_input_dir", type=Path)
parser.add_argument("--output_dir", type=str, default="./outputs/")

parser.add_argument("--dataset", type=str, default='padchest')
parser.add_argument("--vae_dataset", type=str, default='padchest-trained')
parser.add_argument("--classifier_dataset", type=str, default='padchest-finetuned')
parser.add_argument("--vae_filter", type=str, default='all-data')
parser.add_argument("--classifier_filter", type=str, default='frontal_only')
parser.add_argument("--window", type=str)
parser.add_argument("--stride", type=str)
parser.add_argument("--min_periods", type=int, default=150)
parser.add_argument("--ref_frontal_only", type=int, default=1)

parser.add_argument("--include_metadata", type=int, default=True)

parser.add_argument("--lateral_add_date", type=str, default=None)
parser.add_argument("--indist_remove_date", type=str, default=None)

parser.add_argument("--peds_weight", type=float, default=0)
parser.add_argument("--peds_start_date", type=str, default=None)
parser.add_argument("--peds_end_date", type=str, default=None)

parser.add_argument("--replacement", type=int, default=1)
parser.add_argument("--sample_size", type=int, default=1000)
parser.add_argument("--n_samples", type=int, default=20)

parser.add_argument("--generate_name", type=int, default=0)

parser.add_argument("--num_workers", type=int, default=-1)
parser.add_argument("--dbg", type=int, default=0)

parser.add_argument("--bad_q", type=float, default=0)
parser.add_argument("--bad_start_date", type=str, default=None)
parser.add_argument("--bad_end_date", type=str, default=None)
parser.add_argument("--bad_sample_start_date", type=str, default=None)
parser.add_argument("--bad_sample_end_date", type=str, default=None)

parser.add_argument("--good_q", type=float, default=0)
parser.add_argument("--good_start_date", type=str, default=None)
parser.add_argument("--good_end_date", type=str, default=None)
parser.add_argument("--good_sample_start_date", type=str, default=None)
parser.add_argument("--good_sample_end_date", type=str, default=None)

parser.add_argument("--start_date", type=str, default=None)
parser.add_argument("--end_date", type=str, default=None)

args = parser.parse_args()

if args.dataset != "mgb":
    raise NotImplementedError("unrecognized dataset")

name = "output"
fname = output_path.joinpath(name + ".csv")

num_cpus = os.cpu_count()
if args.num_workers < 0:
    args.num_workers = num_cpus

print("loading dataset vae results")
vae_pred_file = args.vae_input_path.joinpath('vae', args.vae_dataset, args.vae_filter, 'preds.jsonl'))
vae_df = helpers.jsonl_files2dataframe(vae_pred_file, desc="reading VAE results", refresh_rate=.1)
vae_df = pd.concat(
    [
        vae_df,
        pd.DataFrame(vae_df['mu'].values.tolist(), columns=[f"mu.{c:0>3}" for c in range(128)])
    ],
    axis=1
)

