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

from model_drift.data.dataset import MGBCXRDataset
from model_drift.data.utils import split_on_date
from model_drift.data import mgb_data
from model_drift.drift.metrics.sampler import Sampler
from model_drift.drift.metrics.performance import ClassificationReportCalculator
from model_drift.drift.metrics import ChiSqDriftCalculator
from model_drift.drift.metrics.numeric import KSDriftCalculator, BasicDriftCalculator
from model_drift.drift.metrics import TabularDriftCalculator
from model_drift import settings, helpers
from pycrumbs import tracked
import warnings
import pandas as pd
import numpy as np

import argparse

DATASET_DIR = Path("/autofs/cluster/qtim/datasets/xray_drift")
PROJECT_DIR = Path("/autofs/cluster/qtim/projects/xray_drift")


def make_index(row: pd.Series):
    return f"{row.PatientID}_{row.AccessionNumber}_{row.SOPInstanceUID}"


@tracked(directory_parameter="output_dir")
def main(output_dir: Path, args: argparse.Namespace) -> None:

    logger = helpers.basic_logging()

    if args.dataset != "mgb":
        raise NotImplementedError("unrecognized dataset")

    output_dir.mkdir(exist_ok=True)
    name = "output"
    fname = output_dir.joinpath(name + ".csv")

    num_cpus = os.cpu_count()
    if args.num_workers < 0:
        args.num_workers = num_cpus

    print("loading dataset predicted probabilities")
    label_cols = list(MGBCXRDataset.LABEL_COLUMNS)
    scores_pred_file = args.input_dir.joinpath("preds.jsonl")
    scores_df = helpers.jsonl_files2dataframe([scores_pred_file], desc="reading classifier results", refresh_rate=.1)
    scores_df = pd.concat(
        [
            scores_df,
            pd.DataFrame(scores_df['activation'].values.tolist(), columns=[f"activation.{c}" for c in label_cols])
        ],
        axis=1
    )

    print("loading dicom metadata")
    meta_df = pd.read_csv(args.metadata_csv, index_col=0)
    labels_df = pd.read_csv(DATASET_DIR / "csv" / "labels.csv", index_col=0)
    meta_df = meta_df.merge(labels_df, how="left", on=("StudyInstanceUID", "PatientID", "AccessionNumber"))
    meta_df["StudyDate"] = pd.to_datetime(meta_df["StudyDate"], format='%m/%d/%Y')
    meta_df["index"] = meta_df.apply(make_index, axis=1)

    print("loading dataset vae results")
    vae_pred_file = args.vae_input_dir.joinpath('preds.jsonl')
    vae_df = helpers.jsonl_files2dataframe([vae_pred_file], desc="reading VAE results", refresh_rate=.1)
    vae_df = pd.concat(
        [
            vae_df,
            pd.DataFrame(vae_df['mu'].values.tolist(), columns=[f"mu.{c:0>3}" for c in range(128)])
        ],
        axis=1
    )
    vae_df.drop_duplicates(subset="index", inplace=True)

    merged_df = scores_df.merge(vae_df, on="index", how="left")
    merged_df = merged_df.merge(meta_df, on="index", how="left")
    print(len(merged_df))
    train_df, val_df, test_df = split_on_date(merged_df, [mgb_data.TRAIN_DATE_END, mgb_data.VAL_DATE_END])

    calculators = {
        "FLOAT": [KSDriftCalculator],
        "CAT": [ChiSqDriftCalculator],
        "DBG": [BasicDriftCalculator],
    }

    cols = {}
    if args.include_metadata:
        cols.update({
            "ViewPosition": "CAT",
            "Manufacturer": "CAT",
            "PhotometricInterpretation": "CAT",
            "BitsStored": "CAT",
            "Rows": "FLOAT",
            "Columns": "FLOAT",
            "XRayTubeCurrent": "CAT",
            "Exposure": "CAT",
            "ExposureInuAs": "FLOAT",
            "KVP": "FLOAT",
        })

    cols.update({c: "FLOAT" for c in list(merged_df) if c.startswith("mu.") and 'all' not in c})
    cols.update({c: "FLOAT" for c in list(merged_df) if c.startswith("activation.") and 'all' not in c})

    sampler = Sampler(args.sample_size, replacement=args.replacement)

    ref_df = val_df.copy().assign(in_distro=True)
    dwc = TabularDriftCalculator(ref_df)

    for c, TYPE in cols.items():
        for kls in calculators[TYPE]:
            dwc.add_drift_stat(c, kls)

    dwc.add_drift_stat(
        'performance',
        ClassificationReportCalculator,
        col=("score", "label"),
        target_names=tuple(MGBCXRDataset.LABEL_COLUMNS),
        include_stat_name=False
    )

    dwc.prepare()

    target_df = merged_df.set_index('StudyDate')

    ref_df.to_csv(output_dir.joinpath('ref.csv'))
    target_df.to_csv(output_dir.joinpath('target.csv'))

    print("starting drift experiment!")
    output = dwc.rolling_window_predict(
        target_df,
        sampler=sampler, n_samples=args.n_samples,
        stride=args.stride, window=args.window, min_periods=args.min_periods,
        n_jobs=args.num_workers, backend="threading",
        refresh_rate=.01,
    )
    output.to_csv(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_azure", type=int, dest="run_azure", help="run in AzureML", default=0)

    parser.add_argument("--input_dir", "-i", type=Path)
    parser.add_argument("--vae_input_dir", "-v", type=Path)
    parser.add_argument("--output_dir", "-o", type=Path)
    parser.add_argument("--metadata_csv", "-m", type=Path, default=Path("/autofs/cluster/qtim/datasets/xray_drift/csv/dicom_inventory.csv"))
    parser.add_argument("--study_csv", "-s", type=Path, default=Path("/autofs/cluster/qtim/datasets/xray_drift/csv/dicom_inventory.csv"))

    parser.add_argument("--dataset", type=str, default='mgb')
    parser.add_argument("--vae_dataset", type=str, default='padchest-trained')
    parser.add_argument("--classifier_dataset", type=str, default='padchest-finetuned')
    parser.add_argument("--vae_filter", type=str, default='all-data')
    parser.add_argument("--classifier_filter", type=str, default='frontal_only')
    parser.add_argument("--window", "-w", type=str, default="14D")
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

    main(args.output_dir, args)
