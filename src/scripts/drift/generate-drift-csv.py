#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import model_drift # noqa
except ImportError:
    library_path = str(Path(__file__).parent.parent.parent)
    PYPATH = os.environ.get("PYTHONPATH", "").split(":")
    if library_path not in PYPATH:
        PYPATH.append(library_path)
        os.environ["PYTHONPATH"] = ":".join(PYPATH)

from model_drift import settings, helpers
from model_drift.data.padchest import PadChest, LABEL_MAP
from model_drift.data.utils import split_on_date
from model_drift.drift import Sampler
from model_drift.drift.config import padchest_default_config
from model_drift.drift.performance import ClassificationReportCalculator
from model_drift.helpers import load_score_preds, load_vae_preds, create_ood_dataframe, \
    create_score_based_ood_frame, load_raw_drift_dataframe


def load_padchest(metadata_file):
    pc = PadChest(str(metadata_file))
    pc.prepare()
    return pc.df.copy()


logger = helpers.basic_logging()
warnings.filterwarnings("ignore")

print("~-" * 10)
print("Pandas Version:", pd.__version__)
print("~-" * 10)

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()

parser.add_argument("--run_azure", type=int, dest="run_azure", help="Set to 1 to run in AzureML, 0 to run locally",
                    default=0)

parser.add_argument("--input_dir", type=str, help="Directory containing the input data")
parser.add_argument("--output_dir", type=str, default="./outputs/", help="Directory to output results")

parser.add_argument("--split_dates",
                    nargs=2,
                    type=lambda d: datetime.strptime(d, '%Y-%m-%d').date(),
                    default=settings.PADCHEST_SPLIT_DATES,
                    help="Dates to split dataset for training and testing")

parser.add_argument("--dataset", type=str, default='padchest', help="Name of the dataset")

parser.add_argument("--vae_dataset", type=str, default='padchest-trained', help="Name of the VAE dataset")
parser.add_argument("--classifier_dataset", type=str, default='padchest-finetuned',
                    help="Name of the classifier dataset")
parser.add_argument("--vae_filter", type=str, default='all-data', help="Filter for the VAE dataset")
parser.add_argument("--classifier_filter", type=str, default='frontal_only', help="Filter for the classifier dataset")

parser.add_argument("--vae_preds_file", type=str, default=None, help="Path to the VAE predictions file")
parser.add_argument("--score_preds_file", type=str, default=None, help="Path to the score predictions file")

parser.add_argument("--window", type=str, help="Window for rolling apply")
parser.add_argument("--stride", type=str, help="Stride for rolling apply")
parser.add_argument("--min_periods", type=int, default=150, help="Minimum number of periods required for rolling apply")
parser.add_argument("--ref_frontal_only", type=int, default=1,
                    help="Include only frontal radiographs as reference data")

parser.add_argument("--include_metadata", type=int, default=True, help="Include metadata in the results")

parser.add_argument("--lateral_add_date", type=str, default=None, help="Date to add lateral data")
parser.add_argument("--indist_remove_date", type=str, default=None, help="Date to remove indistinct data")

parser.add_argument("--peds_weight", type=float, default=0, help="Weight for pediatric data")
parser.add_argument("--peds_start_date", type=str, default=None, help="Start date for pediatric data")
parser.add_argument("--peds_end_date", type=str, default=None, help="End date for pediatric data")

parser.add_argument("--replacement", type=int, default=1, help="Whether to allow replacement when sampling data")
parser.add_argument("--sample_size", type=int, default=1000, help="Size of each sample")
parser.add_argument("--n_samples", type=int, default=20, help="Number of samples")

parser.add_argument("--generate_name", type=int, default=0, help="Generate a unique name for the output directory")

parser.add_argument("--num_workers", type=int, default=-1, help="Number of workers for parallel processing")
parser.add_argument("--dbg", type=int, default=0, help="Set to 1 to enable debug mode")
parser.add_argument("--reproduce", type=bool, default=False, help="Set to True to enable reproducibility")

parser.add_argument("--bad_q", type=float, default=0, help="Quantile threshold for bad data")
parser.add_argument("--bad_start_date", type=str, default=None, help="Start date for bad data")
parser.add_argument("--bad_end_date", type=str, default=None, help="End date for bad data")

parser.add_argument("--bad_sample_start_date", type=str, default=None,
                    help="Start date for filtering negative examples based on the activation quantile")
parser.add_argument("--bad_sample_end_date", type=str, default=None,
                    help="End date for filtering negative examples based on the activation quantile")

parser.add_argument("--good_q", type=float, default=0,
                    help="Quantile threshold for filtering positive examples based on the activation")
parser.add_argument("--good_start_date", type=str, default=None,
                    help="Start date for filtering positive examples based on the activation quantile")
parser.add_argument("--good_end_date", type=str, default=None,
                    help="End date for filtering positive examples based on the activation quantile")
parser.add_argument("--good_sample_start_date", type=str, default=None,
                    help="Start date for selecting positive examples based on activation quantile")
parser.add_argument("--good_sample_end_date", type=str, default=None,
                    help="End date for selecting positive examples based on activation quantile")

parser.add_argument("--start_date", type=str, default=None, help="Start date for selecting samples")
parser.add_argument("--end_date", type=str, default=None, help="End date for selecting samples")

args, unknown = parser.parse_known_args()

if len(unknown):
    raise ValueError(f"unknown args: {unknown}")

if args.dataset != "padchest":
    raise NotImplementedError("unrecognized dataset")

input_path = Path(args.input_dir)
output_path = Path(args.output_dir)

reproduce_path = output_path.joinpath('data')
history_path = output_path.joinpath("history")

reproduce_path = output_path.joinpath('data')
ref_csv = reproduce_path.joinpath('ref.csv')
target_csv = reproduce_path.joinpath('target.csv')

name = "metrics"
fname = output_path.joinpath(name + ".csv")

num_cpus = os.cpu_count()
if args.num_workers < 0:
    args.num_workers = num_cpus

if args.run_azure:
    from azureml.core import Run

    run = Run.get_context()
    param_tags = vars(args)
    print(param_tags)
    for k, v in vars(args).items():
        run.tag(k, str(v))

num_cpus = os.cpu_count()
if args.num_workers < 0:
    args.num_workers = num_cpus

label_cols = list(LABEL_MAP)

if (args.reproduce or (not ref_csv.exists() or not target_csv.exists())):

    vae_pred_file = str(input_path.joinpath('vae', args.vae_dataset, args.vae_filter, 'preds.jsonl'))
    scores_pred_file = str(
        input_path.joinpath('classifier', args.classifier_dataset, args.classifier_filter, "preds.jsonl"))
    metadata_file = str(input_path.joinpath(settings.PADCHEST_CSV_FILENAME))

    print("vae_pred_file:", vae_pred_file)
    print("scores_pred_file:", scores_pred_file)
    print("metadata_file:", metadata_file)

    data_df = load_raw_drift_dataframe(vae_pred_file, scores_pred_file, metadata_file, label_cols, load_padchest)

    _, ref_df, test = list(split_on_date(data_df, args.split_dates))

    ref_df = ref_df.assign(in_distro=True)
    if args.ref_frontal_only:
        ref_df = ref_df.query("indistro")

    indistro_data = data_df.query("indistro").copy().assign(in_distro=False)

    targets = {}
    if args.indist_remove_date:
        targets["indistro"] = indistro_data.loc[:args.indist_remove_date]
    else:
        targets["indistro"] = indistro_data

    if args.lateral_add_date is not None:
        nonfrontals_target_df = data_df.query("~Frontal").copy()
        nonfrontals_target_df = nonfrontals_target_df.loc[args.lateral_add_date:]
        targets['lateral'] = nonfrontals_target_df.assign(in_distro=False)

    if args.peds_weight:
        counts = indistro_data.groupby(indistro_data.index.date).count().iloc[:, 0]

        # load peds classifier data
        peds_jsonl_file = str(input_path.joinpath('outside-data', "pediatric-classifier-chxfrnt-preds.jsonl"))
        peds_scores_df = load_score_preds(label_cols, peds_jsonl_file)

        # vae data
        peds_vae_jsonl_file = str(input_path.joinpath('outside-data', "pediatric-vae-preds.jsonl"))
        peds_vae_df = load_vae_preds(peds_vae_jsonl_file)

        # merge peds frames
        peds_data = peds_scores_df.set_index('index').join(peds_vae_df.set_index('index'))

        if args.peds_weight < 1.0:
            w = args.peds_weight / (1 - args.peds_weight)
        else:
            w = args.peds_weight

        peds_data = create_ood_dataframe(peds_data, w, counts, start_date=args.peds_start_date,
                                         end_date=args.peds_end_date,
                                         shuffle=True)

        for c in label_cols:
            if c not in peds_data:
                peds_data[c] = 0
        targets['peds'] = peds_data.assign(in_distro=False)

    if args.bad_q:
        targets['bad_sample_data'] = create_score_based_ood_frame(indistro_data, label_cols, q=args.bad_q,
                                                                  sample_start_date=args.bad_sample_start_date,
                                                                  sample_end_date=args.bad_sample_end_date,
                                                                  ood_start_date=args.bad_start_date,
                                                                  ood_end_date=args.bad_end_date, bottom=True
                                                                  ).assign(in_distro=False)

    if args.good_q:
        targets['good_sample_data'] = create_score_based_ood_frame(indistro_data, label_cols, q=args.good_q,
                                                                   sample_start_date=args.good_sample_start_date,
                                                                   sample_end_date=args.good_sample_end_date,
                                                                   ood_start_date=args.good_start_date,
                                                                   ood_end_date=args.good_end_date, bottom=True
                                                                   ).assign(in_distro=False)

    for name, target in targets.items():
        target["source"] = name

    data_df = pd.concat(targets.values(), sort=True)
    if args.start_date or args.end_date:
        data_df = data_df.loc[args.start_date: args.end_date]

    reproduce_path.mkdir(parents=True, exist_ok=True)
    ref_df.to_csv(str(reproduce_path.joinpath('ref.csv')))
    data_df.to_csv(str(reproduce_path.joinpath('target.csv')))

else:
    ref_df = pd.read_csv(ref_csv, index_col=0)
    data_df = pd.read_csv(target_csv, index_col=0)

ref_df.index = pd.to_datetime(ref_df.index)
data_df.index = pd.to_datetime(data_df.index)

avg = data_df.groupby(data_df.index.date)['in_distro'].mean().mean()
avgs = ', '.join("{}: {:.2%}".format(lab, p)
                 for lab, p in data_df.groupby(data_df.index.date)[label_cols].mean().mean(axis=0).items())
mind = str(data_df.index.min())
maxd = str(data_df.index.max())
print(f"{name}:\n {mind} to {maxd} indistro avg: {avg}\n |{avgs}")
for name, xdf in data_df.groupby("source"):
    avg = data_df.groupby(data_df.index.date)['in_distro'].mean().reindex(xdf.index.unique()).mean()
    avgs = ', '.join("{}: {:.2%}".format(lab, p)
                     for lab, p in data_df.groupby(data_df.index.date)[label_cols].mean()
                     .reindex(xdf.index.unique()).mean(axis=0).items())
    avgs1 = ', '.join("{}: {:.2%}".format(lab, p)
                      for lab, p in xdf.groupby(xdf.index.date)[label_cols].mean().mean(axis=0).items())

    mind = str(xdf.index.min())
    maxd = str(xdf.index.max())
    print(f"{name}:\n {mind} to {maxd} indistro avg: {avg}\n |{avgs}\n *{avgs}")

dwc = padchest_default_config(ref_df)
dwc.add_drift_stat('performance', ClassificationReportCalculator(target_names=tuple(LABEL_MAP)), col=("score", "label"),
                   include_stat_name=False)

dwc.prepare(ref_df)
sampler = Sampler(args.sample_size, replacement=args.replacement)
print("starting drift experiment!")
preds = dwc.rolling_window_predict(data_df,
                                   sampler=sampler, n_samples=args.n_samples,
                                   stride=args.stride, window=args.window, min_periods=args.min_periods,
                                   n_jobs=args.num_workers, backend="threading",
                                   refresh_rate=.01,
                                   output_dir=str(output_path),
                                   )
preds.to_csv(fname)
