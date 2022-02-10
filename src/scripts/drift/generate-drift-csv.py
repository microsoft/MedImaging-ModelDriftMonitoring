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


def create_ood_dataframe(outside_data, pct, counts, start_date=None, end_date=None, shuffle=False):
    # print(counts.index.min(), counts.index.max())
    if start_date is None:
        start_date = counts.index.min()

    if end_date is None:
        end_date = counts.index.max()

    inject_index = pd.date_range(start_date, end_date, freq='D')
    cl = helpers.CycleList(outside_data.index, shuffle=shuffle)
    new_df = {}
    counts = (counts * pct).apply(np.round).reindex(inject_index).fillna(0).astype(int)
    for new_ix, count in counts.items():
        ixes = cl.take(int(count))
        new_df[new_ix] = outside_data.loc[ixes]
    return pd.concat(new_df, axis=0).reset_index(level=1).rename_axis('StudyDate')


def filter_label_by_score(df, q, sample_start_date=None, sample_end_date=None, label_cols=tuple(LABEL_MAP), bad=True):
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


warnings.filterwarnings("ignore")

print("~-" * 10)
print("Pandas Version:", pd.__version__)
print("~-" * 10)

parser = argparse.ArgumentParser()

parser.add_argument("--run_azure", type=int, dest="run_azure", help="run in AzureML", default=0)

parser.add_argument("--input_dir", type=str)
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

if args.dataset != "padchest":
    raise NotImplementedError("unrecognized dataset")

input_path = Path(args.input_dir)
output_path = Path(args.output_dir)

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

name = "output"
fname = output_path.joinpath(name + ".csv")

num_cpus = os.cpu_count()
if args.num_workers < 0:
    args.num_workers = num_cpus

print("loading dataset vae results")
vae_pred_file = str(input_path.joinpath('vae', args.vae_dataset, args.vae_filter, 'preds.jsonl'))
vae_df = helpers.jsonl_files2dataframe(vae_pred_file, desc="reading VAE results", refresh_rate=.1)
vae_df = pd.concat(
    [
        vae_df,
        pd.DataFrame(vae_df['mu'].values.tolist(), columns=[f"mu.{c:0>3}" for c in range(128)])
    ],
    axis=1
)

##

print("loading dataset predicted probabilities")
label_cols = list(LABEL_MAP)
scores_pred_file = str(
    input_path.joinpath('classifier', args.classifier_dataset, args.classifier_filter, "preds.jsonl"))
scores_df = helpers.jsonl_files2dataframe(scores_pred_file, desc="reading classifier results", refresh_rate=.1)
scores_df = pd.concat(
    [
        scores_df,
        pd.DataFrame(scores_df['activation'].values.tolist(), columns=[f"activation.{c}" for c in label_cols])
    ],
    axis=1)

print("loading dataset metadata")
pc = PadChest(str(input_path.joinpath(settings.PADCHEST_CSV_FILENAME)))
pc.prepare()

pc.merge(vae_df, left_on="ImageID", right_on="index", how='inner')
pc.merge(scores_df, left_on="ImageID", right_on="index", how='inner')

train, val, test = pc.split(settings.PADCHEST_SPLIT_DATES, studydate_index=True)

calculators = {
    "FLOAT": [KSDriftCalculator],
    "CAT": [ChiSqDriftCalculator],
    "DBG": [BasicDriftCalculator],
}

cols = {}
if args.include_metadata:
    cols.update({
        'age': "FLOAT",
        'Projection': "CAT",
        "PatientSex_DICOM": "CAT",
        "ViewPosition_DICOM": "CAT",
        "Modality_DICOM": "CAT",
        "Manufacturer_DICOM": "CAT",
        "PhotometricInterpretation_DICOM": "CAT",
        "PixelRepresentation_DICOM": "CAT",
        "PixelAspectRatio_DICOM": "CAT",
        "SpatialResolution_DICOM": "CAT",
        "BitsStored_DICOM": "CAT",
        "WindowCenter_DICOM": "FLOAT",
        "WindowWidth_DICOM": "FLOAT",
        "Rows_DICOM": "FLOAT",
        "Columns_DICOM": "FLOAT",
        "XRayTubeCurrent_DICOM": "CAT",
        "Exposure_DICOM": "CAT",
        "ExposureInuAs_DICOM": "FLOAT",
        "RelativeXRayExposure_DICOM": "FLOAT",
    })

cols.update({'Frontal': "DBG", 'in_distro': "DBG", })
cols.update({c: "FLOAT" for c in list(pc.df) if c.startswith("mu.") and 'all' not in c})
cols.update({c: "FLOAT" for c in list(pc.df) if c.startswith("activation.") and 'all' not in c})

sampler = Sampler(args.sample_size, replacement=args.replacement)

ref_df = val.df.copy().assign(in_distro=True)

if args.ref_frontal_only:
    ref_df = ref_df.query("Frontal")

dwc = TabularDriftCalculator(ref_df)

for c, TYPE in cols.items():
    for kls in calculators[TYPE]:
        dwc.add_drift_stat(c, kls)

dwc.add_drift_stat('performance', ClassificationReportCalculator, col=(
    "score", "label"), target_names=tuple(LABEL_MAP), include_stat_name=False)

dwc.prepare()

target_df = pc.df.set_index('StudyDate')
# if args.dbg:
#     target_df = target_df.loc["2012-01-01": "2013-12-31"]

frontals_target_df = target_df.query("Frontal").copy()
all_frontals = target_df.query("Frontal").copy()
if args.indist_remove_date:
    frontals_target_df = frontals_target_df.loc[:args.indist_remove_date]

targets = {"in dist": frontals_target_df.assign(in_distro=True)}

print("in_distro target_dates", frontals_target_df.index.min(), frontals_target_df.index.max())

if args.lateral_add_date is not None:
    nonfrontals_target_df = target_df.query("~Frontal").copy()
    nonfrontals_target_df = nonfrontals_target_df.loc[args.lateral_add_date:]
    targets['lateral'] = nonfrontals_target_df.assign(in_distro=False)

if args.peds_weight:
    counts = all_frontals.groupby(all_frontals.index.date).count().iloc[:, 0]

    # load peds classifier data
    jsonl_file = str(input_path.joinpath('outside-data', "pediatric-classifier-chxfrnt-preds.jsonl"))
    peds_scores_df = helpers.jsonl_files2dataframe(jsonl_file, desc="reading peds score results", refresh_rate=.1)
    peds_scores_df = pd.concat(
        [
            peds_scores_df,
            pd.DataFrame(peds_scores_df['activation'].values.tolist(), columns=[f"activation.{c}" for c in label_cols])
        ],
        axis=1)

    # vae data
    jsonl_file = str(input_path.joinpath('outside-data', "pediatric-vae-preds.jsonl"))
    peds_vae_df = helpers.jsonl_files2dataframe(jsonl_file, desc="reading peds VAE results", refresh_rate=.1)
    peds_vae_df = pd.concat(
        [
            peds_vae_df,
            pd.DataFrame(peds_vae_df['mu'].values.tolist(), columns=[f"mu.{c:0>3}" for c in range(128)])
        ],
        axis=1
    )
    peds_data = peds_scores_df.set_index('index').join(peds_vae_df.set_index('index'))

    if args.peds_weight < 1.0:
        w = args.peds_weight / (1 - args.peds_weight)
    else:
        w = args.peds_weight
    peds_data = create_ood_dataframe(peds_data, w, counts, start_date=args.peds_start_date, end_date=args.peds_end_date,
                                     shuffle=True)

    for c in label_cols:
        if c not in peds_data:
            peds_data[c] = 0
    targets['peds'] = peds_data.assign(in_distro=False)

if args.bad_q:
    counts = all_frontals.groupby(all_frontals.index.date).count().iloc[:, 0]
    bad_sample_data = filter_label_by_score(all_frontals, args.bad_q, args.bad_sample_start_date,
                                            args.bad_sample_end_date, label_cols=label_cols, bad=True)
    print("len bad_sample_data", len(bad_sample_data))
    bad_sample_data = create_ood_dataframe(bad_sample_data, 1.0, counts, start_date=args.bad_start_date,
                                           end_date=args.bad_end_date, shuffle=True)
    targets['bad_sample_data'] = bad_sample_data.assign(in_distro=False)

if args.good_q:
    counts = all_frontals.groupby(all_frontals.index.date).count().iloc[:, 0]
    good_sample_data = filter_label_by_score(all_frontals, args.good_q, args.good_sample_start_date,
                                             args.good_sample_end_date, label_cols=label_cols, bad=False)
    print("len good_sample_data", len(good_sample_data))
    good_sample_data = create_ood_dataframe(good_sample_data, 1.0, counts, start_date=args.good_start_date,
                                            end_date=args.good_end_date, shuffle=True)
    targets['good_sample_data'] = good_sample_data.assign(in_distro=False)

print("Cleaning fixing types")
converters = {
    "FLOAT": lambda x: pd.to_numeric(x, errors="coerce"),
    "CAT": lambda x: x.apply(str),
    "DBG": lambda x: pd.to_numeric(x, errors="coerce"),
}

for col, TYPE in cols.items():
    target_df[c] = converters[TYPE](target_df[c])

target_df = pd.concat(targets.values(), sort=True)

if args.dbg:
    pass

if args.start_date or args.end_date:
    target_df = target_df.loc[args.start_date: args.end_date]

avg = target_df.groupby(target_df.index.date)['in_distro'].mean().mean()
avgs = ', '.join("{}: {:.2%}".format(lab, p)
                 for lab, p in target_df.groupby(target_df.index.date)[label_cols].mean().mean(axis=0).items())
mind = str(target_df.index.min())
maxd = str(target_df.index.max())
print(f"{name}:\n {mind} to {maxd} indistro avg: {avg}\n |{avgs}")
for name, xdf in targets.items():
    avg = target_df.groupby(target_df.index.date)['in_distro'].mean().reindex(xdf.index.unique()).mean()
    avgs = ', '.join("{}: {:.2%}".format(lab, p)
                     for lab, p in target_df.groupby(target_df.index.date)[label_cols].mean()
                     .reindex(xdf.index.unique()).mean(axis=0).items())
    avgs1 = ', '.join("{}: {:.2%}".format(lab, p)
                      for lab, p in xdf.groupby(xdf.index.date)[label_cols].mean().mean(axis=0).items())

    mind = str(xdf.index.min())
    maxd = str(xdf.index.max())
    print(f"{name}:\n {mind} to {maxd} indistro avg: {avg}\n |{avgs}\n *{avgs}")

# Output target_df and ref_df
reproduce_path = output_path.joinpath('data')
reproduce_path.mkdir(parents=True, exist_ok=True)

ref_df.to_csv(str(reproduce_path.joinpath('ref.csv')))
target_df.to_csv(str(reproduce_path.joinpath('target.csv')))

print("starting drift experiment!")
output = dwc.rolling_window_predict(target_df,
                                    sampler=sampler, n_samples=args.n_samples,
                                    stride=args.stride, window=args.window, min_periods=args.min_periods,
                                    n_jobs=args.num_workers, backend="threading",
                                    refresh_rate=.01,
                                    )
output.to_csv(fname)
