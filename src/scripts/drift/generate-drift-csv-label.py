import json
import os
from pathlib import Path

library_path = str(Path(__file__).parent.parent.parent)
PYPATH = os.environ.get("PYTHONPATH", "").split(":")
if library_path not in PYPATH:
    PYPATH.append(library_path)
    os.environ["PYTHONPATH"] = ":".join(PYPATH)

from model_drift.data.padchest import PadChest
from model_drift.data.padchest import LABEL_MAP
from model_drift.drift.sampler import Sampler
from model_drift.drift.performance import ClassificationReportCalculator
from model_drift.drift.categorical import ChiSqDriftCalculator
from model_drift.drift.numeric import KSDriftCalculator, BasicDriftCalculator
from model_drift.drift.tabular import TabularDriftCalculator
from model_drift import settings, helpers
import warnings
import pandas as pd
import numpy as np

import argparse


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


helpers.basic_logging()
warnings.filterwarnings("ignore")

print("~-" * 10)
print("~-" * 10)

print("Pandas Version:", pd.__version__)

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

# parser.add_argument("--label_mod_ref", type=str, default="No Finding")
parser.add_argument("--label_modifiers", type=str, default=None,
                    help="json str of {label1:[pct, start_date, end_date], ....")

parser.add_argument("--replacement", type=int, default=1)
parser.add_argument("--sample_size", type=int, default=1000)
parser.add_argument("--n_samples", type=int, default=20)

parser.add_argument("--start_date", type=str, default=None)
parser.add_argument("--end_date", type=str, default=None)
parser.add_argument("--mod_end_date", type=str, default=None)
parser.add_argument("--randomize_start_date", type=str, default=None)
parser.add_argument("--randomize_end_date", type=str, default=None)

parser.add_argument("--generate_name", type=int, default=0)

parser.add_argument("--num_workers", type=int, default=-1)
parser.add_argument("--dbg", type=int, default=0)

# parser.add_argument("--start_date")


args = parser.parse_args()

input_path = Path(args.input_dir)
output_path = Path(args.output_dir)

num_cpus = os.cpu_count()
if args.num_workers < 0:
    args.num_workers = num_cpus


def add_arg_tags(args):
    from azureml.core import Run
    run = Run.get_context()
    for k, v in vars(args):
        run.tag(k, v)


if args.run_azure:
    from azureml.core import Run

    run = Run.get_context()
    for k, v in vars(args).items():
        run.tag(k, v)

name = "output"

print(name)
fname = output_path.joinpath(name + ".csv")

num_cpus = os.cpu_count()
if args.num_workers < 0:
    args.num_workers = num_cpus

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
# vae_df.head()
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

##
# scores_df.head()

if args.dataset != "padchest":
    raise NotImplementedError("unrecognized dataset")

print("loading padchest data")
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

print(len(ref_df), len(val.df))
dwc = TabularDriftCalculator(ref_df)

for c, TYPE in cols.items():
    for kls in calculators[TYPE]:
        dwc.add_drift_stat(c, kls)

dwc.add_drift_stat('performance', ClassificationReportCalculator, col=(
    "score", "label"), target_names=tuple(LABEL_MAP), include_stat_name=False)

dwc.prepare()
label_mods = {}
if args.label_modifiers is not None:
    label_mods = json.loads(args.label_modifiers)

# normal_until_date = arg.

target_df = pc.df.query("Frontal").set_index('StudyDate').assign(in_distro=True)

counts = target_df.groupby(target_df.index.date).count().iloc[:, 0]
targets = {}

max_date = counts.index.max()
normal_until_date = min(max_date,
                        pd.to_datetime(args.randomize_start_date or max_date),
                        pd.to_datetime(args.randomize_end_date or max_date))
for label, (pct, start_date, end_date) in label_mods.items():
    normal_until_date = min(pd.to_datetime(start_date or max_date),
                            pd.to_datetime(end_date or max_date),
                            normal_until_date)

normal_until_date = str(normal_until_date.date())
print("normal until date:", normal_until_date)

targets["no-mod"] = target_df.loc[:normal_until_date].assign(inject="No")
extradata = target_df.loc[normal_until_date:].reset_index().rename(columns={"StudyDate": "OriginalStudyDate"})

if args.mod_end_date:
    extradata = extradata.loc[:args.mod_end_date]

for label, (pct, start_date, end_date) in label_mods.items():
    inject_data = extradata[extradata[label] > 0]
    targets[label] = create_ood_dataframe(inject_data, pct, counts,
                                          start_date=start_date, end_date=end_date or max_date,
                                          shuffle=True)
    targets[label].assign(inject=label)

if args.randomize_start_date:
    targets["random"] = create_ood_dataframe(extradata, 1.0, counts,
                                             start_date=args.randomize_start_date,
                                             end_date=args.randomize_end_date or max_date,
                                             shuffle=True)

print("Cleaning fixing types")
converters = {
    "FLOAT": lambda x: pd.to_numeric(x, errors="coerce"),
    "CAT": lambda x: x.apply(str),
    "DBG": lambda x: pd.to_numeric(x, errors="coerce"),
}

for col, TYPE in cols.items():
    target_df[c] = converters[TYPE](target_df[c])

target_df = pd.concat(targets.values(), sort=True)

avgs = ', '.join("{}: {:.0%}".format(lab, p)
                 for lab, p in target_df.groupby(target_df.index.date)[label_cols].mean().mean(axis=0).items())
print("overall", str(target_df.index.min().date()), str(target_df.index.max().date()), "\n *", avgs, "\n")

avgs = ', '.join("{}: {:.0%}".format(lab, p)
                 for lab, p in ref_df.groupby(ref_df.index.date)[label_cols].mean().mean(axis=0).items())
print("ref", str(ref_df.index.min().date()), str(ref_df.index.max().date()), "\n *", avgs, "\n")

for name, xdf in targets.items():
    # avg = target_df.groupby(target_df.index.date)['in_distro'].mean().reindex(xdf.index.unique()).mean()
    avgs = ', '.join("{}: {:.2%}".format(lab, p)
                     for lab, p in target_df.groupby(target_df.index.date)[label_cols].mean()
                     .reindex(xdf.index.unique()).mean(axis=0).items())
    print(name, str(xdf.index.min().date()), str(xdf.index.max().date()))
    print(" *", avgs)
    avgs = ', '.join("{}: {:.2%}".format(lab, p)
                     for lab, p in xdf.groupby(xdf.index.date)[label_cols].mean().mean(axis=0).items())
    print(" *", avgs)

if args.dbg:
    target_df = target_df.loc["2012-11-01":"2015-03-01"]

# Output target_df and ref_df
reproduce_path = output_path.joinpath('data')
reproduce_path.mkdir(parents=True, exist_ok=True)

ref_df.to_csv(str(reproduce_path.joinpath('ref.csv')))
target_df.to_csv(str(reproduce_path.joinpath('target.csv')))

if args.run_azure:
    import matplotlib.pylab as plt
    from azureml.core import Run

    run = Run.get_context()

    fig, ax = plt.subplots(figsize=(10, 8))
    target_df.groupby(target_df.index.date)[label_cols].mean().rolling(30).mean().plot(ax=ax)
    fig.savefig(str(reproduce_path.joinpath('target-fig.png')))

print("starting drift experiment!")
output = dwc.rolling_window_predict(target_df,
                                    sampler=sampler, n_samples=args.n_samples,
                                    stride=args.stride, window=args.window, min_periods=args.min_periods,
                                    n_jobs=args.num_workers, backend="threading",
                                    refresh_rate=.01,
                                    )
output.to_csv(fname)
