
import os
from pathlib import Path

library_path = str(Path(__file__).parent.parent.parent)
PYPATH = os.environ.get("PYTHONPATH", "").split(":")
if library_path not in PYPATH:
    PYPATH.append(library_path)
    os.environ["PYTHONPATH"] = ":".join(PYPATH)

from model_drift.data.utils import rolling_window_dt_apply
from model_drift.data.padchest import PadChest, LABEL_MAP
from model_drift.data.padchest import LABEL_MAP
from model_drift import helpers
from model_drift.drift.sampler import Sampler
from model_drift.drift.performance import ClassificationReportCalculator
from model_drift.drift.categorical import ChiSqDriftCalculator
from model_drift.drift.numeric import KSDriftCalculator, BasicDriftCalculator
from model_drift.drift.tabular import TabularDriftCalculator
from model_drift.data.utils import nested2series
from model_drift import settings, helpers
import warnings
import pandas as pd
import numpy as np
from IPython.display import display


import argparse
from argparse import Namespace

def create_ood_dataframe(outside_data, pct, counts, start_date=None, end_date=None):
    
    print(counts.index.min(), counts.index.max())
    if start_date is None:
        start_date = counts.index.min()
    
    if end_date is None:
        end_date = counts.index.max()
        
    inject_index = pd.date_range(start_date, end_date, freq='D')
    cl = helpers.CycleList(outside_data.index)
    new_df = {}
    counts = (counts*pct).apply(np.round).reindex(inject_index).fillna(0).astype(int)
    for new_ix, count in counts.items():
        ixes = cl.take(int(count))
        new_df[new_ix] = outside_data.loc[ixes]
    return pd.concat(new_df, axis=0).reset_index(level=1).rename_axis('StudyDate')

helpers.basic_logging()
warnings.filterwarnings("ignore")

print("~-"*10)
print("~-"*10)

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

parser.add_argument("--nonfrontal_add_date", type=str, default=None)
parser.add_argument("--frontal_remove_date", type=str, default=None)

parser.add_argument("--peds_weight", type=float, default=0)
parser.add_argument("--peds_start_date", type=str, default=None)
parser.add_argument("--peds_end_date", type=str, default=None)


parser.add_argument("--replacement", type=int, default=1)
parser.add_argument("--sample_size", type=int, default=1000)
parser.add_argument("--n_samples", type=int, default=20)

parser.add_argument("--generate_name", type=int, default=0)

parser.add_argument("--num_workers", type=int, default=-1)
parser.add_argument("--dbg", type=int, default=0)


args = parser.parse_args()

input_path = Path(args.input_dir)
output_path = Path(args.output_dir)


num_cpus = os.cpu_count()
if args.num_workers < 0:
    args.num_workers = num_cpus

def add_arg_tags(args):
    from azureml.core import Run, Model
    run = Run.get_context()
    d = vars(args)
    for k, v in vars(args):
        run.tag(k, v)

if args.run_azure:
    from azureml.core import Run
    run = Run.get_context()
    for k, v in vars(args).items():
        run.tag(k, v)
    

name = "output"

print(name)
fname = output_path.joinpath(name+".csv")

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
scores_pred_file = str(input_path.joinpath('classifier', args.classifier_dataset, args.classifier_filter, "preds.jsonl"))
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


# Display
pd.concat(
    {
        "all": pc.df["StudyDate"].describe(datetime_is_numeric=True),
        "train": train.df["StudyDate"].describe(datetime_is_numeric=True),
        "val": val.df["StudyDate"].describe(datetime_is_numeric=True),
        "test": test.df["StudyDate"].describe(datetime_is_numeric=True),
    },
    axis=1,
)


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
    
cols.update({'Frontal': "DBG", 'in_distro': "DBG",})
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

target_df = pc.df.set_index('StudyDate')
if args.dbg:
    target_df = target_df.loc["2012-01-01": "2013-12-31"]

frontals_target_df = target_df.query("Frontal").copy()
if args.frontal_remove_date:
    frontals_target_df = frontals_target_df.loc[:args.frontal_remove_date]
targets = {"pc-frontal": frontals_target_df.assign(in_distro=True)}

print("in_distro target_dates", frontals_target_df.index.min(), frontals_target_df.index.max())

if args.nonfrontal_add_date is not None:
    nonfrontals_target_df = target_df.query("~Frontal").copy()
    nonfrontals_target_df = nonfrontals_target_df.loc[args.nonfrontal_add_date:]
    targets['pc-nonfrontal'] = nonfrontals_target_df.assign(in_distro=False)


if args.peds_weight:
    
    counts = frontals_target_df.groupby(frontals_target_df.index.date).count().iloc[:,0]
    
    # load peds classifier data
    jsonl_file = str(input_path.joinpath('outside-data', "pediatric-classifier-preds.jsonl"))
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
    
    w = args.peds_weight/(1-args.peds_weight)
    peds_data = create_ood_dataframe(peds_data, w, counts, start_date=args.peds_start_date, end_date=args.peds_end_date)
    targets['peds'] = peds_data.assign(in_distro=False)
    
print("Cleaning fixing types")
converters = {
    "FLOAT": lambda x: pd.to_numeric(x, errors="coerce"),
    "CAT": lambda x: x.apply(str),
    "DBG": lambda x: pd.to_numeric(x, errors="coerce"),
}

for col, TYPE in cols.items():
    target_df[c] = converters[TYPE](target_df[c])

target_df = pd.concat(targets.values(), sort=True)
avg = target_df.groupby(target_df.index.date)['in_distro'].mean().mean()
print("overall", str(target_df.index.min()), str(target_df.index.max()), avg)
for name, xdf in targets.items():
    avg = target_df.groupby(target_df.index.date)['in_distro'].mean().reindex(xdf.index.unique()).mean()
    print(name, str(xdf.index.min()), str(xdf.index.max()), avg)


# Output target_df and ref_df
reproduce_path  = output_path.joinpath('data')
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
