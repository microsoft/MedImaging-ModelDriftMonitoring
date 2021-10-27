import sys
from pathlib import Path
import argparse
import os
from azureml import data
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

d = str(Path(__file__).parent.parent.parent)
PYPATH = os.environ.get("PYTHONPATH", "")
if d not in PYPATH:
    if PYPATH:
        os.environ["PYTHONPATH"] = d + ":" + PYPATH
    else:
        os.environ["PYTHONPATH"] = d
try:
    import model_drift
except ModuleNotFoundError:
    print("*%" * 10)
    print(sys.path)
    print("*%" * 10)
    raise


from model_drift.models.finetune import CheXFinetune
from model_drift.dataset import PadChestDataset
from model_drift.callbacks import ClassifierPredictionWriter
from model_drift.models.finetune import IMAGE_SIZE, CHANNELS

num_gpus = torch.cuda.device_count()
num_cpus = os.cpu_count()
node_rank = os.environ.get("NODE_RANK", 0)
local_rank = os.environ.get("LOCAL_RANK", 0)


print()
print("=" * 5)
print(" Pytorch Lightning Version:", pl.__version__)
print(" Pytorch Version:", torch.__version__)
print(" Num GPUs:", num_gpus)
print(" Num CPUs:", num_cpus)
print(" Node Rank:", node_rank)
print(" Local Rank:", local_rank)
print("=" * 5)
print()


parser = argparse.ArgumentParser()


parser.add_argument("--model", type=str, dest="model", help="path to model or registered model name")
parser.add_argument("--run_azure", type=int, dest="run_azure", help="run in AzureML", default=0)
parser.add_argument("--data_folder", type=str, dest="data_folder", help="data folder mounting point")
parser.add_argument("--batch_size", type=int, dest="batch_size", help="batch_size", default=64)

parser.add_argument(
    "--csv",
    type=str,
    dest="csv",
    help="csv",
    default="PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
)

parser.add_argument(
    "--output_dir",
    type=str,
    dest="output_dir",
    help="output_dir",
    default="outputs",
)

parser.add_argument(
    "--num_workers",
    type=int,
    dest="num_workers",
    help="number of workers for data loacing",
    default=-1,
)

parser.add_argument("--append_run_name", type=int, dest="append_run_name", default=False)


parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()

args.gpus = num_gpus

args.output_dir = args.output_dir.replace("//", "/")

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


if args.run_azure:
    from azureml.core import Run, Model

    run = Run.get_context()
    if args.append_run_name:
        args.output_dir = os.path.join(args.output_dir, run.display_name)
    # Add run context for AML
    ws = run.experiment.workspace
    m = Model(ws, args.model)
    print(f"Downloading azure registered model: {args.model}")
    args.model = m.download(
        exist_ok=True,
        target_dir=os.path.join(args.output_dir, "checkpoint", f"{node_rank}-{local_rank}"),
    )
    print(f"Download Complete! Path: {args.model}")


args.default_root_dir = args.output_dir

model = CheXFinetune.load_from_checkpoint(args.model, predict_mode=True)

if args.num_workers < 0:
    args.num_workers = num_cpus


dataset = PadChestDataset(
    args.data_folder,
    args.csv,
    IMAGE_SIZE,
    True,
    channels=CHANNELS,
    load_labels=False,
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

model.eval()
trainer = pl.Trainer.from_argparse_args(args)
trainer.callbacks.append(ClassifierPredictionWriter(args.output_dir))
_ = trainer.predict(model, dataloader)
