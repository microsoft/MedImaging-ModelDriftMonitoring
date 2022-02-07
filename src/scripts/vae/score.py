import argparse
import os
import pytorch_lightning as pl
import torch
from argparse import Namespace
from pathlib import Path

library_path = str(Path(__file__).parent.parent.parent)
PYPATH = os.environ.get("PYTHONPATH", "").split(":")
if library_path not in PYPATH:
    PYPATH.append(library_path)
    os.environ["PYTHONPATH"] = ":".join(PYPATH)

from model_drift import helpers
from model_drift.models.vae import VAE
from model_drift.data.datamodules import PadChestDataModule, PediatricCheXpertDataModule, MIDRCDataModule
from model_drift.callbacks import VAEPredictionWriter
from model_drift.data.transform import VisionTransformer

# Add your data module here. Two examples are:
data_modules = {
    "padchest": PadChestDataModule,
    "peds": PediatricCheXpertDataModule,
    "midrc": MIDRCDataModule,
}

helpers.basic_logging()

ddp_model_check_key = "_LOCAL_MODEL_PATH_"

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
parser.add_argument("--output_dir", type=str, dest="output_dir", help="output_dir", default="outputs")
parser.add_argument("--add_name_to_output", type=int, dest="add_name_to_output", help="output_dir", default=0)
parser.add_argument("--write_recon", type=int, dest="write_recon", help="write_recon", default=0)

parser.add_argument("--dataset", type=str, dest="dataset",
                    help="dataset", choices=list(data_modules), default='padchest')
temp_args, _ = parser.parse_known_args()
dm_cls = data_modules[temp_args.dataset]
parser = dm_cls.add_argparse_args(parser)

parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()
args.gpus = num_gpus
args.output_dir = args.output_dir.replace("//", "/")

if args.add_name_to_output and args.run_azure:
    run_name = helpers.get_run_name()
    args.output_dir = os.path.join(args.output_dir, run_name)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if args.run_azure:
    args.model = helpers.download_model_azure(args.model, args.output_dir)

args.default_root_dir = args.output_dir

model = VAE.load_from_checkpoint(args.model)
transformer = VisionTransformer.from_argparse_args(Namespace(), **model.hparams.params)

dm = dm_cls.from_argparse_args(args, transforms=transformer.train_transform)
writer = VAEPredictionWriter(args.output_dir, write_recon=args.write_recon)

if args.num_workers < 0:
    args.num_workers = num_cpus

model.eval()
trainer = pl.Trainer.from_argparse_args(args)
trainer.callbacks.append(writer)
_ = trainer.predict(model, dm)

trainer.training_type_plugin.barrier()

writer.merge_prediction_files(trainer)
