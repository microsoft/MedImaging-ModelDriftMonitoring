from pathlib import Path
import argparse
import os
import pytorch_lightning as pl
import torch
import itertools
import tqdm
from argparse import Namespace

library_path = str(Path(__file__).parent.parent.parent)
library_path = "/home/arjunsoin/project/MedImaging-ModelDriftMonitoring/src"
PYPATH = os.environ.get("PYTHONPATH", "").split(":")
if library_path not in PYPATH:
    PYPATH.append(library_path)
    os.environ["PYTHONPATH"] = ":".join(PYPATH)

from model_drift import helpers
from model_drift.models.vae import VAE
from model_drift.data.datamodules import PadChestDataModule
from model_drift.data.datamodules import PediatricCheXpertDataModule
from model_drift.callbacks import VAEPredictionWriter
from model_drift.data.transform import VisionTransformer

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

# Add your dataloader here. Two examples are:
# parser = PadChestDataModule.add_argparse_args(parser)
parser = PediatricCheXpertDataModule.add_argparse_args(parser)
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()
args.gpus = num_gpus
args.output_dir = args.output_dir.replace("//", "/")

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if args.run_azure:
    args.model = helpers.download_model_azure(args.model, args.output_dir)

args.default_root_dir = args.output_dir

model = VAE.load_from_checkpoint(args.model)
transformer = VisionTransformer.from_argparse_args(Namespace(), **model.hparams.params)
# dm = PadChestDataModule.from_argparse_args(args, transforms=transformer.train_transform)
dm = PediatricCheXpertDataModule.from_argparse_args(args, transforms=transformer.train_transform)
writer = VAEPredictionWriter(args.output_dir)

if args.num_workers < 0:
    args.num_workers = num_cpus

model.eval()
trainer = pl.Trainer.from_argparse_args(args)
trainer.callbacks.append(writer)
_ = trainer.predict(model, dm)

trainer.training_type_plugin.barrier()

writer.merge_prediction_files(trainer)



# import argparse
# import os
# import pytorch_lightning as pl
# from model_drift.models.vae import VAE
# from data.dataset import PadChestDataset
# import torch
# from torch.utils.data import DataLoader
# from model_drift.callbacks import VAEPredictionWriter
#
# num_gpus = torch.cuda.device_count()
# num_cpus = os.cpu_count()
# node_rank = os.environ.get("NODE_RANK", 0)
# local_rank = os.environ.get("LOCAL_RANK", 0)
#
# print()
# print("=" * 5)
# print(" Pytorch Lightning Version:", pl.__version__)
# print(" Pytorch Version:", torch.__version__)
# print(" Num GPUs:", num_gpus)
# print(" Num CPUs:", num_cpus)
# print(" Node Rank:", node_rank)
# print(" Local Rank:", local_rank)
# print("=" * 5)
# print()
#
#
# parser = argparse.ArgumentParser()
#
#
# parser.add_argument("--model", type=str, dest="model", help="path to model or registered model name")
# parser.add_argument("--run_azure", type=int, dest="run_azure", help="run in AzureML", default=0)
#
# parser.add_argument("--data_folder", type=str, dest="data_folder", help="data folder mounting point")
#
# parser.add_argument("--batch_size", type=int, dest="batch_size", help="batch_size", default=64)
#
# parser.add_argument(
#     "--csv",
#     type=str,
#     dest="csv",
#     help="csv",
#     default="PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
# )
#
# parser.add_argument(
#     "--output_dir",
#     type=str,
#     dest="output_dir",
#     help="output_dir",
#     default="outputs",
# )
#
# parser.add_argument(
#     "--num_workers",
#     type=int,
#     dest="num_workers",
#     help="number of workers for data loacing",
#     default=-1,
# )
#
# parser.add_argument(
#     "--write_recon",
#     type=int,
#     dest="write_recon",
#     default=0,
#     help="flag to write reconstrucitons and inputs",
# )
#
# parser.add_argument(
#     "--latent_output_dir",
#     type=str,
#     dest="latent_output_dir",
#     default=None,
#     help="Additional location to write latent variable files",
# )
#
#
# parser.add_argument(
#     "--write_grid",
#     type=float,
#     dest="write_grid",
#     default=0,
#     help="probability to write a grid image for each batch "
#     + "0 for never, 1 for always.  Note this creates large images",
# )
#
# parser.add_argument("--append_run_name", type=int, dest="append_run_name", default=False)
#
#
# parser = pl.Trainer.add_argparse_args(parser)
#
# args = parser.parse_args()
#
# args.gpus = num_gpus
#
# args.output_dir = args.output_dir.replace("//", "/")
#
# if not os.path.exists(args.output_dir):
#     os.makedirs(args.output_dir)
#
#
# if args.run_azure:
#     from azureml.core import Run, Model
#
#     run = Run.get_context()
#     if args.append_run_name:
#         args.output_dir = os.path.join(args.output_dir, run.display_name)
#     # Add run context for AML
#     ws = run.experiment.workspace
#     m = Model(ws, args.model)
#     print(f"Downloading azure registered model: {args.model}")
#     args.model = m.download(
#         exist_ok=True,
#         target_dir=os.path.join(args.output_dir, "checkpoint", f"{node_rank}-{local_rank}"),
#     )
#     print(f"Download Complete! Path: {args.model}")
#
#
# args.default_root_dir = args.output_dir
#
# model = VAE.load_from_checkpoint(args.model)
#
# if args.num_workers < 0:
#     args.num_workers = num_cpus
#
#
# dataset = PadChestDataset(
#     args.data_folder,
#     args.csv,
#     model.hparams.image_size,
#     True,
#     channels=model.hparams.channels,
# )
#
# dataloader = DataLoader(
#     dataset=dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     num_workers=args.num_workers,
#     pin_memory=True,
# )
#
# # from tqdm import tqdm
# # for _ in tqdm(dataloader):
# #     pass
#
#
# model.eval()
# trainer = pl.Trainer.from_argparse_args(args)
#
# trainer.callbacks.append(
#     VAEPredictionWriter(
#         args.output_dir,
#         write_grid=args.write_grid,
#         write_recon=args.write_recon,
#         latent_output_dir=args.latent_output_dir,
#     )
# )
#
# _ = trainer.predict(model, dataloader)
