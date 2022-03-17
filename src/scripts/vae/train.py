#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import os
import pytorch_lightning as pl
import torch
import yaml
from pathlib import Path
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

library_path = str(Path(__file__).parent.parent.parent)
PYPATH = os.environ.get("PYTHONPATH", "").split(":")
if library_path not in PYPATH:
    PYPATH.append(library_path)
    os.environ["PYTHONPATH"] = ":".join(PYPATH)

from model_drift.callbacks import IOMonitor
from model_drift.models.vae import VAE
from model_drift.azure_utils import get_azure_logger
from model_drift.data.datamodules import PadChestDataModule, CheXpertDataModule
from model_drift.data.transform import VisionTransformer

num_gpus = torch.cuda.device_count()
num_cpus = os.cpu_count()
node_rank = os.environ.get("NODE_RANK", 0)
local_rank = os.environ.get("LOCAL_RANK", 0)
world_size = os.environ.get("WORLD_SIZE", 1)
global_rank = os.environ.get("RANK", 0)
rank_id = f"{node_rank}-{local_rank}"

print()
print("=" * 5)
print(" Pytorch Lightning Version:", pl.__version__)
print(" Pytorch Version:", torch.__version__)
print(" Num GPUs:", num_gpus)
print(" Num CPUs:", num_cpus)
print(" Node Rank:", node_rank)
print(" Local Rank:", local_rank)
print(" World Size:", world_size)
print(" Global Rank:", global_rank)
print(" Rank ID:", rank_id)
print("=" * 5)
print()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, dest="dataset", help="dataset to train on", default="chexpert",
                    choices=['chexpert', 'padchest'])
known_args, _ = parser.parse_known_args()

print(known_args)

datasets = {"padchest": PadChestDataModule, "chexpert": CheXpertDataModule}

dm_cls = datasets[known_args.dataset]
print(dm_cls)

parser.add_argument("--run_azure", type=int, dest="run_azure", help="run in AzureML", default="0")
parser.add_argument("--output_dir", type=str, dest="output_dir", help="output directory", default="./outputs", )

parser = VisionTransformer.add_argparse_args(parser)
parser = VAE.add_argparse_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
parser = dm_cls.add_argparse_args(parser)

args = parser.parse_args()

if args.num_workers < 0:
    args.num_workers = num_cpus

args.weights_summary = "top" if rank_id == "0-0" else None
output_dir = args.output_dir
if output_dir is not None:
    os.makedirs(output_dir, exist_ok=True)

args.gpus = num_gpus
new_batch_size = max(args.batch_size, num_gpus * args.batch_size)
if new_batch_size != args.batch_size:
    print(f"scaling batch size by device count {args.gpus} " f"from {args.batch_size} to {new_batch_size}")
    args.batch_size = new_batch_size

model_dirpath = os.path.join(output_dir, "checkpoints")
ckpt_path = os.path.join(model_dirpath, "last.ckpt")

os.makedirs(model_dirpath, exist_ok=True)

if os.path.exists(ckpt_path):
    args.resume_from_checkpoint = ckpt_path
    args.num_sanity_val_steps = 0

lr_monitor = LearningRateMonitor(logging_interval="step")
checkpoint_callback = ModelCheckpoint(
    dirpath=model_dirpath,
    filename="{epoch:0>3}",
    save_last=True,
    save_top_k=-1,
    every_n_epochs=1,
)

trainer = pl.Trainer.from_argparse_args(args)
trainer.callbacks.append(checkpoint_callback)
trainer.callbacks.append(lr_monitor)
trainer.callbacks.append(IOMonitor())
# trainer.callbacks.append(GPUStatsMonitor())


if args.run_azure:
    trainer.logger = get_azure_logger()

transformer = VisionTransformer.from_argparse_args(args)
dm = dm_cls.from_argparse_args(args, output_dir=args.output_dir, transforms=transformer.train_transform)
args.image_dims = transformer.dims
params = vars(args)
model = VAE.from_argparse_args(args, params=params)

if args.auto_lr_find:
    lr_finder = trainer.tuner.lr_find(model)
    fig = lr_finder.plot(suggest=True)

    model.base_lr = lr_finder.suggestion()

    if args.run_azure:
        from azureml.core import Run

        run = Run.get_context()
        trainer.logger.experiment.log_figure(run.id, lr_finder.plot(suggest=True), "lr_find.png")
        trainer.logger.experiment.log_param(run.id, "real_base_lr", model.base_lr)

if trainer.is_global_zero:
    with open(os.path.join(args.output_dir, "input.yaml"), 'w') as f:
        yaml.safe_dump(params, f)
    with open(os.path.join(args.output_dir, "model.txt"), 'w') as f:
        print(model, file=f)

trainer.fit(model, dm)

#
# import argparse
# import os
# import pytorch_lightning as pl
# import subprocess
# import torch
#
# from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# from pytorch_lightning.loggers import MLFlowLogger
#
# from model_drift.callbacks import IOMonitor
# from model_drift.models.vae import VAE
#
# print("--- ENV VARIABLES ---")
# for k, v in sorted(os.environ.items()):
#     print(f" {k}={v}")
# print("--------------------")
#
# subprocess.call("nvidia-smi")
#
#
# # Parse incoming parameters
# parser = argparse.ArgumentParser()
# parser.add_argument("--data_folder", type=str, dest="data_folder", help="data folder mounting point")
# parser.add_argument("--run_azure", type=int, dest="run_azure", help="run in AzureML", default="0")
# parser.add_argument(
#     "--output_dir",
#     type=str,
#     dest="output_dir",
#     help="output directory",
#     default="./outputs",
# )
#
# parser.add_argument(
#     "--scale-batch",
#     type=bool,
#     dest="scale_batch",
#     help="scale batch size with from gpu count",
#     default=True,
# )
#
# parser.add_argument(
#     "--save_top_k",
#     type=int,
#     dest="save_top_k",
#     help="if save_top_k == k, the best k models according to the quantity "
#     "monitored will be saved. if save_top_k == 0, no models are saved. "
#     "if save_top_k == -1, all models are saved.",
#     default=-1,
# )
#
#
# num_gpus = torch.cuda.device_count()
# num_cpus = os.cpu_count()
#
# print()
# print("=" * 5)
# print(" Pytorch Lightning Version:", pl.__version__)
# print(" Pytorch Version:", torch.__version__)
# print(" Num GPUs:", num_gpus)
# print(" Num CPUs", num_cpus)
# print("=" * 5)
# print()
#
# parser = VAE.add_model_args(parser)
# parser = pl.Trainer.add_argparse_args(parser)
#
# args = parser.parse_args()
#
# print("Unmodifed args")
# prev = dict()
# for k, v in sorted(vars(args).items()):
#     print(f" {k}: {v}")
#     prev[k] = v
#
# if args.num_workers < 0:
#     args.num_workers = num_cpus
#
# args.weights_summary = "full"
# RUN_AZURE_ML = args.run_azure
#
# output_dir = args.output_dir
# if output_dir is not None:
#     os.makedirs(output_dir, exist_ok=True)
#
# args.gpus = num_gpus
# new_batch_size = max(args.batch_size, num_gpus * args.batch_size)
# if new_batch_size != args.batch_size:
#     print(f"scaling batch size by device count {args.gpus} " f"from {args.batch_size} to {new_batch_size}")
#     args.batch_size = new_batch_size
#
# model_dirpath = os.path.join(output_dir, "checkpoints")
# ckpt_path = os.path.join(model_dirpath, "last.ckpt")
#
# os.makedirs(model_dirpath, exist_ok=True)
#
# if os.path.exists(ckpt_path):
#     args.resume_from_checkpoint = ckpt_path
#     args.num_sanity_val_steps = 0
#
# print("Modified args:")
# for k, v in sorted(vars(args).items()):
#     if k in prev and prev[k] == v:
#         continue
#     print(f" {k}: {prev[k]} -> {v}")
#
#
# vae = VAE(
#     args.train_csv,
#     args.val_csv,
#     args.data_folder,
#     zsize=args.zsize,
#     batch_size=args.batch_size,
#     image_size=args.image_size,
#     layer_count=args.layer_count,
#     channels=args.channels,
#     width=args.width,
#     num_workers=args.num_workers,
#     base_lr=args.base_lr,
#     weight_decay=args.weight_decay,
#     lr_scheduler=args.lr_scheduler,
#     step_size=args.step_size,
#     gamma=args.gamma,
#     min_lr=args.min_lr,
#     cooldown=args.cooldown,
#     kl_coeff=args.kl_coeff,
#     log_recon_images=args.log_recon_images,
# )
#
# checkpoint_callback = ModelCheckpoint(
#     dirpath=model_dirpath,
#     filename="{epoch:0>3}",
#     save_last=True,
#     save_top_k=-1,
#     auto_insert_metric_name=False,
#     monitor=vae.monitor_val,
#     mode="min",
#     verbose=True,
#     every_n_epochs=1,
# )
#
# lr_monitor = LearningRateMonitor(logging_interval="step")
#
# if args.auto_lr_find:
#     args.auto_lr_find = "base_lr"
#
# trainer = pl.Trainer.from_argparse_args(args)
#
#
# if RUN_AZURE_ML:
#     from azureml.core import Run
#
#     # Add run context for AML
#     run = Run.get_context()
#
#     mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()
#     print("ml flow uri:", mlflow_url)
#     mlf_logger = MLFlowLogger(experiment_name=run.experiment.name, tracking_uri=mlflow_url)
#     mlf_logger._run_id = run.id
#     trainer.logger = mlf_logger
#
# trainer.callbacks.append(checkpoint_callback)
# trainer.callbacks.append(lr_monitor)
# trainer.callbacks.append(IOMonitor())
#
#
# if args.auto_lr_find:
#     lr_finder = trainer.tuner.lr_find(vae)
#     fig = lr_finder.plot(suggest=True)
#
#     vae.base_lr = lr_finder.suggestion()
#
#     if RUN_AZURE_ML:
#         trainer.logger.experiment.log_figure(run.id, lr_finder.plot(suggest=True), "lr_find.png")
#         trainer.logger.experiment.log_param(run.id, "real_base_lr", vae.base_lr)
#
#
# if args.auto_scale_batch_size:
#     print("Batch scaling tuning!")
#     new_batch_size = trainer.tuner.scale_batch_size(vae, max_trials=5, init_val=8)
#     vae.batch_size = new_batch_size
#     if RUN_AZURE_ML:
#         trainer.logger.experiment.log_param(run.id, "real_batch_size", vae.batch_size)
#     print("Batch scaling done new batch size:", new_batch_size)
#
# # Train model
# trainer.fit(vae)
