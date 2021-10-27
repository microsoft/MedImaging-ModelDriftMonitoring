import argparse
import os
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import LearningRateMonitor, GPUStatsMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from model_drift.lib import IOMonitor
from model import CheXFinetune

# print("--- ENV VARIABLES ---")
# for k, v in sorted(os.environ.items()):
#     print(f" {k}={v}")
# print("--------------------")

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
parser.add_argument("--run_azure", type=int, dest="run_azure", help="run in AzureML", default="0")
parser.add_argument(
    "--output_dir",
    type=str,
    dest="output_dir",
    help="output directory",
    default="./outputs",
)

parser = CheXFinetune.add_model_args(parser)
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()

if args.num_workers < 0:
    args.num_workers = num_cpus

args.weights_summary = "top" if rank_id == "0-0" else None
RUN_AZURE_ML = args.run_azure

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


if RUN_AZURE_ML:
    from azureml.core import Run, Model

    # Add run context for AML
    run = Run.get_context()

    ws = run.experiment.workspace
    m = Model(ws, args.checkpoint)
    print(f"Downloading azure registered model: {args.checkpoint}")
    args.checkpoint = str(
        m.download(
            exist_ok=True,
            target_dir=os.path.join(args.output_dir, "pretrain", rank_id),
        )
    )
    print(f"Download Complete! Path: {args.checkpoint}")
    mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()

    print("ml flow uri:", mlflow_url)
    mlf_logger = MLFlowLogger(experiment_name=run.experiment.name, tracking_uri=mlflow_url)
    mlf_logger._run_id = run.id
    trainer.logger = mlf_logger


model = CheXFinetune.from_argparse_args(args)


if rank_id == "0-0":
    print(model)

trainer.fit(model)
