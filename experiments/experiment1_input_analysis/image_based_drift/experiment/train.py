import argparse
import os
import pytorch_lightning as pl
import subprocess
import torch
from model import VAE
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from lib import IOMonitor

print("--- ENV VARIABLES ---")
for k, v in sorted(os.environ.items()):
    print(f" {k}={v}")
print("--------------------")

subprocess.call("nvidia-smi")


# Parse incoming parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_folder", type=str, dest="data_folder", help="data folder mounting point"
)
parser.add_argument(
    "--run_azure", type=int, dest="run_azure", help="run in AzureML", default="0"
)
parser.add_argument(
    "--output_dir",
    type=str,
    dest="output_dir",
    help="output directory",
    default="./outputs",
)

parser.add_argument(
    "--scale-batch",
    type=bool,
    dest="scale_batch",
    help="scale batch size with from gpu count",
    default=True,
)

parser.add_argument(
    "--save_top_k",
    type=int,
    dest="save_top_k",
    help="if save_top_k == k, the best k models according to the quantity "
    "monitored will be saved. if save_top_k == 0, no models are saved. "
    "if save_top_k == -1, all models are saved.",
    default=-1,
)


num_gpus = torch.cuda.device_count()
num_cpus = os.cpu_count()

print()
print("=" * 5)
print(" Pytorch Lightning Version:", pl.__version__)
print(" Pytorch Version:", torch.__version__)
print(" Num GPUs:", num_gpus)
print(" Num CPUs", num_cpus)
print("=" * 5)
print()

parser = VAE.add_model_args(parser)
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()

print("Unmodifed args")
prev = dict()
for k, v in sorted(vars(args).items()):
    print(f" {k}: {v}")
    prev[k] = v

if args.num_workers < 0:
    args.num_workers = num_cpus

args.weights_summary = "full"
RUN_AZURE_ML = args.run_azure

output_dir = args.output_dir
if output_dir is not None:
    os.makedirs(output_dir, exist_ok=True)

args.gpus = num_gpus
new_batch_size = max(args.batch_size, num_gpus * args.batch_size)
if new_batch_size != args.batch_size:
    print(
        f"scaling batch size by device count {args.gpus} "
        f"from {args.batch_size} to {new_batch_size}"
    )
    args.batch_size = new_batch_size

model_dirpath = os.path.join(output_dir, "checkpoints")
ckpt_path = os.path.join(model_dirpath, "last.ckpt")

os.makedirs(model_dirpath, exist_ok=True)

if os.path.exists(ckpt_path):
    args.resume_from_checkpoint = ckpt_path
    args.num_sanity_val_steps = 0

print("Modified args:")
for k, v in sorted(vars(args).items()):
    if k in prev and prev[k] == v:
        continue
    print(f" {k}: {prev[k]} -> {v}")


vae = VAE(
    args.train_csv,
    args.val_csv,
    args.data_folder,
    zsize=args.zsize,
    batch_size=args.batch_size,
    image_size=args.image_size,
    layer_count=args.layer_count,
    channels=args.channels,
    width=args.width,
    num_workers=args.num_workers,
    base_lr=args.base_lr,
    weight_decay=args.weight_decay,
    lr_scheduler=args.lr_scheduler,
    step_size=args.step_size,
    gamma=args.gamma,
    min_lr=args.min_lr,
    cooldown=args.cooldown,
    kl_coeff=args.kl_coeff,
    log_recon_images=args.log_recon_images,
)

checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_path,
    filename="{epoch:0>3}",
    save_last=True,
    save_top_k=-1,
    auto_insert_metric_name=False,
    monitor=vae.monitor_val,
    mode="min",
    verbose=True,
    every_n_epochs=1,
)

lr_monitor = LearningRateMonitor(logging_interval="step")

if args.auto_lr_find:
    args.auto_lr_find = "base_lr"

trainer = pl.Trainer.from_argparse_args(args)


if RUN_AZURE_ML:
    from azureml.core import Run

    # Add run context for AML
    run = Run.get_context()

    mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()
    print("ml flow uri:", mlflow_url)
    mlf_logger = MLFlowLogger(
        experiment_name=run.experiment.name, tracking_uri=mlflow_url
    )
    mlf_logger._run_id = run.id
    trainer.logger = mlf_logger

trainer.callbacks.append(checkpoint_callback)
trainer.callbacks.append(lr_monitor)
trainer.callbacks.append(IOMonitor())


if args.auto_lr_find:
    lr_finder = trainer.tuner.lr_find(vae)
    fig = lr_finder.plot(suggest=True)

    vae.base_lr = lr_finder.suggestion()

    if RUN_AZURE_ML:
        trainer.logger.experiment.log_figure(
            run.id, lr_finder.plot(suggest=True), "lr_find.png"
        )
        trainer.logger.experiment.log_param(run.id, "real_base_lr", vae.base_lr)


if args.auto_scale_batch_size:
    print("Batch scaling tuning!")
    new_batch_size = trainer.tuner.scale_batch_size(vae, max_trials=5, init_val=8)
    vae.batch_size = new_batch_size
    if RUN_AZURE_ML:
        trainer.logger.experiment.log_param(run.id, "real_batch_size", vae.batch_size)
    print("Batch scaling done new batch size:", new_batch_size)

# Train model
trainer.fit(vae)
