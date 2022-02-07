import time

import json
import logging
import numpy as np
import os
import random
import torch
import torchvision
import tqdm
from PIL import Image
from collections import defaultdict
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.nn import functional as F


def save_image(im_as_tensor, fn):
    subdir = os.path.dirname(fn)
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    im = np.squeeze(im_as_tensor.cpu().numpy().transpose(1, 2, 0))

    is_int = np.issubdtype(im.dtype, np.integer)
    im = np.clip(im, 0, 255 if is_int else 1)
    if not is_int:
        im = im * 255
    Image.fromarray(im.astype(np.uint8)).save(fn)


def make_grid(images, recons, **make_grid_kwargs):
    make_grid_kwargs.setdefault("normalize", True)
    make_grid_kwargs.setdefault("nrow", 8)

    if make_grid_kwargs["nrow"] % 2 != 0:
        make_grid_kwargs["nrow"] += 1

    stack_size = list(images.shape)
    stack_size[0] *= 2
    grid_im = torch.empty(stack_size).cpu()
    grid_im[::2, ...] = images.cpu()
    grid_im[1::2, ...] = recons.cpu()
    grid_im = torchvision.utils.make_grid(grid_im, **make_grid_kwargs)
    return grid_im


class PredictionWriterBase(BasePredictionWriter):
    PRED_FILENAME_BASE = "preds"
    PRED_FILENAME_EXT = "jsonl"

    def __init__(
            self,
            output_dir: str,
            write_interval="batch", ):
        super().__init__(write_interval=write_interval)

        self.output_dir = output_dir
        self.counts = [0]
        self.logger = logging.getLogger(type(self).__name__)

    def on_predict_start(self, trainer, pl_module) -> None:
        self.counts = [0] * trainer.world_size

    def on_predict_end(self, trainer, pl_module) -> None:
        trainer.training_type_plugin.barrier()
        self.logger.info(f"Node: {trainer.global_rank} Complete! Wrote {self.counts[trainer.global_rank]} lines.")

    def get_full_pred_name(self, trainer, global_rank=None):
        return os.path.join(self.output_dir, self.get_pred_file(trainer, global_rank=global_rank))

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices) -> None:
        pass

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx, ):
        self.counts[trainer.global_rank] += len(batch_indices or [])

    def get_pred_file(self, trainer, global_rank=None):
        global_rank = trainer.global_rank if global_rank is None else global_rank
        return f"{self.PRED_FILENAME_BASE}-{global_rank}.{self.PRED_FILENAME_EXT}"

    def get_pred_all_filenames(self, trainer):
        return [self.get_full_pred_name(trainer, global_rank=i) for i in range(trainer.world_size)]

    @property
    def global_pred_filename(self):
        return f"{self.PRED_FILENAME_BASE}.{self.PRED_FILENAME_EXT}"

    def merge_prediction_files(self, trainer):
        trainer.training_type_plugin.barrier()
        if trainer.is_global_zero:
            counts = defaultdict(int)
            count = 0
            fn_out = os.path.join(self.output_dir, self.global_pred_filename)
            for fn_in in self.get_pred_all_filenames(trainer):
                with open(fn_in, "r") as f_in, open(fn_out, 'a') as f_out:
                    for line in tqdm.tqdm(f_in.readlines(), desc=fn_in):
                        print(line.strip(), file=f_out)
                        counts[fn_in] += 1
                        count += 1
            self.logger.info(f"Wrote {count} predictions to {fn_out}")

            s = "\n".join(f" {k}: {c}" for k, c in counts.items())
            self.logger.info(f"Line counts:\n{s}")


class VAEPredictionWriter(PredictionWriterBase):

    def __init__(
            self,
            output_dir: str,
            write_recon=False,
            write_grid=0,
            write_interval="batch",
    ):
        super().__init__(output_dir=output_dir, write_interval=write_interval)
        self.write_recon = write_recon
        self.write_grid = write_grid

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx, ):
        super().write_on_batch_end(trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)

        images, index, recon_paths = (
            batch["image"],
            batch["index"],
            batch["recon_path"],
        )
        # for prediction in predictions:
        image_recons, mu, logvar = prediction

        mse = F.mse_loss(image_recons, images, reduction="none")
        mse = mse.view(mse.size(0), -1).mean(dim=1).cpu()

        mu = mu.cpu().numpy().tolist()
        logvar = logvar.cpu().numpy().tolist()
        mse = mse.squeeze().cpu().numpy().tolist()

        for idx, m, var, err in zip(index, mu, logvar, mse):
            s = json.dumps({"index": idx, "mu": m, "logvar": var, "error": err})
            with open(self.get_full_pred_name(trainer), "a") as f:
                print(s, file=f)

        if self.write_grid > 0 and (self.write_grid >= 1 or random.random() <= self.write_grid):
            save_image(
                make_grid(images, image_recons),
                os.path.join(self.output_dir, "grids", f"{batch_idx}.png"),
            )

        if self.write_recon:
            for recon_path, recon, image in zip(recon_paths, image_recons, images):
                save_image(
                    recon,
                    os.path.join(self.output_dir, "recon", recon_path),
                )


class ClassifierPredictionWriter(PredictionWriterBase):

    def write_on_batch_end(
            self,
            trainer,
            pl_module,
            prediction,
            batch_indices,
            batch,
            batch_idx,
            dataloader_idx,
    ):
        super().write_on_batch_end(trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)
        index = batch["index"]
        labels = batch["label"]
        raw_scores, activations = prediction

        raw_scores = raw_scores.cpu().numpy().tolist()
        activations = activations.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        for idx, score, activation, lbl in zip(index, raw_scores, activations, labels):
            s = json.dumps({"index": idx, "score": score, "activation": activation, "label": lbl})
            with open(self.get_full_pred_name(trainer), "a") as f:
                print(s, file=f)


class IOMonitor(Callback):
    def __init__(self, prefix="train", *args, **kwargs):
        self.prefix = prefix
        super().__init__(*args, **kwargs)

    def on_train_epoch_start(self, trainer, module, *args, **kwargs):
        self.data_time = time.time()
        self.total_time = time.time()

    def on_train_batch_start(self, trainer, module, *args, **kwargs):
        elapsed = time.time() - self.data_time
        module.log(f"{self.prefix}/time.data", elapsed, on_step=True, on_epoch=True)

        self.batch_time = time.time()

    def on_train_batch_end(self, trainer, module, *args, **kwargs):
        elapsed = time.time() - self.batch_time
        module.log(f"{self.prefix}/time.batch", elapsed, on_step=True, on_epoch=True)

        elapsed = time.time() - self.total_time
        module.log(f"{self.prefix}/time.total", elapsed, on_step=True, on_epoch=True)

        self.total_time = time.time()
        self.data_time = time.time()
