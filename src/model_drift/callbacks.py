import json
import numpy as np
import os
import random
import torch
import torchvision
from PIL import Image
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


class VAEPredictionWriter(BasePredictionWriter):
    def __init__(
            self,
            output_dir: str,
            write_recon=False,
            write_grid=0,
            write_interval="batch",
            latent_output_dir=None,
    ):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.write_recon = write_recon
        self.write_grid = write_grid
        self.latent_output_dir = latent_output_dir

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

        fn_name = f"latent-{trainer.global_rank}.jsonl"
        for idx, m, var, err in zip(index, mu, logvar, mse):
            s = json.dumps({"index": idx, "mu": m, "logvar": var, "error": err})
            with open(f"{self.output_dir}/{fn_name}", "a") as f:
                print(s, file=f)

            if self.latent_output_dir is None:
                continue
            with open(f"{self.latent_output_dir}/{fn_name}", "a") as f:
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
                    os.path.join(self.output_dir, "images", "recon", recon_path),
                )
                save_image(
                    image,
                    os.path.join(self.output_dir, "images", "inputs", recon_path),
                )


class ClassifierPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval="batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir

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
        index = batch["index"]
        raw_scores, activations = prediction
        fn_name = f"scores-{trainer.global_rank}.jsonl"

        raw_scores = raw_scores.cpu().numpy().tolist()
        activations = activations.cpu().numpy().tolist()
        for idx, score, activation in zip(index, raw_scores, activations):
            s = json.dumps({"index": idx, "score": score, "activation": activation})
            with open(f"{self.output_dir}/{fn_name}", "a") as f:
                print(s, file=f)
