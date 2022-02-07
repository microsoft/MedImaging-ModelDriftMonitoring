import torch
import torchvision
from torch.nn import functional as F
from torchmetrics import Metric
from torchvision import utils


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
    grid_im = (utils.make_grid(grid_im, **make_grid_kwargs)
               .numpy()
               .transpose(1, 2, 0))
    return grid_im


class ImageReconLogger(Metric):

    def get_image_shape(self, batch_size=0):
        return [batch_size] + list(self.img_shape)

    def __init__(self, img_size, k=32, ignore_nonfrontal_loss=False):
        super().__init__(compute_on_step=False, dist_sync_on_step=False)

        self.img_shape = img_size
        self.k = k
        self.ignore_nonfrontal_loss = ignore_nonfrontal_loss

        tenor_size = self.get_image_shape()
        def_img = lambda: torch.empty(tenor_size).float()  # noqa

        self.add_state(
            "worst_loss", default=torch.empty((0,)).float(), dist_reduce_fx="cat"
        )
        self.add_state(
            "worst_img",
            default=def_img(),
            dist_reduce_fx="cat",
        )
        self.add_state(
            "worst_recon",
            default=def_img(),
            dist_reduce_fx="cat",
        )

        self.add_state(
            "best_loss", default=torch.empty((0,)).float(), dist_reduce_fx="cat"
        )
        self.add_state(
            "best_img",
            default=def_img(),
            dist_reduce_fx="cat",
        )
        self.add_state(
            "best_recon",
            default=def_img(),
            dist_reduce_fx="cat",
        )

        self.add_state(
            "losses", default=torch.empty((0,)).float(), dist_reduce_fx="cat"
        )
        self.add_state(
            "frontal", default=torch.empty((0,)).float(), dist_reduce_fx="cat"
        )

        self.add_state(
            "recon_loss_frontal",
            default=torch.tensor(0).float(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "recon_loss_lateral",
            default=torch.tensor(0).float(),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "total_frontal", default=torch.tensor(0).float(), dist_reduce_fx="sum"
        )
        self.add_state(
            "total_lateral", default=torch.tensor(0).float(), dist_reduce_fx="sum"
        )

    @staticmethod
    def _topk(loss, images, recon, k, largest):
        loss, idx = torch.topk(loss, min(k, loss.shape[0]), dim=0, largest=largest)
        return loss, images[idx.flatten()], recon[idx.flatten()]

    def update(self, image_batch, image_batch_recon, label, weights):
        raw_loss = F.mse_loss(image_batch_recon, image_batch, reduction="none")
        loss = raw_loss.view(raw_loss.size(0), -1).mean(dim=1)

        weights = weights.squeeze()

        self.recon_loss_frontal += (weights * loss).sum()
        self.recon_loss_lateral += ((1 - weights) * loss).sum()

        self.total_frontal += weights.sum()
        self.total_lateral += (1 - weights).sum()

        self.losses = torch.cat((self.losses, loss), dim=0)
        self.frontal = torch.cat((self.frontal, weights), dim=0)

        self.worst_loss, self.worst_img, self.worst_recon = self._topk(
            torch.cat((self.worst_loss, loss), dim=0),
            torch.cat((self.worst_img, image_batch), dim=0),
            torch.cat((self.worst_recon, image_batch_recon), dim=0),
            self.k,
            True,
        )

        self.best_loss, self.best_img, self.best_recon = self._topk(
            torch.cat((self.best_loss, loss), dim=0),
            torch.cat((self.best_img, image_batch), dim=0),
            torch.cat((self.best_recon, image_batch_recon), dim=0),
            self.k,
            False,
        )

    def compute(self):

        recon_loss_frontal = self.recon_loss_frontal.float() / self.total_frontal.float()
        recon_loss_lateral = self.recon_loss_lateral.float() / self.total_lateral.float()

        if self.ignore_nonfrontal_loss:
            val_loss = recon_loss_frontal
        else:
            val_loss = (self.recon_loss_frontal.float() + self.recon_loss_lateral.float()) / (
                    self.total_frontal.float() + self.total_lateral.float())

        out = {"metrics": {
            "val/weighted_recon_loss": val_loss,
            "val/recon_loss_frontal": recon_loss_frontal,
            "val/recon_loss_lateral": recon_loss_lateral,
        }, "loss": self.losses.view(-1, 1).squeeze(),
            "frontal": self.frontal.view(-1, 1).squeeze(), "grids": self.get_grids() if self.k else None}

        return out

    def get_grids(self):
        self.best_loss = self.best_loss.view(-1, 1)

        self.best_img = self.best_img.view(*self.get_image_shape(-1))
        self.best_recon = self.best_recon.view(*self.get_image_shape(-1))

        self.worst_loss = self.worst_loss.view(-1, 1)
        self.worst_img = self.worst_img.view(*self.get_image_shape(-1))
        self.worst_recon = self.worst_recon.view(*self.get_image_shape(-1))

        self.best_loss, self.best_img, self.best_recon = self._topk(
            self.best_loss, self.best_img, self.best_recon, self.k, False
        )

        self.worst_loss, self.worst_img, self.worst_recon = self._topk(
            self.worst_loss, self.worst_img, self.worst_recon, self.k, True
        )

        grids = {}
        grids["best_grid"] = self._make_grid(self.best_img, self.best_recon)
        grids["best_loss"] = self.best_loss.flatten()

        grids["worst_grid"] = self._make_grid(self.worst_img, self.worst_recon)
        grids["worst_loss"] = self.worst_loss.flatten()

        return grids

    def _make_grid(self, images, recons, **make_grid_kwargs):
        make_grid_kwargs.setdefault("normalize", True)
        make_grid_kwargs.setdefault("nrow", 8)

        if make_grid_kwargs["nrow"] % 2 != 0:
            make_grid_kwargs["nrow"] += 1

        grid_im = torch.empty(self.get_image_shape(self.k * 2)).cpu()
        grid_im[::2, ...] = images.cpu()
        grid_im[1::2, ...] = recons.cpu()
        grid_im = torchvision.utils.make_grid(grid_im, **make_grid_kwargs).numpy().transpose(1, 2, 0)
        return grid_im
