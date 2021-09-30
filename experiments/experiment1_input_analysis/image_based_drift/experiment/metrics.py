import torch
from torch._C import TupleType
import torchvision
from lib import weighted_mean
from torch.nn import functional as F
from torchmetrics import Metric

"""
    @staticmethod
    def _collect(prev_loss, prev_imgs, prev_recon, loss, images, recon, k, largest):

        loss = (
            torch.cat((prev_loss.detach().cpu(), loss.detach().cpu()), dim=0)
            if prev_loss != None
            else loss.detach().cpu()
        )
        images = (
            torch.cat((prev_imgs.detach().cpu(), images.detach().cpu()), dim=0)
            if prev_imgs != None
            else images.detach().cpu()
        )
        recon = (
            torch.cat((prev_recon.detach().cpu(), recon.detach().cpu()), dim=0)
            if prev_recon != None
            else recon.detach().cpu()
        )

        loss, idx = torch.topk(
            loss,
            min(k, loss.shape[0]),
            dim=0,
            largest=largest,
        )

        return loss, images[idx.flatten()], recon[idx.flatten()]
        """


class ImageReconLogger(Metric):
    def get_image_shape(self, batch_size=0):
        return [batch_size] + list(self.img_shape)

    def __init__(self, img_size, k=32):

        super().__init__(compute_on_step=False, dist_sync_on_step=False)

        # print("img_size:", img_size)

        self.img_shape = img_size
        self.k = k

        tenor_size = self.get_image_shape()
        def_img = lambda: torch.empty(tenor_size).float()

        # print("tenor_size:", tenor_size)

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
            "indexes", default=torch.empty((0,)).float(), dist_reduce_fx="cat"
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

    def update(self, image_batch, image_batch_recon, label, indexes, weights):
        raw_loss = F.mse_loss(image_batch_recon, image_batch, reduction="none")
        loss = raw_loss.view(raw_loss.size(0), -1).mean(dim=1)

        weights = weights.squeeze()

        self.recon_loss_frontal += (weights * loss).sum()
        self.recon_loss_lateral += ((1 - weights) * loss).sum()

        self.total_frontal += weights.sum()
        self.total_lateral += (1 - weights).sum()

        self.losses = torch.cat((self.losses, loss), dim=0)
        self.indexes = torch.cat((self.indexes, indexes), dim=0)
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

        # print(f"update {self.recon_loss_frontal}")

    def compute(self):

        # print(f"self.recon_loss_frontal {self.recon_loss_frontal}")
        out = {
            "metrics": {
                "val/recon_loss_frontal": self.recon_loss_frontal.float()
                / self.total_frontal.float(),
                "val/recon_loss_lateral": self.recon_loss_lateral.float()
                / self.total_lateral.float(),
            },
            "loss": self.losses.view(-1, 1).squeeze(),
            "index": self.indexes.view(-1, 1).squeeze(),
            "frontal": self.frontal.view(-1, 1).squeeze(),
        }

        out["grids"] = self.get_grids() if self.k else None

        return out

    def get_grids(self):
        # print(
        #     f"b1 {self.best_loss.shape}, {self.best_img.shape}, {self.best_recon.shape}"
        # )
        # print(
        #     f"w1, {self.worst_loss.shape}, {self.worst_img.shape}, {self.worst_recon.shape}"
        # )

        self.best_loss = self.best_loss.view(-1, 1)

        self.best_img = self.best_img.view(*self.get_image_shape(-1))
        self.best_recon = self.best_recon.view(*self.get_image_shape(-1))

        self.worst_loss = self.worst_loss.view(-1, 1)
        self.worst_img = self.worst_img.view(*self.get_image_shape(-1))
        self.worst_recon = self.worst_recon.view(*self.get_image_shape(-1))

        # print(
        #     f"b2 {self.best_loss.shape}, {self.best_img.shape}, {self.best_recon.shape}"
        # )
        # print(
        #     f"w2, {self.worst_loss.shape}, {self.worst_img.shape}, {self.worst_recon.shape}"
        # )

        self.best_loss, self.best_img, self.best_recon = self._topk(
            self.best_loss, self.best_img, self.best_recon, self.k, False
        )

        self.worst_loss, self.worst_img, self.worst_recon = self._topk(
            self.worst_loss, self.worst_img, self.worst_recon, self.k, True
        )

        # print(
        #     f"b3 {self.best_loss.shape}, {self.best_img.shape}, {self.best_recon.shape}"
        # )
        # print(
        #     f"w3, {self.worst_loss.shape}, {self.worst_img.shape}, {self.worst_recon.shape}"
        # )

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
        # print(f"grid_shape {grid_im.shape},... {images.shape}, {recons.shape}")
        grid_im[::2, ...] = images.cpu()
        grid_im[1::2, ...] = recons.cpu()
        grid_im = (
            torchvision.utils.make_grid(grid_im, **make_grid_kwargs)
            .numpy()
            .transpose(1, 2, 0)
        )
        return grid_im

    # def make_grid(self, loss_text=True, **make_grid_kwargs):
    #     best_im, best_loss = self._make_grid(
    #         self.best_images,
    #         self.best_recons,
    #         self.best_losses,
    #         loss_text=loss_text,
    #         **make_grid_kwargs,
    #     )
    #
    #     worst_im, worst_loss = self._make_grid(
    #         self.worst_images,
    #         self.worst_recons,
    #         self.worst_losses,
    #         loss_text=loss_text,
    #         **make_grid_kwargs,
    #     )
    #
    #     return best_im, best_loss, worst_im, worst_loss

    def get_values_csv(self, sep=","):
        return "\n".join(
            [f"index,mse"] + ["{}{}{:.5f}".format(p, sep, v) for p, v in self.values]
        )

    # def get_recon_stats(self):
    #     best_var = self.best_recons.var(dim=0).cpu().numpy().transpose(1, 2, 0)
    #     best_mean = self.best_recons.mean(dim=0).cpu().numpy().transpose(1, 2, 0)
    #
    #     worst_var = self.worst_recons.var(dim=0).cpu().numpy().transpose(1, 2, 0)
    #     worst_mean = self.worst_recons.mean(dim=0).cpu().numpy().transpose(1, 2, 0)
    #
    #     return best_mean, best_var, worst_mean, worst_var
