import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from azureml.core import Run
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader

from lib import ChestXrayDataset, conv_output_shape, weighted_mean


ImageFile.LOAD_TRUNCATED_IMAGES = True

# Util flags
TEST_LOCAL = False
RUN_AZURE_ML = True


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# Variational Autoencoder
class VAE(LightningModule):
    def __init__(
        self,
        train_csv,
        val_csv,
        data_path,
        batch_size=32,
        zsize=8,
        image_size=128,
        layer_count=3,
        channels=3,
        width=16,
        #
        num_workers=8,
        #
        base_lr=1e-3,
        weight_decay=1e-5,
        #
        lr_scheduler="step",
        step_size=7,
        gamma=0.1,
        min_lr=0,
        cooldown=0,
        kl_coeff=0.1,
        #
        log_recon_images=16,
    ):
        super(VAE, self).__init__()

        self.save_hyperparameters()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.base_lr = base_lr
        self.weight_decay = weight_decay

        self.lr_scheduler = lr_scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        self.cooldown = cooldown

        self.run = None
        self.data_path = data_path

        self.root_folder = self.data_path + os.sep

        h_encoder, w_encoder = image_size, image_size
        self.d = width
        self.zsize = zsize
        self.layer_count = layer_count

        self.kl_coeff = kl_coeff

        encode = []
        mul = 1
        inputs = channels
        for i in range(self.layer_count):
            # print(f"encoder.{i}\n input shape", (inputs, h_encoder, w_encoder))
            h_encoder, w_encoder = conv_output_shape((h_encoder, w_encoder), 4, 2, 1)
            self.encoder_out = (h_encoder, w_encoder)

            encode += [
                nn.Conv2d(inputs, width * mul, 4, 2, 1),
                nn.BatchNorm2d(width * mul),
                nn.ReLU(),
            ]

            inputs = width * mul
            mul *= 2

            # print(f" output shape", (inputs, h_encoder, w_encoder))
        self.encoder = nn.Sequential(*encode)

        # print(f"z", (inputs, h_encoder, w_encoder))
        self.d_max = inputs
        self.fc1 = nn.Linear(
            inputs * self.encoder_out[0] * self.encoder_out[1], self.zsize
        )
        self.fc2 = nn.Linear(
            inputs * self.encoder_out[0] * self.encoder_out[1], self.zsize
        )

        self.d1 = nn.Sequential(
            nn.Linear(self.zsize, inputs * self.encoder_out[0] * self.encoder_out[1]),
            nn.LeakyReLU(0.2),
        )

        decode = []
        mul = inputs // width // 2
        for i in range(1, self.layer_count):
            decode += [
                nn.ConvTranspose2d(inputs, width * mul, 4, 2, 1),
                nn.BatchNorm2d(width * mul),
                nn.LeakyReLU(0.2),
            ]
            inputs = width * mul
            mul //= 2

        decode += [nn.ConvTranspose2d(inputs, channels, 4, 2, 1), nn.Tanh()]

        self.decoder = nn.Sequential(*decode)

        self.train_dataset = ChestXrayDataset(
            self.root_folder, train_csv, image_size, True, channels
        )
        self.val_dataset = ChestXrayDataset(
            self.root_folder, val_csv, image_size, True, channels
        )

        self.monitor_train = "train/loss"
        self.monitor_val = "val/loss"

        self.image_recon_logger = None

        if log_recon_images > 0:
            from metrics import ImageReconLogger

            self.image_recon_logger = ImageReconLogger(
                (channels, image_size, image_size), k=log_recon_images
            )

    def decode(self, x):
        x = x.view(x.shape[0], self.zsize)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, self.encoder_out[0], self.encoder_out[1])
        x = self.decoder(x)

        return x

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu.nan_to_num(0), std.nan_to_num(float("inf")))
        z = q.rsample()
        return p, q, z

    def forward(self, batch):
        # encode
        x = self.encoder(batch)

        # print("encoder out", x.shape)

        # middle part
        x = x.view(x.shape[0], self.d_max * self.encoder_out[0] * self.encoder_out[1])
        mu = self.fc1(x).squeeze()
        logvar = self.fc2(x).squeeze()

        try:
            p, q, z = self.sample(mu, logvar)
        except:  # noqa
            print(f"mu {mu}, logvar {logvar}")
            raise

        # print("z", z.shape)
        # reconstruct
        image_batch_recon = self.decode(z)

        return image_batch_recon, z, p, q

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.base_lr, weight_decay=self.weight_decay
        )

        lr_scheduler = {}
        if self.lr_scheduler == "plateau":

            lr_scheduler["scheduler"] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.gamma,
                patience=self.step_size,
                min_lr=self.min_lr,
                cooldown=self.cooldown,
            )
            lr_scheduler["monitor"] = self.monitor_val
            lr_scheduler["strict"] = True
        else:
            lr_scheduler["scheduler"] = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.step_size, gamma=self.gamma
            )
            lr_scheduler["strict"] = False

        lr_scheduler["interval"] = "epoch"

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def step(self, images, loss_weights=None):

        image_batch_recon, z, p, q = self.forward(images)

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl *= self.kl_coeff

        kl = kl.view(kl.size(0), -1).mean(dim=1)

        recon_loss = F.mse_loss(image_batch_recon, images, reduction="none")
        recon_loss = recon_loss.view(recon_loss.size(0), -1).mean(dim=1)

        recon_loss = weighted_mean(recon_loss, loss_weights)
        kl = weighted_mean(kl, loss_weights)

        loss = kl + recon_loss

        log_dict = {"loss": loss, "recon_loss": recon_loss, "kl": kl}

        return (
            loss,
            log_dict,
            image_batch_recon,
        )

    def training_step(self, image_batch, batch_idx):
        batch, label, index, frontal = image_batch
        loss, logs, image_batch_recon = self.step(batch)
        for log_name, value in logs.items():
            self.log(f"train/{log_name}", value, on_step=True)
        return loss

    def validation_step(self, image_batch, batch_idx):
        batch, label, index, frontal = image_batch
        loss, logs, recon = self.step(batch, loss_weights=frontal)

        for log_name, value in logs.items():
            self.log(
                f"val/{log_name}", value.detach().cpu(), on_step=False, on_epoch=True
            )

        if self.image_recon_logger is not None:
            self.image_recon_logger.update(
                batch.detach(),
                recon.detach(),
                label.detach(),
                index.detach(),
                frontal.detach(),
            )
        #     extra_metrics = self.image_recon_logger(batch, recon, label, index, frontal)
        #     for log_name, value in extra_metrics.items():
        #         self.log(
        #             f"val/{log_name}",
        #             value.detach().cpu(),
        #             on_step=False,
        #             on_epoch=True,
        #         )

        return loss

    def on_train_epoch_start(self):
        if self.image_recon_logger is not None:
            self.image_recon_logger.reset()

    def on_validation_epoch_start(self) -> None:
        if self.image_recon_logger is not None:
            self.image_recon_logger.reset()

    def on_validation_end(self) -> None:

        # print(f"global_zero {self.trainer.is_global_zero}")
        run = Run.get_context()
        epoch = (
            f"{self.current_epoch:0>4}"
            if not self.trainer.sanity_checking
            else "sanity"
        )

        if self.image_recon_logger is not None:
            if 1:
                out = self.image_recon_logger.compute()
                grids = out["grids"]

                values = ["index,frontal,mse"]
                for i, f, l in zip(out["index"], out["frontal"], out["loss"]):
                    values.append(f"{i:f},{f:f},{l:f}")
                self.logger.experiment.log_text(
                    run.id, "\n".join(values), f"{epoch}-values.csv"
                )

                if not self.trainer.sanity_checking:
                    for k, v in out["metrics"].items():
                        self.logger.experiment.log_metric(
                            run.id, k, float(v), step=self.trainer.global_step
                        )

                if grids:
                    self.logger.experiment.log_image(
                        run.id, grids["worst_grid"], f"{epoch}-image-worst.png"
                    )
                    # self.logger.experiment.log_text(
                    #     run.id, worst_loss, f"{epoch}-loss-worst.txt"
                    # )
                    self.logger.experiment.log_image(
                        run.id, grids["best_grid"], f"{epoch}-image-best.png"
                    )
                    # self.logger.experiment.log_text(run.id, best_loss, f"{epoch}-loss-best.txt")

                    # (
                    #     best_mean,
                    #     best_var,
                    #     worst_mean,
                    #     worst_var,
                    # ) = self.image_recon_logger.get_recon_stats()

                    # self.logger.experiment.log_image(
                    #     run.id, best_mean, f"{epoch}-recon-best-mean.png"
                    # )
                    # self.logger.experiment.log_image(
                    #     run.id, best_var, f"{epoch}-recon-best-var.png"
                    # )
                    # self.logger.experiment.log_image(
                    #     run.id, worst_mean, f"{epoch}-recon-worst-mean.png"
                    # )
                    # self.logger.experiment.log_image(
                    #     run.id, worst_var, f"{epoch}-recon-worst-var.png"
            # )

    @staticmethod
    def add_model_args(parser):

        group = parser.add_argument_group("module")

        group.add_argument(
            "--image_size", type=int, dest="image_size", help="Image size", default=128
        )

        group.add_argument(
            "--channels",
            type=int,
            dest="channels",
            help="num input channels",
            default=3,
        )

        group.add_argument(
            "--batch_size", type=int, dest="batch_size", help="batch size", default=32
        )
        group.add_argument("--z", type=int, dest="zsize", help="zsize", default=8)
        group.add_argument(
            "--layer_count",
            type=int,
            dest="layer_count",
            help="layer count for encoder/decoder",
            default=3,
        )
        group.add_argument("--width", type=int, dest="width", help="d", default=16)

        group.add_argument(
            "--train_csv",
            type=str,
            dest="train_csv",
            help="train csv filename",
            default="train.csv",
        )
        group.add_argument(
            "--val_csv",
            type=str,
            dest="val_csv",
            help="validation data csv",
            default="valid.csv",
        )

        group.add_argument(
            "--num_workers",
            type=int,
            dest="num_workers",
            help="number of workers for data loacing",
            default=8,
        )

        group.add_argument(
            "--base_lr",
            type=float,
            dest="base_lr",
            help="base learning rate",
            default=1e-3,
        )

        group.add_argument(
            "--weight_decay",
            type=float,
            dest="weight_decay",
            help="weight decay for optimizer",
            default=1e-5,
        )

        group.add_argument(
            "--gamma",
            type=float,
            dest="gamma",
            help="reduction factor for lr scheduler.  if reduce on plateau is used, this value is used for 'factor'",
            default=0.1,
        )

        group.add_argument(
            "--min_lr",
            type=float,
            dest="min_lr",
            help="minimum learning rate for reduce on plateau",
            default=0,
        )

        group.add_argument(
            "--cooldown",
            type=int,
            dest="cooldown",
            help="cooldown for reduce on plateau",
            default=0,
        )

        group.add_argument(
            "--step_size",
            type=int,
            dest="step_size",
            help="step_size for lr schedulers, if reduce on plateau, this value is used for 'patience'",
            default=7,
        )

        group.add_argument(
            "--lr_scheduler",
            type=str,
            choices=["step", "plateau"],
            dest="lr_scheduler",
            help="lr_scheduler type",
            default="step",
        )

        group.add_argument(
            "--kl_coeff",
            type=float,
            dest="kl_coeff",
            help="kl loss weight",
            default=0.1,
        )

        group.add_argument(
            "--log_recon_images",
            type=int,
            dest="log_recon_images",
            help="log_recon_images",
            default=0,
        )

        return parser
