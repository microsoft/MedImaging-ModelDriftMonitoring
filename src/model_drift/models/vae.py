import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from azureml.core import Run
from .base import VisionModuleBase

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Util flags
TEST_LOCAL = False
RUN_AZURE_ML = True


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# Variational Autoencoder
class VAE(VisionModuleBase):
    def __init__(
            self,
            image_dims=(3, 128, 128),
            zsize=8,
            layer_count=3,
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
            ignore_nonfrontal_loss=False,
            #
            labels=None, params=None
    ):
        super().__init__(labels=labels, params=params)

        self.save_hyperparameters()

        self.num_workers = num_workers
        self.base_lr = base_lr
        self.weight_decay = weight_decay

        self.lr_scheduler = lr_scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        self.cooldown = cooldown

        self.image_dims = image_dims
        channels, h_encoder, w_encoder = image_dims
        self.d = width
        self.zsize = zsize
        self.layer_count = layer_count

        self.kl_coeff = kl_coeff

        encode = []
        mul = 1
        inputs = channels
        for i in range(self.layer_count):
            h_encoder, w_encoder = conv_output_shape((h_encoder, w_encoder), 4, 2, 1)
            self.encoder_out = (h_encoder, w_encoder)

            encode += [
                nn.Conv2d(inputs, width * mul, 4, 2, 1),
                nn.BatchNorm2d(width * mul),
                nn.ReLU(),
            ]

            inputs = width * mul
            mul *= 2
        self.encoder = nn.Sequential(*encode)
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

        self.monitor_train = "train/loss"
        self.monitor_val = "val/loss"

        self.image_recon_logger = None

        self.ignore_nonfrontal_loss = ignore_nonfrontal_loss

        if log_recon_images > 0:
            from model_drift.metrics import ImageReconLogger
            self.image_recon_logger = ImageReconLogger(image_dims, k=log_recon_images,
                                                       ignore_nonfrontal_loss=self.ignore_nonfrontal_loss)

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
        if self.training:
            z = q.rsample()
        else:
            z = mu
        return p, q, z

    def forward(self, batch):
        # encode
        x = self.encoder(batch)

        # middle part
        x = x.view(x.shape[0], self.d_max * self.encoder_out[0] * self.encoder_out[1])
        mu = self.fc1(x).squeeze()
        logvar = self.fc2(x).squeeze()

        try:
            p, q, z = self.sample(mu, logvar)
        except:  # noqa
            print(f"mu {mu}, logvar {logvar}")
            raise

        # reconstruct
        image_batch_recon = self.decode(z)

        return image_batch_recon, z, p, q, logvar

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

    def step(self, images, loss_weights=None):

        image_batch_recon, z, p, q, _ = self.forward(images)

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
        batch = image_batch["image"]

        loss, logs, image_batch_recon = self.step(batch)
        for log_name, value in logs.items():
            self.log(f"train/{log_name}", value, on_step=True)
        return loss

    def validation_step(self, image_batch, batch_idx):
        batch, label, frontal = (
            image_batch["image"],
            image_batch["label"],
            image_batch["frontal"],
        )
        lw = frontal if self.ignore_nonfrontal_loss else None
        loss, logs, recon = self.step(batch, loss_weights=lw)

        for log_name, value in logs.items():
            self.log(
                f"val/{log_name}", value.detach().cpu(), on_step=False, on_epoch=True
            )

        if self.image_recon_logger is not None:
            self.image_recon_logger.update(
                batch.detach(),
                recon.detach(),
                label.detach(),
                frontal.detach(),
            )

        return loss

    def predict_step(self, image_batch, batch_idx, **kwargs):
        batch = image_batch["image"]
        image_batch_recon, z, p, q, logvar = self.forward(batch)
        return image_batch_recon, z, logvar

    def on_train_epoch_start(self):
        if self.image_recon_logger is not None:
            self.image_recon_logger.reset()

    def on_validation_epoch_start(self) -> None:
        if self.image_recon_logger is not None:
            self.image_recon_logger.reset()

    def on_validation_end(self) -> None:

        epoch = (
            f"{self.current_epoch:0>4}"
            if not self.trainer.sanity_checking
            else "sanity"
        )

        if self.image_recon_logger is not None:
            if 1:
                run = Run.get_context()
                out = self.image_recon_logger.compute()
                grids = out["grids"]

                if not self.trainer.sanity_checking:
                    for k, v in out["metrics"].items():
                        self.logger.experiment.log_metric(
                            run.id, k, float(v), step=self.trainer.global_step
                        )

                if grids:
                    self.logger.experiment.log_image(
                        run.id, grids["worst_grid"], f"{epoch}-image-worst.png"
                    )
                    self.logger.experiment.log_image(
                        run.id, grids["best_grid"], f"{epoch}-image-best.png"
                    )

    @staticmethod
    def add_argparse_args(parser):

        group = parser.add_argument_group("module")
        group.add_argument("--image_dims", type=int, dest="image_dims", help="image_dims", default=(1, 128, 128))
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
            "--ignore_nonfrontal_loss",
            type=int,
            dest="ignore_nonfrontal_loss",
            help="ignore_nonfrontal_loss",
            default=0,
        )

        group.add_argument(
            "--log_recon_images",
            type=int,
            dest="log_recon_images",
            help="log_recon_images",
            default=0,
        )

        return parser


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(pad) is not tuple:
        pad = (pad, pad)
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    return h, w


def vae_loss(recon_x, x, mu, logvar):
    # print(x.shape, recon_x.shape)
    BCE = torch.mean((recon_x - x) ** 2)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.1


def weighted_mean(values, weights=None):
    if weights is None:
        return values.mean()
    weights = weights.squeeze()
    values = values.squeeze()

    values *= values
    return values.sum() / weights.sum()
