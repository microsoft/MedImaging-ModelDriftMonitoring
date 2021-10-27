import time

import torch
from pytorch_lightning.callbacks.base import Callback


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
