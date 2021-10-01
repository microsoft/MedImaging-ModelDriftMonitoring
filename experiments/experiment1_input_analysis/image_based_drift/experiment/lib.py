import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import time
from PIL import ImageFile
from pytorch_lightning.callbacks.base import Callback

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        folder_dir,
        dataframe_path,
        image_size,
        normalization,
        channels=3,
    ):

        """
        Init Dataset

        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe_path: CSV
            dataframe_path csv contains all information of images

        image_size: int
            image size to rescale
        normalization: bool
            whether applying normalization with mean and std from ImageNet or not
        """

        self.image_paths = []  # List of image paths
        self.image_labels = []  # List of image labels
        self.image_index = []
        self.frontal = []

        # Define list of image transformations
        image_transformation = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]

        if channels == 1:
            image_transformation.append(
                transforms.Grayscale(num_output_channels=channels)
            )

        image_transformation.append(transforms.ToTensor())

        if normalization and channels == 3:
            # Normalization with mean and std from ImageNet
            image_transformation.append(
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            )

        self.image_transformation = transforms.Compose(image_transformation)

        # Get all image paths and image labels from dataframe
        dataframe = pd.read_csv(dataframe_path, low_memory=False)
        dataframe["is_frontal"] = (dataframe["Frontal/Lateral"] == "Frontal").astype(
            int
        )
        for _, row in dataframe.iterrows():

            # Read in image from path
            # print(row)
            image_path = os.path.join(
                folder_dir, row.Path.partition("CheXpert-v1.0/")[2]
            )
            self.image_paths.append(image_path)
            # if len(row) < 10:
            labels = [0] * 14
            self.frontal.append(row["is_frontal"])
            self.image_labels.append(labels)
            self.image_index.append(row.Path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        """
        Read image at index and convert to torch Tensor
        """

        # Read image
        image_path = self.image_paths[index]
        start = time.process_time()
        image_data = Image.open(image_path).convert("RGB")
        # if os.path.exists(image_path):
        # image_data = cv2.cvtColor(cv2.imread(image_path).astype('uint8'), cv2.COLOR_BGR2RGB)
        # Resize and convert image to torch tensor
        image_data = self.image_transformation(image_data)
        # label = torch.tensor(self.image_labels[index], dtype=torch.long)
        # Return LOADING TIME

        return (
            image_data,
            torch.FloatTensor(self.image_labels[index]),
            torch.LongTensor([index]),
            torch.FloatTensor([self.frontal[index]]),
        )


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
    def on_train_epoch_start(self, trainer, module, *args, **kwargs):
        self.data_time = time.time()
        self.total_time = time.time()

    def on_train_batch_start(self, trainer, module, *args, **kwargs):
        elapsed = time.time() - self.data_time
        module.log("train/time.data", elapsed, on_step=True, on_epoch=True)

        self.batch_time = time.time()

    def on_train_batch_end(self, trainer, module, *args, **kwargs):

        elapsed = time.time() - self.batch_time
        module.log("train/time.batch", elapsed, on_step=True, on_epoch=True)

        elapsed = time.time() - self.total_time
        module.log("train/time.total", elapsed, on_step=True, on_epoch=True)

        self.total_time = time.time()
        self.data_time = time.time()
