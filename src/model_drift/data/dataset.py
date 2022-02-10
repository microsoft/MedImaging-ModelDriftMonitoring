import os

import numpy as np
import pandas as pd
import six
import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _trunc_long_str(s, max_size):
    s = str(s)
    if len(s) <= max_size:
        return s
    n_2 = int(max_size / 2 - 3)
    n_1 = max_size - n_2 - 3
    return "{0}...{1}".format(s[:n_1], s[-n_2:])


def normalize_PIL(image):
    if image.mode in ["I"]:
        image = Image.fromarray((np.array(image) / 256).astype(np.uint8))
    return image.convert("RGB")


class BaseDataset(Dataset):
    def __init__(
            self,
            folder_dir,
            dataframe_or_csv,
            transform=None,
            frontal_only=False,
            image_dir=None,
            labels=None,
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

        self.dataframe_or_csv = dataframe_or_csv
        self.folder_dir = folder_dir
        self.image_dir = image_dir
        self.frontal_only = frontal_only
        self.labels = labels
        self.image_transformation = transform
        self._reset_lists()
        self.prepare_data()

    def read_csv(self, csv):
        return pd.read_csv(csv, low_memory=False)

    def __str__(self) -> str:

        params = [
            f"folder_dir={_trunc_long_str(self.folder_dir, 30)}",
            f"labels = {self.labels}",
            f"frontal_only={self.frontal_only}",
        ]
        return f"{type(self).__name__}(" + ", ".join(params) + ")"

    def prepare_data(self):
        print("Initializing:", str(self))

    def _reset_lists(self):
        self.image_paths = []  # List of image paths
        self.image_labels = []  # List of image labels
        self.image_index = []
        self.frontal = []
        self.recon_image_path = []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        """
        Read image at index and convert to torch Tensor
        """

        # Read image
        image_path = self.image_paths[index]
        image_data_original = self.read_image(image_path)
        # if os.path.exists(image_path):
        # image_data = cv2.cvtColor(cv2.imread(image_path).astype('uint8'), cv2.COLOR_BGR2RGB)
        # Resize and convert image to torch tensor
        image_data = self.image_transformation(image_data_original)
        # label = torch.tensor(self.image_labels[index], dtype=torch.long)
        # Return LOADING TIME

        onp = np.array(image_data_original)

        return {
            "image": image_data,
            "label": torch.FloatTensor(self.image_labels[index]),
            "frontal": torch.FloatTensor([self.frontal[index]]),
            "index": self.image_index[index],
            "recon_path": self.recon_image_path[index],
            "o_mean": torch.tensor([onp.mean()], dtype=torch.float),
            "o_max": torch.tensor([onp.max()], dtype=torch.float),
            "o_min": torch.tensor([onp.min()], dtype=torch.float),
        }

    def read_image(self, image_path):
        try:
            image_data_original = Image.open(image_path)
        except BaseException:
            print(f"\nbad path: {image_path}\n")
            raise
        return normalize_PIL(image_data_original)


class ChestXrayDataset(BaseDataset):
    def prepare_data(self):
        # Get all image paths and image labels from dataframe
        if isinstance(self.dataframe_or_csv, six.string_types):
            print(self.dataframe_or_csv)
            dataframe = pd.read_csv(self.dataframe_or_csv, low_memory=False)
        else:
            dataframe = self.dataframe_or_csv

        dataframe["Frontal"] = dataframe["Frontal/Lateral"] == "Frontal"
        if self.frontal_only:
            dataframe = dataframe[dataframe["Frontal"].astype(bool)]
        for _, row in dataframe.iterrows():
            # Read in image from path
            # print(row)
            image_path = os.path.join(
                self.image_dir or self.folder_dir,
                row.Path.partition("CheXpert-v1.0/")[2],
            )
            self.image_paths.append(image_path)
            # if len(row) < 10:
            labels = [0] * 14
            self.frontal.append(float(row["Frontal"]))
            self.image_labels.append(labels)
            self.image_index.append(row.Path)
            self.recon_image_path.append(row.Path.partition("CheXpert-v1.0/")[2])


class PediatricChestXrayDataset(BaseDataset):
    def prepare_data(self):
        # Get all image paths and image labels from dataframe
        if isinstance(self.dataframe_or_csv, six.string_types):
            # print(self.dataframe_or_csv)
            dataframe = pd.read_csv(self.dataframe_or_csv, low_memory=False)
        else:
            dataframe = self.dataframe_or_csv

        for _, row in dataframe.iterrows():
            # Read in image from path
            image_path = os.path.join(
                self.image_dir or self.folder_dir,
                row.Path.partition("Pediatric_Chest_X-ray_Pneumonia/")[2],
            )
            self.image_paths.append(image_path)

            labels = []
            # Labels come from column after path
            for col in row[1:]:
                if col == 1:
                    labels.append(1)
                else:
                    labels.append(0)
            self.image_labels.append(labels)
            self.frontal.append(1.0)
            self.image_index.append(row.Path)
            self.recon_image_path.append(row.Path.partition("Pediatric_Chest_X-ray_Pneumonia/")[2])


class PadChestDataset(BaseDataset):

    def __init__(self, folder_dir, *args, **kwargs):
        kwargs.setdefault("image_dir", os.path.join(folder_dir, "png"))
        super().__init__(folder_dir, *args, **kwargs)

    def prepare_data(self):
        # Get all image paths and image labels from dataframe
        # Get all image paths and image labels from dataframe
        if isinstance(self.dataframe_or_csv, six.string_types):
            dataframe = pd.read_csv(self.dataframe_or_csv, low_memory=False)
        else:
            dataframe = self.dataframe_or_csv

        if self.labels is not None:
            dataframe["binary_label"] = dataframe[self.labels].apply(list, axis=1)

        if "Frontal" not in dataframe:
            dataframe["Frontal"] = dataframe["Projection"].isin(["PA", "AP", "AP_horizontal"])

        if self.frontal_only:
            dataframe = dataframe[dataframe["Frontal"].astype(bool)]

        for index, row in dataframe.iterrows():
            # Read in image from path
            # print(row)
            image_path = os.path.join(str(int(row["ImageDir"])), str(row["ImageID"]))
            self.image_paths.append(os.path.join(self.image_dir, image_path))
            # if len(row) < 10:
            labels = row["binary_label"] if self.labels is not None else [0]
            self.frontal.append(float(row["Frontal"]))
            self.image_labels.append(labels)
            self.image_index.append(row["ImageID"])
            self.recon_image_path.append(image_path)


class MIDRCDataset(BaseDataset):
    def prepare_data(self):
        # Get all image paths and image labels from dataframe
        if isinstance(self.dataframe_or_csv, six.string_types):
            print(self.dataframe_or_csv)
            dataframe = pd.read_csv(self.dataframe_or_csv, low_memory=False)
        else:
            dataframe = self.dataframe_or_csv

        dataframe["Frontal"] = True
        if self.frontal_only:
            dataframe = dataframe[dataframe["Frontal"].astype(bool)]

        for _, row in dataframe.iterrows():
            # Read in image from path
            image_path = os.path.join(
                self.image_dir or self.folder_dir, 'png',
                row['ImageId'][:-3] + 'png',
            )
            self.image_paths.append(image_path)
            labels = row[self.labels].astype(int).tolist()
            self.frontal.append(float(row["Frontal"]))
            self.image_labels.append(labels)
            self.image_index.append(row['ImageId'][:-3] + 'png')
            self.recon_image_path.append(row['ImageId'][:-3] + 'png')
