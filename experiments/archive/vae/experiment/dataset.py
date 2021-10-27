import os
from numpy.lib.type_check import imag
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        dataframe_path,
        image_size=128,
        normalization=True,
        channels=3,
        frontal_only=False,
        image_dir=None,
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

        self.image_size = image_size
        self.channels = channels
        self.dataframe_path = dataframe_path
        self.folder_dir = folder_dir
        self.image_dir = image_dir
        self.normalization = normalization
        self.frontal_only = frontal_only

        print("Initializing:", str(self))

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

        self._reset_lists()
        self.prepare_data()

    def __str__(self) -> str:

        params = [
            f"folder_dir={_trunc_long_str(self.folder_dir, 30)}",
            f"dataframe_path = {_trunc_long_str(self.dataframe_path, 30)}",
            f"image_size={self.image_size}",
            f"normalization={self.normalization}",
            f"channels={self.channels}",
            f"frontal_only={self.frontal_only}",
        ]
        return f"{type(self).__name__}(" + ", ".join(params) + ")"

    def prepare_data(self):
        raise NotImplementedError()

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
        except:
            print(f"\nbad path: {image_path}\n")
            image_data_original = Image.fromarray(
                np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            )
        return normalize_PIL(image_data_original)


class ChestXrayDataset(BaseDataset):
    def prepare_data(self):
        # Get all image paths and image labels from dataframe
        dataframe = pd.read_csv(
            os.path.join(self.folder_dir, self.dataframe_path), low_memory=False
        )
        dataframe["is_frontal"] = dataframe["Frontal/Lateral"] == "Frontal"
        if self.frontal_only:
            dataframe = dataframe[dataframe["is_frontal"].astype(bool)]
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
            self.frontal.append(float(row["is_frontal"]))
            self.image_labels.append(labels)
            self.image_index.append(row.Path)
            self.recon_image_path.append(row.Path.partition("CheXpert-v1.0/")[2])


class PadChestDataset(BaseDataset):
    def __init__(
        self,
        folder_dir,
        dataframe_path,
        image_size,
        normalization=True,
        channels=3,
        image_dir=None,
    ):
        image_dir = image_dir or os.path.join(folder_dir, "png")
        super().__init__(
            folder_dir,
            dataframe_path,
            image_size,
            normalization=normalization,
            channels=channels,
            image_dir=image_dir,
        )

    bad_files = [
        "216840111366964012283393834152009026160348294_00-014-160.png",
        "216840111366964012283393834152009033102258826_00-059-087.png",
        "216840111366964012339356563862009041122518701_00-061-032.png",
        "216840111366964012339356563862009047085820744_00-054-000.png",
        "216840111366964012339356563862009068084200743_00-045-105.png",
        "216840111366964012339356563862009072111404053_00-043-192.png",
        "216840111366964012373310883942009111121552024_00-072-099.png",
        "216840111366964012373310883942009117084022290_00-064-025.png",
        "216840111366964012373310883942009170084120009_00-097-074.png",
        "216840111366964012373310883942009203115626970_00-031-135.png",
        "216840111366964012487858717522009251095944293_00-018-154.png",
        "216840111366964012558082906712009300162151055_00-078-079.png",
        "216840111366964012558082906712009327122220177_00-102-064.png",
        "216840111366964012558082906712009330202206556_00-102-040.png",
        "216840111366964012734950068292010154110220411_04-008-052.png",
        "216840111366964012734950068292010166125223829_04-006-138.png",
        "216840111366964012819207061112010306085429121_04-020-102.png",
        "216840111366964012819207061112010307142602253_04-014-084.png",
        "216840111366964012819207061112010314122154282_04-013-126.png",
        "216840111366964012819207061112010315104455352_04-024-184.png",
        "216840111366964012819207061112010320134721426_04-022-028.png",
        "216840111366964012819207061112010322100558680_04-001-153.png",
        "216840111366964012819207061112010322154706609_04-021-011.png",
        "216840111366964012904401302362010328092649206_04-014-085.png",
        "216840111366964012904401302362010329193325676_04-016-070.png",
        "216840111366964012904401302362010333080926354_04-019-163.png",
        "216840111366964012922382741642010350171324419_04-000-122.png",
        "216840111366964012922382741642011010093926179_00-126-037.png",
        "216840111366964012948363412702011018092612949_00-124-038.png",
        "216840111366964012959786098432011032091803456_00-172-113.png",
        "216840111366964012959786098432011033083840143_00-176-115.png",
        "216840111366964012959786098432011054135834306_00-176-162.png",
        "216840111366964012989926673512011068092304604_00-163-066.png",
        "216840111366964012989926673512011069111543722_00-165-111.png",
        "216840111366964012989926673512011074122523403_00-163-058.png",
        "216840111366964012989926673512011083122446341_00-158-003.png",
        "216840111366964012989926673512011101135816654_00-184-188.png",
        "216840111366964012989926673512011101154138555_00-191-086.png",
        "216840111366964012989926673512011132200139442_00-157-099.png",
        "216840111366964013076187734852011178154626671_00-145-086.png",
        "216840111366964013076187734852011259174838161_00-131-007.png",
        "216840111366964013076187734852011291090445391_00-196-188.png",
    ]

    def prepare_data(self):

        # Get all image paths and image labels from dataframe
        dataframe = pd.read_csv(
            os.path.join(self.folder_dir, self.dataframe_path), low_memory=False
        )
        dataframe = dataframe[~dataframe["ImageID"].isin(self.bad_files)]
        dataframe["is_frontal"] = True  ## TODO
        if self.frontal_only:
            dataframe = dataframe[dataframe["is_frontal"].astype(bool)]
        for index, row in dataframe.iterrows():

            # Read in image from path
            # print(row)
            image_path = os.path.join(str(int(row["ImageDir"])), str(row["ImageID"]))
            self.image_paths.append(os.path.join(self.image_dir, image_path))
            # if len(row) < 10:
            labels = [0] * 14
            self.frontal.append(float(row["is_frontal"]))
            self.image_labels.append(labels)
            self.image_index.append(row["ImageID"])
            self.recon_image_path.append(image_path)
