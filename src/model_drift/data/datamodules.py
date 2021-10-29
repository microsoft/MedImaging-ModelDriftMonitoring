import argparse
import os
import pytorch_lightning as pl
import yaml
from model_drift import settings
from model_drift.data.padchest import PadChest, LABEL_MAP, BAD_FILES
from torch.utils.data import DataLoader


def _split_dates(s):
    if s is None:
        return tuple(settings.PADCHEST_SPLIT_DATES)
    try:
        return tuple([ss.strip() for ss in s.split(",")])
    except:
        raise argparse.ArgumentTypeError("Dates must be date1,date2")


class PadChestDataModule(pl.LightningDataModule):

    def __init__(self, data_folder, csv_file=None,
                 transforms=None,

                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None,

                 label_map_yaml=None,
                 bad_files_yaml=None,
                 split_dates=settings.PADCHEST_SPLIT_DATES,
                 batch_size=32,
                 num_workers=-1,
                 train_kwargs=None,
                 val_kwargs=None,
                 test_kwargs=None,

                 output_dir='./',

                 frontal_only=False,
                 train_frontal_only=None,
                 val_frontal_only=None,
                 test_frontal_only=None,
                 ):
        super().__init__()
        if transforms is None and (train_transforms is None or val_transforms is None or test_transforms is None):
            raise ValueError("transforms is not specified you must specify transforms for train, val and test")

        if label_map_yaml is not None:
            with open(label_map_yaml, "r") as f:
                label_map = yaml.safe_load(f)
        else:
            label_map = LABEL_MAP

        if bad_files_yaml is not None:
            with open(bad_files_yaml, "r") as f:
                bad_files = yaml.safe_load(f)
        else:
            bad_files = BAD_FILES

        self.label_map = label_map
        self.bad_files = bad_files
        self.split_dates = split_dates

        self.train_kwargs = train_kwargs or {}
        self.val_kwargs = val_kwargs or {}
        self.test_kwargs = test_kwargs or {}

        if train_frontal_only is None:
            train_frontal_only = frontal_only

        if val_frontal_only is None:
            val_frontal_only = frontal_only

        if test_frontal_only is None:
            test_frontal_only = frontal_only

        self.train_kwargs['frontal_only'] = train_frontal_only
        self.val_kwargs['frontal_only'] = val_frontal_only
        self.test_kwargs['frontal_only'] = test_frontal_only

        self.csv_file = csv_file or os.path.join(data_folder, "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")

        self.train_transforms = train_transforms or transforms
        self.val_transforms = val_transforms or transforms
        self.test_transforms = test_transforms or transforms

        self.data_folder = data_folder
        self.batch_size = batch_size

        if num_workers < 0:
            num_workers = os.cpu_count()
        self.num_workers = num_workers
        self.output_dir = output_dir

    @property
    def labels(self):
        return list(self.label_map)

    def setup(self, stage=None) -> None:
        self.train, self.val, self.test = PadChest.splits(self.csv_file, self.split_dates, label_map=self.label_map,
                                                          bad_files=self.bad_files)

        self.train_dataset = self.train.to_dataset(self.data_folder, labels=self.labels,
                                                   transform=self.train_transforms,
                                                   **self.train_kwargs)

        self.val_dataset = self.val.to_dataset(self.data_folder, labels=self.labels, transform=self.val_transforms,
                                               **self.val_kwargs)

        self.test_dataset = self.test.to_dataset(self.data_folder, labels=self.labels, transform=self.test_transforms,
                                                 **self.test_kwargs)

        if self.trainer.is_global_zero:
            self.save()

    def save(self):
        output_dir = os.path.join(self.output_dir, "data")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, "info.yml"), 'w') as f:
            yaml.safe_dump({
                "split_dates": self.split_dates,
                "csv_file": self.csv_file,
                'train_kwargs': self.train_kwargs,
                'val_kwargs': self.val_kwargs,
                'test_kwargs': self.test_kwargs,
                'batch_size': self.batch_size,
                "dataset_len": {
                    "train": len(self.train_dataset),
                    "val": len(self.val_dataset),
                    "test": len(self.test_dataset)
                }
            }, f)

        with open(os.path.join(output_dir, "label_map.yml"), 'w') as f:
            yaml.safe_dump(self.label_map)

        with open(os.path.join(output_dir, "bad_files.yml"), 'w') as f:
            yaml.safe_dump(self.bad_files)

        self.train.save_df(os.path.join(output_dir, "train.csv"))
        self.train.save_df(os.path.join(output_dir, "val.csv"))
        self.train.save_df(os.path.join(output_dir, "test.csv"))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @classmethod
    def add_argparse_args(cls, parser, **kwargs):
        group = parser.add_argument_group("data")
        group.add_argument(
            "--data_folder", type=str, dest="data_folder", help="data folder")
        group.add_argument(
            "--label_map_yaml", type=str, dest="label_map_yaml", help="yaml file with a new label mapping",
            default=None)
        group.add_argument(
            "--bad_files_yaml", type=str, dest="bad_files_yaml", help="yaml file with bad files", default=None)

        parser.add_argument('--split_dates', help="split dates", dest="split_dates", type=_split_dates, nargs=2,
                            default=settings.PADCHEST_SPLIT_DATES_STR)
        group.add_argument("--batch_size", type=int, dest="batch_size", help="batch_size", default=64)
        group.add_argument("--num_workers", type=int, dest="num_workers", help="number of workers for loading",
                           default=-1, )
        group.add_argument("--frontal_only", type=int, dest="frontal_only", help="",
                           default=0, )
        group.add_argument("--train_frontal_only", type=int, dest="train_frontal_only", help="",
                           default=None, )
        group.add_argument("--val_frontal_only", type=int, dest="val_frontal_only", help="",
                           default=None, )
        group.add_argument("--test_frontal_only", type=int, dest="test_frontal_only", help="",
                           default=None, )

        return parser
