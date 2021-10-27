import os
from numpy.lib.type_check import imag
from six import b
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from azureml.core import Run
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from torchmetrics.classification.accuracy import Accuracy
from torchvision import models
from collections import OrderedDict
from model_drift.dataset import PadChestDataset
from torchmetrics import AUROC, Recall, Specificity, MetricCollection

IMAGE_SIZE = 320
CHANNELS = 3


class CheXFinetune(LightningModule):
    def __init__(
        self,
        train_csv=None,
        val_csv=None,
        data_folder=None,
        num_workers=8,
        batch_size=32,
        checkpoint="iter_662400.pth.tar",
        num_classes=10,
        learning_rate=0.001,
        step_size=7,
        gamma=0.1,
        freeze_backbone=False,
    ):
        super().__init__()
        self.label_cols = []
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma

        self.data_folder = data_folder

        self.save_hyperparameters()

        # Load pre-trained CheXpert model to be fine-tuned
        new_state_dict = OrderedDict()
        model = models.densenet121(pretrained=True)

        model.classifier.weight = torch.nn.Parameter(torch.randn(14, 1024))
        model.classifier.bias = torch.nn.Parameter(torch.randn(14))
        checkpoint = torch.load(checkpoint)

        for k, v in checkpoint["model_state"].items():
            if k[:13] == "module.model.":
                name = k[13:]  # remove `module.model`
            else:
                name = k
            new_state_dict[name] = v

        checkpoint["model_state"] = new_state_dict
        model.load_state_dict(checkpoint["model_state"])

        # Add new last layer for fine-tuning
        num_ftrs = model.classifier.in_features
        self.backbone = model
        self.backbone.classifier = nn.Linear(num_ftrs, num_classes)
        self.activation = nn.Sigmoid()
        self.criterion = nn.BCELoss()

        self.val_metrics = MetricCollection(
            [
                # Accuracy(num_classes=num_classes, average='none'),
                Recall(num_classes=num_classes, average="none"),
                Specificity(num_classes=num_classes, average="none"),
                AUROC(num_classes=num_classes, average=None, compute_on_step=False),
            ],
            prefix="val/",
        )

        # TODO: Freeze Backbone
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        self.label_cols = PadChestDataset.label_cols

    def forward(self, images):
        return self.backbone(images)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]

        activations = self.activation(self.forward(images))
        loss = self.criterion(activations, labels)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]

        activations = self.activation(self.forward(images))
        loss = self.criterion(activations, labels)
        self.val_metrics.update(activations, labels.to(torch.int))
        self.log("val/loss", loss, on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_end(self) -> None:
        metrics = {}
        for k, v in self.val_metrics.compute().items():
            if len(v) > 1:
                metrics[f"{k}.mean"] = v.mean()
                if len(v) == len(self.label_cols):
                    for label, vv in zip(self.label_cols, v):
                        metrics[f"{k}.{label}"] = vv
                else:
                    for label, vv in enumerate(v):
                        metrics[f"{k}.{label}"] = vv
            else:
                metrics[k] = v
        self.log_dict(metrics)

    def predict_step(self, batch, batch_idx, **kwargs):
        images = batch["image"]
        return self.forward(images)

    def train_dataloader(self):
        dataset = PadChestDataset(
            self.data_folder,
            self.train_csv,
            IMAGE_SIZE,
            True,
            channels=CHANNELS,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = PadChestDataset(
            self.data_folder,
            self.val_csv,
            IMAGE_SIZE,
            True,
            channels=CHANNELS,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=list(filter(lambda p: p.requires_grad, self.parameters())),
            lr=self.learning_rate,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    @staticmethod
    def add_model_args(parser):

        group = parser.add_argument_group("module")
        group.add_argument("--data_folder", type=str, dest="data_folder", help="data folder mounting point")
        group.add_argument("--batch_size", type=int, dest="batch_size", help="batch size", default=8)
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
            "--checkpoint",
            type=str,
            dest="checkpoint",
            help="checkpoint to fine tune from",
            default="iter_662400.pth.tar",
        )

        group.add_argument(
            "--csv_root",
            type=str,
            dest="csv_root",
            help="csv_root",
            default=None,
        )
        group.add_argument(
            "--num_classes",
            type=int,
            dest="num_classes",
            help="number of output classes",
            default=10,
        )

        group.add_argument(
            "--num_workers",
            type=int,
            dest="num_workers",
            help="number of workers for data loading",
            default=8,
        )

        group.add_argument(
            "--learning_rate",
            type=float,
            dest="learning_rate",
            help="base learning rate",
            default=1e-3,
        )

        group.add_argument(
            "--freeze_backbone",
            type=int,
            dest="freeze_backbone",
            help="freeze_backbone",
            default=0,
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
            "--step_size",
            type=int,
            dest="step_size",
            help="step_size for lr schedulers, if reduce on plateau, this value is used for 'patience'",
            default=7,
        )

        return parser

    @classmethod
    def from_argparse_args(cls, args):

        return cls(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            data_folder=args.data_folder,
            checkpoint=args.checkpoint,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            learning_rate=args.learning_rate,
            step_size=args.step_size,
            gamma=args.gamma,
            freeze_backbone=args.freeze_backbone,
        )
