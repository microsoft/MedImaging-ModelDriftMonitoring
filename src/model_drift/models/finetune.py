from collections import OrderedDict

import torch
import torch.nn as nn
from torchmetrics import AUROC, MetricCollection
from torchvision import models

from .base import VisionModuleBase


class CheXFinetune(VisionModuleBase):
    def __init__(
            self,
            pretrained=None,
            num_classes=10,
            learning_rate=0.001,
            step_size=7,
            gamma=0.1,
            freeze_backbone=False,
            labels=None, params=None
    ):
        super().__init__(labels=labels, params=params)
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        # Transformation

        self.save_hyperparameters()

        model = models.densenet121(pretrained=bool(pretrained))
        if pretrained:
            # Load pre-trained CheXpert model to be fine-tuned
            model.classifier.weight = torch.nn.Parameter(torch.randn(14, 1024))
            model.classifier.bias = torch.nn.Parameter(torch.randn(14))

            new_state_dict = OrderedDict()
            pretrained = torch.load(pretrained)

            if "model_state" in pretrained:
                state_key = "model_state"
                prefixes = ["module.model."]
            elif "state_dict" in pretrained:
                state_key = "state_dict"
                prefixes = ['model.model.', "model."]

            for k, v in pretrained[state_key].items():
                for prefix in prefixes:
                    if k.startswith(prefix):
                        k = k[len(prefix):]  # remove prefix
                        break
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict)

        # Add new last layer for fine-tuning
        num_ftrs = model.classifier.in_features
        self.backbone = model
        self.backbone.classifier = nn.Linear(num_ftrs, num_classes)
        self.activation = nn.Sigmoid()
        self.criterion = nn.BCELoss()

        self.val_metrics = MetricCollection(
            [
                # Accuracy(num_classes=num_classes, average='none'),
                # Recall(num_classes=num_classes, average="none"),
                # Specificity(num_classes=num_classes, average="none"),
                AUROC(num_classes=num_classes, average=None, compute_on_step=False),
            ],
            prefix="val/",
        )
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

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
                if len(v) == len(self.labels):
                    for label, vv in zip(self.labels, v):
                        metrics[f"{k}.{label}"] = vv
                else:
                    for label, vv in enumerate(v):
                        metrics[f"{k}.{label}"] = vv
            else:
                metrics[k] = v
        self.log_dict(metrics)

    def predict_step(self, batch, batch_idx, **kwargs):
        images = batch["image"]
        raw_scores = self.forward(images)
        return raw_scores, self.activation(raw_scores)

    # def train_dataloader(self):
    #     dataset = PadChestDataset(
    #         self.data_folder,
    #         self.train_csv,
    #         IMAGE_SIZE,
    #         True,
    #         channels=CHANNELS,
    #     )
    #     return DataLoader(
    #         dataset=dataset,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         num_workers=self.num_workers,
    #         pin_memory=True,
    #     )
    #
    # def val_dataloader(self):
    #     dataset = PadChestDataset(
    #         self.data_folder,
    #         self.val_csv,
    #         IMAGE_SIZE,
    #         True,
    #         channels=CHANNELS,
    #     )
    #
    #     return DataLoader(
    #         dataset=dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=True,
    #     )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=list(filter(lambda p: p.requires_grad, self.parameters())),
            lr=self.learning_rate,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    @classmethod
    def add_model_args(cls, parser):
        parser = cls.add_common_args(parser)
        group = parser.add_argument_group("module")
        group.add_argument(
            "--pretrained", type=str, dest="pretrained", help="model to fine tune from",
            default=None, )
        group.add_argument(
            "--num_classes", type=int, dest="num_classes", help="number of output classes", default=10, )

        group = parser.add_argument_group("optimization")
        group.add_argument(
            "--learning_rate", type=float, dest="learning_rate", help="base learning rate", default=1e-3, )
        group.add_argument(
            "--freeze_backbone", type=int, dest="freeze_backbone", help="freeze_backbone", default=0, )
        group.add_argument(
            "--weight_decay", type=float, dest="weight_decay", help="weight decay for optimizer", default=1e-5, )
        group.add_argument("--gamma", type=float, dest="gamma", default=0.1,
                           help="reduction factor for lr scheduler"
                                "if reduce on plateau is used, this value is used for 'factor'")
        group.add_argument("--step_size", type=int, dest="step_size",
                           help="step_size for lr schedulers, if reduce on plateau, this value is used for 'patience'",
                           default=7, )
        return parser

    # @classmethod
    # def from_argparse_args(cls, args):
    #     kwargs = cls.get_kwargs(args)
    #     return cls(**kwargs)
