import warnings
from pytorch_lightning.utilities.argparse import from_argparse_args
from torchvision import transforms


class Transformer(object):
    pass


class VisionTransformer(Transformer):

    def __init__(self, image_size, normalize="imagenet", channels=3, **kwargs):
        self.image_size = image_size
        self.normalize = normalize
        self.channels = channels

    @property
    def normalization(self):
        if self.normalize == "imagenet":
            if self.channels == 3:
                return [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            else:
                warnings.warn(
                    "ImageNet normalization requires 3 channels, skipping normalization")
        return []

    @property
    def train_transform(self):
        image_transformation = [
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
        ]
        if self.channels == 1:
            image_transformation.append(transforms.Grayscale(num_output_channels=self.channels))
        image_transformation.append(transforms.ToTensor())
        image_transformation += self.normalization
        return transforms.Compose(image_transformation)

    @property
    def infer_transform(self):
        return self.train_transform

    @classmethod
    def add_argparse_args(cls, parser):
        group = parser.add_argument_group("transform")
        group.add_argument("--image_size", type=int, dest="image_size", help="image_size", default=320)
        group.add_argument("--channels", type=int, dest="channels", help="channels", default=3)
        group.add_argument("--normalize", type=str, dest="normalize", help="normalize",
                           default="imagenet", )

        return parser

    @property
    def dims(self):
        return (self.channels, self.image_size, self.image_size)

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)
