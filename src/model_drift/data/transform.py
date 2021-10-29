from torchvision import transforms
from pytorch_lightning.utilities.argparse import from_argparse_args, get_init_arguments_and_types

class Transformer(object):
    pass


class VisionTransformer(Transformer):

    def __init__(self, image_size, normalize="imagenet", channels=3, **kwargs):
        self.image_size = image_size
        self.normalize = normalize
        self.channels = channels

    @property
    def normalization(self):
        if self.normalize == True or self.normalize == "imagenet":
            if not self.channels == 3:
                raise ValueError("ImageNet normalization requires 3 channels")
            return [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
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
        group.add_argument("--normalization", type=str, dest="normalization", help="normalization",
                           default="imagenet", )

        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)
