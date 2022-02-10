from torchvision import transforms

from data.dataset import IMAGENET_MEAN, IMAGENET_STD


def get_transform(image_size, normalization=(IMAGENET_MEAN, IMAGENET_STD), channels=3, **kwargs):
    # Define list of image transformations
    image_transformation = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]

    if channels == 1:
        image_transformation.append(transforms.Grayscale(num_output_channels=channels))

    image_transformation.append(transforms.ToTensor())

    if normalization and channels == 3:
        # Normalization with mean and std from ImageNet
        image_transformation.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    return transforms.Compose(image_transformation)
