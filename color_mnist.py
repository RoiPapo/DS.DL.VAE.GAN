from typing import Optional

from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from typing import Callable, Optional


def wrapper(x):
    class _wrap(x):
        def __init__(
                self,
                root: str,
                train: bool,
                target_transform: Optional[Callable] = None
        ) -> None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            download = True if train else False
            self.wrap = MNIST(root, train, transform, target_transform, download)

    return _wrap


if __name__ == '__main__':
    transform_obj = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ColorMNISTDataset = wrapper(datasets.MNIST)
    train_ColorMNISTDataset = ColorMNISTDataset(root='./data/', train=True)
    test_ColorMNISTDataset = ColorMNISTDataset(root='./data/', train=False)

    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transform_obj,
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transform_obj,
                                  download=True)
    print("moo")
