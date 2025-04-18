import torch
import torchvision
from torchvision import transforms

from data.seed import generator
from utils.root_folder import find_project_root

_ROOT_FILE = find_project_root() / "data"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

cifar10_dataset_train = torchvision.datasets.CIFAR10(
    root=_ROOT_FILE,
    train=True,
    transform=transform,
    download=True,
)

cifar10_dataset_test = torchvision.datasets.CIFAR10(
    root=_ROOT_FILE,
    train=False,
    transform=transform,
    download=True,
)


def get_dataloaders(batch_size=64):
    cifar10_dataloader_train = torch.utils.data.DataLoader(
        dataset=cifar10_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    cifar10_dataloader_test = torch.utils.data.DataLoader(
        dataset=cifar10_dataset_test,
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
    )

    return cifar10_dataloader_train, cifar10_dataloader_test
