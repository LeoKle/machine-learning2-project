import torch
import torchvision
from torchvision import transforms

from data.seed import generator
from utils.root_folder import find_project_root

_ROOT_FILE = find_project_root() / "data"

mnist_dataset_train = torchvision.datasets.MNIST(
    root=_ROOT_FILE,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.01307), (0.3081))]
    ),
    download=True,
)

mnist_dataset_test = torchvision.datasets.MNIST(
    root=_ROOT_FILE,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.01307), (0.3081))]
    ),
)


def get_dataloaders(batch_size=64):
    mnist_dataloader_train = torch.utils.data.DataLoader(
        dataset=mnist_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    mnist_dataloader_test = torch.utils.data.DataLoader(
        dataset=mnist_dataset_test,
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
    )

    return mnist_dataloader_train, mnist_dataloader_test
