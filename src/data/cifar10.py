from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from data.seed import generator
from utils.root_folder import find_project_root


def get_cifar10_dataloaders(batch_size=64, transform=None, root=None, download=True):
    transform = transform or transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    root = root or (find_project_root() / "data")

    train_dataset = datasets.CIFAR10(
        root=root, train=True, transform=transform, download=download
    )
    test_dataset = datasets.CIFAR10(root=root, train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, generator=generator
    )

    return train_loader, test_loader
