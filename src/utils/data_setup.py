import torch
from data.mnist import get_mnist_dataloaders
from data.cifar10 import get_cifar10_dataloaders

def prepare_dataset(dataset_type: str, batch_size: int):
    if dataset_type == "MNIST":
        train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)
        dummy_input = torch.randn(1, 1, 28, 28)
        encoder_path = "resultsAECNN2_MNIST/MNIST_encoder_weights.pth"
    elif dataset_type == "CIFAR10":
        train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)
        dummy_input = torch.randn(1, 3, 32, 32)
        encoder_path = "resultsAECNN2_CIFAR10/CIFAR10_encoder_weights.pth"
    else:
        raise ValueError("Unsupported dataset_type")

    return train_loader, test_loader, dummy_input, encoder_path

def prepare_dataset_for_cnn(dataset_type: str, batch_size: int):
    if dataset_type == "MNIST":
        train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)
        dummy_input = torch.randn(1, 1, 28, 28)
    elif dataset_type == "CIFAR10":
        train_loader, test_loader = get_cifar10_dataloaders(batch_size=batch_size)
        dummy_input = torch.randn(1, 3, 32, 32)
    else:
        raise ValueError("Unsupported dataset_type")
    return train_loader, test_loader, dummy_input