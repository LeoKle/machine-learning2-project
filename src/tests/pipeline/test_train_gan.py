import unittest
import torch
import torch.nn as nn

from data.mnist import get_mnist_dataloaders
from data.cifar10 import get_cifar10_dataloaders
from models.gan.discriminator import Discriminator
from models.gan.generator import Generator
from pipeline.train_gan import GanTrainingPipeline


class TestGanPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loss_function_discriminator = nn.BCELoss()
        cls.loss_function_generator = nn.BCELoss()

    def setUp(self):
        """Setup for each test method (will be called for both MNIST and CIFAR10 tests)"""
        if not hasattr(self, "dataset"):
            return  # Base case for non-parameterized tests

        if self.dataset == "MNIST":
            self.discriminator = Discriminator(dataset="MNIST")
            self.generator = Generator(dataset="MNIST")
            self.dataloader_train, self.dataloader_test = get_mnist_dataloaders()
        elif self.dataset == "CIFAR10":
            self.discriminator = Discriminator(dataset="CIFAR10")
            self.generator = Generator(dataset="CIFAR10")
            self.dataloader_train, self.dataloader_test = get_cifar10_dataloaders()

        self.optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_generator = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

        self.pipeline = GanTrainingPipeline(
            dataloader_train=self.dataloader_train,
            dataloader_test=self.dataloader_test,
            generator=self.generator,
            discriminator=self.discriminator,
            loss_function_discriminator=self.loss_function_discriminator,
            loss_function_generator=self.loss_function_generator,
            optimizer_discriminator=self.optimizer_discriminator,
            optimizer_generator=self.optimizer_generator,
        )

    def test_epoch_mnist(self):
        """Test with MNIST dataset"""
        self.dataset = "MNIST"
        self.setUp()
        self.pipeline.train(1)

    def test_epoch_cifar10(self):
        """Test with CIFAR10 dataset"""
        self.dataset = "CIFAR10"
        self.setUp()
        self.pipeline.train(1)