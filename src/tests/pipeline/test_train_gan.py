import unittest
import torch
import torch.nn as nn

from data.mnist import get_mnist_dataloaders
from models.gan.discriminator import Discriminator
from models.gan.generator import Generator
from pipeline.train_gan import GanTrainingPipeline


class TestGanPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.discriminator = Discriminator(dataset="MNIST")
        cls.generator = Generator(dataset="MNIST")

        cls.loss_function_discriminator = nn.BCELoss()
        cls.loss_function_generator = nn.BCELoss()

        cls.optimizer_discriminator = torch.optim.Adam(
            cls.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        cls.optimizer_generator = torch.optim.Adam(
            cls.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

        cls.dataloader_train, cls.dataloader_test = get_mnist_dataloaders()

        cls.pipeline = GanTrainingPipeline(
            dataloader_train=cls.dataloader_train,
            dataloader_test=cls.dataloader_test,
            generator=cls.generator,
            discriminator=cls.discriminator,
            loss_function_discriminator=cls.loss_function_discriminator,
            loss_function_generator=cls.loss_function_generator,
            optimizer_discriminator=cls.optimizer_discriminator,
            optimizer_generator=cls.optimizer_generator,
        )

    def test_epoch(self):
        self.pipeline.train(1)


if __name__ == "__main__":
    unittest.main()
