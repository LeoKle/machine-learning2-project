import unittest
import torch

from models.gan.discriminator import Discriminator


class TestDiscriminator(unittest.TestCase):
    def test_discriminator_mnist(self):
        model = Discriminator(dataset="MNIST")
        dummy_input = torch.randn(4, 1, 28, 28)
        output = model(dummy_input)

        self.assertEqual(output.shape, (4, 1))

    def test_discriminator_cifar10(self):
        model = Discriminator(dataset="CIFAR10")
        dummy_input = torch.randn(4, 3, 32, 32)
        output = model(dummy_input)

        self.assertEqual(output.shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
