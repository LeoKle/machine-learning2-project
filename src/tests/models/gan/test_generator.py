import unittest
import torch

from models.gan.generator import Generator


class TestGenerator(unittest.TestCase):
    def test_generator_mnist(self):
        model = Generator(dataset="MNIST")
        dummy_input = torch.randn(4, 100)
        output = model(dummy_input)

        self.assertEqual(output.shape, (4, 1, 28, 28))

    def test_generator_cifar10(self):
        model = Generator(dataset="CIFAR10")
        dummy_input = torch.randn(4, 100)
        output = model(dummy_input)

        self.assertEqual(output.shape, (4, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
