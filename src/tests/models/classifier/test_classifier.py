import unittest
import torch

from models.classifier.classifier_linear import Classifier


class TestClassifier(unittest.TestCase):
    def test_classifier_mnist(self):
        model = Classifier(dataset="MNIST")
        dummy_input = torch.randn(4, 1, 28, 28)  # batch_size, channels, height, width
        output = model(dummy_input)

        self.assertEqual(output.shape, (4, 10))  # batch_size, num_classes

    def test_classifier_cifar10(self):
        model = Classifier(dataset="CIFAR10")
        dummy_input = torch.randn(4, 3, 32, 32)
        output = model(dummy_input)

        self.assertEqual(output.shape, (4, 10))


if __name__ == "__main__":
    unittest.main()
