import unittest
import torch
from models.autoencoder.autoencoder import Autoencoder
from models.classifier.classifier import Classifier
from models.classifier.encoder_classifier import EncoderClassifier

class TestEncoderClassifier(unittest.TestCase):
    def test_mnist(self):
        encoder = Autoencoder(dataset_type="MNIST").encoder
        classifier = Classifier(dataset="MNIST")
        model = EncoderClassifier(encoder, classifier)
        
        dummy_input = torch.randn(4, 1, 28, 28)  # MNIST shape
        output = model(dummy_input)
        self.assertEqual(output.shape, (4, 10))  # batch_size, num_classes

    def test_cifar10(self):
        encoder = Autoencoder(dataset_type="CIFAR10").encoder
        classifier = Classifier(dataset="CIFAR10")
        model = EncoderClassifier(encoder, classifier)
        
        dummy_input = torch.randn(4, 3, 32, 32)  # CIFAR10 shape
        output = model(dummy_input)
        self.assertEqual(output.shape, (4, 10))  # batch_size, num_classes

if __name__ == "__main__":
    unittest.main()