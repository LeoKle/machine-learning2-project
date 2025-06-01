import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from data.mnist import get_mnist_dataloaders
from models.classifier.classifier import Classifier
from pipeline.train_classifier import ClassifierTrainingPipeline


class TestClassifierPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataloader_train, cls.dataloader_test = get_mnist_dataloaders(batch_size=32)

        cls.model = Classifier(dataset="MNIST")
        cls.loss_function = nn.NLLLoss()
        cls.optimizer = optim.Adam(cls.model.parameters(), lr=0.001)

        cls.pipeline = ClassifierTrainingPipeline(
            dataloader_train=cls.dataloader_train,
            dataloader_test=cls.dataloader_test,
            model=cls.model,
            loss_function=cls.loss_function,
            optimizer=cls.optimizer,
        )
        cls.pipeline.current_epoch = 1

    def test_training_components(self):
        self.assertIsInstance(self.pipeline.model, nn.Module)
        self.assertIsInstance(self.pipeline.loss_function, nn.Module)
        self.assertIsInstance(self.pipeline.optimizer, optim.Optimizer)

    def test_train_epoch(self):
        for batch, labels in self.dataloader_train:
            self.pipeline.train_epoch(batch, labels)
            break

    def test_evaluation(self):
        self.pipeline.current_epoch = 1
        loss, accuracy = self.pipeline.evaluate()
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 100)

    def test_full_training(self):
        test_save_path = Path("test_model.pth")

        self.pipeline.train(epoch=1, save_path=test_save_path)
        self.assertTrue(test_save_path.exists())
        test_save_path.unlink()


if __name__ == "__main__":
    unittest.main()
