import unittest
import torch
import torch.nn as nn


from data.cifar10 import get_cifar10_dataloaders
from data.mnist import get_mnist_dataloaders
from models.autoencoder.autoencoder_CNN2 import AutoencoderCNN2
from pipeline.train_autoencoder import AutoencoderTrainingPipline


class TestAutoencoderPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.model = AutoencoderCNN2(dataset_type="MNIST")

        cls.loss_function = nn.MSELoss()
        cls.optimizer = torch.optim.Adam(cls.model.parameters(), lr=1e-3)
        cls.dataloader_train, cls.dataloader_test = get_mnist_dataloaders()

        cls.pipeline = AutoencoderTrainingPipline(
            dataloader_train=cls.dataloader_train,
            dataloader_test=cls.dataloader_test,
            model=cls.model,
            loss_function=cls.loss_function,
            optimizer=cls.optimizer,
            epochs=100,
            save_path="resultsAECNN2_MNIST",
        )

    def test_epoch(self):
        self.pipeline.train()
        self.pipeline.evaluate_and_save()


if __name__ == "__main__":
    unittest.main()
