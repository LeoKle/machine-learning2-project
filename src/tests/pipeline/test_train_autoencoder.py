import unittest
import torch
import torch.nn as nn


from data.cifar10 import get_cifar10_dataloaders
from data.mnist import get_mnist_dataloaders
from models.autoencoder.autoencoder_CNN3 import AutoencoderCNN3
from pipeline.train_autoencoder import AutoencoderTrainingPipline


class TestAutoencoderPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_type = "CIFAR10"  


        cls.model = AutoencoderCNN3(dataset_type=cls.dataset_type, drop_prob=0.3)

        cls.loss_function = nn.MSELoss()
        cls.optimizer = torch.optim.Adam(cls.model.parameters(), lr=1e-3)
        cls.dataloader_train, cls.dataloader_test = get_cifar10_dataloaders()

        cls.pipeline = AutoencoderTrainingPipline(
            dataloader_train=cls.dataloader_train,
            dataloader_test=cls.dataloader_test,
            model=cls.model,
            loss_function=cls.loss_function,
            optimizer=cls.optimizer,
            epochs=100,
            save_path="resultsAECNN3_CIFAR10",
            dataset_type=cls.dataset_type, 

        )

    def test_epoch(self):
        self.pipeline.train()



if __name__ == "__main__":
    unittest.main()
