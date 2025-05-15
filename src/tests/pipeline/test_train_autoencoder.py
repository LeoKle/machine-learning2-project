import unittest
import torch
import torch.nn as nn

from data.mnist import get_mnist_dataloaders
from models.autoencoder.autoencoder import Autoencoder
from pipeline.train_autoencoder import AutoencoderTrainingPipline



class TestAutoencoderPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        
        cls.model = Autoencoder(dataset_type="MNIST")

        cls.loss_function = nn.MSELoss()
        cls.optimizer = torch.optim.Adam(cls.model.parameters(), lr=0.001)

        cls.pipeline = AutoencoderTrainingPipline(
            dataloader_train=cls.dataloader_train,
            dataloader_test=cls.dataloader_test,
            model=cls.model,
            loss_function=cls.loss_function,
            optimizer=cls.optimizer,
            epochs=1,   
            save_path="resultsAE"
        )
    
    def test_epoch(self):
        self.pipeline.train()
        self.pipeline.evaluate_and_save()



if __name__ == "__main__":
    unittest.main()
