import unittest
from pathlib import Path
import torch

from utils.data_setup import prepare_dataset
from utils.plotter import Plotter
from utils.device import DEVICE
from models.autoencoder.autoencoder_CNN2 import AutoencoderCNN2
from models.classifier.encoder_classifier import EncoderClassifier
from models.classifier.classifier_resnet import ClassifierResNet


class TestPlotFromModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_type = "MNIST"  # or "MNIST"
        cls.model_path = Path("output/best_classifier_epoch_43.pth")  # Path

        # Data setup
        cls.train_loader, cls.test_loader, cls.dummy_input, _ = prepare_dataset(cls.dataset_type, batch_size=64)

        # Build encoder + classifier
        cls.encoder = AutoencoderCNN2(dataset_type=cls.dataset_type).encoder
        with torch.no_grad():
            encoder_output = cls.encoder(cls.dummy_input.to(DEVICE))
            cls.encoder_output_size = encoder_output.view(1, -1).size(1)

        cls.classifier = ClassifierResNet(input_size=cls.encoder_output_size)
        cls.model = EncoderClassifier(
            encoder=cls.encoder,
            classifier=cls.classifier,
            fine_tune_encoder=False,
            img_channels=cls.dummy_input.shape[1],
            img_size=cls.dummy_input.shape[2]
        ).to(DEVICE)

        cls.model.load_state_dict(torch.load(cls.model_path, map_location=DEVICE))
        cls.model.eval()

        cls.output_dir = Path("output/plot_predictions_output")
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def test_plot_random_predictions(self):
        for i in range(1, 11):
            output_file = self.output_dir / f"{self.dataset_type}_predictions_{i}.png"
            Plotter.plot_predictions(self.model, self.test_loader, self.dataset_type, DEVICE, output_file, show=False)
            self.assertTrue(output_file.exists(), f"Plot not created: {output_file}")


if __name__ == "__main__":
    unittest.main()
