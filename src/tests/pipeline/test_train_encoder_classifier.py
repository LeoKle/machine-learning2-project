import unittest
from pipeline.train_encoder_classifier import train_encoder_classifier_model
from pathlib import Path
import torch

class TestEncoderClassifierTraining(unittest.TestCase):
    def test_training_and_saving(self):
        model, model_path_dir, plot_path_dir = train_encoder_classifier_model()

        for epoch in [10, 20]:
            model_path = model_path_dir / f"classifier_epoch_{epoch}.pth"
            self.assertTrue(model_path.exists(), f"Missing model file for epoch {epoch}")

        plot_path = plot_path_dir / "classifier_metrics.png"
        self.assertTrue(plot_path.exists(), "Missing classifier metrics plot")

        print(f"Model paths saved to: {model_path_dir}")
        print(f"Plots saved to: {plot_path_dir}")

if __name__ == "__main__":
    unittest.main()