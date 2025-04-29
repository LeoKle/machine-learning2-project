from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from classes.tracker import DataDict
from utils.plotter import Plotter


class TestPlotter(unittest.TestCase):
    def setUp(self):
        self.mnist_tensor = torch.rand(1, 28, 28)  # simulate a MNIST data
        self.cifar_tensor = torch.rand(3, 32, 32)  # simulate a CIFAR-10 data

    @patch("matplotlib.pyplot.show")
    def test_plot_cifar_tensor(self, mock_show):
        Plotter.show_image(self.cifar_tensor, show=True)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_mnist_tensor(self, mock_show):
        Plotter.show_image(self.mnist_tensor, show=True)
        mock_show.assert_called_once()

    def test_plot_saves_to_file(self):
        # Create a temporary file path
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_plot.png"

            Plotter.show_image(self.mnist_tensor, output_file_name=output_path)

            self.assertTrue(output_path.exists(), "Plot file was not created.")

    def test_plot_metrics(self):
        metric_dict: DataDict = {"epochs": [0, 1, 2], "accuracy": [0.5, 0.6, 0.7]}
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_plot.png"

            Plotter.plot_metrics(metric_dict, output_path)

            self.assertTrue(output_path.exists)


if __name__ == "__main__":
    unittest.main()
