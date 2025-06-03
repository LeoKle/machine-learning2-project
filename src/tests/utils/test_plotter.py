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
        self.batch_cifar = torch.rand(4, 3, 32, 32)
        self.batch_mnist = torch.rand(9, 1, 28, 28)

    @patch("utils.plotter.plt.show")
    def test_plot_cifar_tensor(self, mock_show):
        Plotter.show_image(self.cifar_tensor, show=True)
        mock_show.assert_called_once()

    @patch("utils.plotter.plt.show")
    def test_plot_mnist_tensor(self, mock_show):
        Plotter.show_image(self.mnist_tensor, show=True)
        mock_show.assert_called_once()

    @patch("utils.plotter.plt.show")
    def test_plot_batch_cifar_tensor(self, mock_show):
        Plotter.show_image(self.batch_cifar, show=True)
        mock_show.assert_called_once()

    @patch("utils.plotter.plt.show")
    def test_plot_batch_mnist_tensor(self, mock_show):
        Plotter.show_image(self.batch_mnist, show=True)
        mock_show.assert_called_once()

    def test_plot_saves_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_plot.png"
            Plotter.show_image(self.mnist_tensor, output_file_name=output_path)
            self.assertTrue(output_path.exists(), "Plot file was not created.")

    def test_plot_batch_saves_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "batch_plot.png"
            Plotter.show_image(self.batch_mnist, output_file_name=output_path)
            self.assertTrue(output_path.exists(), "Batch plot file was not created.")

    @patch("utils.plotter.plt.show")
    def test_plot_metrics(self, mock_show):
        metric_dict: DataDict = {"epochs": [0, 1, 2], "accuracy": [0.5, 0.6, 0.7]}
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_plot.png"
            Plotter.plot_metrics(metric_dict, output_path)
            self.assertTrue(output_path.exists(), "Metrics plot file was not created.")

    def test_invalid_tensor_shape(self):
        invalid_tensor = torch.rand(5, 5, 5, 5, 5)  # 5D tensor
        with self.assertRaises(ValueError):
            Plotter.show_image(invalid_tensor)

    def test_invalid_channel_count(self):
        invalid_channels = torch.rand(2, 28, 28)  # neither 1 nor 3
        with self.assertRaises(ValueError):
            Plotter.show_image(invalid_channels)


if __name__ == "__main__":
    unittest.main()
