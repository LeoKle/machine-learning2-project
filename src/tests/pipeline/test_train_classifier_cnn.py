import unittest
from pathlib import Path
import torch
from utils.plotter import Plotter
from pipeline.train_classifier_cnn import train_classifier_cnn
from data.mnist import get_mnist_dataloaders
from data.cifar10 import get_cifar10_dataloaders
from utils.device import DEVICE

class TestLinearOnEncodedTraining(unittest.TestCase):
    def test_training_and_saving(self):
        dataset_type = "MNIST"  # Change to "MNIST" for MNIST dataset

        if dataset_type == "MNIST":
            train_loader, test_loader = get_mnist_dataloaders(batch_size=32)
            dummy_input = torch.randn(1, 1, 28, 28)
        elif dataset_type == "CIFAR10":
            train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)
            dummy_input = torch.randn(1, 3, 32, 32)
        else:
            raise ValueError("Unsupported dataset_type. Use 'MNIST' or 'CIFAR10'.")

        model, pipeline = train_classifier_cnn(train_loader, test_loader, dummy_input)

        model_save_dir = Path(f"output/results_cnn_raw_{dataset_type}_paths")
        plot_save_dir = Path(f"output/results_cnn_raw_{dataset_type}_pngs")
        model_save_dir.mkdir(parents=True, exist_ok=True)
        plot_save_dir.mkdir(parents=True, exist_ok=True)

        def plot_callback(metrics, epoch):
            Plotter.plot_loss_progression(metrics, list(range(1, epoch + 1)), plot_save_dir, dataset_type)

        best_model_path, best_epoch = pipeline.train(
            max_epochs=100, 
            save_every=10,
            model_save_dir=model_save_dir,
            metrics_save_dir=plot_save_dir,
            plot_callback=plot_callback
        )

        if best_model_path is not None:
            print(f"Loading best model from epoch {best_epoch} for confusion matrix...")
            model.load_state_dict(torch.load(best_model_path))

        Plotter.plot_metrics(pipeline.tracker.get_metrics(), plot_save_dir / "classifier_metrics.png")
        Plotter.plot_accuracy(
            accuracy_values=pipeline.tracker.get_metrics()["accuracy"],
            output_file_name=plot_save_dir / "classifier_accuracy.png",  dataset_type=dataset_type
        )

        Plotter.plot_predictions(model, test_loader, dataset_type, DEVICE, plot_save_dir / f"{dataset_type}_predictions_1.png", show=False)
        Plotter.plot_predictions(model, test_loader, dataset_type, DEVICE, plot_save_dir / f"{dataset_type}_predictions_2.png", show=False)

        if dataset_type == "MNIST":
            Plotter.plot_confusion_matrix_mnist(model, test_loader, DEVICE, plot_save_dir / "mnist_confusion_matrix.png")
        elif dataset_type == "CIFAR10":
            Plotter.plot_confusion_matrix_cifar10(model, test_loader, DEVICE, plot_save_dir / "cifar10_confusion_matrix.png")

        for e in [10]:
            self.assertTrue((model_save_dir / f"classifier_epoch_{e}.pth").exists())

        self.assertTrue((plot_save_dir / "classifier_metrics.png").exists())
        self.assertTrue((plot_save_dir / "epoch_metrics.txt").exists()) 

if __name__ == "__main__":
    unittest.main()
