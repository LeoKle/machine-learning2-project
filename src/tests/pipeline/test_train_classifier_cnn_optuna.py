import unittest
from pathlib import Path
import torch
from utils.plotter import Plotter
from pipeline.train_classifier_cnn import train_classifier_cnn, tune_hyperparameters_cnn
from utils.device import DEVICE
from utils.data_setup import prepare_dataset_for_cnn


class TestCNNTrainingWithOptuna(unittest.TestCase):
    def test_training_and_saving(self):
        dataset_type = "MNIST"  # or "CIFAR10"

        # Step 1: Tune hyperparameters
        best_params = tune_hyperparameters_cnn(dataset_type)
        batch_size = best_params["batch_size"]
        lr = best_params["lr"]

        # Step 2: Prepare data and model
        train_loader, test_loader, dummy_input = prepare_dataset_for_cnn(
            dataset_type, batch_size
        )
        model, pipeline = train_classifier_cnn(
            train_loader, test_loader, dummy_input, lr=lr
        )

        # Step 3: Define save paths
        model_save_dir = Path(f"output/results_cnn_optuna_{dataset_type}_paths")
        plot_save_dir = Path(f"output/results_cnn_optuna_{dataset_type}_pngs")
        model_save_dir.mkdir(parents=True, exist_ok=True)
        plot_save_dir.mkdir(parents=True, exist_ok=True)

        def plot_callback(metrics, epoch):
            Plotter.plot_loss_progression(
                metrics, list(range(1, epoch + 1)), plot_save_dir, dataset_type
            )

        # Step 4: Train
        best_model_path, best_epoch = pipeline.train(
            max_epochs=2,
            save_every=10,
            model_save_dir=model_save_dir,
            metrics_save_dir=plot_save_dir,
            plot_callback=plot_callback,
        )

        # Step 5: Evaluation plots
        if best_model_path is not None:
            print(f"Loading best model from epoch {best_epoch} for confusion matrix...")
            model.load_state_dict(torch.load(best_model_path))

        Plotter.plot_metrics(
            pipeline.tracker.get_metrics(), plot_save_dir / "metrics.png"
        )
        Plotter.plot_accuracy(
            pipeline.tracker.get_metrics()["accuracy"],
            plot_save_dir / "accuracy.png",
            dataset_type,
        )
        Plotter.plot_predictions(
            model,
            test_loader,
            dataset_type,
            DEVICE,
            plot_save_dir / "predictions.png",
            show=False,
        )

        if dataset_type == "MNIST":
            Plotter.plot_confusion_matrix_mnist(
                model, test_loader, DEVICE, plot_save_dir / "confusion_matrix.png"
            )
        else:
            Plotter.plot_confusion_matrix_cifar10(
                model, test_loader, DEVICE, plot_save_dir / "confusion_matrix.png"
            )

        # Assertions
        self.assertTrue((plot_save_dir / "metrics.png").exists())
        self.assertTrue((plot_save_dir / "accuracy.png").exists())
        self.assertTrue((plot_save_dir / "confusion_matrix.png").exists())


if __name__ == "__main__":
    unittest.main()
