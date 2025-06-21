import unittest
import torch
from pathlib import Path
from utils.plotter import Plotter
from pipeline.train_classifier_linear import (
    train_classifier_linear,
    tune_hyperparameters,
)
from utils.data_setup import prepare_dataset
from utils.device import DEVICE


class TestLinearOnEncodedTraining(unittest.TestCase):

    def test_train_with_best_optuna_params(self):
        dataset_type = "CIFAR10"  # Change to "CIFAR10" as needed

        # Step 1: Run Optuna and get best params
        best_params = tune_hyperparameters(dataset_type)
        batch_size = best_params["batch_size"]
        lr = best_params["lr"]

        # Step 2: Prepare dataset and model inputs
        train_loader, test_loader, dummy_input, encoder_path = prepare_dataset(
            dataset_type, batch_size
        )

        # Step 3: Train model using the best params
        model, pipeline = train_classifier_linear(
            train_loader, test_loader, dummy_input, encoder_path, lr=lr
        )

        model_save_dir = Path(f"output/results_linear_with_optuna_{dataset_type}_paths")
        plot_save_dir = Path(f"output/results_linear_with_optuna_{dataset_type}_pngs")
        model_save_dir.mkdir(parents=True, exist_ok=True)
        plot_save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\nTraining with best Optuna parameters: batch_size = {batch_size}, lr = {lr:.10f}\n"
        )

        def plot_callback(metrics, epoch):
            Plotter.plot_loss_progression(
                metrics=metrics,
                epochs=list(range(1, epoch + 1)),
                output_file_name=plot_save_dir,
                dataset_type=dataset_type,
            )

        best_model_path, best_epoch = pipeline.train(
            max_epochs=100,
            save_every=10,
            model_save_dir=model_save_dir,
            metrics_save_dir=plot_save_dir,
            plot_callback=plot_callback,
        )

        if best_model_path is not None:
            print(f"Loading best model from epoch {best_epoch} for confusion matrix...")
            model.load_state_dict(torch.load(best_model_path))

        # Step 4: Visualizations
        Plotter.plot_metrics(
            pipeline.tracker.get_metrics(),
            plot_save_dir / "classifier_metrics.png",
            show=False,
        )
        Plotter.plot_accuracy(
            accuracy_values=pipeline.tracker.get_metrics()["accuracy"],
            output_file_name=plot_save_dir / "classifier_accuracy.png",
            dataset_type=dataset_type,
        )
        Plotter.plot_predictions(
            model,
            test_loader,
            dataset_type,
            DEVICE,
            plot_save_dir / f"{dataset_type}_predictions_1.png",
            show=False,
        )
        Plotter.plot_predictions(
            model,
            test_loader,
            dataset_type,
            DEVICE,
            plot_save_dir / f"{dataset_type}_predictions_2.png",
            show=False,
        )

        if dataset_type == "MNIST":
            Plotter.plot_confusion_matrix_mnist(
                model, test_loader, DEVICE, plot_save_dir / "mnist_confusion_matrix.png"
            )
        else:
            Plotter.plot_confusion_matrix_cifar10(
                model,
                test_loader,
                DEVICE,
                plot_save_dir / "cifar10_confusion_matrix.png",
            )

        # Step 5: Assertions
        self.assertTrue((plot_save_dir / "classifier_metrics.png").exists())
        self.assertTrue((plot_save_dir / "epoch_metrics.txt").exists())


if __name__ == "__main__":
    unittest.main()
