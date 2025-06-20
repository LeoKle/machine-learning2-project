import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from models.autoencoder.autoencoder_CNN2 import AutoencoderCNN2
from pipeline.train_autoencoder import AutoencoderTrainingPipline
from data.cifar10 import get_cifar10_dataloaders
from data.mnist import get_mnist_dataloaders


class AutoencoderOptunaStudy:
    def __init__(self, dataset="CIFAR10"):
        self.dataset = dataset
        self.train_loader, self.test_loader = get_cifar10_dataloaders()
        self.study = optuna.create_study(
            direction="minimize",  # Minimierung des Rekonstruktionsfehlers
            study_name=f"autoencoder_{dataset.lower()}",
            storage="sqlite:///autoencoder_optuna.db",
            load_if_exists=True,
        )

    def objective(self, trial: optuna.Trial):
        # ---- Hyperparameter
        lr = trial.suggest_float("lr", 6e-4, 2e-3, log=True)
        drop_prob = trial.suggest_float("drop_prob", 0.0, 0.5)
        epochs = trial.suggest_int("epochs", 50, 100)

        # ---- Modell, Optimizer, Loss
        # model = Autoencoder(dataset_type=self.dataset)
        model = AutoencoderCNN2(dataset_type=self.dataset, drop_prob=drop_prob)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_function = nn.MSELoss()

        # ---- Pipeline
        save_path = f"optuna_results_cifar/trial_{trial.number}"
        pipeline = AutoencoderTrainingPipline(
            dataloader_train=self.train_loader,
            dataloader_test=self.test_loader,
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            save_path=save_path,
            drop_prob=drop_prob,
            epochs=epochs,
            dataset_type=self.dataset,
        )

        pipeline.train()

        # Rückgabe: Test-Loss am Ende (Ziel: minimieren)
        final_test_loss = pipeline.evaluate_epoch_loss()
        return final_test_loss

    def optimize(self, n_trials=20):
        self.study.optimize(self.objective, n_trials=n_trials)

    def best_params(self):
        return self.study.best_trial.params


if __name__ == "__main__":
    study = AutoencoderOptunaStudy()
    study.optimize(n_trials=5)
    print(study.best_params())
