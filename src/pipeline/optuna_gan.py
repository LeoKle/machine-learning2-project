import os
from typing import Literal
import optuna
import torch
import torch.nn as nn

from data.cifar10 import get_cifar10_dataloaders
from data.mnist import get_mnist_dataloaders
from pipeline.train_gan import GanTrainingPipeline


class OptunaStudy:
    def __init__(
        self,
        generator_model: nn.Module,
        discriminator_model: nn.Module,
        dataset: Literal["MNIST", "CIFAR10"] = "MNIST",
    ):
        self.dataset = dataset
        if self.dataset == "MNIST":
            self.study_name = "gan_mnist"
            self.dataloader_train, self.dataloader_test = get_mnist_dataloaders()
        elif self.dataset == "CIFAR10":
            self.study_name = "gan_cifar10"
            self.dataloader_train, self.dataloader_test = get_cifar10_dataloaders()
        else:
            raise ValueError("Unsupported dataset: choose 'MNIST' or 'CIFAR10'")

        self.generator = generator_model(dataset=self.dataset)
        self.discriminator = discriminator_model(dataset=self.dataset)

        self.study = optuna.create_study(
            directions=["maximize", "minimize"],
            study_name=f"dcgan",
            storage="sqlite:///dcgan.db",
            load_if_exists=True,
        )

        self.output_dir = f"output/{self.study.study_name}"
        os.makedirs(self.output_dir, exist_ok=True)

    def objective(self, trial: optuna.Trial) -> float:
        self.trial_dir = self.output_dir + f"/{trial.number}"
        os.makedirs(self.trial_dir, exist_ok=True)

        n_epochs = trial.suggest_int("n_epochs", 350, 350)

        lr_gen = trial.suggest_float("lr_gen", 2e-4, 5e-4, log=True)
        beta1_gen = trial.suggest_float("beta1_gen", 0.5, 0.5)
        beta2_gen = trial.suggest_float("beta2_gen", 0.999, 0.999)

        lr_disc = trial.suggest_float("lr_dis", 1e-4, 2e-4, log=True)
        beta1_disc = trial.suggest_float("beta1_dis", 0.5, 0.5)
        beta2_disc = trial.suggest_float("beta2_dis", 0.999, 0.999)

        loss_function_generator = nn.BCELoss()
        loss_function_discriminator = nn.BCELoss()
        loss_choice = trial.suggest_categorical("loss_generator", ["mse"])
        if loss_choice == "bce":
            loss_function_generator = nn.BCELoss()
        elif loss_choice == "mse":
            loss_function_generator = nn.MSELoss()

        loss_disc_choice = trial.suggest_categorical(
            "loss_discriminator", ["bce", "mse"]
        )
        if loss_disc_choice == "bce":
            loss_function_discriminator = nn.BCELoss()
        elif loss_disc_choice == "mse":
            loss_function_discriminator = nn.MSELoss()

        optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_disc, betas=(beta1_disc, beta2_disc)
        )
        optimizer_generator = torch.optim.Adam(
            self.generator.parameters(), lr=lr_gen, betas=(beta1_gen, beta2_gen)
        )

        pipeline = GanTrainingPipeline(
            dataloader_train=self.dataloader_train,
            dataloader_test=self.dataloader_test,
            generator=self.generator,
            discriminator=self.discriminator,
            loss_function_discriminator=loss_function_discriminator,
            loss_function_generator=loss_function_generator,
            optimizer_discriminator=optimizer_discriminator,
            optimizer_generator=optimizer_generator,
            output_dir=self.trial_dir,
        )

        print(f"Starting training with params: {trial.params}")

        pipeline.train(n_epochs)

        torch.save(
            pipeline.generator.state_dict(),
            self.trial_dir + "/generator.pth",
        )

        # return pipeline.metrics.get_last_inception_score()[0]

        return (
            pipeline.metrics.get_last_inception_score()[0],
            pipeline.metrics.get_last_fid_score(),
        )

    def optimize(self, n_trials: int):
        self.study.optimize(self.objective, n_trials=n_trials)

    def best_trials(self):
        return self.study.best_trials
