import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import optuna
from models.classifier.classifier_cnn import CNNClassifier
from utils.device import DEVICE
from classes.tracker import Tracker
from classes.metrics import Metrics
from utils import data_setup
from pathlib import Path


class ClassifierTrainingPipeline:
    def __init__(
        self, dataloader_train, dataloader_test, model, loss_function, optimizer, tracker_output_dir="output/tracker_output",
    ):
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.model = model.to(DEVICE)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.tracker = Tracker(tracker_output_dir)
        self.current_epoch = 0
        self.metrics_file = None

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_batches = 0
        for batch, labels in self.dataloader_train:
            self.optimizer.zero_grad()
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            outputs = self.model(batch)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        self.tracker.track("train_loss", avg_loss, self.current_epoch)
        return avg_loss

    def _write_metrics_header(self):
        if self.metrics_file:
            with open(self.metrics_file, "w") as f:
                f.write(
                    "Epoch | Train Loss | Test Loss  | Accuracy | Error Rate | Precision | Recall | Specificity |  NPV   |  FPR   |  FNR   | F1 Score | Fbeta Score\n"
                )

    def _append_metrics_row(self, epoch, avg_train_loss, test_loss, metrics):
        if self.metrics_file:
            accuracy = metrics["accuracy"][-1]
            error_rate = metrics["error_rate"][-1]
            precision = metrics["precision"][-1]
            recall = metrics["recall"][-1]
            specificity = metrics["specificity"][-1]
            npv = metrics["NPV"][-1]
            fpr = metrics["FPR"][-1]
            fnr = metrics["FNR"][-1]
            f1_score = metrics["f1_score"][-1]
            fbeta_score = metrics["fbeta_score"][-1]
            with open(self.metrics_file, "a") as f:
                f.write(
                    f"{epoch:>5} |  {avg_train_loss:.4f}    |   {test_loss:.4f}   |  {accuracy:.2f}%  |   "
                    f"{error_rate:.4f}   |  {precision:.4f}   | {recall:.4f} |   {specificity:.4f}    | "
                    f"{npv:.4f} | {fpr:.4f} | {fnr:.4f} |  {f1_score:.4f}  | {fbeta_score:.4f}\n"
                )

    def train(
        self,
        max_epochs=100,
        save_every=10,
        model_save_dir=None,
        metrics_save_dir=None,
        plot_callback=None,
    ):
        test_losses = []
        extra_epochs = 0
        triggered_extra = False
        epoch = 1
        best_model_path = None
        best_epoch = None

        if model_save_dir is not None:
            model_save_dir = Path(model_save_dir)
            model_save_dir.mkdir(parents=True, exist_ok=True)

        if metrics_save_dir is not None:
            metrics_save_dir = Path(metrics_save_dir)
            metrics_save_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_file = metrics_save_dir / "epoch_metrics.txt"
            self._write_metrics_header()

        while True:
            self.current_epoch = epoch
            avg_train_loss = self.train_epoch()
            test_loss = self.evaluate()
            test_losses.append(test_loss)
            metrics = self.tracker.get_metrics()
            self._append_metrics_row(epoch, avg_train_loss, test_loss, metrics)
            accuracy = metrics["accuracy"][-1]

            print(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}, Acc = {accuracy:.2f}%"
            )

            if plot_callback:
                plot_callback(metrics, epoch)

            if model_save_dir is not None and epoch % save_every == 0:
                torch.save(
                    self.model.state_dict(),
                    model_save_dir / f"classifier_epoch_{epoch}.pth",
                )

            if not triggered_extra and epoch >= 4:
                if (
                    test_losses[-1]
                    > test_losses[-2]
                    > test_losses[-3]
                    > test_losses[-4]
                ):
                    best_epoch = epoch - 3
                    best_model_path = (
                        model_save_dir / f"best_classifier_epoch_{best_epoch}.pth"
                    )
                    print(
                        f"Early stopping: Test loss increased. Saving best checkpoint at epoch {best_epoch}."
                    )
                    torch.save(self.model.state_dict(), best_model_path)
                    if plot_callback:
                        plot_callback(self.tracker.get_metrics(), epoch)
                    extra_epochs = 5
                    triggered_extra = True

            if triggered_extra:
                extra_epochs -= 1
                if extra_epochs == 0:
                    print(f"Stopping after 5 extra epochs. Final epoch: {epoch}")
                    torch.save(
                        self.model.state_dict(),
                        model_save_dir / f"classifier_final_epoch_{epoch}.pth",
                    )
                    if plot_callback:
                        plot_callback(self.tracker.get_metrics(), epoch)
                    break

            epoch += 1
            if epoch > max_epochs:
                print("Max epochs reached.")
                break

        return best_model_path, best_epoch

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        tp = tn = fp = fn = 0

        with torch.no_grad():
            for batch, labels in self.dataloader_test:
                batch, labels = batch.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(batch)
                loss = self.loss_function(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    if t == p:
                        if t == 1:
                            tp += 1
                        else:
                            tn += 1
                    else:
                        if p == 1:
                            fp += 1
                        else:
                            fn += 1

        avg_loss = test_loss / len(self.dataloader_test)
        self.tracker.track("test_loss", avg_loss, self.current_epoch)
        self.tracker.track(
            "accuracy", Metrics.accuracy(tp, tn, fp, fn), self.current_epoch
        )
        self.tracker.track(
            "error_rate", Metrics.error_rate(tp, tn, fp, fn), self.current_epoch
        )
        self.tracker.track(
            "precision", Metrics.precision(tp, tn, fp, fn), self.current_epoch
        )
        self.tracker.track("recall", Metrics.recall(tp, tn, fp, fn), self.current_epoch)
        self.tracker.track(
            "specificity", Metrics.specificity(tp, tn, fp, fn), self.current_epoch
        )
        self.tracker.track(
            "NPV", Metrics.negative_predictive_value(tp, tn, fp, fn), self.current_epoch
        )
        self.tracker.track(
            "FPR", Metrics.false_positive_rate(tp, tn, fp, fn), self.current_epoch
        )
        self.tracker.track(
            "FNR", Metrics.false_negative_rate(tp, tn, fp, fn), self.current_epoch
        )
        self.tracker.track(
            "f1_score", Metrics.f1_score(tp, tn, fp, fn), self.current_epoch
        )
        self.tracker.track(
            "fbeta_score", Metrics.fbeta_score(tp, tn, fp, fn), self.current_epoch
        )
        return avg_loss


def train_classifier_cnn(
    train_loader, test_loader, dummy_input, lr=9.140946480582927e-05
):
    in_channels = dummy_input.shape[1]
    if in_channels == 1:
        dataset_type = "MNIST"
    elif in_channels == 3:
        dataset_type = "CIFAR10"
    else:
        raise ValueError("Unsupported input shape.")

    model = CNNClassifier(dataset=dataset_type).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    pipeline = ClassifierTrainingPipeline(
        train_loader, test_loader, model, loss_fn, optimizer, tracker_output_dir=f"output/tracker_output_cnn_{dataset_type}"
    )
    return model, pipeline


def tune_hyperparameters_cnn(dataset_type="CIFAR10"):
    def objective(trial):
        # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64]) # MNIST
        # lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True) # MNIST
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])  # CIFAR10
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)  # CIFAR10

        train_loader, test_loader, dummy_input = data_setup.prepare_dataset_for_cnn(
            dataset_type, batch_size
        )
        model, pipeline = train_classifier_cnn(
            train_loader, test_loader, dummy_input, lr=lr
        )

        pipeline.train(max_epochs=7)
        return pipeline.tracker.get_metrics()["accuracy"][-1]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    output_path = Path(f"output/optuna_best_params_cnn_{dataset_type}.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"Best Accuracy: {study.best_trial.value:.2f}%\n")
        for key, val in study.best_trial.params.items():
            f.write(f"{key}: {val}\n")

    print(f"Best hyperparameters saved to {output_path.resolve()}")
    return study.best_trial.params
