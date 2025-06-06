from pathlib import Path
from typing import DefaultDict, List
from matplotlib import pyplot as plt
import torch

from classes.tracker import DataDict


class Plotter:
    @staticmethod
    def show_image(
        data: torch.Tensor, output_file_name: Path = None, show: bool = False
    ):
        """displays / saves a MNIST / CIFAR10 tensor as image"""
        data = data.cpu()

        fig = plt.figure()

        if data.ndim == 3:
            if data.shape[0] == 1:
                image = data.squeeze(0).numpy()
                plt.imshow(image, cmap="gray")
            elif data.shape[0] == 3:
                image = data.permute(1, 2, 0).numpy()
                plt.imshow(image)
            else:
                raise ValueError(f"Unsupported number of channels: {data.shape[0]}")
        elif data.ndim == 2:
            plt.imshow(data.numpy(), cmap="gray")
        else:
            raise ValueError(f"Unsupported tensor shape: {data.shape}")

        plt.axis("off")

        if output_file_name:
            fig.savefig(output_file_name, bbox_inches="tight", pad_inches=0)

        if show:
            plt.show()

        plt.close(fig)

    @staticmethod
    def plot_metrics(metrics: DataDict, output_file_name: Path):
        fig, ax = plt.subplots()

        # remove epochs from data
        if "epoch" in metrics:
            metrics.pop("epoch")

        for metric, values in metrics.items():
            ax.plot(values, label=metric)

        ax.set_title("Metrics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(output_file_name)
        plt.show()

    @staticmethod
    def plot_loss_progression(metrics: DataDict, epochs: list[int], output_file_name: Path):
        train_losses = metrics.get("train_loss", [])
        test_losses = metrics.get("test_loss", [])

        for e in epochs:
            plt.figure()
            plt.plot(range(1, e + 1), train_losses[:e], 'r-', label='Training loss')
            plt.plot(range(1, e + 1), test_losses[:e], 'b-', label='Validation loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Training vs Validation Loss (1 to {e})")
            plt.legend()
            plt.grid(True)
            output_file = output_file_name / f"loss_epochs_up_to_{e}.png"
            plt.savefig(output_file)
            plt.close()

    @staticmethod
    def plot_accuracy(accuracy_values, output_file_name: Path):
        import matplotlib.pyplot as plt

        epochs = list(range(1, len(accuracy_values) + 1))

        plt.figure()
        plt.plot(epochs, accuracy_values, 'g-', label='Validation Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Validation Accuracy (1 to {})".format(len(accuracy_values)))
        plt.legend()
        plt.grid(True)
        plt.savefig(output_file_name)
        plt.close()