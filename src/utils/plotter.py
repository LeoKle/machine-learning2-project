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
