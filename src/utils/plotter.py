import math
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

        if data.ndim == 3:
            # Make it a batch of 1 image
            data = data.unsqueeze(0)  # [1, C, H, W]

        num_images = data.shape[0]
        fig = plt.figure(figsize=(num_images * 2, 2))

        for i in range(num_images):
            img = data[i]
            ax = fig.add_subplot(1, num_images, i + 1)
            ax.axis("off")

            if img.shape[0] == 1:
                image = img.squeeze(0).numpy()
                ax.imshow(image, cmap="gray")
            elif img.shape[0] == 3:
                image = img.permute(1, 2, 0).numpy()
                ax.imshow(image)
            else:
                raise ValueError(f"Unsupported number of channels: {img.shape[0]}")

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
