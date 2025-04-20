from pathlib import Path
from matplotlib import pyplot as plt
import torch


class Plotter:
    @staticmethod
    def plot(data: torch.Tensor, output_file_name: Path = None, show: bool = False):
        data = data.cpu()

        if data.ndim == 3:
            # (C, H, W)
            if data.shape[0] == 1:
                # MNIST: (1, 28, 28)
                image = data.squeeze(0).numpy()
                plt.imshow(image, cmap="gray")
            elif data.shape[0] == 3:
                # CIFAR-10: (3, 32, 32)
                image = data.permute(1, 2, 0).numpy()  # (H, W, C)
                plt.imshow(image)
            else:
                raise ValueError(f"Unsupported number of channels: {data.shape[0]}")
        elif data.ndim == 2:
            # (H, W) already
            plt.imshow(data.numpy(), cmap="gray")
        else:
            raise ValueError(f"Unsupported tensor shape: {data.shape}")

        plt.axis("off")

        if output_file_name:
            plt.savefig(output_file_name)

        if show:
            plt.show()
