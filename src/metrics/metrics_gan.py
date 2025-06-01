import torch
from torchmetrics.image.inception import InceptionScore

from utils.device import DEVICE


class GanMetrics:
    def __init__(self, device: str | torch.device = DEVICE):
        self.device = device
        self.inception_score = InceptionScore(normalize=True).to(self.device)

    def update(self, images: torch.Tensor):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)  # [B, 1, H, W] → [B, 3, H, W]

        if images.dtype != torch.float32:
            images = images.to(torch.float32)

        self.inception_score.update(images.to(self.device))

    def compute(self):
        mean_std_tensor = self.inception_score.compute()
        mean, std = mean_std_tensor[0].item(), mean_std_tensor[1].item()
        # print(f"Inception Score: {mean:.4f} ± {std:.4f}")
        return mean, std

    def reset(self):
        self.inception_score.reset()
