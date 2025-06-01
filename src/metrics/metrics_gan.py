import torch
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

from utils.device import DEVICE


class GanMetrics:
    def __init__(self, device: str | torch.device = DEVICE):
        self.device = device

        self.inception_score = InceptionScore(normalize=True).to(self.device)
        self.is_image_count = 0

        self.fid = FrechetInceptionDistance(normalize=True).to(self.device)
        self.fid_image_count = 0

    def __preprocess_images(self, images: torch.Tensor):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)  # [B, 1, H, W] → [B, 3, H, W]

        if images.dtype != torch.float32:
            images = images.to(torch.float32)

        return images.to(self.device)

    def update_is(self, images: torch.Tensor):
        images = self.__preprocess_images(images)

        self.inception_score.update(images)
        self.is_image_count += images.shape[0]

    def update_fid(self, images: torch.Tensor, real: bool):
        self.fid.update(images.to(self.device), real=real)
        if real:
            self.num_images_fid_real += images.size(0)
        else:
            self.num_images_fid_fake += images.size(0)

    def compute(self):
        mean_std_tensor = self.inception_score.compute()
        mean, std = mean_std_tensor[0].item(), mean_std_tensor[1].item()

        fid_score = self.fid.compute().item()
        print(f"Inception Score on {self.is_image_count}: {mean:.4f} ± {std:.4f}")
        print(f"FID Score on {self.fid_image_count}: {fid_score:.4f}")

        return mean, std, fid_score

    def reset(self):
        self.inception_score.reset()
        self.is_image_count = 0

        self.fid.reset()
        self.fid_image_count = 0
