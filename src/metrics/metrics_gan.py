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
        self.fid_image_count_real = 0
        self.fid_image_count_fake = 0

        self.__last_inception_score = 0
        self.__last_inception_score_std = 0
        self.__last_fid_score = 0

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
        images = self.__preprocess_images(images)

        self.fid.update(images.to(self.device), real=real)
        if real:
            self.fid_image_count_real += images.size(0)
        else:
            self.fid_image_count_fake += images.size(0)

    def compute(self):
        mean_std_tensor = self.inception_score.compute()
        mean, std = mean_std_tensor[0].item(), mean_std_tensor[1].item()
        self.__last_inception_score = mean
        self.__last_inception_score_std = std

        fid_score = self.fid.compute().item()
        self.__last_fid_score = fid_score
        print(f"Inception Score on {self.is_image_count}: {mean:.4f} ± {std:.4f}")
        print(
            f"FID Score on {self.fid_image_count_real + self.fid_image_count_fake}: {fid_score:.4f}"
        )

        return mean, std, fid_score

    def reset(self):
        self.inception_score.reset()
        self.is_image_count = 0

        self.fid.reset()
        self.fid_image_count_real = 0
        self.fid_image_count_fake = 0

    def get_last_inception_score(self):
        return self.__last_inception_score, self.__last_inception_score_std

    def get_last_fid_score(self):
        return self.__last_fid_score
