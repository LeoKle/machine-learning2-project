import torch
import torch.nn as nn

from pathlib import Path
from classes.tracker import Tracker
from utils.device import DEVICE
from utils.plotter import Plotter
from tqdm import tqdm
from typing import Union
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np




class AutoencoderTrainingPipline:
    def __init__(
        self,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_test: torch.utils.data.DataLoader,
        model: nn.Module,
        loss_function: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        save_path: Union[str, Path] = "results",
        drop_prob: float = 0.0,
        epochs: int = 10,
        dataset_type: str = "CIFAR10",
        
        ):
        
        self.epochs = epochs
        self.drop_prob = drop_prob
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.model = model.to(DEVICE)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.current_epoch = 0
        self.tracker = Tracker()
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.last_inputs = None
        self.last_reconstructions = None
        self.last_latents = None
        self.dataset_type = dataset_type

    def train(self):
        self.model.train()
        train_losses = []
        test_losses = []

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.model.train()

            batch_losses = []
            loop = tqdm(self.dataloader_train, desc=f"[{self.dataset_type}] Epoch {epoch+1}/{self.epochs}")

            for batch, _ in loop:
                loss = self.train_epoch(batch)
                batch_losses.append(loss)
                loop.set_postfix(loss=loss)

            avg_train_loss = torch.tensor(batch_losses).mean().item()
            train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.4f}")

            avg_test_loss = self.evaluate_epoch_loss()
            test_losses.append(avg_test_loss)
            print(f"Epoch {epoch+1}: Avg Test Loss = {avg_test_loss:.4f}")

        self._save_loss_plot(train_losses, test_losses)
        self._save_min_loss_distance(train_losses, test_losses)
        self.evaluate_and_save()
  

    def train_epoch(self, batch: torch.Tensor) -> float:
        self.optimizer.zero_grad()

        batch = batch.to(DEVICE)
        outputs = self.model(batch)

        loss = self.loss_function(outputs, batch)
        loss.backward()
        self.optimizer.step()

        self.tracker.track("train_loss", loss.item(), self.current_epoch)
        return loss.item()
   

    def evaluate_and_save(self):
        self.model.eval()

        all_inputs = []
        all_recons = []

        with torch.no_grad():
            for batch, _ in self.dataloader_test:
                batch = batch.to(DEVICE)
                latents = self.model.encoder(batch)
                reconstructions = self.model.decoder(latents)

                all_inputs.append(batch.cpu())
                all_recons.append(reconstructions.cpu())

        all_inputs = torch.cat(all_inputs, dim=0)
        all_recons = torch.cat(all_recons, dim=0)

        num_examples = min(32, all_inputs.size(0))
        inputs_subset = all_inputs[:num_examples]
        recons_subset = all_recons[:num_examples]

        self._save_image_grid(inputs_subset, self.save_path / f"{self.dataset_type}_originals_grid.png")
        self._save_image_grid(recons_subset, self.save_path / f"{self.dataset_type}_reconstructions_grid.png")

        self.last_latents = self.model.encoder(inputs_subset.to(DEVICE)).cpu()

        torch.save(self.model.encoder, self.save_path / f"{self.dataset_type}_encoder_full.pt")
        torch.save(self.model.encoder.state_dict(), self.save_path / f"{self.dataset_type}_encoder_weights.pth")

    def evaluate_epoch_loss(self) -> float:
        self.model.eval()
        batch_losses = []

        with torch.no_grad():
            for batch, _ in self.dataloader_test:
                batch = batch.to(DEVICE)
                outputs = self.model(batch)
                loss = self.loss_function(outputs, batch)
                batch_losses.append(loss.item())

        return torch.tensor(batch_losses).mean().item()




    def _save_image_grid(self, images: torch.Tensor, path: Path, nrow: int = 8):
        grid = make_grid(images, nrow=nrow, normalize=True, pad_value=1)
        img = TF.to_pil_image(grid)
        img.save(path)


    def _save_min_loss_distance(self, train_losses, test_losses):

        distances = np.abs(np.array(train_losses) - np.array(test_losses))
        min_dist = np.min(distances)
        min_epoch = np.argmin(distances) + 1  
        result_str = f"Min. Loss Distance: {min_dist:.6f} at Epoch {min_epoch}"
    
        print(result_str)

        with open(self.save_path / f"{self.dataset_type}_min_loss_distance.txt", "w") as f:
            f.write(result_str)


    def _save_loss_plot(self, train_losses, test_losses):
        from classes.tracker import DataDict 

        loss_dict: DataDict = {
            "Train Loss": train_losses,
            "Test Loss": test_losses,
        }
        Plotter.plot_metrics(loss_dict, self.save_path / f"{self.dataset_type}_loss.png")

    