import torch
import torch.nn as nn

from pathlib import Path
from classes.tracker import Tracker
from utils.device import DEVICE
from utils.plotter import Plotter
from tqdm import tqdm
from typing import Union




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
        dataset_type: str = "MNIST",
        epochs: int = 10,


        
        ):
        
        self.epochs = epochs
        self.dataset_type = dataset_type
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



        self.current_epoch = 0

        self.last_inputs = None
        self.last_reconstructions = None
        self.last_latents = None




    def train(self):
        self.model.train()
        losses = []

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            loop = tqdm(self.dataloader_train, desc=f"[{self.dataset_type}] Epoch {epoch+1}/{self.epochs}")

            for batch, _ in loop:
                loss = self.train_epoch(batch)
                epoch_loss += loss
                loop.set_postfix(loss=loss)

            avg_loss = epoch_loss / len(self.dataloader_train)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        self._save_loss_plot(losses)



    def train_epoch(self, batch: torch.Tensor) -> float:
        self.model.train()
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

        with torch.no_grad():
            for batch, _ in self.dataloader_test:
                batch = batch.to(DEVICE)
                latents = self.model.encoder(batch)
                reconstructions = self.model.decoder(latents)

                self.last_inputs = batch.cpu()
                self.last_reconstructions = reconstructions.cpu()
                self.last_latents = latents.cpu()
                break

        # Speichere Original & Rekonstruktion
        Plotter.show_image(self.last_inputs[0], self.save_path / f"{self.dataset_type}_original.png")
        Plotter.show_image(self.last_reconstructions[0], self.save_path / f"{self.dataset_type}_reconstruction.png")

        # Speichere latente Repräsentationen
        torch.save(self.last_latents, self.save_path / f"{self.dataset_type}_latent.pt")
        print(f"[INFO] Rekonstruktion und Latents gespeichert in: {self.save_path}")

    def _save_loss_plot(self, losses):
        from classes.tracker import DataDict  # Lokaler Import, falls DataDict verwendet wird

        loss_dict: DataDict = {
            "epochs": list(range(1, len(losses) + 1)),
            "loss": losses
        }
        Plotter.plot_metrics(loss_dict, self.save_path / f"{self.dataset_type}_loss.png")