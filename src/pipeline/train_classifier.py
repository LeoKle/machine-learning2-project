import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from utils.device import DEVICE
from utils.plotter import Plotter
from classes.tracker import Tracker

class ClassifierTrainingPipeline:
    def __init__(
        self,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_test: torch.utils.data.DataLoader,
        model: nn.Module,
        loss_function: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
    ):
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.model = model.to(DEVICE)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.tracker = Tracker()
        self.current_epoch = 0

    def train_epoch(self, batch: torch.Tensor, labels: torch.Tensor):
        self.model.zero_grad()
        batch, labels = batch.to(DEVICE), labels.to(DEVICE)

        outputs = self.model(batch)
        loss = self.loss_function(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.tracker.track("train_loss", loss.item(), self.current_epoch)
        
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100. * correct / labels.size(0)
        self.tracker.track("train_accuracy", accuracy, self.current_epoch)

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch, labels in self.dataloader_test:
                batch, labels = batch.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(batch)
                loss = self.loss_function(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = test_loss / len(self.dataloader_test)
        accuracy = 100. * correct / total
        
        self.tracker.track("test_loss", avg_loss, self.current_epoch)
        self.tracker.track("test_accuracy", accuracy, self.current_epoch)
        
        return avg_loss, accuracy

    def train(self, epoch: int, save_path: Optional[Path] = None):
        print(f"Starting training on {DEVICE}...")
        
        for epoch in range(1, epoch + 1):
            self.current_epoch = epoch
            self.model.train()
            
            print(f"\nEpoch {epoch}/{epoch}")
            for batch, labels in self.dataloader_train:
                self.train_epoch(batch, labels)
            
            test_loss, test_accuracy = self.evaluate()
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")

            if save_path:
                torch.save(self.model.state_dict(), save_path)
        
        metrics = self.tracker.get_metrics()
        plot_path = save_path.parent / "output/classifier_metrics.png" if save_path else Path("output/classifier_metrics.png")
        Plotter.plot_metrics(metrics, plot_path)
        
        print("Training completed!")