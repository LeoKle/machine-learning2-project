from pathlib import Path
from typing import DefaultDict, List
from matplotlib import pyplot as plt
import torch
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import random

from classes.tracker import DataDict

class Plotter:
    @staticmethod
    def show_image(data: torch.Tensor, output_file_name: Path = None, show: bool = False):
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
    def plot_metrics(metrics: DataDict, output_file_name: Path, show=False):
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
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_loss_progression(metrics: DataDict, epochs: list[int], output_file_name: Path, dataset_type: str):
        train_losses = metrics.get("train_loss", [])
        test_losses = metrics.get("test_loss", [])

        for e in epochs:
            plt.figure()
            plt.plot(range(1, e + 1), train_losses[:e], 'r-', label='Training loss')
            plt.plot(range(1, e + 1), test_losses[:e], 'b-', label='Validation loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dataset_type} Training vs Validation Loss")
            plt.legend()
            plt.grid(True)
            output_file = output_file_name / f"loss_epochs_up_to_{e}.png"
            plt.savefig(output_file)
            plt.close()

    @staticmethod
    def plot_accuracy(accuracy_values, output_file_name: Path, dataset_type: str):

        epochs = list(range(1, len(accuracy_values) + 1))

        plt.figure()
        plt.plot(epochs, accuracy_values, 'g-', label='Validation Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{dataset_type} Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_file_name)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(model, dataloader, device, class_names, title, output_file):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        #cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        #sns.heatmap(cm_normalized, annot=True, fmt=".4f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    @staticmethod
    def plot_confusion_matrix_mnist(model, dataloader, device, output_file: Path):
        Plotter.plot_confusion_matrix(
            model, dataloader, device,
            class_names=[str(i) for i in range(10)],
            title="MNIST Confusion Matrix",
            output_file=output_file)

    @staticmethod
    def plot_confusion_matrix_cifar10(model, dataloader, device, output_file: Path):
        Plotter.plot_confusion_matrix(
            model, dataloader, device,
            class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
            title="CIFAR10 Confusion Matrix", 
            output_file=output_file)
        
    @staticmethod
    def plot_predictions(model, dataloader, dataset_type, device, output_file_name=None, show=False):

        if dataset_type.upper() == "MNIST":
            class_names = [str(i) for i in range(10)]
        else:
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Collect all images and labels for random selection
        images_list, labels_list = [], []
        for images, labels in dataloader:
            images_list.append(images)
            labels_list.append(labels)
        images = torch.cat(images_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        idxs = random.sample(range(images.shape[0]), 9)
        selected_images = images[idxs]
        selected_labels = labels[idxs]

        # Get predictions for selected images
        model.eval()
        with torch.no_grad():
            imgs_on_device = selected_images.to(device)
            outputs = model(imgs_on_device)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu()

        def unnormalize(img):
            if dataset_type.upper() == "MNIST":
                return img * 0.5 + 0.5
            else:
                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
                return img * std + mean

        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            img = unnormalize(selected_images[i])
            if img.shape[0] == 1:
                ax.imshow(img.squeeze(0).numpy(), cmap="gray")
            else:
                ax.imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
            true_lbl = class_names[selected_labels[i].item()]
            pred_lbl = class_names[preds[i].item()]
            correct = (selected_labels[i].item() == preds[i].item())
            color = "green" if correct else "red"
            ax.set_title(f"Label: {true_lbl}\nPred: {pred_lbl}", color=color, fontsize=11)
            ax.axis("off")

        # fig.suptitle(f"Random {dataset_type} Images", fontsize=14)
        plt.tight_layout()
        if output_file_name:
            plt.savefig(output_file_name, dpi=200)
        if show:
            plt.show()
        plt.close(fig)
        
        