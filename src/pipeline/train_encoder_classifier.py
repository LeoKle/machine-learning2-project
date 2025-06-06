import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from utils.device import DEVICE
from models.autoencoder.autoencoder_CNN2 import AutoencoderCNN2
from models.classifier.classifier import Classifier
from models.classifier.encoder_classifier import EncoderClassifier
from pipeline.train_classifier import ClassifierTrainingPipeline
from data.mnist import get_mnist_dataloaders
from data.cifar10 import get_cifar10_dataloaders
from utils.plotter import Plotter

def train_encoder_classifier_model(
    dataset_type="MNIST",
    epochs=20,
    save_epochs=(10, 20),
    model_save_dir=None,
    plot_save_dir=None
    ):
    # Set default save paths
    if model_save_dir is None:
        model_save_dir = f"results_classifier_{dataset_type}_paths"
    if plot_save_dir is None:
        plot_save_dir = f"results_classifier_{dataset_type}_pngs"

    # Data loading
    if dataset_type.upper() == "MNIST":
        train_loader, test_loader = get_mnist_dataloaders(batch_size=32)
        dummy_input = torch.randn(1, 1, 28, 28).to(DEVICE)
        encoder_path = "resultsAECNN2_MNIST/MNIST_encoder_weights.pth"
    elif dataset_type.upper() == "CIFAR10":
        train_loader, test_loader = get_cifar10_dataloaders(batch_size=32)
        dummy_input = torch.randn(1, 3, 32, 32).to(DEVICE)
        encoder_path = "resultsAECNN2_CIFAR10/CIFAR10_encoder_weights.pth"
    else:
        raise ValueError("Unsupported dataset_type. Use 'MNIST' or 'CIFAR10'.")

    autoencoder = AutoencoderCNN2(dataset_type=dataset_type)
    encoder = autoencoder.encoder
    encoder.load_state_dict(torch.load(encoder_path))

    with torch.no_grad():
        encoder_output = encoder(dummy_input)
        encoder_output_size = encoder_output.view(1, -1).size(1)

    classifier = Classifier(input_size=encoder_output_size)
    combined_model = EncoderClassifier(encoder, classifier).to(DEVICE)

    optimizer = optim.Adam(combined_model.parameters(), lr=0.0001)
    loss_fn = nn.NLLLoss()

    # Paths
    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    plot_save_dir = Path(plot_save_dir)
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    # Training pipeline
    pipeline = ClassifierTrainingPipeline(
        dataloader_train=train_loader,
        dataloader_test=test_loader,
        model=combined_model,
        loss_function=loss_fn,
        optimizer=optimizer
    )

    for epoch in range(1, epochs + 1):
        pipeline.current_epoch = epoch
        combined_model.train()

        total_loss = 0
        total_batches = 0
        for batch, labels in train_loader:
            loss = pipeline.train_epoch(batch, labels)
            total_loss += loss
            total_batches += 1

        avg_train_loss = total_loss / total_batches
        pipeline.tracker.track("train_loss", avg_train_loss, epoch)

        test_loss, test_accuracy = pipeline.evaluate()
        print(f"[{dataset_type}] Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {test_loss:.4f}, Test Acc = {test_accuracy:.2f}%")

        if epoch in save_epochs:
            torch.save(combined_model.state_dict(), model_save_dir / f"classifier_epoch_{epoch}.pth")

    # Save full metrics plot
    metrics = pipeline.tracker.get_metrics()
    Plotter.plot_metrics(metrics, plot_save_dir / "classifier_metrics.png")

    # Save clean loss plot
    selected_epochs = [i for i in save_epochs if i <= len(metrics["train_loss"])]
    # train_losses = [metrics["train_loss"][i-1] for i in selected_epochs]
    # test_losses = [metrics["test_loss"][i-1] for i in selected_epochs]
    # Plotter.plot_selected_losses(selected_epochs, train_losses, test_losses,
    #                              plot_save_dir / f"loss_epochs_{'_'.join(map(str, selected_epochs))}.png")

    # Save progressive loss plots
    Plotter.plot_loss_progression(metrics, selected_epochs, plot_save_dir)

    # Save accuracy plot
    if "test_accuracy" in metrics:
        Plotter.plot_accuracy(metrics["test_accuracy"], plot_save_dir / "accuracy_epochs_1_to_20.png")

    return combined_model, model_save_dir, plot_save_dir
