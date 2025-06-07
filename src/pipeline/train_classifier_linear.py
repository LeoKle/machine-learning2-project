import torch
import torch.nn as nn
import torch.optim as optim
from utils.device import DEVICE
from models.autoencoder.autoencoder_CNN2 import AutoencoderCNN2
from models.classifier.classifier_linear import Classifier
from models.classifier.encoder_classifier import EncoderClassifier
from classes.tracker import Tracker

class ClassifierTrainingPipeline:
    def __init__(self, dataloader_train, dataloader_test, model, loss_function, optimizer):
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.model = model.to(DEVICE)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.tracker = Tracker()
        self.current_epoch = 0

    def train_epoch(self, batch, labels):
        self.model.zero_grad()
        batch, labels = batch.to(DEVICE), labels.to(DEVICE)
        outputs = self.model(batch)
        loss = self.loss_function(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

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

def train_classifier_linear(train_loader, test_loader, dummy_input, encoder_path):
    autoencoder = AutoencoderCNN2()
    encoder = autoencoder.encoder
    encoder.load_state_dict(torch.load(encoder_path))

    with torch.no_grad():
        encoder_output = encoder(dummy_input.to(DEVICE))
        encoder_output_size = encoder_output.view(1, -1).size(1)

    classifier = Classifier(input_size=encoder_output_size)
    combined_model = EncoderClassifier(encoder, classifier).to(DEVICE)

    optimizer = optim.Adam(combined_model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()

    pipeline = ClassifierTrainingPipeline(
        dataloader_train=train_loader,
        dataloader_test=test_loader,
        model=combined_model,
        loss_function=loss_fn,
        optimizer=optimizer
    )

    return combined_model, pipeline
