import torch
import torch.nn as nn
import torch.optim as optim
from utils.device import DEVICE
from models.autoencoder.autoencoder_CNN2 import AutoencoderCNN2
from models.classifier.classifier_linear import Classifier
from models.classifier.encoder_classifier import EncoderClassifier
from classes.tracker import Tracker
from classes.metrics import Metrics

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
        #correct = 0
        #total = 0
        tp = tn = fp = fn = 0

        with torch.no_grad():
            for batch, labels in self.dataloader_test:
                batch, labels = batch.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(batch)
                loss = self.loss_function(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                #total += labels.size(0)
                #correct += predicted.eq(labels).sum().item()

                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    if t == p:
                        if t == 1:
                            tp += 1
                        else:
                            tn += 1
                    else:
                        if p == 1:
                            fp += 1
                        else:
                            fn += 1

        avg_loss = test_loss / len(self.dataloader_test)
        #accuracy = 100. * correct / total

        self.tracker.track("test_loss", avg_loss, self.current_epoch)
        #self.tracker.track("test_accuracy", accuracy, self.current_epoch)
        self.tracker.track("accuracy", Metrics.accuracy(tp, tn, fp, fn), self.current_epoch)
        self.tracker.track("error_rate", Metrics.error_rate(tp, tn, fp, fn), self.current_epoch)
        self.tracker.track("precision", Metrics.precision(tp, tn, fp, fn), self.current_epoch)
        self.tracker.track("recall", Metrics.recall(tp, tn, fp, fn), self.current_epoch)
        self.tracker.track("specificity", Metrics.specificity(tp, tn, fp, fn), self.current_epoch)
        self.tracker.track("NPV", Metrics.negative_predictive_value(tp, tn, fp, fn), self.current_epoch)
        self.tracker.track("FPR", Metrics.false_positive_rate(tp, tn, fp, fn), self.current_epoch)
        self.tracker.track("FNR", Metrics.false_negative_rate(tp, tn, fp, fn), self.current_epoch)
        self.tracker.track("f1_score", Metrics.f1_score(tp, tn, fp, fn), self.current_epoch)
        self.tracker.track("fbeta_score", Metrics.fbeta_score(tp, tn, fp, fn), self.current_epoch)

        return avg_loss

def train_classifier_linear(train_loader, test_loader, dummy_input, encoder_path):

    in_channels = dummy_input.shape[1]
    if in_channels == 1:
        dataset_type = "MNIST"
    elif in_channels == 3:
        dataset_type = "CIFAR10"
    else:
        raise ValueError("Unsupported input shape. Expecting channels = 1 or 3.")
    
    autoencoder = AutoencoderCNN2(dataset_type=dataset_type)
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
