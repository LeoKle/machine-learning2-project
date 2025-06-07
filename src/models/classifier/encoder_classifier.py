import torch.nn as nn
import torch
from models.classifier.classifier_linear import Classifier

class EncoderClassifier(nn.Module):
    def __init__(self, encoder, classifier=None, num_classes=10, fine_tune_encoder=True):
        super().__init__()
        self.encoder = encoder

        if not fine_tune_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get encoder output size
        self.encoder_output_size = self.get_encoder_output_size()

        if classifier is None:
            self.classifier = Classifier(input_size=self.encoder_output_size, num_classes=num_classes)
        else:
            self.classifier = classifier

    def get_encoder_output_size(self):
        with torch.no_grad():
            in_channels = next(self.encoder.parameters()).shape[1]
            dummy_input = torch.randn(1, in_channels, 28, 28)  # default for MNIST
            output = self.encoder(dummy_input)
            return output.view(1, -1).size(1)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)