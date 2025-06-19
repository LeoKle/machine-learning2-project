import torch.nn as nn
import torch
from models.classifier.classifier_linear import Classifier, ClassifierDeep
from models.classifier.classifier_resnet import ClassifierResNet, ClassifierLinear


class EncoderClassifier(nn.Module):
    def __init__(
        self,
        encoder,
        classifier=None,
        num_classes=10,
        img_channels=1,
        img_size=28,
        fine_tune_encoder=True,
    ):
        super().__init__()
        self.encoder = encoder

        if not fine_tune_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.img_channels = img_channels
        self.img_size = img_size
        self.encoder_output_size = self.get_encoder_output_size()

        if classifier is None:
            self.classifier = ClassifierResNet(
                input_size=self.encoder_output_size, num_classes=num_classes
            )
        else:
            self.classifier = classifier

    def get_encoder_output_size(self):
        with torch.no_grad():
            dummy_input = torch.randn(
                1, self.img_channels, self.img_size, self.img_size
            )
            output = self.encoder(dummy_input)
            return output.view(1, -1).size(1)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)
