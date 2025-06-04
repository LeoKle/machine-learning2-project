import torch.nn as nn
import torch

class EncoderClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        
        # Determine input channels from classifier's dataset
        target_channels = 3 if classifier.dataset == "CIFAR10" else 1
        
        # Determine encoder's input channels from first Conv2d layer
        first_conv = None
        for module in encoder.modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break
        
        if first_conv is None:
            raise ValueError("Encoder must contain at least one Conv2d layer")
            
        encoder_channels = first_conv.in_channels
        self.needs_channel_adapter = (encoder_channels != target_channels)
        
        if self.needs_channel_adapter:
            self.channel_adapter = nn.Conv2d(
                in_channels=target_channels,
                out_channels=encoder_channels,
                kernel_size=1
            )
        
        self.dim_adapter = nn.Linear(self._get_encoder_output_size(),
                                     784 if classifier.dataset == "MNIST" else 3072)
        

        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def _get_encoder_output_size(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 1 if self.classifier.dataset == "MNIST" else 3, 
                                      28 if self.classifier.dataset == "MNIST" else 32, 
                                      28 if self.classifier.dataset == "MNIST" else 32)
            if self.needs_channel_adapter:
                dummy_input = self.channel_adapter(dummy_input)
            dummy_output = self.encoder(dummy_input)
            return dummy_output.view(-1).shape[0]
    
    def forward(self, x):
        if self.needs_channel_adapter:
            x = self.channel_adapter(x)
        
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        features = self.dim_adapter(features) 
        return self.classifier(features)