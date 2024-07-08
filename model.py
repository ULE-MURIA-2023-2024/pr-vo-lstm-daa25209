import torch
import torch.nn as nn
from torchvision.models import resnet50 as resnet
from torchvision.models import ResNet50_Weights as weights
from typing import Callable

class VisualOdometryModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = False,
        lstm_dropout: float = 0.2
    ) -> None:
        super(VisualOdometryModel, self).__init__()

        # Load pre-trained ResNet model
        self.cnn_model = resnet(weights=weights.DEFAULT)
        resnet_output = list(self.cnn_model.children())[-1].in_features
        self.cnn_model.fc = nn.Identity()

        # Freeze the weights of the ResNet layers
        for param in self.cnn_model.parameters():
            param.requires_grad = False

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=resnet_output,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )

        # Fully connected layers to generate translation (3) and rotation (4)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), 7)

    def resnet_transforms(self) -> Callable:
        return weights.DEFAULT.transforms(antialias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(batch_size * seq_length, channels, height, width)

        with torch.no_grad():
            features = self.cnn_model(x)

        features = features.view(batch_size, seq_length, -1)

        lstm_out, _ = self.lstm(features)

        outputs = self.fc(lstm_out[:, -1, :])
        return outputs

