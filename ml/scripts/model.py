from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyCNN(nn.Module):
    """
    Input:  (B, 1, n_mels=128, T)
    Output: logits (B, 24) for 12 major + 12 minor
    """

    def __init__(self, num_classes: int = 24):
        super().__init__()

        # Conv blocks treat mel-spectrogram as an "image"
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Pool more in frequency than time (key depends heavily on pitch)
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        # Global average pooling -> fixed size regardless of time length
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,1,128,T)
        returns: logits (B,24)
        """
        x = self.conv1(x)  # -> (B,32,64,T/2)
        x = self.conv2(x)  # -> (B,64,32,T/4)
        x = self.conv3(x)  # -> (B,128,16,T/4)
        x = self.gap(x)    # -> (B,128,1,1)
        x = x.squeeze(-1).squeeze(-1)  # -> (B,128)
        logits = self.fc(x)            # -> (B,24)
        return logits


def predict_key_logits(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """
    Convenience helper:
    X: (1,1,128,T) or (B,1,128,T)
    returns logits (B,24)
    """
    model.eval()
    with torch.no_grad():
        return model(X)


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)
