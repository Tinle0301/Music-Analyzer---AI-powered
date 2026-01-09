import torch
from ml.scripts.model import KeyCNN

# Example batch: 2 samples, 1 channel, 128 mel bins, 517 frames
X = torch.randn(2, 1, 128, 517)

m = KeyCNN()
logits = m(X)

print("logits shape:", logits.shape)  # should be (2, 24)
