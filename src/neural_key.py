from __future__ import annotations

import torch
import numpy as np
import librosa

from ml.scripts.model import KeyCNN
from ml.scripts.label_map import class_to_key


def audio_to_mel_tensor(
    path: str,
    sr: int = 22050,
    clip_seconds: float = 12.0,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> torch.Tensor:
    """Match the preprocessing used in KeyDataset."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = librosa.util.normalize(y)

    clip_samples = int(sr * clip_seconds)
    if len(y) >= clip_samples:
        y = y[:clip_samples]
    else:
        reps = int(np.ceil(clip_samples / max(len(y), 1)))
        y = np.tile(y, reps)[:clip_samples]

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    mu = S_db.mean()
    sigma = S_db.std() + 1e-9
    S_norm = (S_db - mu) / sigma

    X = torch.from_numpy(S_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    # shape: (1,1,128,T)
    return X


def load_model(ckpt_path: str, device: str = "cpu") -> KeyCNN:
    model = KeyCNN(num_classes=24)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_key(
    model: KeyCNN,
    audio_path: str,
    device: str = "cpu",
) -> tuple[str, str, float, float]:
    """
    Returns:
      tonic, mode, top1_prob, margin (top1 - top2)
    """
    X = audio_to_mel_tensor(audio_path).to(device)
    logits = model(X)
    probs = torch.softmax(logits, dim=-1).squeeze(0)  # (24,)

    top2 = torch.topk(probs, k=2)
    top1_id = int(top2.indices[0].item())
    top1_prob = float(top2.values[0].item())
    top2_prob = float(top2.values[1].item())
    margin = top1_prob - top2_prob

    tonic, mode = class_to_key(top1_id)
    return tonic, mode, top1_prob, margin
