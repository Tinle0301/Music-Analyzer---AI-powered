from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

from ml.scripts.label_map import key_to_class


# ----------------------------
# Data record definition
# ----------------------------
@dataclass(frozen=True)
class TrackItem:
    path: str       # path to audio file
    tonic: str      # e.g., "E"
    mode: str       # "major" or "minor"


# ----------------------------
# Utility: parse a simple annotation file
# ----------------------------
def load_manifest(manifest_path: str, root_dir: Optional[str] = None) -> List[TrackItem]:
    """
    Manifest format (TSV recommended):
      relative/or/absolute/path<TAB>TONIC<TAB>MODE

    Example line:
      audio/track_001.wav    E    major
    """
    items: List[TrackItem] = []
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"Bad manifest line (need 3 tab-separated columns): {line}"
                )

            rel_path, tonic, mode = parts
            audio_path = rel_path
            if root_dir is not None and not os.path.isabs(audio_path):
                audio_path = os.path.join(root_dir, rel_path)

            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            items.append(TrackItem(path=audio_path, tonic=tonic.strip(), mode=mode.strip().lower()))

    return items


# ----------------------------
# Dataset: audio -> mel spectrogram tensor
# ----------------------------
class KeyDataset(Dataset):
    """
    Returns:
      X: torch.FloatTensor shaped (1, n_mels, time_frames)
      y: int class id in [0..23]
    """

    def __init__(
        self,
        items: List[TrackItem],
        sr: int = 22050,
        clip_seconds: float = 12.0,
        n_mels: int = 128,
        hop_length: int = 512,
        n_fft: int = 2048,
        fmin: float = 30.0,
        fmax: Optional[float] = None,
        training: bool = True,
        seed: int = 1234,
    ):
        self.items = items
        self.sr = sr
        self.clip_seconds = clip_seconds
        self.clip_samples = int(sr * clip_seconds)
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.training = training

        random.seed(seed)

    def __len__(self) -> int:
        return len(self.items)

    def _random_crop(self, y: np.ndarray) -> np.ndarray:
        """Random crop/pad waveform to fixed length (clip_samples)."""
        if len(y) >= self.clip_samples:
            if self.training:
                start = random.randint(0, len(y) - self.clip_samples)
            else:
                start = max(0, (len(y) - self.clip_samples) // 2)
            return y[start:start + self.clip_samples]

        # If too short: pad (repeat-pad is often better than zero-pad for music)
        pad_len = self.clip_samples - len(y)
        if len(y) > 0:
            reps = int(np.ceil(self.clip_samples / len(y)))
            y_rep = np.tile(y, reps)[:self.clip_samples]
            return y_rep
        return np.zeros(self.clip_samples, dtype=np.float32)

    def _wave_to_mel(self, y: np.ndarray) -> np.ndarray:
        """
        Convert waveform -> log-mel spectrogram:
          mel shape: (n_mels, frames)
        """
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)  # log scale

        # Normalize per example (helps training stability)
        mu = S_db.mean()
        sigma = S_db.std() + 1e-9
        S_norm = (S_db - mu) / sigma
        return S_norm.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.items[idx]

        # Load audio
        y, _sr = librosa.load(item.path, sr=self.sr, mono=True)
        if y.size == 0:
            # Return zeros if file is weird, avoids crashing training
            y = np.zeros(self.clip_samples, dtype=np.float32)
        y = librosa.util.normalize(y)

        # Crop/pad to fixed length
        y_clip = self._random_crop(y)

        # Convert to mel features
        mel = self._wave_to_mel(y_clip)  # (n_mels, frames)

        # Torch tensor shape for CNN: (channels=1, n_mels, frames)
        X = torch.from_numpy(mel).unsqueeze(0)

        # Label
        y_class = key_to_class(item.tonic, item.mode)
        return X, y_class
