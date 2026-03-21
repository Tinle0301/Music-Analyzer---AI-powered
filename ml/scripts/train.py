from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ml.scripts.dataset import load_manifest, KeyDataset
from ml.scripts.model import KeyCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def make_loaders(
    manifest_path: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_split: float,
    sr: int,
    clip_seconds: float,
    n_mels: int,
    hop_length: int,
    n_fft: int,
) -> Tuple[DataLoader, DataLoader]:
    items = load_manifest(manifest_path, root_dir=None)

    idxs = list(range(len(items)))
    train_idxs, val_idxs = train_test_split(
        idxs, test_size=val_split, random_state=seed, shuffle=True
    )

    # Build datasets (same items, different training flag -> random crop vs center crop)
    train_ds = KeyDataset(
        [items[i] for i in train_idxs],
        sr=sr,
        clip_seconds=clip_seconds,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        training=True,
        seed=seed,
    )
    val_ds = KeyDataset(
        [items[i] for i in val_idxs],
        sr=sr,
        clip_seconds=clip_seconds,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        training=False,
        seed=seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for X, y in tqdm(loader, desc="train", leave=False):
        X = X.to(device)
        y = y.to(device)

        optim.zero_grad(set_to_none=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for X, y in tqdm(loader, desc="val", leave=False):
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="ml/data/manifest.tsv", help="TSV: path<TAB>tonic<TAB>mode")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=0)  # mac safe default
    ap.add_argument("--seed", type=int, default=1234)

    # Feature params (must match inference later)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--clip_seconds", type=float, default=12.0)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--n_fft", type=int, default=2048)

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--save_dir", type=str, default="ml/models")
    ap.add_argument("--save_name", type=str, default="key_cnn_best.pt")
    args = ap.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple Silicon
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    train_loader, val_loader = make_loaders(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_split=args.val_split,
        sr=args.sr,
        clip_seconds=args.clip_seconds,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
    )

    model = KeyCNN(num_classes=24).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = -1.0
    best_path = os.path.join(args.save_dir, args.save_name)

    # Save training config alongside the model for reproducibility
    config_path = os.path.join(args.save_dir, "train_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}={v}\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optim, loss_fn, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        # Save best checkpoint by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  ✅ Saved best model -> {best_path} (val acc {best_val_acc:.3f})")

    print(f"\nDone. Best val acc: {best_val_acc:.3f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
