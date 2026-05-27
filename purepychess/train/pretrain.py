"""Phase 1: supervised pretraining from npz shards.

Usage
-----
    python -m purepychess.train.pretrain \\
        data/shards \\
        checkpoints/pretrain \\
        --epochs 3 --lr 1e-3 --batch-size 256

The training loop iterates all shards once per epoch, loading one shard
at a time (lazy, low memory).  A cosine LR schedule decays from *lr* to
1e-5 across all batches.  A checkpoint is saved every CHECKPOINT_INTERVAL
positions and a final ``weights.pt`` is written at the end.

Metric logged every checkpoint
-------------------------------
  policy_acc — percentage of moves where argmax(policy_logits) == Lichess move
  (proxy for how well the network is imitating human play)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from purepychess.model import ChessNet

CHECKPOINT_INTERVAL = 500_000   # positions between checkpoints


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShardDataset(Dataset):
    """Loads a single npz shard into memory."""

    def __init__(self, shard_path: str) -> None:
        data = np.load(shard_path)
        # Cast float16 → float32 here so DataLoader workers don't need to
        self.states = data["state"].astype(np.float32)   # (N, 20, 8, 8)
        self.moves  = data["move"].astype(np.int64)       # (N,)
        self.wdls   = data["wdl"].astype(np.int64)        # (N,)  0/1/2

    def __len__(self) -> int:
        return len(self.moves)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, int]:
        return self.states[idx], self.moves[idx], self.wdls[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_one_shard(
    model: ChessNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    criterion: torch.nn.Module,
    device: torch.device,
    positions_seen: int,
    last_checkpoint: int,
    out: Path,
) -> tuple[int, int]:  # (positions_seen, last_checkpoint)
    """Train over one shard's DataLoader.  Returns updated counters."""
    correct_policy = 0
    total_policy   = 0

    for states, target_moves, target_wdl in loader:
        states       = states.to(device, non_blocking=True)
        target_moves = target_moves.to(device, non_blocking=True)
        target_wdl   = target_wdl.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            policy_logits, value_logits = model(states)
            loss_policy = criterion(policy_logits, target_moves)
            loss_value  = criterion(value_logits, target_wdl)
            loss        = loss_policy + loss_value

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        preds = policy_logits.argmax(dim=1)
        correct_policy += (preds == target_moves).sum().item()
        total_policy   += len(target_moves)
        positions_seen += len(target_moves)

        if positions_seen - last_checkpoint >= CHECKPOINT_INTERVAL:
            ckpt = out / f"checkpoint_{positions_seen:012d}.pt"
            torch.save(model.state_dict(), ckpt)
            acc = correct_policy / total_policy if total_policy else 0.0
            print(
                f"  @ {positions_seen:,} pos | "
                f"policy_acc={acc:.4f} | "
                f"loss={loss.item():.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            last_checkpoint = positions_seen
            correct_policy  = 0
            total_policy    = 0

    return positions_seen, last_checkpoint


def pretrain(
    shard_dir: str,
    output_dir: str,
    epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> None:
    """Supervised pretraining from npz shards.

    Parameters
    ----------
    shard_dir:   Directory containing ``shard_*.npz`` files.
    output_dir:  Directory for checkpoints and final ``weights.pt``.
    epochs:      Number of full passes over all shards.
    lr:          Peak learning rate (Adam); cosine-decayed to 1e-5.
    batch_size:  Mini-batch size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(Path(shard_dir).glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {shard_dir}")

    model     = ChessNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    criterion = nn.CrossEntropyLoss()

    # Build cosine scheduler over total expected batches
    total_batches = (
        sum(
            math.ceil(len(ShardDataset(str(p))) / batch_size)
            for p in shard_paths
        )
        * epochs
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_batches, 1), eta_min=1e-5
    )

    positions_seen  = 0
    last_checkpoint = 0
    model.train()

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")

        for shard_path in shard_paths:
            dataset = ShardDataset(str(shard_path))
            loader  = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=(device.type == "cuda"),
                num_workers=4,
                persistent_workers=True,
            )
            positions_seen, last_checkpoint = _train_one_shard(
                model, loader, optimizer, scaler, scheduler, criterion,
                device, positions_seen, last_checkpoint, out,
            )

    final = out / "weights.pt"
    torch.save(model.state_dict(), final)
    print(f"\nTraining complete. Saved final weights → {final}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised pretraining for purePyChess."
    )
    parser.add_argument("shard_dir",   help="Directory of shard_*.npz files")
    parser.add_argument("output_dir",  help="Directory for checkpoints")
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=256)
    args = parser.parse_args()
    pretrain(
        args.shard_dir,
        args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    _cli()
