"""Phase 2: multiprocessing self-play fine-tuning.

Architecture
------------
* **Main process** — owns the GPU model, optimizer, replay buffer, and training
  loop.
* **Worker processes** (NUM_SELFPLAY_WORKERS) — each runs CPU-based MCTS games
  and pushes completed game records to the main process via a
  ``multiprocessing.Queue``.
* Weight updates are periodically pushed from the main process to a shared
  file that workers reload at the start of each new game.

Windows / spawn compatibility
------------------------------
All worker-launch code is guarded by ``if __name__ == "__main__":`` via the
:func:`run_selfplay` call site.  Workers themselves are fully serialisable
(no lambda / local class objects passed to ``mp.Process``).

Usage
-----
    if __name__ == "__main__":
        from purepychess.train.selfplay import run_selfplay
        run_selfplay("checkpoints/pretrain/weights.pt")
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import chess
import numpy as np
import torch
import torch.nn as nn

from purepychess.encoding import board_to_tensor
from purepychess.mcts import MCTS
from purepychess.model import ChessNet
from purepychess.train.config import NUM_SELFPLAY_WORKERS, REPLAY_BUFFER_SIZE
from purepychess.train.replay import ReplayBuffer

_rng = np.random.default_rng(seed=None)  # non-deterministic by design

# Hyper-parameters
TRAIN_INTERVAL  = 256        # new positions received before one training step
BATCH_SIZE      = 512        # mini-batch sampled from replay buffer
EVAL_INTERVAL   = 10_000     # training steps between checkpoints
WEIGHT_PUSH_INTERVAL = 1_000 # training steps between weight pushes to workers


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _worker(
    worker_id: int,
    weights_path: str,
    game_queue: mp.Queue,
    stop_event: mp.Event,
    simulations: int,
) -> None:
    """Worker entry point.  Plays games until stop_event is set."""
    device = torch.device("cpu")  # workers use CPU; GPU stays in main process
    model  = ChessNet().to(device)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    mcts = MCTS(model, device, simulations=simulations)

    while not stop_event.is_set():
        # Reload weights from shared file before each game
        try:
            sd = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(sd)
        except Exception:
            pass  # use stale weights if file is mid-write

        record = _play_one_game(mcts)
        game_queue.put(record)


def _play_one_game(mcts: MCTS) -> list[tuple[np.ndarray, dict[int, int], int]]:
    """Play one self-play game.

    Returns a list of ``(state, sparse_policy, outcome)`` tuples, one per
    position in the game.  ``outcome`` is the game result from white's
    perspective: 0=white win, 1=draw, 2=black win.
    """
    board:      chess.Board          = chess.Board()
    prev_board: Optional[chess.Board] = None
    history: list[tuple[np.ndarray, dict[int, int]]] = []

    while not board.is_game_over(claim_draw=True):
        state        = board_to_tensor(board, prev_board)
        visit_counts = mcts.search(board, prev_board)

        if visit_counts:
            indices = list(visit_counts.keys())
            counts  = np.array([visit_counts[i] for i in indices], dtype=np.float64)
            probs   = counts / counts.sum()
            chosen  = int(_rng.choice(indices, p=probs))
            move    = chess.Move(chosen // 64, chosen % 64)
        else:
            move = next(iter(board.legal_moves))

        history.append((state, dict(visit_counts)))
        prev_board = board.copy()
        board.push(move)

    result = board.result(claim_draw=True)
    if result == "1-0":
        outcome = 0
    elif result == "0-1":
        outcome = 2
    else:
        outcome = 1

    return [(s, sp, outcome) for s, sp in history]


# ---------------------------------------------------------------------------
# Main training step
# ---------------------------------------------------------------------------

def _train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    replay_buffer: ReplayBuffer,
    device: torch.device,
) -> float:
    """Sample a mini-batch and update the model.  Returns the loss value."""
    if len(replay_buffer) < BATCH_SIZE:
        return 0.0

    states, policies, outcomes = replay_buffer.sample(BATCH_SIZE)
    states   = states.to(device, non_blocking=True)
    policies = policies.to(device, non_blocking=True)
    outcomes = outcomes.to(device, non_blocking=True)

    model.train()

    with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
        policy_logits, value_logits = model(states)

        # Policy loss: KL-divergence via soft cross-entropy
        log_probs   = torch.nn.functional.log_softmax(policy_logits, dim=1)
        loss_policy = -(policies * log_probs).sum(dim=1).mean()

        # Value loss: cross-entropy with hard WDL labels
        loss_value = nn.CrossEntropyLoss()(value_logits, outcomes)

        loss = loss_policy + loss_value

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    model.eval()

    return float(loss.item())


# ---------------------------------------------------------------------------
# Self-play loop
# ---------------------------------------------------------------------------

def _drain_queue_once(
    game_queue: mp.Queue,
    replay_buffer: ReplayBuffer,
) -> tuple[int, int]:
    """Drain all currently available game records.  Returns (games, positions)."""
    games     = 0
    positions = 0
    while not game_queue.empty():
        try:
            record = game_queue.get_nowait()
        except Exception:
            break
        replay_buffer.add_game(record)
        positions += len(record)
        games     += 1
    return games, positions


def _run_training_steps(
    new_positions:  int,
    model:          nn.Module,
    optimizer:      torch.optim.Optimizer,
    scaler:         torch.amp.GradScaler,
    replay_buffer:  ReplayBuffer,
    shared_weights: str,
    ckpt_dir:       Path,
    train_steps:    int,
    device:         torch.device,
) -> tuple[int, int]:  # returns (train_steps, remaining_positions)
    """Run as many training steps as *new_positions* allows."""
    while new_positions >= TRAIN_INTERVAL:
        loss = _train_step(model, optimizer, scaler, replay_buffer, device)
        train_steps  += 1
        new_positions -= TRAIN_INTERVAL

        if train_steps % EVAL_INTERVAL == 0:
            ckpt = ckpt_dir / f"step_{train_steps:08d}.pt"
            torch.save(model.state_dict(), str(ckpt))
            print(f"  Checkpoint saved: {ckpt} (loss={loss:.4f})")

        if train_steps % WEIGHT_PUSH_INTERVAL == 0:
            torch.save(model.state_dict(), shared_weights)

    return train_steps, new_positions


def run_selfplay(
    initial_weights: str,
    output_dir: str = "selfplay_output",
    simulations: int = 200,
    max_games: int = 10_000,
) -> None:
    """Run the self-play fine-tuning loop.

    IMPORTANT: On Windows (spawn start method) this function must be called
    inside a ``if __name__ == "__main__":`` guard.

    Parameters
    ----------
    initial_weights:
        Path to a ``weights.pt`` produced by Phase 1 pretraining.
    output_dir:
        Directory for shared weights, checkpoints, and final weights.
    simulations:
        MCTS simulations per move for workers.
    max_games:
        Stop after this many completed self-play games.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out    = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    model = ChessNet().to(device)
    sd    = torch.load(initial_weights, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    optimizer     = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler        = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    replay_buffer = ReplayBuffer(maxlen=REPLAY_BUFFER_SIZE)

    # Shared weights file workers read from
    shared_weights = str(out / "shared_weights.pt")
    torch.save(model.state_dict(), shared_weights)

    game_queue: mp.Queue = mp.Queue(maxsize=200)
    stop_event: mp.Event = mp.Event()

    workers: list[mp.Process] = []
    for wid in range(NUM_SELFPLAY_WORKERS):
        p = mp.Process(
            target=_worker,
            args=(wid, shared_weights, game_queue, stop_event, simulations),
            daemon=True,
        )
        p.start()
        workers.append(p)

    games_received  = 0
    new_positions   = 0
    train_steps     = 0

    print(
        f"Self-play started: {NUM_SELFPLAY_WORKERS} worker(s), "
        f"target {max_games} games, device={device}"
    )

    try:
        while games_received < max_games:
            games, positions = _drain_queue_once(game_queue, replay_buffer)
            games_received += games
            new_positions  += positions

            if games > 0 and games_received % 100 == 0:
                print(
                    f"  Games received: {games_received} | "
                    f"Buffer: {len(replay_buffer):,} | "
                    f"Train steps: {train_steps}"
                )

            train_steps, new_positions = _run_training_steps(
                new_positions, model, optimizer, scaler,
                replay_buffer, shared_weights, ckpt_dir, train_steps, device,
            )

    finally:
        stop_event.set()
        for p in workers:
            p.join(timeout=10)

    final = out / "final_weights.pt"
    torch.save(model.state_dict(), str(final))
    print(f"\nSelf-play complete. Final weights → {final}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: self-play fine-tuning for purePyChess."
    )
    parser.add_argument("initial_weights", help="Path to pretrained weights.pt")
    parser.add_argument(
        "--output-dir", default="selfplay_output",
        help="Directory for output files"
    )
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--max-games",   type=int, default=10_000)
    args = parser.parse_args()

    # Windows spawn guard
    run_selfplay(
        args.initial_weights,
        output_dir=args.output_dir,
        simulations=args.simulations,
        max_games=args.max_games,
    )


if __name__ == "__main__":
    _cli()
