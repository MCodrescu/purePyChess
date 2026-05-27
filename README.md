# purePyChess

A pure-Python chess engine powered by a residual neural network and Monte Carlo Tree Search (MCTS). The engine uses a policy + WDL value network trained on Lichess game data and refined through self-play, then communicates over the UCI protocol for compatibility with chess GUIs.

> **Note:** This project was developed with the assistance of GitHub Copilot (AI pair programming). The architecture, training pipeline, and codebase were designed and implemented collaboratively between a human author and an AI coding assistant.

## Features

- **ResNet policy+value network** — 10-block, 128-filter residual architecture with WDL (win/draw/loss) output
- **Leaf-batched MCTS** — batches leaf evaluations to keep GPU utilization high; transposition table for position reuse
- **Supervised pretraining** — trains on filtered Lichess PGN exports (rated games, configurable Elo floor)
- **Self-play fine-tuning** — multiprocessing workers generate games on CPU; main process trains on GPU
- **UCI interface** — plug into Arena, Lichess local analysis, or any UCI-compatible GUI
- **Auto weight download** — weights are fetched from Hugging Face Hub on first use and cached locally

## Installation

**Inference only** (play chess, use the UCI engine):

```bash
pip install .
```

**With training support**:

```bash
pip install ".[train]"
```

Requires Python ≥ 3.10.

## Quick Start

```python
import chess
import purepychess

engine = purepychess.Engine()          # downloads weights automatically on first run
board  = chess.Board()
move   = engine.get_best_move(board)   # returns a chess.Move
print(board.san(move))
```

Pre-download weights manually:

```python
purepychess.download_weights()
```

## UCI Engine

After installation, launch as a UCI engine from the command line:

```bash
purepychess
```

Point any UCI-compatible GUI (Arena, BanksiaGUI, Lichess board analysis, etc.) at this executable.

## Training Pipeline

Training runs in three phases:

### Phase 0 — Preprocess

Convert a Lichess monthly PGN export (`.bz2`) into binary shards:

```bash
python -m purepychess.train.preprocess /path/to/lichess.pgn.bz2 data/shards/
```

Filters to rated games with average Elo ≥ `TARGET_ELO` (default 1800) and time control ≥ 3 minutes. Writes `float16` `.npz` shards of 100k positions each.

### Phase 1 — Supervised Pretraining

Train on the shards produced in Phase 0:

```bash
python -m purepychess.train.pretrain data/shards/ checkpoints/
```

### Phase 2 — Self-Play Fine-Tuning

```bash
python -m purepychess.train.selfplay checkpoints/latest.pt output/
```

Spawns `NUM_SELFPLAY_WORKERS` CPU worker processes for game generation; the main process owns the GPU for training.

## Project Structure

```
purepychess/
  __init__.py        # public API: Engine, play(), download_weights()
  config.py          # inference constants (MCTS_SIMULATIONS, HF_REPO_ID, …)
  encoding.py        # board → (20,8,8) tensor; move ↔ index
  model.py           # ResNet ChessNet
  mcts.py            # leaf-batched MCTS + transposition table
  engine.py          # UCI CLI entry point
  weights.py         # Hugging Face Hub weight management

purepychess/train/
  config.py          # training hyperparameters
  preprocess.py      # Lichess PGN → npz shards
  pretrain.py        # supervised training loop
  selfplay.py        # self-play fine-tuning loop
  replay.py          # replay buffer with sparse policy storage

tests/               # pytest suite
```

## Configuration

| Constant | File | Default | Description |
|---|---|---|---|
| `MCTS_SIMULATIONS` | `config.py` | 400 | Simulations per move at inference |
| `BATCH_LEAF_SIZE` | `config.py` | 16 | Leaves batched per GPU call |
| `HF_REPO_ID` | `config.py` | *(set before use)* | Hugging Face model repo |
| `TARGET_ELO` | `train/config.py` | 1800 | Minimum average Elo for training games |
| `MIN_TIME_CONTROL` | `train/config.py` | 180 s | Minimum time control for training games |
| `REPLAY_BUFFER_SIZE` | `train/config.py` | 500,000 | Self-play replay buffer capacity |
| `NUM_SELFPLAY_WORKERS` | `train/config.py` | 4 | Parallel self-play worker processes |

## Running Tests

```bash
pytest
```

## Requirements

- Python ≥ 3.10
- [PyTorch](https://pytorch.org/) (CPU or CUDA)
- [python-chess](https://python-chess.readthedocs.io/) ≥ 1.0
- numpy
- huggingface_hub
- h5py *(training only)*

## AI Development Disclosure

This project was built using **GitHub Copilot** (powered by Claude Sonnet) as an AI coding assistant. The overall architecture, design decisions, and requirements were provided by the human author; the AI assisted with code generation, debugging, and implementation details throughout the development process. All code has been reviewed by the author.

## License

MIT License — see [LICENSE](LICENSE) for the full text.
