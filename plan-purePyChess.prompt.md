# Plan: purePyChess — Python Chess Engine with Neural Network

## Overview
Pure-Python chess engine using a ResNet policy+value network (WDL output).
Phase 0: offline preprocessing of Lichess PGN → binary cache.
Phase 1: supervised pretraining on cached binary data.
Phase 2: self-play fine-tuning with leaf-batched MCTS + multiprocessing.
All positions encoded from white's perspective. Configurable target Elo filter.

## Dependencies
Base: torch, python-chess, numpy, huggingface_hub
Train extra: h5py

## Package Structure
```
purepychess/                    ← base package (pip install purepychess)
  __init__.py                   # public API: Engine, get_best_move(), play()
  config.py                     # inference constants only
  encoding.py                   # board → (20,8,8) tensor, move → index, history
  model.py                      # ResNet policy+value (WDL)
  mcts.py                       # leaf-batched MCTS + transposition table
  engine.py                     # UCI CLI entry point
  weights.py                    # HuggingFace Hub auto-download + local cache

purepychess/train/              ← pip install purepychess[train]
  __init__.py
  config.py                     # TARGET_ELO, NUM_SELFPLAY_WORKERS, etc.
  preprocess.py                 # Lichess PGN → npz shards (offline, run once)
  pretrain.py                   # supervised training from shards
  selfplay.py                   # multiprocessing self-play loop
  replay.py                     # replay buffer, sparse policy storage

tests/
  conftest.py                   # shared fixtures (UniformStubModel, cpu_device, …)
  test_encoding.py
  test_model.py
  test_mcts.py
  test_weights.py
  test_engine.py
  train/
    test_preprocess.py
    test_pretrain.py
    test_replay.py

pyproject.toml
pytest.ini
plan-purePyChess.prompt.md      ← this file
```

## pyproject.toml
```
[project]
name = "purepychess"
requires-python = ">=3.10"
dependencies = ["torch", "python-chess", "numpy", "huggingface_hub"]

[project.optional-dependencies]
train = ["h5py"]

[project.scripts]
purepychess = "purepychess.engine:main"
```

## config.py (base — inference only)
- MCTS_SIMULATIONS = 400
- BATCH_LEAF_SIZE = 16
- HF_REPO_ID = "your-hf-username/purepychess-weights"

## train/config.py (train extra only)
- TARGET_ELO = 1800
- MIN_TIME_CONTROL = 180
- REPLAY_BUFFER_SIZE = 500_000
- NUM_SELFPLAY_WORKERS = 4

## weights.py — HuggingFace Weight Management
- On first Engine instantiation, check local cache (~/.cache/purepychess/weights.pt)
- If not present: call huggingface_hub.hf_hub_download(HF_REPO_ID, "weights.pt") → downloads and caches automatically
- If download fails (no internet, repo missing, auth required): raise a clear RuntimeError with instructions: "Could not download weights. Check internet connection or manually place weights.pt in ~/.cache/purepychess/. Download from: https://huggingface.co/{HF_REPO_ID}"
- huggingface_hub handles resumable downloads, caching, and versioning natively
- Subsequent instantiations: load from local cache instantly (no network call)
- Expose purepychess.download_weights() for users who want to pre-cache manually

## __init__.py — Public API (base package)
Expose three simple entry points so new users need almost no documentation:
```python
import purepychess, chess

# 1. Get best move from a position
engine = purepychess.Engine()          # auto-downloads weights on first call
board  = chess.Board()
move   = engine.get_best_move(board)   # returns chess.Move

# 2. Play a full game (engine vs engine or engine vs human hook)
purepychess.play()

# 3. Launch as UCI engine (for chess GUIs)
# $ purepychess   ← installed CLI command
```
Engine constructor accepts optional simulations= kwarg to override MCTS_SIMULATIONS.

## Board Representation (encoding.py)
- Always encode from WHITE's perspective (no board flipping for black to move)
  - Network learns both sides simultaneously; side-to-move plane disambiguates
- Encode board as (20, 8, 8) float32 tensor:
  - Planes 0-5:   white piece presence (pawn, knight, bishop, rook, queen, king) — absolute white perspective; plane layout never changes regardless of side to move
  - Planes 6-11:  black piece presence (pawn, knight, bishop, rook, queen, king) — absolute white perspective; plane layout never changes regardless of side to move
  - Planes 12-13: history — white/black AGGREGATE piece presence from t-1 (all white pieces OR'd into plane 12; all black pieces OR'd into plane 13)
  - Planes 14-17: castling rights (KQkq)
  - Plane 18:     en passant file (entire column set to 1)
  - Plane 19:     side to move (all 1s = white, all 0s = black)
- History tracking: caller passes previous board state as prev_board argument
- Move encoding: flat index from_sq * 64 + to_sq → 4096 indices
  - Underpromotions always treated as queen promotion (out of scope)
  - Illegal moves masked to -inf before softmax during training and search

## Neural Network (model.py)
- Input: (20, 8, 8)
- Stem: Conv2d(20, 128, 3, padding=1) + BN + ReLU
- Body: 10 residual blocks (Conv→BN→ReLU→Conv→BN→skip→ReLU), 128 filters
- Policy head: Conv(128,2,1)→BN→ReLU→Flatten→FC(in=128, out=4096) → raw logits  [2 channels × 8×8 = 128 features]
- Value head: Conv(128,1,1)→BN→ReLU→Flatten→FC(in=64, out=256)→ReLU→FC(in=256, out=3) → raw WDL logits  [1 channel × 8×8 = 64 features]
  - Scalar for MCTS: apply softmax to raw logits at inference time → p_win - p_loss
- Loss = nn.CrossEntropyLoss(policy_logits, move_played) + nn.CrossEntropyLoss(wdl_logits, wdl_target)
  (nn.CrossEntropyLoss expects raw logits and applies log_softmax internally — do NOT apply softmax before the loss)
- GPU: device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
- Mixed precision: torch.amp.autocast('cuda') + torch.amp.GradScaler('cuda') for ~2x throughput

## Phase 0: Preprocessing (preprocess.py) — run once offline
- Input: Lichess monthly PGN bz2 file path
- Filter games: Rated header ∈ {"Yes", "true"} (case-insensitive), avg_elo >= TARGET_ELO, time_control >= MIN_TIME_CONTROL
- For each passing game, iterate all positions:
  - Encode (20,8,8) board tensor (including 1-step history)
  - Record move index (int16)
  - Record WDL label (uint8: 0=white win, 1=draw, 2=black win)
- Write batches of 100k positions to numbered npz shards: {state: float16, move: int16, wdl: uint8}
- Store float16 (halved disk/memory); cast to float32 at DataLoader time
- Output: data/shards/shard_0000.npz, shard_0001.npz, ...

## Phase 1: Supervised Pretraining (pretrain.py)
- ShardDataset: iterates npz shards lazily (one shard in memory at a time)
- DataLoader: pin_memory=True, num_workers=4, persistent_workers=True, shuffle within shard
- WDL target always from white's perspective (consistent with encoding)
  - white win → class 0, draw → class 1, black win → class 2
- Optimizer: Adam, lr=1e-3, weight_decay=1e-4, cosine decay to 1e-5 over training
- Save checkpoint every 500k positions processed
- Log policy accuracy (% of moves matching Lichess move) as proxy metric

## Phase 2: MCTS with Leaf Batching (mcts.py)

### Transposition Table
- Dict mapping Zobrist hash (via chess.polyglot.zobrist_hash(board)) → MCTSNode
- On expansion: check table first; if hit, reuse existing node's N/Q/P stats
- Cap table size at ~1M entries (evict oldest on overflow via OrderedDict)

### Leaf Batching
- During simulation, do NOT call network immediately on each leaf
- Instead, mark leaf with "virtual loss" (temporarily decrement Q to discourage parallel selection of same node)
- Accumulate BATCH_LEAF_SIZE pending leaves before doing a single batched network forward pass
- After GPU call returns all values + policies, backpropagate all simultaneously
- This keeps GPU utilization high (batch size >> 1 per forward pass)

### MCTS Algorithm
- Selection: UCB-PUCT: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)), c_puct=1.5
- Expansion: check board.is_game_over(claim_draw=True) first; if terminal, assign value directly (checkmate → -1 for side just moved, draw → 0) without network evaluation
- Expansion (non-terminal): check transposition table; if miss, add to pending leaf batch
- Evaluation: batched GPU network call once BATCH_LEAF_SIZE leaves accumulated
- Backprop: walk each leaf's path to root, update N and Q; remove virtual loss
- Root noise: Dirichlet(alpha=0.3) mixed into root priors at frac=0.25
- Move selection: sample proportional to N^(1/temp); temp=1 first 30 moves, then greedy

## Phase 3: Self-Play with Multiprocessing (selfplay.py)

### Architecture
- Main process: owns the network weights (on GPU if available), replay buffer, and training loop
- Worker processes (NUM_SELFPLAY_WORKERS): each runs CPU-only MCTS games using a copy of weights loaded from a shared file; GPU stays exclusively in main process
- Communication: workers send completed game records to main process via multiprocessing.Queue
- Main process: pulls records from queue, adds to replay buffer, trains, periodically writes updated weights to the shared file for workers to reload
- Windows guard: all worker launch code wrapped in `if __name__ == "__main__":` for spawn compatibility
- Weight push interval: every WEIGHT_PUSH_INTERVAL = 1_000 training steps

### Per-game data
- Each position produces: (state_tensor, sparse_policy, outcome)
  - sparse_policy: dict {move_index: visit_count} — only visited moves stored (typically <50 of 4096)
  - outcome: 0/1/2 (white win/draw/black win) — same WDL encoding as pretraining
- Replay buffer (replay.py): deque of REPLAY_BUFFER_SIZE entries storing sparse policies
  - At batch sample time: convert sparse dict → dense (4096,) tensor on the fly

### Training loop
- After every 256 new positions arrive from workers, sample a mini-batch of 512 from replay buffer
- Policy loss: soft cross-entropy (KL divergence) against MCTS visit-count distribution
- Value loss: cross-entropy against hard WDL outcome label
- Compute loss, backprop with GradScaler, step optimizer
- Every 10k training steps: save checkpoint

## Engine Interface (engine.py)
- UCI protocol over stdin/stdout (compatible with Arena, Lichess analysis board, etc.)
- Key commands: uci, isready, position fen/moves, go movetime/nodes, stop, quit
- go handler: run MCTS for specified time/nodes, return bestmove

## Verification Steps
1. encoding.py: encode starting position → verify 32 non-zero cells across piece planes; verify history planes all-zero for first move; decode move index back to uci string
2. model.py: forward pass (batch=4) random tensors → policy shape (4,4096), value shape (4,3), value rows sum to 1.0
3. preprocess.py: process 1000 games → verify shard written, spot-check 5 positions manually against board
4. pretrain.py: overfit on single shard (100k positions) → policy loss should drop below 3.0 within 1 epoch
5. mcts.py: mate-in-1 position → MCTS finds it within 50 simulations; transposition table hit rate > 0 after 200 simulations
6. selfplay.py: launch 2 workers, confirm Queue receives game records, replay buffer populates
7. engine.py: pipe "uci\nisready\nposition startpos\ngo nodes 100\n" → receive "uciok", "readyok", "bestmove [move]"

## Decisions
- python-chess for move gen and Zobrist hashing (chess.polyglot.zobrist_hash(board) — board.zobrist_hash() removed in python-chess ≥ 1.0)
- White's-perspective encoding throughout (consistent, simpler than current-player flip)
- 2 history planes (aggregate per-color presence from t-1) — lightweight repetition signal; planes 12 and 13 are each a single OR'd bitmap, not 6 per-type planes
- Sparse policy storage in replay buffer — avoids ~10GB RAM for dense 4096-float vectors
- npz shards for binary cache — no extra dependency (numpy built-in), easy to parallelize reads
- Underpromotions out of scope — always promote to queen
- TARGET_ELO in train/config.py — single value to change, affects preprocess and pretrain filters
- Leaf batching batch size (BATCH_LEAF_SIZE=16) is tunable in config.py
- Weights hosted on HuggingFace Hub; auto-downloaded to ~/.cache/purepychess/ on first use
- Training split as optional [train] extra — base install is lean and inference-only
- Workers use CPU exclusively; main process owns the GPU — avoids CUDA multiprocessing pitfalls
