"""Phase 0: offline preprocessing of Lichess PGN → binary npz shards.

Run once before training:
    python -m purepychess.train.preprocess \\
        /path/to/lichess_db_standard_rated_2024-01.pgn.bz2 \\
        data/shards

Output shards: data/shards/shard_0000.npz, shard_0001.npz, …
Each shard contains up to SHARD_SIZE positions with keys:
    state : float16  (N, 20, 8, 8)  — board tensor (halved disk / RAM)
    move  : int16    (N,)            — flat move index (from_sq*64 + to_sq)
    wdl   : uint8    (N,)            — 0=white win, 1=draw, 2=black win

Cast state to float32 inside the DataLoader before feeding to the network.
"""

from __future__ import annotations

import argparse
import bz2
import re
from pathlib import Path

import chess.pgn
import numpy as np

from purepychess.encoding import board_to_tensor, move_to_index
from purepychess.train.config import MIN_TIME_CONTROL, TARGET_ELO

SHARD_SIZE = 100_000


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _parse_time_control(tc: str) -> int:
    """Parse a Lichess time-control string to its base seconds.

    Examples: ``"600+5"`` → 600, ``"300"`` → 300, ``"-"`` → 0.
    """
    if not tc or tc == "-":
        return 0
    m = re.match(r"(\d+)", tc)
    return int(m.group(1)) if m else 0


def _game_passes_filter(game: chess.pgn.Game) -> bool:
    """Return True if *game* satisfies Elo and time-control thresholds."""
    headers = game.headers

    # Lichess exports set "Rated" header to "Yes" / "No" / "true" / "false"
    rated_raw = headers.get("Rated", "").lower()
    if rated_raw not in {"yes", "true"}:
        return False

    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
    except ValueError:
        return False

    if (white_elo + black_elo) / 2 < TARGET_ELO:
        return False

    if _parse_time_control(headers.get("TimeControl", "")) < MIN_TIME_CONTROL:
        return False

    return True


def _result_to_wdl(result: str) -> int:
    """Convert a PGN result string to a WDL label from white's POV."""
    if result == "1-0":
        return 0   # white win
    if result == "0-1":
        return 2   # black win
    return 1       # draw / unknown


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

def _write_shard(
    out: Path,
    idx: int,
    states: list,
    moves: list,
    wdls: list,
) -> None:
    path = out / f"shard_{idx:04d}.npz"
    np.savez_compressed(
        path,
        state=np.stack(states),                  # float16
        move=np.array(moves, dtype=np.int16),
        wdl=np.array(wdls, dtype=np.uint8),
    )
    print(f"  Wrote {path} ({len(states):,} positions)")


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess(pgn_path: str, output_dir: str) -> None:
    """Convert a Lichess PGN (.pgn or .pgn.bz2) file to npz shards.

    Parameters
    ----------
    pgn_path:
        Path to the Lichess monthly PGN file (plain or bz2-compressed).
    output_dir:
        Directory where shard files are written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    states: list[np.ndarray] = []
    moves:  list[int]        = []
    wdls:   list[int]        = []
    shard_idx = 0
    total     = 0

    opener = bz2.open if str(pgn_path).endswith(".bz2") else open

    with opener(pgn_path, "rt", encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            if not _game_passes_filter(game):
                continue

            wdl_label = _result_to_wdl(game.headers.get("Result", "*"))
            board     = game.board()
            prev_board: chess.Board | None = None

            for move in game.mainline_moves():
                tensor    = board_to_tensor(board, prev_board).astype(np.float16)
                move_idx  = move_to_index(move)

                states.append(tensor)
                moves.append(move_idx)
                wdls.append(wdl_label)
                total += 1

                prev_board = board.copy()
                board.push(move)

                if len(states) >= SHARD_SIZE:
                    _write_shard(out, shard_idx, states, moves, wdls)
                    shard_idx += 1
                    states, moves, wdls = [], [], []

    # Flush remaining positions
    if states:
        _write_shard(out, shard_idx, states, moves, wdls)
        shard_idx += 1

    print(
        f"\nPreprocessing complete: {total:,} positions → "
        f"{shard_idx} shard(s) in {out}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Lichess PGN file to purePyChess npz shards."
    )
    parser.add_argument("pgn_path",    help="Path to .pgn or .pgn.bz2 file")
    parser.add_argument("output_dir",  help="Directory to write npz shards")
    args = parser.parse_args()
    preprocess(args.pgn_path, args.output_dir)


if __name__ == "__main__":
    _cli()
