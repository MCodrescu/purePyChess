"""UCI engine entry point for purePyChess.

Run directly:
    python -m purepychess.engine

Or via the installed script:
    purepychess

Supported UCI commands
----------------------
uci          → uciok
isready      → readyok
ucinewgame   → reset board + MCTS
position fen <fen> [moves <uci…>]
position startpos [moves <uci…>]
go [movetime <ms>] [nodes <n>]  → bestmove <uci>
stop         → (no-op for synchronous search)
quit         → exit
"""

from __future__ import annotations

import sys

import chess
import torch

from purepychess.config import MCTS_SIMULATIONS
from purepychess.mcts import MCTS
from purepychess.model import ChessNet
from purepychess.weights import load_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_position(line: str) -> tuple[chess.Board, chess.Board | None, int]:
    """Parse a ``position`` command and return (board, prev_board, move_number)."""
    tokens = line.split()
    board  = chess.Board()
    idx    = 1

    if idx < len(tokens) and tokens[idx] == "startpos":
        board = chess.Board()
        idx  += 1
    elif idx < len(tokens) and tokens[idx] == "fen":
        idx += 1
        fen_parts: list[str] = []
        while idx < len(tokens) and tokens[idx] != "moves":
            fen_parts.append(tokens[idx])
            idx += 1
        board = chess.Board(" ".join(fen_parts))

    prev_board: chess.Board | None = None
    move_number = 0

    if idx < len(tokens) and tokens[idx] == "moves":
        idx += 1
        for uci_str in tokens[idx:]:
            prev_board  = board.copy()
            board.push(chess.Move.from_uci(uci_str))
            move_number += 1

    return board, prev_board, move_number


def _parse_go_nodes(line: str) -> int | None:
    """Extract ``nodes N`` from a ``go`` command line, or return ``None``."""
    tokens = line.split()
    try:
        i = tokens.index("nodes")
        return int(tokens[i + 1])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Read UCI commands from stdin and write responses to stdout."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChessNet()
    load_weights(model, device)
    model.eval()

    mcts        = MCTS(model, device, simulations=MCTS_SIMULATIONS)
    board       = chess.Board()
    prev_board: chess.Board | None = None
    move_number = 0

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        if line == "uci":
            print("id name purePyChess")
            print("id author purePyChess")
            print("uciok")

        elif line == "isready":
            print("readyok")

        elif line == "ucinewgame":
            board       = chess.Board()
            prev_board  = None
            move_number = 0
            mcts        = MCTS(model, device, simulations=MCTS_SIMULATIONS)

        elif line.startswith("position"):
            board, prev_board, move_number = _parse_position(line)

        elif line.startswith("go"):
            nodes = _parse_go_nodes(line)
            if nodes is not None:
                mcts.simulations = nodes
            move = mcts.get_best_move(board, prev_board, move_number)
            print(f"bestmove {move.uci()}")

        elif line == "stop":
            pass  # synchronous search — nothing to stop

        elif line == "quit":
            break

        sys.stdout.flush()


if __name__ == "__main__":
    main()
