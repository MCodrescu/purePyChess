"""purePyChess — Pure-Python chess engine with a ResNet policy+value network.

Quick start
-----------
    import purepychess, chess

    # Get a single best move
    engine = purepychess.Engine()          # auto-downloads weights on first call
    board  = chess.Board()
    move   = engine.get_best_move(board)   # returns chess.Move

    # Play a full engine-vs-engine game
    moves = purepychess.play()

    # Pre-cache weights manually
    purepychess.download_weights()

    # Launch as a UCI engine (via installed script)
    #   $ purepychess
"""

from __future__ import annotations

from typing import Optional

import chess
import torch

from purepychess.config import MCTS_SIMULATIONS

__all__ = ["Engine", "play", "download_weights"]


class Engine:
    """High-level interface to the purePyChess neural chess engine.

    Parameters
    ----------
    simulations:
        Number of MCTS simulations per move.  Defaults to ``MCTS_SIMULATIONS``
        from :mod:`purepychess.config`.
    """

    def __init__(self, simulations: int = MCTS_SIMULATIONS) -> None:
        from purepychess.model import ChessNet
        from purepychess.mcts import MCTS
        from purepychess.weights import load_weights

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        _model = ChessNet()
        load_weights(_model, self.device)
        _model.eval()

        self._mcts        = MCTS(_model, self.device, simulations=simulations)
        self._prev_board: Optional[chess.Board] = None
        self._move_number: int = 0

    def get_best_move(
        self,
        board: chess.Board,
        prev_board: Optional[chess.Board] = None,
    ) -> chess.Move:
        """Return the engine's best move for *board*.

        If *prev_board* is not provided the engine uses its internal one-step
        history (useful when calling repeatedly across a game).
        """
        prev = prev_board if prev_board is not None else self._prev_board
        move = self._mcts.get_best_move(board, prev, self._move_number)
        self._prev_board  = board.copy()
        self._move_number += 1
        return move

    def reset(self) -> None:
        """Reset internal history (call before starting a new game)."""
        self._prev_board  = None
        self._move_number = 0


def play(simulations: int = MCTS_SIMULATIONS) -> list[chess.Move]:
    """Play a full engine-vs-engine game and return the move list.

    Parameters
    ----------
    simulations:
        MCTS simulations per move for both sides.
    """
    engine_white = Engine(simulations=simulations)
    engine_black = Engine(simulations=simulations)
    board        = chess.Board()
    moves: list[chess.Move] = []
    prev_board: Optional[chess.Board] = None

    while not board.is_game_over(claim_draw=True):
        engine = engine_white if board.turn == chess.WHITE else engine_black
        move   = engine.get_best_move(board, prev_board)
        prev_board = board.copy()
        board.push(move)
        moves.append(move)

    return moves


def download_weights() -> None:
    """Pre-cache model weights from HuggingFace Hub."""
    from purepychess.weights import download_weights as _dw
    _dw()
