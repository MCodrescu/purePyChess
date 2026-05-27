"""Board and move encoding for purePyChess.

Board is always encoded from white's perspective:
  Planes 0-5:   white piece presence (pawn, knight, bishop, rook, queen, king)
  Planes 6-11:  black piece presence (pawn, knight, bishop, rook, queen, king)
  Planes 12-13: history — white/black aggregate piece presence from t-1
  Planes 14-17: castling rights (K, Q, k, q)
  Plane 18:     en passant file (entire column set to 1)
  Plane 19:     side to move (all 1s = white, all 0s = black)

Move encoding: from_sq * 64 + to_sq → flat index in [0, 4096).
Underpromotions are always treated as queen promotions (out of scope).
"""

from __future__ import annotations

import chess
import numpy as np

_PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def _fill_piece_planes(
    planes: np.ndarray,
    board: chess.Board,
    white_offset: int,
    black_offset: int,
) -> None:
    """Write piece-presence bitmaps for *board* into *planes* in-place."""
    for i, pt in enumerate(_PIECE_ORDER):
        for sq in board.pieces(pt, chess.WHITE):
            r, f = divmod(sq, 8)
            planes[white_offset + i, r, f] = 1.0
        for sq in board.pieces(pt, chess.BLACK):
            r, f = divmod(sq, 8)
            planes[black_offset + i, r, f] = 1.0


def board_to_tensor(
    board: chess.Board,
    prev_board: chess.Board | None = None,
) -> np.ndarray:
    """Encode *board* as a (20, 8, 8) float32 tensor from white's perspective.

    Square index mapping: sq = rank * 8 + file, with rank 0 = rank 1 (white's
    back rank) and file 0 = a-file, matching python-chess square numbering.
    """
    planes = np.zeros((20, 8, 8), dtype=np.float32)

    # Planes 0-5: white pieces; 6-11: black pieces
    _fill_piece_planes(planes, board, white_offset=0, black_offset=6)

    # Planes 12-13: history (aggregate white/black piece presence from t-1)
    if prev_board is not None:
        for pt in _PIECE_ORDER:
            for sq in prev_board.pieces(pt, chess.WHITE):
                r, f = divmod(sq, 8)
                planes[12, r, f] = 1.0
            for sq in prev_board.pieces(pt, chess.BLACK):
                r, f = divmod(sq, 8)
                planes[13, r, f] = 1.0

    # Planes 14-17: castling rights (K, Q, k, q)
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[14] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[15] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[16] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[17] = 1.0

    # Plane 18: en passant file (entire column)
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        planes[18, :, ep_file] = 1.0

    # Plane 19: side to move
    if board.turn == chess.WHITE:
        planes[19] = 1.0

    return planes


def move_to_index(move: chess.Move) -> int:
    """Encode *move* as flat index ``from_sq * 64 + to_sq``."""
    return move.from_square * 64 + move.to_square


def index_to_move(index: int) -> chess.Move:
    """Decode flat index back to a :class:`chess.Move` (queen promotion assumed)."""
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a (4096,) boolean mask that is ``True`` for each legal move."""
    mask = np.zeros(4096, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask
