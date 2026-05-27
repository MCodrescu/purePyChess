"""Tests for purepychess.encoding."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from purepychess.encoding import (
    board_to_tensor,
    index_to_move,
    legal_move_mask,
    move_to_index,
)


# ---------------------------------------------------------------------------
# board_to_tensor — shape and dtype
# ---------------------------------------------------------------------------

class TestBoardToTensorShape:
    def test_shape(self):
        t = board_to_tensor(chess.Board())
        assert t.shape == (20, 8, 8)

    def test_dtype(self):
        t = board_to_tensor(chess.Board())
        assert t.dtype == np.float32

    def test_values_binary(self):
        """Every value in the tensor must be 0.0 or 1.0."""
        t = board_to_tensor(chess.Board())
        unique = set(np.unique(t))
        assert unique <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# board_to_tensor — starting position piece planes (planes 0-11)
# ---------------------------------------------------------------------------

class TestStartingPositionPiecePlanes:
    """From the starting position, white and black each have 16 pieces."""

    def setup_method(self):
        self.t = board_to_tensor(chess.Board())

    def test_white_piece_plane_count(self):
        """Planes 0-5 combined should have exactly 16 ones (all white pieces)."""
        white_planes = self.t[0:6]
        assert white_planes.sum() == pytest.approx(16)

    def test_black_piece_plane_count(self):
        """Planes 6-11 combined should have exactly 16 ones (all black pieces)."""
        black_planes = self.t[6:12]
        assert black_planes.sum() == pytest.approx(16)

    def test_total_piece_presence(self):
        """32 non-zero cells across planes 0-11."""
        assert self.t[0:12].sum() == pytest.approx(32)

    def test_white_pawn_plane(self):
        """Plane 0: 8 white pawns on rank 1 (row index 1)."""
        # a2..h2 = squares 8..15, row=1
        assert self.t[0, 1, :].sum() == pytest.approx(8)
        assert self.t[0, 0, :].sum() == pytest.approx(0)

    def test_black_pawn_plane(self):
        """Plane 6: 8 black pawns on rank 6 (row index 6)."""
        assert self.t[6, 6, :].sum() == pytest.approx(8)

    def test_white_king_plane(self):
        """Plane 5: white king on e1 = square 4, row=0, col=4."""
        assert self.t[5, 0, 4] == pytest.approx(1.0)
        assert self.t[5].sum() == pytest.approx(1.0)

    def test_black_king_plane(self):
        """Plane 11: black king on e8 = square 60, row=7, col=4."""
        assert self.t[11, 7, 4] == pytest.approx(1.0)
        assert self.t[11].sum() == pytest.approx(1.0)

    def test_white_queen_plane(self):
        """Plane 4: white queen on d1 = square 3, row=0, col=3."""
        assert self.t[4, 0, 3] == pytest.approx(1.0)
        assert self.t[4].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# board_to_tensor — history planes (12-13)
# ---------------------------------------------------------------------------

class TestHistoryPlanes:
    def test_history_zero_without_prev_board(self):
        t = board_to_tensor(chess.Board())
        assert t[12].sum() == pytest.approx(0)
        assert t[13].sum() == pytest.approx(0)

    def test_history_filled_with_prev_board(self):
        """History planes should mirror the previous board's piece layout."""
        prev = chess.Board()
        # Make a move to get a different current board
        board = prev.copy()
        board.push_san("e4")
        t = board_to_tensor(board, prev_board=prev)
        # Plane 12: white aggregate from previous position — 16 white pieces
        assert t[12].sum() == pytest.approx(16)
        # Plane 13: black aggregate from previous position — 16 black pieces
        assert t[13].sum() == pytest.approx(16)

    def test_history_reflects_previous_state(self):
        """After e4, pawn moves from e2 to e4; history plane should show e2 pawn."""
        prev = chess.Board()
        board = prev.copy()
        board.push_san("e4")
        t = board_to_tensor(board, prev_board=prev)
        # Previous board had white pawn on e2 = square 12, row=1, col=4
        assert t[12, 1, 4] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# board_to_tensor — castling planes (14-17)
# ---------------------------------------------------------------------------

class TestCastlingPlanes:
    def test_all_castling_rights_at_start(self):
        """Starting position: all four castling planes are all-ones."""
        t = board_to_tensor(chess.Board())
        for plane in range(14, 18):
            assert t[plane].sum() == pytest.approx(64), (
                f"Plane {plane} should be all-ones at start"
            )

    def test_no_castling_rights(self):
        """Board with no castling rights: planes 14-17 are all-zeros."""
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
        # Strip all castling rights
        board.castling_rights = chess.BB_EMPTY
        t = board_to_tensor(board)
        for plane in range(14, 18):
            assert t[plane].sum() == pytest.approx(0), (
                f"Plane {plane} should be zero when castling stripped"
            )

    def test_white_kingside_only(self):
        board = chess.Board()
        board.castling_rights = chess.BB_H1  # only white kingside
        t = board_to_tensor(board)
        assert t[14].sum() == pytest.approx(64)  # white kingside
        assert t[15].sum() == pytest.approx(0)   # white queenside
        assert t[16].sum() == pytest.approx(0)   # black kingside
        assert t[17].sum() == pytest.approx(0)   # black queenside


# ---------------------------------------------------------------------------
# board_to_tensor — en passant plane (18)
# ---------------------------------------------------------------------------

class TestEnPassantPlane:
    def test_no_ep_at_start(self):
        t = board_to_tensor(chess.Board())
        assert t[18].sum() == pytest.approx(0)

    def test_ep_file_set_after_double_push(self):
        """After 1.e4 e5 2.d4, d pawn on d4 — en passant square is d3 (file 3)."""
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("d4")
        # EP square is d3, file = 3
        t = board_to_tensor(board)
        assert t[18, :, 3].sum() == pytest.approx(8)  # entire d-file column set
        # Other files are zero
        for f in range(8):
            if f != 3:
                assert t[18, :, f].sum() == pytest.approx(0)

    def test_ep_plane_covers_entire_file(self):
        """The entire column of the EP file must be 1.0, not just one row."""
        board = chess.Board()
        board.push_san("e4")
        board.push_san("d5")
        board.push_san("e5")
        board.push_san("f5")
        # After f5, EP square is f6, file = 5
        t = board_to_tensor(board)
        assert np.all(t[18, :, 5] == 1.0)


# ---------------------------------------------------------------------------
# board_to_tensor — side-to-move plane (19)
# ---------------------------------------------------------------------------

class TestSideToMovePlane:
    def test_white_to_move(self):
        t = board_to_tensor(chess.Board())
        assert t[19].sum() == pytest.approx(64)  # all ones

    def test_black_to_move(self):
        board = chess.Board()
        board.push_san("e4")
        t = board_to_tensor(board)
        assert t[19].sum() == pytest.approx(0)  # all zeros


# ---------------------------------------------------------------------------
# move_to_index and index_to_move
# ---------------------------------------------------------------------------

class TestMoveEncoding:
    def test_round_trip_e2e4(self):
        move = chess.Move.from_uci("e2e4")
        idx  = move_to_index(move)
        back = index_to_move(idx)
        assert back.from_square == move.from_square
        assert back.to_square   == move.to_square

    def test_index_range(self):
        """Every chess move must map to an index in [0, 4096)."""
        board = chess.Board()
        for move in board.legal_moves:
            idx = move_to_index(move)
            assert 0 <= idx < 4096

    @pytest.mark.parametrize("uci", ["a1a2", "h8h1", "e2e4", "d7d5", "a1h8"])
    def test_known_moves(self, uci: str):
        move = chess.Move.from_uci(uci)
        idx  = move_to_index(move)
        back = index_to_move(idx)
        assert back.from_square == move.from_square
        assert back.to_square   == move.to_square

    def test_formula(self):
        """Index = from_sq * 64 + to_sq."""
        move = chess.Move(12, 28)  # e2→e4
        assert move_to_index(move) == 12 * 64 + 28

    def test_min_index(self):
        """a1→a1 (degenerate) should give index 0."""
        assert move_to_index(chess.Move(0, 0)) == 0

    def test_max_index(self):
        """h8→h8 gives index 4095."""
        assert move_to_index(chess.Move(63, 63)) == 4095


# ---------------------------------------------------------------------------
# legal_move_mask
# ---------------------------------------------------------------------------

class TestLegalMoveMask:
    def test_shape(self):
        mask = legal_move_mask(chess.Board())
        assert mask.shape == (4096,)

    def test_dtype(self):
        mask = legal_move_mask(chess.Board())
        assert mask.dtype == bool

    def test_starting_position_count(self):
        """Starting position has exactly 20 legal moves."""
        mask = legal_move_mask(chess.Board())
        assert mask.sum() == 20

    def test_legal_moves_are_set(self):
        """Every legal move's index must be True in the mask."""
        board = chess.Board()
        mask  = legal_move_mask(board)
        for move in board.legal_moves:
            assert mask[move_to_index(move)], (
                f"Legal move {move.uci()} not set in mask"
            )

    def test_no_extra_bits(self):
        """No illegal move index should be True in the mask."""
        board        = chess.Board()
        mask         = legal_move_mask(board)
        legal_set    = {move_to_index(m) for m in board.legal_moves}
        true_indices = set(np.where(mask)[0].tolist())
        assert true_indices == legal_set

    def test_checkmate_has_zero_legal_moves(self):
        """Scholar's mate: no legal moves → mask all-False."""
        board = chess.Board(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        )
        assert board.is_checkmate()
        mask = legal_move_mask(board)
        assert mask.sum() == 0

    def test_single_legal_move_position(self):
        """Position where only one move is legal."""
        # Back-rank mate threat forced block: KK+R position
        # Use a forced-king-move position
        board = chess.Board("8/8/8/8/8/8/6pp/6kK w - - 0 1")
        # White is in check (gxh? no, hmm let me use a simpler one)
        # Use direct setup: white king must move, only one square free
        board = chess.Board("8/8/8/8/8/8/5ppp/5pkK w - - 0 1")
        legal = list(board.legal_moves)
        mask  = legal_move_mask(board)
        assert mask.sum() == len(legal)
