"""Tests for purepychess.engine (UCI parsing helpers)."""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock, patch

import chess
import pytest

from purepychess.engine import _parse_go_nodes, _parse_position


# ---------------------------------------------------------------------------
# _parse_position — startpos
# ---------------------------------------------------------------------------

class TestParsePositionStartpos:
    def test_startpos_no_moves(self):
        board, prev, move_num = _parse_position("position startpos")
        assert board.fen() == chess.Board().fen()
        assert prev is None
        assert move_num == 0

    def test_startpos_one_move(self):
        board, prev, move_num = _parse_position("position startpos moves e2e4")
        assert move_num == 1
        assert prev is not None
        # prev should be the starting position
        assert prev.fen() == chess.Board().fen()
        # board should have e4 pawn
        assert board.piece_at(chess.E4) is not None
        assert board.piece_at(chess.E4).piece_type == chess.PAWN

    def test_startpos_two_moves(self):
        board, prev, move_num = _parse_position(
            "position startpos moves e2e4 e7e5"
        )
        assert move_num == 2
        assert board.piece_at(chess.E5) is not None

    def test_startpos_four_moves(self):
        board, prev, move_num = _parse_position(
            "position startpos moves e2e4 e7e5 g1f3 b8c6"
        )
        assert move_num == 4
        assert board.turn == chess.WHITE  # after 4 half-moves, white to move

    def test_prev_board_is_second_to_last_state(self):
        """prev_board should be the position *before* the last move."""
        board, prev, _ = _parse_position(
            "position startpos moves e2e4 e7e5"
        )
        # prev is the position after e2e4 (before e7e5)
        assert prev.piece_at(chess.E4) is not None  # e4 pawn present
        assert prev.piece_at(chess.E5) is None      # e5 pawn not yet played


# ---------------------------------------------------------------------------
# _parse_position — fen
# ---------------------------------------------------------------------------

class TestParsePositionFen:
    def test_fen_only(self):
        fen = "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"
        board, prev, move_num = _parse_position(f"position fen {fen}")
        assert board.fen() == fen
        assert prev is None
        assert move_num == 0

    def test_fen_with_moves(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        board, prev, move_num = _parse_position(
            f"position fen {fen} moves e7e5"
        )
        assert move_num == 1
        assert board.piece_at(chess.E5) is not None

    def test_startpos_board_is_same_as_chess_board(self):
        board, _, _ = _parse_position("position startpos")
        expected = chess.Board()
        assert board.fen() == expected.fen()

    def test_fen_preserves_castling_rights(self):
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        board, _, _ = _parse_position(f"position fen {fen}")
        assert board.has_kingside_castling_rights(chess.WHITE)
        assert board.has_queenside_castling_rights(chess.WHITE)
        assert board.has_kingside_castling_rights(chess.BLACK)
        assert board.has_queenside_castling_rights(chess.BLACK)

    def test_fen_preserves_en_passant(self):
        # After 1.e4 e5 2.d4 — EP square is d3
        fen = "rnbqkbnr/ppp1pppp/8/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3"
        board, _, _ = _parse_position(f"position fen {fen}")
        assert board.ep_square == chess.D6


# ---------------------------------------------------------------------------
# _parse_go_nodes
# ---------------------------------------------------------------------------

class TestParseGoNodes:
    def test_nodes_present(self):
        assert _parse_go_nodes("go nodes 100") == 100

    def test_nodes_large(self):
        assert _parse_go_nodes("go nodes 999999") == 999_999

    def test_movetime_only_returns_none(self):
        assert _parse_go_nodes("go movetime 1000") is None

    def test_bare_go_returns_none(self):
        assert _parse_go_nodes("go") is None

    def test_nodes_with_extra_params(self):
        assert _parse_go_nodes("go movetime 5000 nodes 50") == 50

    def test_nodes_value_is_int(self):
        result = _parse_go_nodes("go nodes 42")
        assert isinstance(result, int)
        assert result == 42

    def test_empty_string_returns_none(self):
        assert _parse_go_nodes("") is None


# ---------------------------------------------------------------------------
# Full UCI I/O loop (smoke test via mocked stdin/stdout)
# ---------------------------------------------------------------------------

class TestUCILoop:
    """Verify the main() loop responds correctly to key commands.

    The actual network is mocked to avoid requiring weights on disk.
    """

    def _run_uci(self, commands: list[str]) -> list[str]:
        """Run main() feeding *commands* as stdin and capture stdout lines."""
        import purepychess.engine as eng_mod

        fake_mcts = MagicMock()
        fake_mcts.get_best_move.return_value = chess.Move.from_uci("e2e4")
        fake_mcts.simulations = 400

        captured: list[str] = []

        def fake_print(*args, **kwargs):
            captured.append(" ".join(str(a) for a in args))

        stdin_text = "\n".join(commands) + "\n"

        with (
            patch.object(eng_mod, "ChessNet", return_value=MagicMock()),
            patch.object(eng_mod, "load_weights"),
            patch.object(eng_mod, "MCTS", return_value=fake_mcts),
            patch("sys.stdin", io.StringIO(stdin_text)),
            patch("builtins.print", side_effect=fake_print),
        ):
            eng_mod.main()

        return captured

    def test_uci_command(self):
        output = self._run_uci(["uci", "quit"])
        assert any("uciok" in line for line in output)
        assert any("id name" in line for line in output)

    def test_isready_command(self):
        output = self._run_uci(["isready", "quit"])
        assert any("readyok" in line for line in output)

    def test_go_returns_bestmove(self):
        output = self._run_uci([
            "position startpos",
            "go nodes 10",
            "quit",
        ])
        assert any(line.startswith("bestmove") for line in output)

    def test_ucinewgame_does_not_crash(self):
        output = self._run_uci(["ucinewgame", "isready", "quit"])
        assert any("readyok" in line for line in output)

    def test_stop_does_not_crash(self):
        output = self._run_uci(["stop", "quit"])
        assert isinstance(output, list)  # just no exception

    def test_position_then_go(self):
        output = self._run_uci([
            "position startpos moves e2e4",
            "go nodes 5",
            "quit",
        ])
        assert any(line.startswith("bestmove") for line in output)

    def test_full_sequence(self):
        """Simulate a typical arena / GUI handshake."""
        output = self._run_uci([
            "uci",
            "isready",
            "ucinewgame",
            "position startpos",
            "go nodes 100",
            "quit",
        ])
        lines = "\n".join(output)
        assert "uciok"    in lines
        assert "readyok"  in lines
        assert "bestmove" in lines
