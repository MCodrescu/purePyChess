"""Tests for purepychess.train.preprocess."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path
from unittest.mock import patch

import chess.pgn
import numpy as np
import pytest

from purepychess.train.preprocess import (
    _game_passes_filter,
    _parse_time_control,
    _result_to_wdl,
    preprocess,
)


# ---------------------------------------------------------------------------
# _parse_time_control
# ---------------------------------------------------------------------------

class TestParseTimeControl:
    @pytest.mark.parametrize("tc, expected", [
        ("600+5",    600),
        ("600",      600),
        ("300+0",    300),
        ("180+2",    180),
        ("60+0",     60),
        ("0+2",      0),
        ("-",        0),
        ("",         0),
        ("garbage",  0),
    ])
    def test_various_inputs(self, tc, expected):
        assert _parse_time_control(tc) == expected


# ---------------------------------------------------------------------------
# _result_to_wdl
# ---------------------------------------------------------------------------

class TestResultToWdl:
    def test_white_win(self):
        assert _result_to_wdl("1-0") == 0

    def test_black_win(self):
        assert _result_to_wdl("0-1") == 2

    def test_draw(self):
        assert _result_to_wdl("1/2-1/2") == 1

    def test_unknown(self):
        assert _result_to_wdl("*") == 1  # treated as draw

    @pytest.mark.parametrize("result, wdl", [
        ("1-0", 0),
        ("0-1", 2),
        ("1/2-1/2", 1),
    ])
    def test_parametrized(self, result, wdl):
        assert _result_to_wdl(result) == wdl


# ---------------------------------------------------------------------------
# _game_passes_filter
# ---------------------------------------------------------------------------

def _make_game(
    rated: str = "Yes",
    white_elo: int = 2000,
    black_elo: int = 2000,
    time_control: str = "600+5",
    result: str = "1-0",
) -> chess.pgn.Game:
    """Construct a minimal chess.pgn.Game with the given headers."""
    game = chess.pgn.Game()
    game.headers["Rated"]       = rated
    game.headers["WhiteElo"]    = str(white_elo)
    game.headers["BlackElo"]    = str(black_elo)
    game.headers["TimeControl"] = time_control
    game.headers["Result"]      = result
    return game


class TestGamePassesFilter:
    def test_good_game_passes(self):
        assert _game_passes_filter(_make_game()) is True

    def test_unrated_fails(self):
        assert _game_passes_filter(_make_game(rated="No")) is False

    def test_false_rated_fails(self):
        assert _game_passes_filter(_make_game(rated="false")) is False

    def test_low_elo_fails(self):
        assert _game_passes_filter(_make_game(white_elo=1200, black_elo=1200)) is False

    def test_exactly_target_elo_passes(self):
        """Average Elo exactly at TARGET_ELO (1800) should pass."""
        assert _game_passes_filter(_make_game(white_elo=1800, black_elo=1800)) is True

    def test_just_below_target_elo_fails(self):
        assert _game_passes_filter(_make_game(white_elo=1799, black_elo=1799)) is False

    def test_short_time_control_fails(self):
        assert _game_passes_filter(_make_game(time_control="60+0")) is False

    def test_exactly_min_time_control_passes(self):
        """Time control of exactly MIN_TIME_CONTROL (180s) should pass."""
        assert _game_passes_filter(_make_game(time_control="180")) is True

    def test_bullet_fails(self):
        assert _game_passes_filter(_make_game(time_control="120+1")) is False

    def test_missing_elo_fails(self):
        game = _make_game()
        del game.headers["WhiteElo"]
        # Missing header → int("0") = 0 → average 0 < TARGET_ELO
        assert _game_passes_filter(game) is False

    def test_non_numeric_elo_fails(self):
        game = _make_game()
        game.headers["WhiteElo"] = "?"
        assert _game_passes_filter(game) is False

    def test_true_rated_header_passes(self):
        """Some exports use 'true' instead of 'Yes'."""
        assert _game_passes_filter(_make_game(rated="true")) is True


# ---------------------------------------------------------------------------
# preprocess — shard writing
# ---------------------------------------------------------------------------

_MINIMAL_PGN = textwrap.dedent("""\
    [Event "Rated Blitz game"]
    [Rated "Yes"]
    [WhiteElo "2100"]
    [BlackElo "2000"]
    [TimeControl "300+3"]
    [Result "1-0"]

    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6
    8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7
    14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6
    20. Nbd2 Nxd6 21. Nc4 Nxc4 22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7
    25. Nxf7 Rxe1+ 26. Qxe1 Kxf7 27. Qe3 Qg5 28. Qxg5 hxg5 29. b3 Ke6
    30. a3 Kd6 31. axb4 cxb4 32. Ra5 Nd5 33. f3 Bc8 34. Kf2 Bf5 35. Ra7 g6
    36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3 39. Kd2 Kb5 40. Rd6 Kc5 41. Ra6 Nf2
    42. g4 Bd3 43. Re6 1-0

    [Event "Rated Classical game"]
    [Rated "Yes"]
    [WhiteElo "1900"]
    [BlackElo "1850"]
    [TimeControl "1800+30"]
    [Result "1/2-1/2"]

    1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 e6 5. e3 a6 6. b3 Bb4 7. Bd2 Nbd7
    8. Bd3 O-O 9. O-O Bd6 10. Qc2 e5 11. cxd5 cxd5 12. dxe5 Nxe5 13. Nxe5 Bxe5
    14. f4 Bd6 15. e4 dxe4 16. Nxe4 Nxe4 17. Bxe4 Re8 18. Rac1 Bf5 19. Bxf5
    Qd2 20. Rxc8 Raxc8 21. Be6 Rxe6 22. Qxf5 Re2 23. Rf2 Rxf2 24. Kxf2 Qxd2+
    25. Kg3 Qe3+ 26. Kh4 Qe1+ 27. Kg5 Qe3+ 28. Kh4 Qe1+ 1/2-1/2
""")


class TestPreprocess:
    def test_shard_created(self, tmp_path):
        """preprocess() should create at least one shard file."""
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(_MINIMAL_PGN, encoding="utf-8")
        out_dir  = tmp_path / "shards"

        preprocess(str(pgn_file), str(out_dir))

        shards = list(out_dir.glob("shard_*.npz"))
        assert len(shards) >= 1

    def test_shard_has_correct_keys(self, tmp_path):
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(_MINIMAL_PGN, encoding="utf-8")
        out_dir  = tmp_path / "shards"

        preprocess(str(pgn_file), str(out_dir))

        shard = np.load(next(out_dir.glob("shard_*.npz")))
        assert "state" in shard
        assert "move"  in shard
        assert "wdl"   in shard

    def test_shard_state_shape(self, tmp_path):
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(_MINIMAL_PGN, encoding="utf-8")
        out_dir  = tmp_path / "shards"

        preprocess(str(pgn_file), str(out_dir))

        shard = np.load(next(out_dir.glob("shard_*.npz")))
        assert shard["state"].shape[1:] == (20, 8, 8)
        assert shard["state"].dtype == np.float16

    def test_shard_move_dtype(self, tmp_path):
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(_MINIMAL_PGN, encoding="utf-8")
        out_dir  = tmp_path / "shards"

        preprocess(str(pgn_file), str(out_dir))

        shard = np.load(next(out_dir.glob("shard_*.npz")))
        assert shard["move"].dtype == np.int16

    def test_shard_wdl_dtype_and_range(self, tmp_path):
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(_MINIMAL_PGN, encoding="utf-8")
        out_dir  = tmp_path / "shards"

        preprocess(str(pgn_file), str(out_dir))

        shard = np.load(next(out_dir.glob("shard_*.npz")))
        wdl   = shard["wdl"]
        assert wdl.dtype == np.uint8
        assert np.all((wdl >= 0) & (wdl <= 2))

    def test_move_indices_in_valid_range(self, tmp_path):
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(_MINIMAL_PGN, encoding="utf-8")
        out_dir  = tmp_path / "shards"

        preprocess(str(pgn_file), str(out_dir))

        shard = np.load(next(out_dir.glob("shard_*.npz")))
        moves = shard["move"].astype(np.int32)
        assert np.all(moves >= 0)
        assert np.all(moves < 4096)

    def test_low_elo_game_filtered_out(self, tmp_path):
        """A game below TARGET_ELO should not generate any positions."""
        low_elo_pgn = textwrap.dedent("""\
            [Event "Rated Blitz game"]
            [Rated "Yes"]
            [WhiteElo "1000"]
            [BlackElo "1000"]
            [TimeControl "300+3"]
            [Result "1-0"]

            1. e4 e5 2. Nf3 Nc6 1-0
        """)
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(low_elo_pgn, encoding="utf-8")
        out_dir = tmp_path / "shards"

        preprocess(str(pgn_file), str(out_dir))

        shards = list(out_dir.glob("shard_*.npz"))
        # No positions → no shards written
        assert len(shards) == 0

    def test_unrated_game_filtered_out(self, tmp_path):
        unrated_pgn = textwrap.dedent("""\
            [Event "Casual game"]
            [Rated "No"]
            [WhiteElo "2200"]
            [BlackElo "2200"]
            [TimeControl "600+0"]
            [Result "1-0"]

            1. e4 e5 1-0
        """)
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(unrated_pgn, encoding="utf-8")
        out_dir = tmp_path / "shards"

        preprocess(str(pgn_file), str(out_dir))

        assert len(list(out_dir.glob("shard_*.npz"))) == 0

    def test_consistent_position_count(self, tmp_path):
        """Number of recorded positions matches the total moves in passing games."""
        # Use a single short game with known move count: 3 half-moves
        short_pgn = textwrap.dedent("""\
            [Event "Rated Classical game"]
            [Rated "Yes"]
            [WhiteElo "2000"]
            [BlackElo "2000"]
            [TimeControl "600+5"]
            [Result "1-0"]

            1. e4 e5 2. Qh5 1-0
        """)
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(short_pgn, encoding="utf-8")
        out_dir  = tmp_path / "shards"

        preprocess(str(pgn_file), str(out_dir))

        shards = list(out_dir.glob("shard_*.npz"))
        assert len(shards) == 1
        shard = np.load(shards[0])
        # 3 moves → 3 positions
        assert len(shard["move"]) == 3
