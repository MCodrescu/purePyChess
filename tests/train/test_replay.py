"""Tests for purepychess.train.replay (ReplayBuffer)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from purepychess.train.replay import ReplayBuffer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_state() -> np.ndarray:
    return np.zeros((20, 8, 8), dtype=np.float32)


def _make_sparse_policy(**visit_counts: int) -> dict[int, int]:
    """Create a sparse policy dict.  Keys are move indices, values are counts."""
    return dict(visit_counts)


def _fill_buffer(buf: ReplayBuffer, n: int, outcome: int = 0) -> None:
    for i in range(n):
        buf.add(
            _make_state(),
            {i % 4096: 10, (i + 1) % 4096: 5},
            outcome,
        )


# ---------------------------------------------------------------------------
# Basic add / len
# ---------------------------------------------------------------------------

class TestReplayBufferAdd:
    def test_empty_at_construction(self):
        buf = ReplayBuffer(maxlen=100)
        assert len(buf) == 0

    def test_add_single_entry(self):
        buf = ReplayBuffer(maxlen=100)
        buf.add(_make_state(), {0: 1}, 0)
        assert len(buf) == 1

    def test_add_multiple_entries(self):
        buf = ReplayBuffer(maxlen=100)
        for _ in range(10):
            buf.add(_make_state(), {0: 1}, 1)
        assert len(buf) == 10

    def test_len_does_not_exceed_maxlen(self):
        buf = ReplayBuffer(maxlen=5)
        _fill_buffer(buf, 10)
        assert len(buf) == 5

    def test_oldest_entry_evicted(self):
        """After filling past capacity, oldest entries are gone."""
        buf = ReplayBuffer(maxlen=3)
        for i in range(5):
            buf.add(_make_state(), {i: 1}, i)
        # Buffer should contain entries with outcome 2, 3, 4 (newest 3)
        outcomes = [entry[2] for entry in buf._buf]
        assert outcomes == [2, 3, 4]


# ---------------------------------------------------------------------------
# add_game
# ---------------------------------------------------------------------------

class TestAddGame:
    def test_add_game_adds_all_positions(self):
        buf  = ReplayBuffer(maxlen=100)
        game = [(_make_state(), {i: 1}, 0) for i in range(10)]
        buf.add_game(game)
        assert len(buf) == 10

    def test_add_empty_game(self):
        buf = ReplayBuffer(maxlen=100)
        buf.add_game([])
        assert len(buf) == 0

    def test_add_game_preserves_outcome(self):
        buf  = ReplayBuffer(maxlen=100)
        game = [(_make_state(), {0: 1}, 2)]  # black win
        buf.add_game(game)
        assert buf._buf[0][2] == 2


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

class TestReplayBufferSample:
    def test_sample_raises_when_underfull(self):
        buf = ReplayBuffer(maxlen=100)
        _fill_buffer(buf, 5)
        with pytest.raises(ValueError, match="Buffer has only"):
            buf.sample(10)

    def test_sample_returns_correct_batch_size(self):
        buf = ReplayBuffer(maxlen=1000)
        _fill_buffer(buf, 100)
        states, policies, outcomes = buf.sample(32)
        assert states.shape   == (32, 20, 8, 8)
        assert policies.shape == (32, 4096)
        assert outcomes.shape == (32,)

    def test_states_tensor_dtype(self):
        buf = ReplayBuffer(maxlen=100)
        _fill_buffer(buf, 20)
        states, _, _ = buf.sample(10)
        assert states.dtype == torch.float32

    def test_policies_tensor_dtype(self):
        buf = ReplayBuffer(maxlen=100)
        _fill_buffer(buf, 20)
        _, policies, _ = buf.sample(10)
        assert policies.dtype == torch.float32

    def test_outcomes_tensor_dtype(self):
        buf = ReplayBuffer(maxlen=100)
        _fill_buffer(buf, 20)
        _, _, outcomes = buf.sample(10)
        assert outcomes.dtype == torch.long

    def test_policy_rows_sum_to_one_or_zero(self):
        """Each policy row should either sum to ~1 (non-empty sparse) or 0."""
        buf = ReplayBuffer(maxlen=100)
        _fill_buffer(buf, 50)
        _, policies, _ = buf.sample(20)
        row_sums = policies.sum(dim=1)
        for s in row_sums:
            assert s.item() == pytest.approx(1.0, abs=1e-5) or s.item() == pytest.approx(0.0)

    def test_outcomes_in_valid_range(self):
        buf = ReplayBuffer(maxlen=100)
        for outcome in [0, 1, 2]:
            buf.add(_make_state(), {0: 1}, outcome)
        # Need at least 3 entries; fill more
        _fill_buffer(buf, 50, outcome=1)
        _, _, outcomes = buf.sample(20)
        assert outcomes.min().item() >= 0
        assert outcomes.max().item() <= 2

    def test_dense_policy_normalised(self):
        """Sparse policy {idx: 10, idx2: 10} → dense should give equal probs."""
        buf = ReplayBuffer(maxlen=100)
        idx_a, idx_b = 100, 200
        for _ in range(20):
            buf.add(_make_state(), {idx_a: 10, idx_b: 10}, 0)
        _, policies, _ = buf.sample(10)
        # Each sampled row should have policies[row, idx_a] == policies[row, idx_b] == 0.5
        for row in policies:
            assert row[idx_a].item() == pytest.approx(0.5, abs=1e-5)
            assert row[idx_b].item() == pytest.approx(0.5, abs=1e-5)

    def test_empty_sparse_policy_gives_zero_row(self):
        """An empty sparse policy should produce a zero policy row."""
        buf = ReplayBuffer(maxlen=100)
        for _ in range(20):
            buf.add(_make_state(), {}, 0)
        _, policies, _ = buf.sample(10)
        assert (policies == 0.0).all()

    def test_sample_is_random(self):
        """Two samples from the same buffer should not always be identical."""
        buf = ReplayBuffer(maxlen=1000)
        for i in range(200):
            buf.add(_make_state(), {i % 4096: i + 1}, i % 3)
        _, p1, o1 = buf.sample(50)
        _, p2, o2 = buf.sample(50)
        # Extremely unlikely both are identical with 200 diverse entries
        assert not torch.equal(o1, o2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestReplayBufferEdgeCases:
    def test_sample_exactly_full_buffer(self):
        buf = ReplayBuffer(maxlen=10)
        _fill_buffer(buf, 10)
        states, policies, outcomes = buf.sample(10)
        assert len(states) == 10

    def test_maxlen_one(self):
        buf = ReplayBuffer(maxlen=1)
        buf.add(_make_state(), {0: 1}, 0)
        buf.add(_make_state(), {1: 1}, 1)  # evicts first
        assert len(buf) == 1
        states, _, outcomes = buf.sample(1)
        assert outcomes[0].item() == 1  # only the newest entry remains
