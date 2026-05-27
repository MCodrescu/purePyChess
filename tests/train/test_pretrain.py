"""Tests for purepychess.train.pretrain (ShardDataset and training utilities)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from purepychess.train.pretrain import ShardDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_shard(path: Path, n: int = 50) -> Path:
    """Write a minimal npz shard with *n* random positions."""
    shard = path / "shard_0000.npz"
    rng   = np.random.default_rng(seed=42)
    np.savez_compressed(
        shard,
        state=rng.random((n, 20, 8, 8), dtype=None).astype(np.float16),
        move=rng.integers(0, 4096, size=n, dtype=np.int16),
        wdl=rng.integers(0, 3, size=n, dtype=np.uint8),
    )
    return shard


@pytest.fixture()
def shard_file(tmp_path) -> Path:
    return _write_shard(tmp_path)


# ---------------------------------------------------------------------------
# ShardDataset
# ---------------------------------------------------------------------------

class TestShardDataset:
    def test_len(self, shard_file):
        ds = ShardDataset(str(shard_file))
        assert len(ds) == 50

    def test_getitem_state_shape(self, shard_file):
        ds    = ShardDataset(str(shard_file))
        state, move, wdl = ds[0]
        assert state.shape == (20, 8, 8)

    def test_getitem_state_dtype(self, shard_file):
        """States are cast to float32 inside ShardDataset."""
        ds    = ShardDataset(str(shard_file))
        state, _, _ = ds[0]
        assert state.dtype == np.float32

    def test_getitem_move_dtype(self, shard_file):
        ds = ShardDataset(str(shard_file))
        _, move, _ = ds[0]
        assert move.dtype == np.int64

    def test_getitem_wdl_dtype(self, shard_file):
        ds = ShardDataset(str(shard_file))
        _, _, wdl = ds[0]
        assert wdl.dtype == np.int64

    def test_getitem_move_range(self, shard_file):
        ds = ShardDataset(str(shard_file))
        for i in range(len(ds)):
            _, move, _ = ds[i]
            assert 0 <= int(move) < 4096

    def test_getitem_wdl_range(self, shard_file):
        ds = ShardDataset(str(shard_file))
        for i in range(len(ds)):
            _, _, wdl = ds[i]
            assert int(wdl) in {0, 1, 2}

    def test_indexing_last_element(self, shard_file):
        ds = ShardDataset(str(shard_file))
        state, move, wdl = ds[len(ds) - 1]
        assert state.shape == (20, 8, 8)

    def test_all_states_finite(self, shard_file):
        ds = ShardDataset(str(shard_file))
        for i in range(len(ds)):
            state, _, _ = ds[i]
            assert np.isfinite(state).all()

    def test_dataset_usable_with_dataloader(self, shard_file):
        """ShardDataset should work inside a DataLoader without error."""
        from torch.utils.data import DataLoader
        ds     = ShardDataset(str(shard_file))
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        batch  = next(iter(loader))
        states, moves, wdls = batch
        assert states.shape == (8, 20, 8, 8)
        assert moves.shape  == (8,)
        assert wdls.shape   == (8,)

    def test_dataloader_state_dtype(self, shard_file):
        from torch.utils.data import DataLoader
        ds     = ShardDataset(str(shard_file))
        loader = DataLoader(ds, batch_size=8)
        states, _, _ = next(iter(loader))
        assert states.dtype == torch.float32

    def test_different_indices_give_different_data(self, shard_file):
        """Two different indices should (generally) return different states."""
        ds = ShardDataset(str(shard_file))
        s0, _, _ = ds[0]
        s1, _, _ = ds[1]
        # With random data, this almost certainly differs
        assert not np.array_equal(s0, s1)


# ---------------------------------------------------------------------------
# pretrain function (smoke test with tiny data)
# ---------------------------------------------------------------------------

class TestPretrainSmoke:
    def test_pretrain_runs_without_error(self, tmp_path):
        """pretrain() should complete and write a weights.pt file."""
        from purepychess.train.pretrain import pretrain

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_shard(shard_dir, n=32)

        out_dir = tmp_path / "out"
        pretrain(
            str(shard_dir),
            str(out_dir),
            epochs=1,
            lr=1e-3,
            batch_size=16,
        )
        assert (out_dir / "weights.pt").exists()

    def test_pretrain_weights_loadable(self, tmp_path):
        """The saved weights.pt must be loadable as a ChessNet state dict."""
        from purepychess.model import ChessNet
        from purepychess.train.pretrain import pretrain

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_shard(shard_dir, n=32)

        out_dir = tmp_path / "out"
        pretrain(str(shard_dir), str(out_dir), epochs=1, batch_size=16)

        model = ChessNet()
        state_dict = torch.load(
            out_dir / "weights.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)  # should not raise

    def test_pretrain_raises_on_missing_shards(self, tmp_path):
        from purepychess.train.pretrain import pretrain
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        out_dir   = tmp_path / "out"
        with pytest.raises(FileNotFoundError, match="No shard"):
            pretrain(str(empty_dir), str(out_dir))
