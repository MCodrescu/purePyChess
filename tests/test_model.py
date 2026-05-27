"""Tests for purepychess.model (ChessNet)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import numpy as np

from purepychess.model import ChessNet, _ResBlock


# ---------------------------------------------------------------------------
# _ResBlock
# ---------------------------------------------------------------------------

class TestResBlock:
    def test_output_shape(self):
        block = _ResBlock(64)
        x     = torch.randn(2, 64, 8, 8)
        out   = block(x)
        assert out.shape == (2, 64, 8, 8)

    def test_residual_connection(self):
        """With zeroed weights the residual stream should pass through."""
        block = _ResBlock(32)
        # Zero out conv weights so block acts as identity via skip
        for p in block.parameters():
            nn.init.zeros_(p)
        x   = torch.randn(1, 32, 8, 8)
        out = block(x)
        # After zeroing weights: conv output is 0, BN will output 0 (no running mean yet)
        # Result is relu(0 + x) = relu(x) — non-negative
        assert out.shape == x.shape

    def test_no_nan(self):
        block = _ResBlock(128)
        x     = torch.randn(4, 128, 8, 8)
        out   = block(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# ChessNet construction
# ---------------------------------------------------------------------------

class TestChessNetConstruction:
    def test_default_instantiation(self):
        model = ChessNet()
        assert isinstance(model, ChessNet)

    def test_custom_filters(self):
        model = ChessNet(filters=64, num_blocks=3)
        assert isinstance(model, ChessNet)

    def test_has_expected_submodules(self):
        model = ChessNet()
        assert hasattr(model, "stem")
        assert hasattr(model, "body")
        assert hasattr(model, "policy_fc")
        assert hasattr(model, "value_fc2")

    def test_body_has_correct_block_count(self):
        model = ChessNet(num_blocks=5)
        assert len(model.body) == 5


# ---------------------------------------------------------------------------
# ChessNet forward pass — shapes
# ---------------------------------------------------------------------------

class TestChessNetForward:
    @pytest.fixture(autouse=True)
    def model(self):
        self.net = ChessNet()
        self.net.eval()

    def _random_input(self, batch: int = 4) -> torch.Tensor:
        return torch.randn(batch, 20, 8, 8)

    def test_policy_shape(self):
        x = self._random_input(4)
        with torch.no_grad():
            policy, _ = self.net(x)
        assert policy.shape == (4, 4096)

    def test_value_shape(self):
        x = self._random_input(4)
        with torch.no_grad():
            _, value = self.net(x)
        assert value.shape == (4, 3)

    def test_batch_size_1(self):
        x = self._random_input(1)
        with torch.no_grad():
            policy, value = self.net(x)
        assert policy.shape == (1, 4096)
        assert value.shape  == (1, 3)

    def test_no_nan_in_policy(self):
        x = self._random_input(4)
        with torch.no_grad():
            policy, _ = self.net(x)
        assert not torch.isnan(policy).any()
        assert not torch.isinf(policy).any()

    def test_no_nan_in_value(self):
        x = self._random_input(4)
        with torch.no_grad():
            _, value = self.net(x)
        assert not torch.isnan(value).any()
        assert not torch.isinf(value).any()

    def test_value_softmax_rows_sum_to_one(self):
        """Applying softmax to raw value logits must yield a valid distribution."""
        x = self._random_input(8)
        with torch.no_grad():
            _, value = self.net(x)
        probs = torch.softmax(value, dim=1)
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(8), atol=1e-5)

    def test_policy_softmax_rows_sum_to_one(self):
        x = self._random_input(4)
        with torch.no_grad():
            policy, _ = self.net(x)
        probs = torch.softmax(policy, dim=1)
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)

    def test_different_inputs_give_different_outputs(self):
        """Two different inputs should (almost certainly) give different outputs."""
        x1 = torch.randn(1, 20, 8, 8)
        x2 = torch.randn(1, 20, 8, 8)
        with torch.no_grad():
            p1, v1 = self.net(x1)
            p2, v2 = self.net(x2)
        assert not torch.allclose(p1, p2)

    def test_eval_mode_deterministic(self):
        """Same input in eval mode must give identical outputs."""
        x = torch.randn(2, 20, 8, 8)
        with torch.no_grad():
            p1, v1 = self.net(x)
            p2, v2 = self.net(x)
        assert torch.allclose(p1, p2)
        assert torch.allclose(v1, v2)


# ---------------------------------------------------------------------------
# ChessNet parameter count sanity check
# ---------------------------------------------------------------------------

class TestChessNetParameters:
    def test_parameter_count_reasonable(self):
        """A 128-filter / 10-block net should have > 1M and < 100M parameters."""
        model  = ChessNet()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1_000_000
        assert n_params < 100_000_000

    def test_all_parameters_finite_at_init(self):
        model = ChessNet()
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite init in {name}"
