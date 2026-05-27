"""Shared pytest fixtures for purePyChess tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Stub neural network that returns deterministic uniform outputs
# ---------------------------------------------------------------------------

class UniformStubModel(nn.Module):
    """Returns zeros for both policy logits and value logits.

    Zeros → softmax gives uniform distribution.
    Value scalar: softmax([0,0,0]) = [1/3, 1/3, 1/3] → p_win - p_loss = 0.
    """

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        return torch.zeros(b, 4096), torch.zeros(b, 3)


@pytest.fixture()
def stub_model() -> UniformStubModel:
    """A tiny stub model usable in place of ChessNet."""
    m = UniformStubModel()
    m.eval()
    return m


@pytest.fixture()
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def dummy_state() -> np.ndarray:
    """A (20, 8, 8) float32 zero array standing in for a board tensor."""
    return np.zeros((20, 8, 8), dtype=np.float32)


@pytest.fixture()
def dummy_sparse_policy() -> dict[int, int]:
    """A small sparse policy dict: two moves visited."""
    return {0: 10, 64: 5}
