"""Replay buffer for Phase 2 self-play fine-tuning.

Positions are stored with *sparse* policy dicts to avoid the ~10 GB RAM
cost of dense 4096-float vectors.  Dense tensors are materialised on-the-fly
inside :meth:`ReplayBuffer.sample`.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch

from purepychess.train.config import REPLAY_BUFFER_SIZE


class ReplayBuffer:
    """Fixed-capacity replay buffer storing (state, sparse_policy, outcome) tuples.

    Parameters
    ----------
    maxlen:
        Maximum number of positions to retain.  Oldest entries are evicted
        when the buffer is full.
    """

    def __init__(self, maxlen: int = REPLAY_BUFFER_SIZE) -> None:
        self._buf: deque[tuple[np.ndarray, dict[int, int], int]] = deque(
            maxlen=maxlen
        )

    # ------------------------------------------------------------------
    # Adding data
    # ------------------------------------------------------------------

    def add(
        self,
        state: np.ndarray,
        sparse_policy: dict[int, int],
        outcome: int,
    ) -> None:
        """Add a single position.

        Parameters
        ----------
        state:
            (20, 8, 8) float32 board tensor.
        sparse_policy:
            ``{move_index: visit_count}`` — only visited moves are stored.
        outcome:
            0 = white win, 1 = draw, 2 = black win (white-perspective WDL).
        """
        self._buf.append((state, sparse_policy, outcome))

    def add_game(self, game_record: list[tuple[np.ndarray, dict[int, int], int]]) -> None:
        """Add all positions from a completed game record."""
        for state, sparse_policy, outcome in game_record:
            self.add(state, sparse_policy, outcome)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random mini-batch.

        Returns
        -------
        states:
            ``(batch_size, 20, 8, 8)`` float32 tensor.
        policies:
            ``(batch_size, 4096)`` float32 tensor (dense, normalised to sum=1).
        outcomes:
            ``(batch_size,)`` long tensor with values in {0, 1, 2}.
        """
        if len(self._buf) < batch_size:
            raise ValueError(
                f"Buffer has only {len(self._buf)} entries, need {batch_size}"
            )
        samples = random.sample(self._buf, batch_size)
        states_list, sparse_list, outcomes_list = zip(*samples)

        state_batch   = torch.tensor(np.stack(states_list), dtype=torch.float32)
        outcome_batch = torch.tensor(outcomes_list, dtype=torch.long)

        # Convert sparse dicts → dense normalised vectors
        policy_batch = torch.zeros(batch_size, 4096, dtype=torch.float32)
        for i, sp in enumerate(sparse_list):
            total = sum(sp.values())
            if total > 0:
                for move_idx, count in sp.items():
                    policy_batch[i, move_idx] = count / total

        return state_batch, policy_batch, outcome_batch

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buf)
