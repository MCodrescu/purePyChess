"""ResNet policy + value network for purePyChess.

Architecture
------------
Input  : (batch, 20, 8, 8) float32 tensor
Stem   : Conv2d(20 → 128, 3×3, pad=1) + BN + ReLU
Body   : 10 residual blocks (128 filters, 3×3 convolutions)
Policy : Conv(128→2, 1×1) → BN → ReLU → Flatten(128) → FC(4096)  — raw logits
Value  : Conv(128→1, 1×1) → BN → ReLU → Flatten(64) → FC(256) → ReLU → FC(3)  — raw WDL logits

Loss (training)
---------------
  nn.CrossEntropyLoss(policy_logits, move_played)
+ nn.CrossEntropyLoss(wdl_logits,    wdl_target)

Do NOT apply softmax before the loss; CrossEntropyLoss applies log_softmax internally.

Scalar for MCTS
---------------
  softmax(wdl_logits)  →  p = [p_win, p_draw, p_loss]
  value = p_win - p_loss   (white's perspective)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """One residual block: Conv→BN→ReLU→Conv→BN→skip→ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class ChessNet(nn.Module):
    """ResNet policy + value head for chess positions."""

    def __init__(
        self,
        in_channels: int = 20,
        filters: int = 128,
        num_blocks: int = 10,
    ) -> None:
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )

        # Body
        self.body = nn.Sequential(*[_ResBlock(filters) for _ in range(num_blocks)])

        # Policy head: 2 channels × 8×8 = 128 features → 4096 logits
        self.policy_conv = nn.Conv2d(filters, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc   = nn.Linear(2 * 8 * 8, 4096)

        # Value head: 1 channel × 8×8 = 64 features → 3 WDL logits
        self.value_conv  = nn.Conv2d(filters, 1, 1, bias=False)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_relu  = nn.ReLU(inplace=True)
        self.value_fc1   = nn.Linear(1 * 8 * 8, 256)
        self.value_relu2 = nn.ReLU(inplace=True)
        self.value_fc2   = nn.Linear(256, 3)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits, value_logits)`` of shapes (B, 4096) and (B, 3)."""
        x = self.stem(x)
        x = self.body(x)

        # Policy head
        p = self.policy_relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = self.value_relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = self.value_relu2(self.value_fc1(v))
        value_logits = self.value_fc2(v)

        return policy_logits, value_logits
