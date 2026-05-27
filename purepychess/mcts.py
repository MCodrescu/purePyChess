"""Leaf-batched MCTS with transposition table for purePyChess.

Algorithm overview
------------------
1. Selection  — UCB-PUCT walk from root to an unexpanded or terminal leaf.
2. Expansion  — legal moves initialised with network policy priors.
3. Evaluation — leaves are batched (BATCH_LEAF_SIZE) into a single GPU call.
4. Backprop   — visit counts and Q values updated along each leaf's path.

Virtual loss is applied to leaves entering the batch so that concurrent
selections during the same batch iteration are spread across different nodes.

Transposition table
-------------------
Dict: Zobrist hash → MCTSNode, capped at MAX_TABLE_SIZE entries (LRU eviction
via collections.OrderedDict).  On expansion, if the child's hash is already in
the table the existing node's N/Q/P stats are reused.

Move selection (after search)
------------------------------
  temp=1  (first 30 moves): sample proportional to N
  temp=0  (remaining moves): argmax N
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional

import chess
import chess.polyglot
import numpy as np
import torch

from purepychess.config import BATCH_LEAF_SIZE, MCTS_SIMULATIONS
from purepychess.encoding import board_to_tensor, legal_move_mask, move_to_index

C_PUCT            = 1.5
MAX_TABLE_SIZE    = 1_000_000
DIRICHLET_ALPHA   = 0.3
DIRICHLET_FRAC    = 0.25
TEMP_THRESHOLD    = 30        # move number below which temp=1

_rng = np.random.default_rng(seed=None)  # non-deterministic by design


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class MCTSNode:
    """Single node in the search tree."""

    __slots__ = ("N", "Q", "P", "children", "is_terminal", "terminal_value")

    def __init__(self) -> None:
        self.N: int   = 0
        self.Q: float = 0.0
        self.P: float = 0.0                    # prior probability from policy head
        self.children: dict[int, MCTSNode] = {}  # move_index → child node
        self.is_terminal: bool  = False
        self.terminal_value: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _result_to_value(result: str) -> float:
    """Convert a python-chess result string to a scalar from white's POV."""
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    return 0.0  # draw or unknown


def _add_dirichlet_noise(root: MCTSNode) -> None:
    """Mix Dirichlet(alpha=0.3) noise into root priors at fraction 0.25."""
    n = len(root.children)
    if n == 0:
        return
    noise = _rng.dirichlet([DIRICHLET_ALPHA] * n)
    for (child, ni) in zip(root.children.values(), noise):
        child.P = (1.0 - DIRICHLET_FRAC) * child.P + DIRICHLET_FRAC * float(ni)


def _apply_virtual_loss(path: list[tuple[MCTSNode, int]]) -> None:
    """Decrement Q along *path* to discourage parallel re-selection."""
    for node, move_idx in path:
        child = node.children[move_idx]
        child.N += 1
        child.Q -= 1.0


def _remove_virtual_loss(path: list[tuple[MCTSNode, int]]) -> None:
    """Reverse virtual loss previously applied to *path*."""
    for node, move_idx in path:
        child = node.children[move_idx]
        child.N -= 1
        child.Q += 1.0


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    """Leaf-batched MCTS engine."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        simulations: int = MCTS_SIMULATIONS,
    ) -> None:
        self.model      = model
        self.device     = device
        self.simulations = simulations
        self.table: OrderedDict[int, MCTSNode] = OrderedDict()

    # ------------------------------------------------------------------
    # Transposition table helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, zobrist: int) -> tuple[MCTSNode, bool]:
        """Return (node, is_new).  Evicts LRU entry when table is full."""
        if zobrist in self.table:
            self.table.move_to_end(zobrist)
            return self.table[zobrist], False
        node = MCTSNode()
        self.table[zobrist] = node
        if len(self.table) > MAX_TABLE_SIZE:
            self.table.popitem(last=False)
        return node, True

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_path(
        self,
        root: MCTSNode,
        board: chess.Board,
    ) -> tuple[list[tuple[MCTSNode, int]], MCTSNode, chess.Board]:
        """Walk the tree from *root* to a leaf using UCB-PUCT.

        Returns ``(path, leaf_node, board_at_leaf)`` where *path* is a list of
        ``(parent_node, move_index)`` pairs leading to the leaf.
        """
        path: list[tuple[MCTSNode, int]] = []
        node = root
        b    = board.copy()

        while True:
            if node.is_terminal or not node.children:
                return path, node, b

            # UCB-PUCT: use node.N as the parent visit count
            parent_n = node.N if node.N > 0 else 1
            best_score  = -float("inf")
            best_idx    = -1
            best_child  = node  # placeholder

            for move_idx, child in node.children.items():
                u = C_PUCT * child.P * math.sqrt(parent_n) / (1 + child.N)
                score = child.Q + u
                if score > best_score:
                    best_score = score
                    best_idx   = move_idx
                    best_child = child

            path.append((node, best_idx))
            move = chess.Move(best_idx // 64, best_idx % 64)
            b.push(move)
            node = best_child

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _expand(
        self,
        node: MCTSNode,
        board: chess.Board,
        policy_logits: np.ndarray,
    ) -> None:
        """Populate *node*.children using masked + normalised policy priors."""
        mask = legal_move_mask(board)
        logits = policy_logits.copy()
        logits[~mask] = -1e9

        # Stable softmax over legal moves only
        logits -= logits.max()
        probs = np.exp(logits)
        probs[~mask] = 0.0
        total = probs.sum()
        if total > 0.0:
            probs /= total

        for move in board.legal_moves:
            idx = move_to_index(move)
            child_board = board.copy()
            child_board.push(move)
            child_hash  = chess.polyglot.zobrist_hash(child_board)
            child, _    = self._get_or_create(child_hash)
            child.P     = float(probs[idx])
            node.children[idx] = child

    # ------------------------------------------------------------------
    # Back-propagation
    # ------------------------------------------------------------------

    def _backprop(
        self,
        path: list[tuple[MCTSNode, int]],
        value: float,
    ) -> None:
        """Update N and Q along *path*.  *value* is from white's perspective."""
        for node, move_idx in reversed(path):
            child   = node.children[move_idx]
            child.N += 1
            # Incremental mean update
            child.Q += (value - child.Q) / child.N
        # Also update root visit count
        if path:
            root_node = path[0][0]
            root_node.N += 1

    # ------------------------------------------------------------------
    # Batched network evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_batch(
        self,
        boards: list[chess.Board],
        prev_boards: list[Optional[chess.Board]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single batched forward pass.  Returns numpy arrays."""
        tensors = [
            board_to_tensor(b, p)
            for b, p in zip(boards, prev_boards)
        ]
        batch = torch.tensor(
            np.stack(tensors), dtype=torch.float32, device=self.device
        )
        with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
            policy_logits, value_logits = self.model(batch)
        return policy_logits.cpu().numpy(), value_logits.cpu().numpy()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _init_root(
        self,
        root: MCTSNode,
        board: chess.Board,
        prev_board: Optional[chess.Board],
    ) -> None:
        """Initialise *root* node with network policy if not already done."""
        if board.is_game_over(claim_draw=True):
            root.is_terminal    = True
            root.terminal_value = _result_to_value(board.result(claim_draw=True))
        else:
            policy_arr, _ = self._evaluate_batch([board], [prev_board])
            self._expand(root, board, policy_arr[0])
            root.N = 1

    def _collect_leaf_batch(
        self,
        root: MCTSNode,
        board: chess.Board,
        remaining: int,
    ) -> tuple[
        list[MCTSNode],
        list[list[tuple[MCTSNode, int]]],
        list[chess.Board],
        int,   # simulations consumed
    ]:
        """Collect up to BATCH_LEAF_SIZE leaves, handling trivial cases inline.

        Returns ``(leaves, paths, boards, consumed)``.
        """
        batch_leaves: list[MCTSNode]                        = []
        batch_paths:  list[list[tuple[MCTSNode, int]]]     = []
        batch_boards: list[chess.Board]                    = []
        consumed = 0

        for _ in range(min(BATCH_LEAF_SIZE, remaining)):
            path, leaf, leaf_board = self._select_path(root, board)
            consumed += 1

            if leaf.is_terminal:
                self._backprop(path, leaf.terminal_value)
                continue

            if leaf.children:          # transposition hit
                self._backprop(path, leaf.Q)
                continue

            if leaf_board.is_game_over(claim_draw=True):
                leaf.is_terminal    = True
                leaf.terminal_value = _result_to_value(
                    leaf_board.result(claim_draw=True)
                )
                self._backprop(path, leaf.terminal_value)
                continue

            _apply_virtual_loss(path)
            batch_leaves.append(leaf)
            batch_paths.append(path)
            batch_boards.append(leaf_board)

        return batch_leaves, batch_paths, batch_boards, consumed

    def _evaluate_and_backprop_batch(
        self,
        batch_leaves: list[MCTSNode],
        batch_paths:  list[list[tuple[MCTSNode, int]]],
        batch_boards: list[chess.Board],
    ) -> None:
        """Run batched network call and backpropagate all results."""
        none_list: list[Optional[chess.Board]] = [None] * len(batch_boards)
        policy_arr, value_arr = self._evaluate_batch(batch_boards, none_list)

        for leaf, path, leaf_board, pol, val in zip(
            batch_leaves, batch_paths, batch_boards, policy_arr, value_arr
        ):
            _remove_virtual_loss(path)
            self._expand(leaf, leaf_board, pol)
            wdl   = _softmax(val)
            value = float(wdl[0] - wdl[2])
            self._backprop(path, value)

    def search(
        self,
        board: chess.Board,
        prev_board: Optional[chess.Board] = None,
    ) -> dict[int, int]:
        """Run MCTS for ``self.simulations`` playouts.

        Returns ``{move_index: visit_count}`` for root's direct children.
        """
        root_hash    = chess.polyglot.zobrist_hash(board)
        root, is_new = self._get_or_create(root_hash)

        if is_new or not root.children:
            self._init_root(root, board, prev_board)

        _add_dirichlet_noise(root)

        sim_count = 0
        while sim_count < self.simulations:
            remaining = self.simulations - sim_count
            leaves, paths, boards, consumed = self._collect_leaf_batch(
                root, board, remaining
            )
            sim_count += consumed

            if leaves:
                self._evaluate_and_backprop_batch(leaves, paths, boards)

        return {idx: child.N for idx, child in root.children.items()}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_best_move(
        self,
        board: chess.Board,
        prev_board: Optional[chess.Board] = None,
        move_number: int = 0,
    ) -> chess.Move:
        """Run search and return the selected move.

        Temperature is 1 for the first ``TEMP_THRESHOLD`` moves (sampling
        proportional to N) and 0 (greedy argmax) afterwards.
        """
        visit_counts = self.search(board, prev_board)

        if not visit_counts:
            return next(iter(board.legal_moves))

        indices = list(visit_counts.keys())
        counts  = np.array([visit_counts[i] for i in indices], dtype=np.float64)

        if move_number >= TEMP_THRESHOLD:
            best_idx = indices[int(np.argmax(counts))]
        else:
            probs    = counts / counts.sum()
            best_idx = int(_rng.choice(indices, p=probs))

        return chess.Move(best_idx // 64, best_idx % 64)
