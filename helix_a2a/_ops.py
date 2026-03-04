"""
Low-level wrappers around torch.ops.helix_a2a.* C++ ops.

This module is the single point of contact with the compiled extension.
Higher-level functions in __init__.py delegate here.
"""
from __future__ import annotations

from typing import Tuple

import torch

import helix_a2a._C  # noqa: F401  — triggers TORCH_LIBRARY registration


def get_workspace_size_per_rank(cp_size: int) -> int:
    """Return workspace size in bytes for one rank."""
    return torch.ops.helix_a2a.get_helix_workspace_size_per_rank(cp_size)


def initialize_workspace(workspace: torch.Tensor, cp_rank: int,
                         cp_size: int) -> None:
    """Initialize FIFO buffers in the workspace tensor."""
    torch.ops.helix_a2a.initialize_helix_workspace(workspace, cp_rank, cp_size)


def alltoall_native(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    workspace: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute the Helix all-to-all kernel."""
    return torch.ops.helix_a2a.alltoall_helix_native(
        partial_o, softmax_stats, workspace, cp_rank, cp_size
    )
