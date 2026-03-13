"""
helix_a2a — Standalone Helix All-to-All CUDA extension.

Public API:
    workspace_size(cp_size) -> int
    allocate_workspace(cp_size, cp_rank, device, *, mnnvl, cpu_group) -> Tensor
    init_workspace(workspace, cp_rank, cp_size) -> None
    alltoall(partial_o, softmax_stats, workspace, cp_rank, cp_size) -> (Tensor, Tensor)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from helix_a2a._ops import (
    alltoall_native,
    get_workspace_size_per_rank,
    initialize_workspace,
)

__version__ = "0.1.0"

__all__ = [
    "workspace_size",
    "allocate_workspace",
    "init_workspace",
    "alltoall",
]


def workspace_size(cp_size: int) -> int:
    """Return the workspace size **in bytes** per rank for the given CP group size.

    >>> helix_a2a.workspace_size(cp_size=4)
    16778240
    """
    return get_workspace_size_per_rank(cp_size)


def allocate_workspace(
    cp_size: int,
    cp_rank: int,
    device: str = "cuda",
    *,
    mnnvl: bool | str = False,
    cpu_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Allocate a workspace tensor of shape ``[cp_size, ws_elems_per_rank]``.

    After allocation, call :func:`init_workspace` followed by a cross-rank
    barrier before the first :func:`alltoall` call.  Typical setup::

        workspace = helix_a2a.allocate_workspace(cp_size, cp_rank, mnnvl=True, cpu_group=group)
        helix_a2a.init_workspace(workspace, cp_rank, cp_size)
        dist.barrier(group)   # ensure ALL ranks finish init before any alltoall
        # ... now safe to call helix_a2a.alltoall()

    Args:
        cp_size: Context-parallel group size.
        cp_rank: This rank's position in the CP group.
        device: Device for plain allocation (default ``"cuda"``).
        mnnvl: ``False`` for device memory, ``True`` for MNNVL, ``"auto"``
            for MNNVL when multi-node + aarch64.
        cpu_group: Process group for MNNVL handle exchange (Gloo or NCCL).

    Returns:
        ``torch.int64`` tensor of shape ``[cp_size, ws_elems_per_rank]``.
    """
    if mnnvl is True or (isinstance(mnnvl, str) and mnnvl == "auto"):
        from helix_a2a.mnnvl import allocate_mnnvl_workspace
        return allocate_mnnvl_workspace(cp_size, cp_rank, cpu_group, mnnvl)

    ws_bytes = workspace_size(cp_size)
    ws_elems_per_rank = (ws_bytes + 7) // 8
    return torch.zeros(
        cp_size, ws_elems_per_rank, dtype=torch.long, device=device
    )


def init_workspace(
    workspace: torch.Tensor, cp_rank: int, cp_size: int
) -> None:
    """Initialize the workspace (FIFO reset). Call once before the first alltoall.

    This resets the FIFO buffers in the **local** workspace row
    (``workspace[cp_rank]``).  The operation is synchronous: when this
    function returns, the GPU memset is guaranteed to have completed on
    the current device.

    .. important::
        With MNNVL workspaces, **all ranks** must complete
        ``init_workspace`` before **any** rank calls :func:`alltoall`.
        Insert a cross-rank barrier (e.g. ``dist.barrier()``) between
        ``init_workspace`` and the first ``alltoall`` call.
    """
    initialize_workspace(workspace, cp_rank, cp_size)
    torch.cuda.current_stream().synchronize()


def alltoall(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    workspace: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform the Helix all-to-all exchange.

    Args:
        partial_o: ``[..., cp_size, D]`` — half or bfloat16.
        softmax_stats: ``[..., cp_size, S]`` — float32, S >= 2 and even.
        workspace: From :func:`allocate_workspace`, shape ``[cp_size, ws_elems]``.
        cp_rank: Rank in CP group.
        cp_size: CP group size.

    Returns:
        ``(partial_o_out, softmax_stats_out)`` with same shapes as inputs.
    """
    return alltoall_native(partial_o, softmax_stats, workspace, cp_rank, cp_size)
