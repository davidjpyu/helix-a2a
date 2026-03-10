# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DCP All-to-All with Helix Native backend support.

Drop-in replacement for vllm/v1/attention/ops/dcp_alltoall.py.
Adds helix_a2a fused LL128 kernel as an alternative to NCCL all_to_all_single.

Backend selection via environment variable:
    HELIX_A2A_BACKEND=nccl    → original NCCL path (default, safe)
    HELIX_A2A_BACKEND=native  → helix_a2a fused kernel

Usage:
    export HELIX_A2A_BACKEND=native
    vllm serve model --tp 16 --dcp 16 --dcp-comm-backend a2a
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.attention.ops.common import CPTritonContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helix Native backend (lazy init)
# ---------------------------------------------------------------------------

_HELIX_BACKEND = os.environ.get("HELIX_A2A_BACKEND", "nccl").lower()
_helix_workspace = None
_helix_initialized = False


def _get_helix_workspace(cp_rank: int, cp_size: int, device_group):
    """Lazy-initialize the helix_a2a workspace (once per process)."""
    global _helix_workspace, _helix_initialized

    if _helix_initialized:
        return _helix_workspace

    import helix_a2a

    mnnvl_setting = os.environ.get("HELIX_A2A_USE_MNNVL", "auto")
    if mnnvl_setting == "0":
        mnnvl = False
    elif mnnvl_setting == "1":
        mnnvl = True
    else:
        mnnvl = "auto"

    cpu_group = None
    if mnnvl is not False:
        try:
            cpu_group = dist.new_group(
                ranks=list(range(cp_size)),
                backend="gloo",
            )
        except Exception:
            cpu_group = None
            logger.warning("Failed to create Gloo group for MNNVL; "
                           "falling back to device memory workspace")
            mnnvl = False

    workspace = helix_a2a.allocate_workspace(
        cp_size=cp_size,
        cp_rank=cp_rank,
        mnnvl=mnnvl,
        cpu_group=cpu_group,
    )
    helix_a2a.init_workspace(workspace, cp_rank, cp_size)

    _helix_workspace = workspace
    _helix_initialized = True
    logger.info("helix_a2a workspace initialized: cp_rank=%d cp_size=%d "
                "mnnvl=%s shape=%s", cp_rank, cp_size, mnnvl, workspace.shape)
    return workspace


# ---------------------------------------------------------------------------
# Original functions (unchanged from upstream vLLM)
# ---------------------------------------------------------------------------

def _lse_weighted_combine(
    outputs: torch.Tensor,
    lses: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference implementation for LSE-weighted combination.

    This is a pure PyTorch implementation used for testing and validation.
    For GPU execution, use dcp_lse_combine_triton instead.

    Args:
        outputs: Partial attention outputs [N, B, H, D]
                 N = number of KV shards (ranks)
                 B = batch size (num_tokens)
                 H = number of heads per rank
                 D = head dimension
        lses: Log-sum-exp values [N, B, H]
        return_lse: If True, also return the global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H, D], and optionally global LSE [B, H]
    """
    N, B, H, D = outputs.shape

    # Handle NaN and inf in LSEs
    lses = torch.where(
        torch.isnan(lses) | torch.isinf(lses),
        torch.tensor(float("-inf"), device=lses.device, dtype=lses.dtype),
        lses,
    )

    # Compute max LSE for numerical stability
    lse_max, _ = lses.max(dim=0)  # [B, H]
    lse_max = torch.where(
        lse_max == float("-inf"),
        torch.zeros_like(lse_max),
        lse_max,
    )

    # Compute weights: softmax over the N dimension
    if is_lse_base_on_e:
        weights = torch.exp(lses - lse_max.unsqueeze(0))  # [N, B, H]
    else:
        weights = torch.pow(2.0, lses - lse_max.unsqueeze(0))  # [N, B, H]

    # Handle NaN weights
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)

    # Normalize weights
    weight_sum = weights.sum(dim=0, keepdim=True)  # [1, B, H]
    weights = weights / weight_sum.clamp(min=1e-10)  # [N, B, H]

    # Weighted combination: sum over N dimension
    result = (outputs * weights.unsqueeze(-1)).sum(dim=0)  # [B, H, D]

    if return_lse:
        if is_lse_base_on_e:
            global_lse = torch.log(weight_sum.squeeze(0)) + lse_max  # [B, H]
        else:
            global_lse = torch.log2(weight_sum.squeeze(0)) + lse_max  # [B, H]
        return result, global_lse

    return result


@triton.jit
def _dcp_lse_combine_kernel(
    # Input pointers
    recv_output_ptr,
    recv_lse_ptr,
    # Output pointers
    out_ptr,
    out_lse_ptr,
    # Strides for recv_output [N, B, H_local, D]
    ro_stride_N,
    ro_stride_B,
    ro_stride_H,
    ro_stride_D,
    # Strides for recv_lse [N, B, H_local]
    rl_stride_N,
    rl_stride_B,
    rl_stride_H,
    # Strides for output [B, H_local, D]
    o_stride_B,
    o_stride_H,
    o_stride_D,
    # Constants
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """
    Triton kernel for LSE-weighted combination of partial attention outputs.

    After All-to-All, each rank has:
    - recv_output [N, B, H_local, D]: partial outputs from all KV shards
    - recv_lse [N, B, H_local]: partial LSEs from all KV shards

    This kernel computes the weighted combination locally (no communication).

    Grid: (B, H_local)
    Each program handles one (batch, head) and processes all D elements.
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)

    # Base offset for this (batch, head)
    base_lse_offset = batch_idx * rl_stride_B + head_idx * rl_stride_H
    base_out_offset = batch_idx * ro_stride_B + head_idx * ro_stride_H

    # First pass: find max LSE for numerical stability
    lse_max = -float("inf")
    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    # Second pass: compute sum of exp(lse - max)
    lse_sum = 0.0
    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            lse_sum += tl.exp(lse_val - lse_max)
        else:
            lse_sum += tl.exp2(lse_val - lse_max)

    # Compute global LSE
    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    # Third pass: weighted combination across D dimension
    d_offsets = tl.arange(0, HEAD_DIM)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for n in tl.static_range(N):
        lse_offset = n * rl_stride_N + base_lse_offset
        lse_val = tl.load(recv_lse_ptr + lse_offset)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            weight = tl.exp(lse_val - global_lse)
        else:
            weight = tl.exp2(lse_val - global_lse)
        weight = tl.where(weight != weight, 0.0, weight)

        out_offsets = n * ro_stride_N + base_out_offset + d_offsets * ro_stride_D
        out_vals = tl.load(recv_output_ptr + out_offsets)
        acc += out_vals.to(tl.float32) * weight

    # Store result
    final_offsets = (
        batch_idx * o_stride_B + head_idx * o_stride_H + d_offsets * o_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)

    if RETURN_LSE:
        tl.store(out_lse_ptr + base_lse_offset, global_lse)


def dcp_lse_combine_triton(
    recv_output: torch.Tensor,
    recv_lse: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-accelerated LSE-weighted combination for DCP A2A.

    Args:
        recv_output: [N, B, H_local, D] - partial outputs from all KV shards
        recv_lse: [N, B, H_local] - partial LSEs from all KV shards
        return_lse: If True, also return the global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H_local, D]
        If return_lse=True, also returns global_lse [B, H_local]
    """
    N, B, H_local, D = recv_output.shape

    out = torch.empty(
        (B, H_local, D), device=recv_output.device, dtype=recv_output.dtype
    )

    if return_lse:
        out_lse = torch.empty(
            (B, H_local), device=recv_lse.device, dtype=recv_lse.dtype
        )
    else:
        out_lse = torch.empty(1, device=recv_lse.device, dtype=recv_lse.dtype)

    ro_stride_N, ro_stride_B, ro_stride_H, ro_stride_D = recv_output.stride()
    rl_stride_N, rl_stride_B, rl_stride_H = recv_lse.stride()
    o_stride_B, o_stride_H, o_stride_D = out.stride()

    grid = (B, H_local, 1)

    _dcp_lse_combine_kernel[grid](
        recv_output,
        recv_lse,
        out,
        out_lse,
        ro_stride_N,
        ro_stride_B,
        ro_stride_H,
        ro_stride_D,
        rl_stride_N,
        rl_stride_B,
        rl_stride_H,
        o_stride_B,
        o_stride_H,
        o_stride_D,
        N=N,
        HEAD_DIM=D,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
    )

    if return_lse:
        return out, out_lse
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def dcp_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Combine partial attention outputs across DCP ranks using All-to-All.

    Each rank holds attention output for all heads but only a local shard
    of the KV cache. This function:
    1. Exchanges partial outputs across ranks via All-to-All
    2. Exchanges LSE values via All-to-All
    3. Combines them with exact LSE-weighted reduction (Triton kernel)

    When HELIX_A2A_BACKEND=native, steps 1-2 are fused into a single
    helix_a2a.alltoall() call using a custom LL128 FIFO kernel.

    Tensor flow:
        Input:  cp_attn_out [B, H, D] - all heads, local KV shard
        Reshape: [N, B, H/N, D] - split heads across ranks
        A2A:    Two all_to_all_single calls (output and LSE)
        Combine: recv [N, B, H/N, D] + lse [N, B, H/N] -> [B, H/N, D]

    Args:
        cp_attn_out: [B, H, D] where B=num_tokens, H=total_heads, D=head_dim
        cp_attn_lse: [B, H] log-sum-exp values (fp32)
        cp_group: GroupCoordinator for DCP communication
        ctx: CPTritonContext (unused, for signature compatibility)
        return_lse: If True, also return the combined global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H/N, D] (head-scattered)
        If return_lse=True, also returns global_lse [B, H/N]
    """
    world_size = cp_group.world_size

    if world_size == 1:
        if return_lse:
            return cp_attn_out, cp_attn_lse
        return cp_attn_out

    if _HELIX_BACKEND == "native":
        return _dcp_a2a_helix_native(
            cp_attn_out, cp_attn_lse, cp_group,
            return_lse=return_lse,
            is_lse_base_on_e=is_lse_base_on_e,
        )

    return _dcp_a2a_nccl(
        cp_attn_out, cp_attn_lse, cp_group,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )


# ---------------------------------------------------------------------------
# NCCL path (original upstream)
# ---------------------------------------------------------------------------

def _dcp_a2a_nccl(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Original NCCL path — two all_to_all_single calls."""
    world_size = cp_group.world_size

    local_output = cp_attn_out.contiguous()
    local_lse = cp_attn_lse.contiguous()

    B, H, D = local_output.shape
    H_per_rank = H // world_size

    # Reshape for All-to-All: [B, H, D] -> [N, B, H/N, D]
    # Split heads into N chunks, each destined for a different rank
    send_output = (
        local_output.view(B, world_size, H_per_rank, D).permute(1, 0, 2, 3).contiguous()
    )
    recv_output = torch.empty_like(send_output)

    # Same for LSE: [B, H] -> [N, B, H/N]
    send_lse = local_lse.view(B, world_size, H_per_rank).permute(1, 0, 2).contiguous()
    recv_lse = torch.empty_like(send_lse)

    # All-to-All for partial attention outputs and LSE values (async overlap)
    work_output = dist.all_to_all_single(
        recv_output.view(-1),
        send_output.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    work_lse = dist.all_to_all_single(
        recv_lse.view(-1),
        send_lse.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    work_output.wait()
    work_lse.wait()

    # LSE-weighted combination via Triton kernel (local, no communication)
    return dcp_lse_combine_triton(
        recv_output,
        recv_lse,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )


# ---------------------------------------------------------------------------
# Helix Native path — single fused LL128 A2A kernel
# ---------------------------------------------------------------------------

def _dcp_a2a_helix_native(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Helix Native path: single fused LL128 A2A kernel replaces both
    NCCL all_to_all_single calls.

    helix_a2a.alltoall expects:
        partial_o:      [..., cp_size, D]  half/bf16
        softmax_stats:  [..., cp_size, S]  float32, S >= 2 and even

    vLLM provides:
        cp_attn_out:  [B, H, D]   (H = total_heads across all ranks)
        cp_attn_lse:  [B, H]      (float32)

    Reshape strategy:
        output: [B, H, D] -> view [B, N, H/N, D] -> permute [B, H/N, N, D]
        lse:    [B, H]    -> view [B, N, H/N]     -> permute [B, H/N, N] -> pad to S=2
    """
    import helix_a2a

    world_size = cp_group.world_size
    cp_rank = cp_group.rank_in_group

    workspace = _get_helix_workspace(cp_rank, world_size, cp_group.device_group)

    local_output = cp_attn_out.contiguous()  # [B, H, D]
    local_lse = cp_attn_lse.contiguous()     # [B, H]

    B, H, D = local_output.shape
    H_per_rank = H // world_size

    if cp_rank == 0:
        logger.error("[HELIX_DEBUG] rank=%d input: out=%s dtype=%s lse=%s dtype=%s "
                     "B=%d H=%d D=%d H_per_rank=%d world_size=%d is_base_e=%s",
                     cp_rank, local_output.shape, local_output.dtype,
                     local_lse.shape, local_lse.dtype,
                     B, H, D, H_per_rank, world_size, is_lse_base_on_e)

    # --- Reshape output: [B, H, D] -> [B*H_per_rank, N, D] ---
    # view splits H into (N, H/N), permute puts N as the scatter dim
    send_output = (
        local_output.view(B, world_size, H_per_rank, D)
        .permute(0, 2, 1, 3)                  # [B, H_per_rank, N, D]
        .contiguous()
    )

    # --- Reshape LSE: [B, H] -> [B*H_per_rank, N, 2] ---
    # helix_a2a needs softmax_stats with S >= 2 and even; pad LSE scalar to S=2
    send_lse = (
        local_lse.view(B, world_size, H_per_rank)
        .permute(0, 2, 1)                     # [B, H_per_rank, N]
        .contiguous()
    )
    send_stats = torch.zeros(
        B, H_per_rank, world_size, 2,
        dtype=torch.float32, device=local_lse.device,
    )
    if is_lse_base_on_e:
        send_stats[..., 0] = send_lse
    else:
        send_stats[..., 0] = send_lse * 1.44269504089  # log2(e)

    # Flatten leading dims: [B*H_per_rank, N, D] and [B*H_per_rank, N, 2]
    flat_output = send_output.reshape(B * H_per_rank, world_size, D)
    flat_stats = send_stats.reshape(B * H_per_rank, world_size, 2)

    if cp_rank == 0:
        logger.error("[HELIX_DEBUG] rank=%d flat_output=%s dtype=%s "
                     "flat_stats=%s dtype=%s workspace=%s",
                     cp_rank, flat_output.shape, flat_output.dtype,
                     flat_stats.shape, flat_stats.dtype, workspace.shape)

    # --- Fused A2A ---
    torch.cuda.synchronize()
    logger.error("[HELIX_DEBUG] rank=%d calling helix_a2a.alltoall...", cp_rank)
    recv_output, recv_stats = helix_a2a.alltoall(
        flat_output, flat_stats, workspace,
        cp_rank=cp_rank, cp_size=world_size,
    )
    torch.cuda.synchronize()
    logger.error("[HELIX_DEBUG] rank=%d alltoall returned! recv_output=%s recv_stats=%s",
                 cp_rank, recv_output.shape, recv_stats.shape)

    # --- Reshape back for Triton combine: [N, B, H_per_rank, D] ---
    recv_output = (
        recv_output.reshape(B, H_per_rank, world_size, D)
        .permute(2, 0, 1, 3)                  # [N, B, H_per_rank, D]
        .contiguous()
    )
    # Extract LSE from stats slot 0: [N, B, H_per_rank]
    recv_lse = (
        recv_stats.reshape(B, H_per_rank, world_size, 2)[..., 0]
        .permute(2, 0, 1)                     # [N, B, H_per_rank]
        .contiguous()
    )
    if not is_lse_base_on_e:
        # Convert back from base-e to base-2
        recv_lse = recv_lse * 0.693147180559945  # ln(2)

    return dcp_lse_combine_triton(
        recv_output,
        recv_lse,
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )
