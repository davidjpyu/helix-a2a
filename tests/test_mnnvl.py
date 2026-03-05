"""Phase 4: Multi-GPU MNNVL workspace test — Helix native vs NCCL all_to_all_single.

Run with torchrun:

    torchrun --nproc-per-node=N tests/test_mnnvl.py

where N is the number of GPUs (must be >= 2).

Container requirements:
    x86_64 (H200): --ipc=host --cap-add=SYS_PTRACE
    aarch64 (GB200): --privileged

Set HELIX_A2A_USE_MNNVL=1 to force MNNVL on H200 (intra-node via pidfd).
On GB200 (aarch64), MNNVL auto-detects when multi-node.

Requirements: SM 90+ GPU (H200/GB200), helix_a2a installed, cuda-python.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Tuple

import torch
import torch.distributed as dist

import helix_a2a


# ── NCCL reference ──────────────────────────────────────────────────────


def nccl_alltoall_reference(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NCCL all-to-all matching Helix's [B, cp_size, D] layout.

    Permute [B, N, D] -> [N, B, D] before flattening so NCCL chunks
    align with Helix's per-rank slicing, then permute back on receive.
    """
    send_o = partial_o.transpose(0, 1).contiguous()
    send_s = softmax_stats.transpose(0, 1).contiguous()

    recv_o = torch.empty_like(send_o)
    recv_s = torch.empty_like(send_s)

    dist.all_to_all_single(recv_o.reshape(-1), send_o.reshape(-1))
    dist.all_to_all_single(recv_s.reshape(-1), send_s.reshape(-1))

    return (
        recv_o.transpose(0, 1).contiguous(),
        recv_s.transpose(0, 1).contiguous(),
    )


# ── Test functions ──────────────────────────────────────────────────────


def test_mnnvl_allocation(
    rank: int, cp_size: int, cpu_group: dist.ProcessGroup
) -> torch.Tensor:
    """Allocate MNNVL workspace and verify tensor properties."""
    workspace = helix_a2a.allocate_workspace(
        cp_size=cp_size, cp_rank=rank, mnnvl=True, cpu_group=cpu_group,
    )

    assert workspace.dtype == torch.int64, (
        f"Expected int64, got {workspace.dtype}"
    )
    assert workspace.dim() == 2, f"Expected 2D, got {workspace.dim()}D"
    assert workspace.shape[0] == cp_size, (
        f"Expected {cp_size} rows, got {workspace.shape[0]}"
    )
    assert workspace.shape[1] > 0, "Workspace has 0 columns"

    dist.barrier()
    if rank == 0:
        print(
            f"  workspace shape: {list(workspace.shape)}, "
            f"stride: {list(workspace.stride())}"
        )
        print("  [PASS] MNNVL allocation")
    return workspace


def test_mnnvl_init(
    rank: int, cp_size: int, workspace: torch.Tensor
) -> None:
    """Verify init_workspace succeeds on MNNVL-backed memory."""
    helix_a2a.init_workspace(workspace, cp_rank=rank, cp_size=cp_size)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print("  [PASS] MNNVL workspace init")


def test_mnnvl_correctness(
    rank: int,
    cp_size: int,
    workspace: torch.Tensor,
    dtype: torch.dtype,
    B: int = 4,
    D: int = 128,
    S: int = 2,
) -> bool:
    """Run alltoall with MNNVL workspace and compare output vs NCCL."""
    dtype_name = "bf16" if dtype == torch.bfloat16 else "fp16"

    # All ranks generate the same full tensor (same seed), then each rank
    # extracts its column as the send buffer.
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    full_o = torch.randn(B, cp_size, cp_size, D, dtype=dtype, device="cuda")
    full_s = torch.randn(
        B, cp_size, cp_size, S, dtype=torch.float32, device="cuda"
    )

    send_o = full_o[:, :, rank, :].contiguous()  # [B, cp_size, D]
    send_s = full_s[:, :, rank, :].contiguous()  # [B, cp_size, S]

    # NCCL reference
    nccl_recv_o, nccl_recv_s = nccl_alltoall_reference(
        send_o.clone(), send_s.clone()
    )
    torch.cuda.synchronize()
    dist.barrier()

    # Re-init workspace FIFO state before each alltoall
    helix_a2a.init_workspace(workspace, cp_rank=rank, cp_size=cp_size)
    torch.cuda.synchronize()
    dist.barrier()

    # Helix native with MNNVL workspace
    helix_recv_o, helix_recv_s = helix_a2a.alltoall(
        send_o.clone(), send_s.clone(), workspace,
        cp_rank=rank, cp_size=cp_size,
    )
    torch.cuda.synchronize()
    dist.barrier()

    o_diff = (helix_recv_o.float() - nccl_recv_o.float()).abs().max().item()
    s_diff = (helix_recv_s - nccl_recv_s).abs().max().item()

    o_tol = 1e-3
    s_tol = 1e-5

    passed = o_diff < o_tol and s_diff < s_tol
    status = "PASS" if passed else "FAIL"

    if rank == 0:
        print(
            f"  [{status}] {dtype_name}: partial_o max_diff={o_diff:.6f} "
            f"(tol={o_tol}), softmax_stats max_diff={s_diff:.8f} "
            f"(tol={s_tol})"
        )

    return passed


def test_mnnvl_data_exchange(
    rank: int,
    cp_size: int,
    workspace: torch.Tensor,
) -> bool:
    """Verify actual data exchange: each rank sends unique data."""
    D = 64
    N = 4

    helix_a2a.init_workspace(workspace, cp_rank=rank, cp_size=cp_size)
    torch.cuda.synchronize()
    dist.barrier()

    # Each rank fills its send buffer with a unique value
    po = torch.full(
        (N, cp_size, D), fill_value=float(rank + 1),
        dtype=torch.bfloat16, device="cuda",
    )
    ss = torch.full(
        (N, cp_size, 2), fill_value=float(rank + 1) * 0.1,
        dtype=torch.float32, device="cuda",
    )

    po_out, ss_out = helix_a2a.alltoall(
        po, ss, workspace, cp_rank=rank, cp_size=cp_size,
    )
    torch.cuda.synchronize()
    dist.barrier()

    assert po_out.shape == po.shape, (
        f"Output shape mismatch: {po_out.shape} vs {po.shape}"
    )
    assert ss_out.shape == ss.shape, (
        f"Stats shape mismatch: {ss_out.shape} vs {ss.shape}"
    )

    # After alltoall, po_out[:, j, :] should contain data from rank j
    # (which was filled with value j+1)
    passed = True
    for j in range(cp_size):
        expected_val = float(j + 1)
        actual = po_out[:, j, :].float()
        diff = (actual - expected_val).abs().max().item()
        if diff > 0.1:
            if rank == 0:
                print(
                    f"  [FAIL] data_exchange: rank {j} data mismatch, "
                    f"diff={diff:.4f}"
                )
            passed = False

    if rank == 0:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] data exchange (cp_size={cp_size})")

    return passed


# ── Main ────────────────────────────────────────────────────────────────


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cp_size = world_size

    # Gloo group for MNNVL handle exchange (NCCL can't carry Python objects)
    cpu_group = dist.new_group(backend="gloo")

    if rank == 0:
        gpu_name = torch.cuda.get_device_name(0)
        sm = torch.cuda.get_device_capability(0)
        ws_bytes = helix_a2a.workspace_size(cp_size)
        print("Running Phase 4 MNNVL workspace tests")
        print(f"  GPUs: {world_size}x {gpu_name} (SM {sm[0]}.{sm[1]})")
        print(f"  arch: {platform.machine()}")
        print(f"  cp_size: {cp_size}")
        print(
            f"  workspace_size: {ws_bytes:,} bytes "
            f"({ws_bytes / 1024 / 1024:.1f} MiB)"
        )
        mnnvl_env = os.environ.get("HELIX_A2A_USE_MNNVL", "(unset)")
        print(f"  HELIX_A2A_USE_MNNVL: {mnnvl_env}")
        print()

    all_passed = True

    # Test 1: MNNVL allocation
    if rank == 0:
        print("Allocating MNNVL workspace...")
    workspace = test_mnnvl_allocation(rank, cp_size, cpu_group)

    # Test 2: workspace init
    test_mnnvl_init(rank, cp_size, workspace)

    if rank == 0:
        print()
        print("Running correctness tests (MNNVL workspace vs NCCL)...")

    # Test 3: correctness vs NCCL for bf16 and fp16
    for dtype in [torch.bfloat16, torch.float16]:
        passed = test_mnnvl_correctness(rank, cp_size, workspace, dtype)
        if not passed:
            all_passed = False

    # Test 4: data exchange verification
    if rank == 0:
        print()
        print("Running data exchange test...")
    passed = test_mnnvl_data_exchange(rank, cp_size, workspace)
    if not passed:
        all_passed = False

    dist.barrier()

    if rank == 0:
        print()
        if all_passed:
            print("All Phase 4 MNNVL tests PASSED!")
        else:
            print("Some tests FAILED — see above for details.")

    dist.destroy_process_group()

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
