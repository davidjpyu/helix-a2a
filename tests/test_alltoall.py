"""Phase 3: Multi-GPU correctness test — Helix native vs NCCL all_to_all_single.

Run with torchrun:

    torchrun --nproc-per-node=N tests/test_alltoall.py

where N is the number of GPUs (must be >= 2).

Container requirements:
    --ipc=host --cap-add=SYS_PTRACE   (for POSIX FD handle exchange via pidfd)

Requirements: SM 90+ GPU (H200/GB200), helix_a2a installed, cuda-python.
"""
from __future__ import annotations

import ctypes
import os
import sys
from typing import List, Tuple

import torch
import torch.distributed as dist

import helix_a2a

# ── IPC workspace via cuMem driver API ──────────────────────────────────


def _get_cuda_driver():
    try:
        from cuda.bindings import driver as drv
        return drv
    except ImportError:
        from cuda import cuda as drv
        return drv


def _check_cu(result, ctx: str = ""):
    if isinstance(result, tuple):
        err, *rest = result
        cuda = _get_cuda_driver()
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver error: {err} in {ctx}")
        return rest[0] if len(rest) == 1 else tuple(rest) if rest else None
    cuda = _get_cuda_driver()
    if result != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {result} in {ctx}")
    return None


def _resolve_posix_fds(
    all_fds: List[int], all_pids: List[int], local_rank: int
) -> List[int]:
    """Translate remote POSIX FDs to local FDs via pidfd syscalls.

    Requires --cap-add=SYS_PTRACE in Docker containers.
    Matches vllm-a2a's helix_mnnvl_workspace._resolve_posix_fds.
    """
    SYS_pidfd_open = 434
    SYS_pidfd_getfd = 438

    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall

    resolved: List[int] = []
    for i, (pid, fd) in enumerate(zip(all_pids, all_fds)):
        if i == local_rank:
            resolved.append(fd)
            continue

        pidfd = syscall(SYS_pidfd_open, pid, 0)
        if pidfd < 0:
            err = ctypes.get_errno()
            raise RuntimeError(
                f"pidfd_open({pid}) failed: errno {err} ({os.strerror(err)}). "
                f"Ensure all ranks are on the same node and --ipc=host is set."
            )

        remote_fd = syscall(SYS_pidfd_getfd, pidfd, fd, 0)
        if remote_fd < 0:
            err = ctypes.get_errno()
            hint = ""
            if err == 1:
                hint = (
                    " Add --cap-add=SYS_PTRACE to your docker run command."
                )
            raise RuntimeError(
                f"pidfd_getfd(pidfd={pidfd}, fd={fd}) failed: errno {err} "
                f"({os.strerror(err)}).{hint}"
            )
        resolved.append(remote_fd)

    return resolved


def _ptr_to_strided_tensor(
    base_ptr: int, cols: int, stride_u64: int, rows: int, dev_id: int
) -> torch.Tensor:
    """Wrap a raw GPU pointer as a 2D strided int64 tensor via DLPack."""
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.c_void_p
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    class _DLDataType(ctypes.Structure):
        _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                     ("lanes", ctypes.c_uint16)]

    class _DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

    class _DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p), ("device", _DLDevice),
            ("ndim", ctypes.c_int), ("dtype", _DLDataType),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_size_t),
        ]

    class _DLManagedTensor(ctypes.Structure):
        pass

    _DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))

    @_DELETER
    def _noop(p):
        pass

    _DLManagedTensor._fields_ = [
        ("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DELETER),
    ]

    Arr2 = ctypes.c_int64 * 2
    shape_arr = Arr2(rows, cols)
    stride_arr = Arr2(stride_u64, 1)

    dlt = _DLTensor()
    dlt.data = ctypes.c_void_p(base_ptr)
    dlt.device = _DLDevice(device_type=2, device_id=dev_id)
    dlt.ndim = 2
    dlt.dtype = _DLDataType(code=0, bits=64, lanes=1)
    dlt.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
    dlt.strides = ctypes.cast(stride_arr, ctypes.POINTER(ctypes.c_int64))
    dlt.byte_offset = 0

    mt = _DLManagedTensor()
    mt.dl_tensor = dlt
    mt.manager_ctx = None
    mt.deleter = _noop

    cap_ptr = PyCapsule_New(ctypes.pointer(mt), b"dltensor", None)
    capsule = ctypes.cast(cap_ptr, ctypes.py_object).value

    tensor = torch.utils.dlpack.from_dlpack(capsule)

    tensor._dlpack_keepalive = (shape_arr, stride_arr, mt, capsule)

    return tensor


def allocate_ipc_workspace(cp_rank: int, cp_size: int) -> torch.Tensor:
    """Set up cross-rank workspace via cuMem driver API + POSIX FD handles.

    Each rank allocates shareable physical memory, exports a POSIX FD,
    exchanges FDs + PIDs via all_gather_object, translates remote FDs via
    pidfd syscalls, then maps all ranks into a contiguous VA range.
    Matches vllm-a2a's _allocate_mnnvl_memory for x86_64.
    """
    cuda = _get_cuda_driver()

    _ = torch.empty(1, device="cuda")

    dev = _check_cu(cuda.cuCtxGetDevice(), "cuCtxGetDevice")
    dev_id = int(dev)

    location = cuda.CUmemLocation()
    location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = dev_id

    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = (
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )
    prop.location = location

    gran_flag = cuda.CUmemAllocationGranularity_flags
    granularity = _check_cu(
        cuda.cuMemGetAllocationGranularity(
            prop=prop,
            option=gran_flag.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        ),
        "cuMemGetAllocationGranularity",
    )

    ws_bytes = helix_a2a.workspace_size(cp_size)
    ws_elems = (ws_bytes + 7) // 8
    row_bytes = ws_elems * 8
    aligned_row = ((row_bytes + granularity - 1) // granularity) * granularity

    local_handle = _check_cu(
        cuda.cuMemCreate(aligned_row, prop, 0), "cuMemCreate"
    )

    local_fd = _check_cu(
        cuda.cuMemExportToShareableHandle(
            local_handle, prop.requestedHandleTypes, 0
        ),
        "cuMemExportToShareableHandle",
    )

    all_fds: List[int] = [None] * cp_size
    all_pids: List[int] = [None] * cp_size
    dist.all_gather_object(all_fds, int(local_fd))
    dist.all_gather_object(all_pids, os.getpid())

    resolved_fds = _resolve_posix_fds(all_fds, all_pids, cp_rank)

    total_va = aligned_row * cp_size
    base_ptr = _check_cu(
        cuda.cuMemAddressReserve(total_va, granularity, 0, 0),
        "cuMemAddressReserve",
    )
    base_int = int(base_ptr)

    madesc = cuda.CUmemAccessDesc()
    madesc.location = location
    madesc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    for i in range(cp_size):
        rank_va = base_int + aligned_row * i
        if i == cp_rank:
            _check_cu(
                cuda.cuMemMap(rank_va, aligned_row, 0, local_handle, 0),
                f"cuMemMap(local, rank={i})",
            )
        else:
            imported = _check_cu(
                cuda.cuMemImportFromShareableHandle(
                    resolved_fds[i], prop.requestedHandleTypes
                ),
                f"cuMemImportFromShareableHandle(rank={i})",
            )
            _check_cu(
                cuda.cuMemMap(rank_va, aligned_row, 0, imported, 0),
                f"cuMemMap(remote, rank={i})",
            )
        _check_cu(
            cuda.cuMemSetAccess(rank_va, aligned_row, [madesc], 1),
            f"cuMemSetAccess(rank={i})",
        )

    stride_u64 = aligned_row // 8
    workspace = _ptr_to_strided_tensor(
        base_int, ws_elems, stride_u64, cp_size, dev_id
    )
    return workspace


# ── NCCL reference ──────────────────────────────────────────────────────


def nccl_alltoall_reference(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two dist.all_to_all_single calls — one for partial_o, one for stats."""
    recv_o = torch.empty_like(partial_o)
    recv_s = torch.empty_like(softmax_stats)

    dist.all_to_all_single(recv_o.reshape(-1), partial_o.reshape(-1))
    dist.all_to_all_single(recv_s.reshape(-1), softmax_stats.reshape(-1))

    return recv_o, recv_s


# ── Test logic ──────────────────────────────────────────────────────────


def run_test(
    rank: int,
    cp_size: int,
    workspace: torch.Tensor,
    dtype: torch.dtype,
    B: int = 4,
    D: int = 128,
    S: int = 2,
) -> bool:
    """Run a single correctness comparison for the given dtype.

    All ranks generate the same full [B, cp_size, cp_size, D] tensor (same
    seed). Each rank extracts column ``rank`` as its send buffer — so rank r
    sends full[:, :, r, :], which is [B, cp_size, D]. After all-to-all, rank r
    should receive full[:, r, :, :] (row r from each sender).
    """
    dtype_name = "bf16" if dtype == torch.bfloat16 else "fp16"

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    full_o = torch.randn(B, cp_size, cp_size, D, dtype=dtype, device="cuda")
    full_s = torch.randn(B, cp_size, cp_size, S, dtype=torch.float32, device="cuda")

    send_o = full_o[:, :, rank, :].contiguous()  # [B, cp_size, D]
    send_s = full_s[:, :, rank, :].contiguous()  # [B, cp_size, S]

    nccl_recv_o, nccl_recv_s = nccl_alltoall_reference(
        send_o.clone(), send_s.clone()
    )
    torch.cuda.synchronize()
    dist.barrier()

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


# ── Main ────────────────────────────────────────────────────────────────


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cp_size = world_size

    if rank == 0:
        gpu_name = torch.cuda.get_device_name(0)
        sm = torch.cuda.get_device_capability(0)
        ws_bytes = helix_a2a.workspace_size(cp_size)
        print(f"Running Phase 3 multi-GPU correctness tests")
        print(f"  GPUs: {world_size}x {gpu_name} (SM {sm[0]}.{sm[1]})")
        print(f"  cp_size: {cp_size}")
        print(f"  workspace_size: {ws_bytes:,} bytes ({ws_bytes/1024/1024:.1f} MiB)")
        print()

    if rank == 0:
        print("Setting up IPC workspace via cuMem driver API...")
    workspace = allocate_ipc_workspace(rank, cp_size)

    helix_a2a.init_workspace(workspace, cp_rank=rank, cp_size=cp_size)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print(
            f"  workspace shape: {list(workspace.shape)}, "
            f"stride: {list(workspace.stride())}"
        )
        print()

    all_passed = True

    for dtype in [torch.bfloat16, torch.float16]:
        passed = run_test(rank, cp_size, workspace, dtype)
        if not passed:
            all_passed = False

    dist.barrier()

    if rank == 0:
        print()
        if all_passed:
            print("All Phase 3 correctness tests PASSED!")
        else:
            print("Some tests FAILED — see above for details.")

    dist.destroy_process_group()

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
