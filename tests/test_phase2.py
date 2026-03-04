"""Phase 2 tests: verify real kernel compiles and runs on a single GPU.

Requires: SM 90+ GPU (Hopper/Blackwell).
Run:  python tests/test_phase2.py
  or: pytest tests/test_phase2.py -v
"""
import sys

import torch
import helix_a2a


def test_workspace_size_matches_manual_computation():
    """workspace_size must match the value derived from the GPU's SM count."""
    device = torch.cuda.current_device()
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    for cp_size in [2, 4, 8]:
        max_group = min(cp_size, 8)
        blocks_per_ch = ((cp_size + max_group - 1) // max_group) * 2
        max_channels = max(sm_count // blocks_per_ch, 1)

        FIFO_TOTAL = 512 * 1024
        FIFO_INFO_SIZE = 256  # sizeof(HelixFifoInfo) with __align__(256)
        fifo = FIFO_TOTAL * cp_size * max_channels
        info = FIFO_INFO_SIZE * cp_size * max_channels * 2
        expected = fifo + info

        actual = helix_a2a.workspace_size(cp_size)
        print(f"  workspace_size({cp_size}): expected={expected}, actual={actual} "
              f"(sm_count={sm_count}, maxCh={max_channels})")
        assert actual == expected, (
            f"workspace_size({cp_size}) = {actual}, expected {expected} "
            f"(sm_count={sm_count}, maxChannels={max_channels})"
        )


def test_workspace_size_differs_from_phase1_stub():
    """Real kernel should return a different (larger) value than Phase 1 stub."""
    phase1_stub_cp4 = 16_778_240
    actual = helix_a2a.workspace_size(4)
    assert actual != phase1_stub_cp4, (
        f"workspace_size(4) = {actual} matches Phase 1 stub — "
        "old .so may be cached; do a clean rebuild"
    )


def test_init_workspace():
    """init_workspace should run without error and set FIFO to 0xFF."""
    cp_size = 4
    workspace = helix_a2a.allocate_workspace(cp_size=cp_size, cp_rank=0)
    helix_a2a.init_workspace(workspace, cp_rank=0, cp_size=cp_size)

    local_row = workspace[0]
    ws_bytes = helix_a2a.workspace_size(cp_size)
    fifo_u64_count = (512 * 1024 * cp_size *
                      _max_channels(cp_size)) // 8
    fifo_region = local_row[:fifo_u64_count]
    assert (fifo_region == -1).all(), "FIFO region should be 0xFF (all bits set = -1 as int64)"

    info_region = local_row[fifo_u64_count:]
    assert (info_region == 0).all(), "Info region should be zeroed"


def test_alltoall_launches():
    """alltoall should launch the real kernel without error on a single rank."""
    cp_size = 4
    workspace = helix_a2a.allocate_workspace(cp_size=cp_size, cp_rank=0)
    helix_a2a.init_workspace(workspace, cp_rank=0, cp_size=cp_size)

    partial_o = torch.randn(2, cp_size, 128, dtype=torch.bfloat16, device="cuda")
    softmax_stats = torch.randn(2, cp_size, 2, dtype=torch.float32, device="cuda")

    out, stats = helix_a2a.alltoall(
        partial_o, softmax_stats, workspace, cp_rank=0, cp_size=cp_size
    )
    torch.cuda.synchronize()

    assert out.shape == partial_o.shape
    assert stats.shape == softmax_stats.shape
    assert out.dtype == partial_o.dtype
    assert stats.dtype == softmax_stats.dtype


def test_alltoall_input_validation():
    """alltoall should reject bad inputs with clear errors."""
    cp_size = 4
    workspace = helix_a2a.allocate_workspace(cp_size=cp_size, cp_rank=0)
    helix_a2a.init_workspace(workspace, cp_rank=0, cp_size=cp_size)

    good_o = torch.randn(2, cp_size, 128, dtype=torch.bfloat16, device="cuda")
    good_s = torch.randn(2, cp_size, 2, dtype=torch.float32, device="cuda")

    # Wrong dtype for partial_o
    bad_o = good_o.float()
    try:
        helix_a2a.alltoall(bad_o, good_s, workspace, 0, cp_size)
        assert False, "Should have raised"
    except RuntimeError:
        pass

    # Wrong dtype for softmax_stats
    bad_s = good_s.half()
    try:
        helix_a2a.alltoall(good_o, bad_s, workspace, 0, cp_size)
        assert False, "Should have raised"
    except RuntimeError:
        pass

    # Mismatched cp_size dimension
    bad_o2 = torch.randn(2, 3, 128, dtype=torch.bfloat16, device="cuda")
    try:
        helix_a2a.alltoall(bad_o2, good_s, workspace, 0, cp_size)
        assert False, "Should have raised"
    except RuntimeError:
        pass


def _max_channels(cp_size: int) -> int:
    device = torch.cuda.current_device()
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    max_group = min(cp_size, 8)
    blocks_per_ch = ((cp_size + max_group - 1) // max_group) * 2
    return max(sm_count // blocks_per_ch, 1)


if __name__ == "__main__":
    print("Running helix_a2a Phase 2 tests...")
    tests = [
        ("workspace_size matches manual computation",
         test_workspace_size_matches_manual_computation),
        ("workspace_size differs from Phase 1 stub",
         test_workspace_size_differs_from_phase1_stub),
        ("init_workspace sets FIFO correctly",
         test_init_workspace),
        ("alltoall launches without error",
         test_alltoall_launches),
        ("alltoall rejects bad inputs",
         test_alltoall_input_validation),
    ]
    for name, fn in tests:
        fn()
        print(f"[PASS] {name}")
    print(f"\nAll {len(tests)} Phase 2 tests passed!")
    sys.exit(0)
