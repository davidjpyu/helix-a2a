#!/usr/bin/env python3
"""
Micro-benchmark: Full per-layer DCP vs Helix reduction pipeline.

Unlike benchmark_a2a_vs_rs.py (which only measures raw NCCL collectives),
this script measures the COMPLETE per-layer reduction pipeline including
memory copies and Triton kernels:

DCP:   contiguous(lse) -> AllGather(lse) -> Triton correct -> ReduceScatter(out)
Helix: contiguous(out,lse) -> permute+contiguous -> A2A(out) + A2A(lse) -> Triton combine
Helix Native: permute+pad -> helix_a2a.alltoall (fused) -> Triton combine

Uses srun with SLURM env vars for multi-node launch (bypasses Ray), giving
full nsys visibility into all GPU processes.

Usage:
    # Via sbatch (recommended):
    sbatch benchmark/sbatch/pipeline_bench.sbatch

    # Direct srun (16 GPUs, 4 nodes x 4 GPUs):
    srun --nodes=4 --ntasks=16 --ntasks-per-node=4 ... \\
        python benchmark_helix_pipeline.py --dcp 2

    # With native A2A backend:
    srun ... python benchmark_helix_pipeline.py --dcp 2 --a2a-backend both

    # Wrap with nsys for timeline analysis:
    srun ... nsys profile -t cuda,nvtx,nccl --cuda-graph-trace=node -s none \\
        -o /path/to/output \\
        python benchmark_helix_pipeline.py --dcp 2
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

# vLLM imports (requires helix-vllm container)
from vllm.v1.attention.ops.helix import helix_lse_combine_triton
from vllm.v1.attention.ops.common import correct_attn_out

try:
    import helix_a2a
    HAS_HELIX_A2A = True
except ImportError:
    HAS_HELIX_A2A = False


# ============================================================================
# Model configurations
# ============================================================================

# H = total Q heads AFTER TPA sharding, as seen by the reduction pipeline.
# For Nemotron-49B: 64 Q heads total. With TP=16/DCP=N: TPA = 16/N, H = 64/TPA = 64*N/16
# For DeepSeek-V2-Lite: 16 Q heads total (MLA: nope_heads=16, kv_lora_rank=512)
MODEL_CONFIGS = {
    "nemotron49b": {
        "name": "Nemotron-49B (GQA)",
        "total_q_heads": 64,
        "head_dim": 128,
        "dtype_out": "bfloat16",
    },
    "v2lite": {
        "name": "DeepSeek-V2-Lite (MLA)",
        "total_q_heads": 16,
        "head_dim": 512,  # kv_lora_rank for MLA
        "dtype_out": "bfloat16",
    },
}


# ============================================================================
# Correctness verification
# ============================================================================

def verify_correctness(
    out: torch.Tensor,       # [B, H, D]
    lse: torch.Tensor,       # [B, H]
    group: dist.ProcessGroup,
    world_size: int,
    rank_in_group: int,
    native_workspace: torch.Tensor = None,
) -> dict:
    """
    Run both DCP and Helix pipelines with the same input and verify
    the outputs match. Returns dict with max absolute/relative errors.

    Both pipelines should produce the same [B, H/N, D] output:
    - DCP: AG(lse) -> correct(out) -> RS(out)
    - Helix: permute -> A2A(out,lse) -> combine
    - Helix Native: permute+pad -> fused A2A -> combine
    """
    B, H, D = out.shape
    N = world_size
    H_per_rank = H // N

    # --- DCP pipeline ---
    out_dcp = out.clone()
    lse_dcp = lse.clone()

    lse_contig = lse_dcp.contiguous()
    lses = torch.empty(
        (N,) + lse_contig.shape,
        device=lse_contig.device, dtype=lse_contig.dtype,
    )
    dist.all_gather_into_tensor(
        lses.view(N, -1), lse_contig.view(-1), group=group,
    )
    corrected_out, _ = correct_attn_out(
        out_dcp, lses, rank_in_group, ctx=None, is_lse_base_on_e=True,
    )
    # ReduceScatter along HEAD dimension (dim=1), matching vLLM's
    # cp_group.reduce_scatter(out, dim=1). reduce_scatter_tensor always
    # scatters along dim 0, so we movedim before and after.
    rs_input = corrected_out.movedim(0, 1).contiguous()  # [H, B, D]
    rs_output = torch.empty(
        (H_per_rank, B, D), device=rs_input.device, dtype=rs_input.dtype,
    )
    dist.reduce_scatter_tensor(rs_output, rs_input, group=group)
    dcp_result = rs_output.movedim(0, 1).contiguous()  # [B, H/N, D]

    # --- Helix pipeline ---
    out_helix = out.clone()
    lse_helix = lse.clone()

    out_contig = out_helix.contiguous()
    lse_contig2 = lse_helix.contiguous()
    send_output = out_contig.view(B, N, H_per_rank, D).permute(1, 0, 2, 3).contiguous()
    send_lse = lse_contig2.view(B, N, H_per_rank).permute(1, 0, 2).contiguous()
    recv_output = torch.empty_like(send_output)
    recv_lse = torch.empty_like(send_lse)

    work_out = dist.all_to_all_single(
        recv_output.view(-1), send_output.view(-1), group=group, async_op=True,
    )
    work_lse = dist.all_to_all_single(
        recv_lse.view(-1), send_lse.view(-1), group=group, async_op=True,
    )
    work_out.wait()
    work_lse.wait()

    helix_result = helix_lse_combine_triton(
        recv_output, recv_lse, return_lse=False, is_lse_base_on_e=True,
    )

    # --- Compare Helix vs DCP ---
    torch.cuda.synchronize()

    assert dcp_result.shape == helix_result.shape, (
        f"Shape mismatch: DCP={dcp_result.shape} vs Helix={helix_result.shape}"
    )

    abs_diff = (dcp_result.float() - helix_result.float()).abs()
    max_abs = abs_diff.max().item()

    dcp_abs = dcp_result.float().abs()
    max_val = max(dcp_abs.max().item(), 1e-8)
    max_rel = (max_abs / max_val)

    dcp_nan = dcp_result.isnan().any().item()
    helix_nan = helix_result.isnan().any().item()
    dcp_inf = dcp_result.isinf().any().item()
    helix_inf = helix_result.isinf().any().item()

    result = {
        'max_abs_error': max_abs,
        'max_rel_error': max_rel,
        'mean_abs_error': abs_diff.mean().item(),
        'dcp_has_nan': dcp_nan,
        'helix_has_nan': helix_nan,
        'dcp_has_inf': dcp_inf,
        'helix_has_inf': helix_inf,
        'dcp_shape': list(dcp_result.shape),
        'helix_shape': list(helix_result.shape),
        'dcp_mean': dcp_result.float().mean().item(),
        'helix_mean': helix_result.float().mean().item(),
    }

    # --- Helix Native comparison ---
    if HAS_HELIX_A2A and native_workspace is not None:
        partial_o = send_output.permute(1, 2, 0, 3).reshape(-1, N, D).contiguous()
        lse_1d = send_lse.permute(1, 2, 0).reshape(-1, N)
        softmax_stats = torch.stack(
            [lse_1d, torch.zeros_like(lse_1d)], dim=-1,
        ).contiguous()

        recv_o, recv_s = helix_a2a.alltoall(
            partial_o, softmax_stats, native_workspace,
            cp_rank=rank_in_group, cp_size=N,
        )
        recv_out_n = recv_o.reshape(B, H_per_rank, N, D) \
                           .permute(2, 0, 1, 3).contiguous()
        recv_lse_n = recv_s[..., 0].reshape(B, H_per_rank, N) \
                                    .permute(2, 0, 1).contiguous()
        native_result = helix_lse_combine_triton(
            recv_out_n, recv_lse_n, return_lse=False, is_lse_base_on_e=True,
        )
        torch.cuda.synchronize()

        abs_diff_native = (dcp_result.float() - native_result.float()).abs()
        result['native_max_abs_error'] = abs_diff_native.max().item()
        result['native_mean_abs_error'] = abs_diff_native.mean().item()
        result['native_has_nan'] = native_result.isnan().any().item()
        result['native_has_inf'] = native_result.isinf().any().item()
        result['native_shape'] = list(native_result.shape)
        result['native_mean'] = native_result.float().mean().item()

    return result


# ============================================================================
# Per-operation timed benchmarks
# ============================================================================

def benchmark_dcp_pipeline(
    out: torch.Tensor,       # [B, H, D]
    lse: torch.Tensor,       # [B, H]
    group: dist.ProcessGroup,
    world_size: int,
    rank_in_group: int,
    warmup: int = 50,
    iters: int = 200,
) -> dict:
    """
    Benchmark the DCP reduction pipeline:
      1. contiguous(lse)
      2. AllGather(lse) -> [N, B, H]
      3. Triton correct (in-place reweight of out using gathered lses)
      4. ReduceScatter(out) -> [B, H/N, D]
    """
    B, H, D = out.shape

    op_names = ['total', 'ag_lse', 'triton', 'rs_out']
    times = {k: [] for k in op_names}

    for i in range(warmup + iters):
        out_copy = out.clone()
        lse_copy = lse.clone()

        ev_total_s = torch.cuda.Event(enable_timing=True)
        ev_ag_s = torch.cuda.Event(enable_timing=True)
        ev_ag_e = torch.cuda.Event(enable_timing=True)
        ev_tri_s = torch.cuda.Event(enable_timing=True)
        ev_tri_e = torch.cuda.Event(enable_timing=True)
        ev_rs_s = torch.cuda.Event(enable_timing=True)
        ev_rs_e = torch.cuda.Event(enable_timing=True)
        ev_total_e = torch.cuda.Event(enable_timing=True)

        ev_total_s.record()

        # Step 1: AllGather LSE
        ev_ag_s.record()
        lse_contig = lse_copy.contiguous()
        lses = torch.empty(
            (world_size,) + lse_contig.shape,
            device=lse_contig.device, dtype=lse_contig.dtype,
        )
        dist.all_gather_into_tensor(
            lses.view(world_size, -1),
            lse_contig.view(-1),
            group=group,
        )
        ev_ag_e.record()

        # Step 2: Triton correct (in-place reweight using gathered LSEs)
        # Pass ctx=None so correct_attn_out creates a fresh CPTritonContext
        # each call. This avoids the CPTritonContext replay bug where the
        # cached kernel drops constexpr args on the second invocation.
        # Triton's .cubin cache means re-compilation is effectively free
        # after the first call.
        ev_tri_s.record()
        corrected_out, corrected_lse = correct_attn_out(
            out_copy, lses, rank_in_group, ctx=None, is_lse_base_on_e=True,
        )
        ev_tri_e.record()

        # Step 3: ReduceScatter along HEAD dim (dim=1), matching vLLM's
        # cp_group.reduce_scatter(out, dim=1). vLLM's reduce_scatter does:
        #   movedim(0, dim).contiguous() → RS → movedim(0, dim).contiguous()
        # Both movedim+contiguous calls are real costs included in timing.
        ev_rs_s.record()
        rs_input = corrected_out.movedim(0, 1).contiguous()  # [H, B, D]
        rs_output = torch.empty(
            (H // world_size, B, D),
            device=rs_input.device, dtype=rs_input.dtype,
        )
        dist.reduce_scatter_tensor(
            rs_output, rs_input, group=group,
        )
        # Final movedim+contiguous: vLLM returns output_tensor.movedim(0, dim).contiguous()
        # This converts [H/N, B, D] → [B, H/N, D] to match Helix's output layout.
        dcp_final = rs_output.movedim(0, 1).contiguous()  # [B, H/N, D]  # noqa: F841
        ev_rs_e.record()

        ev_total_e.record()
        torch.cuda.synchronize()

        if i >= warmup:
            times['ag_lse'].append(ev_ag_s.elapsed_time(ev_ag_e))
            times['triton'].append(ev_tri_s.elapsed_time(ev_tri_e))
            times['rs_out'].append(ev_rs_s.elapsed_time(ev_rs_e))
            times['total'].append(ev_total_s.elapsed_time(ev_total_e))

    return {k: _summarize(v) for k, v in times.items()}


def benchmark_helix_pipeline(
    out: torch.Tensor,       # [B, H, D]
    lse: torch.Tensor,       # [B, H]
    group: dist.ProcessGroup,
    world_size: int,
    rank_in_group: int,
    warmup: int = 50,
    iters: int = 200,
) -> dict:
    """
    Benchmark the Helix reduction pipeline:
      1. permute + contiguous copies (reshape for A2A)
      2. A2A(output) + A2A(lse) [both async, then wait]
      3. Triton LSE-weighted combine
    """
    B, H, D = out.shape
    H_per_rank = H // world_size

    op_names = ['total', 'permute_copy', 'a2a', 'triton']
    times = {k: [] for k in op_names}

    for i in range(warmup + iters):
        out_copy = out.clone()
        lse_copy = lse.clone()

        ev_total_s = torch.cuda.Event(enable_timing=True)
        ev_pc_s = torch.cuda.Event(enable_timing=True)
        ev_pc_e = torch.cuda.Event(enable_timing=True)
        ev_a2a_s = torch.cuda.Event(enable_timing=True)
        ev_a2a_e = torch.cuda.Event(enable_timing=True)
        ev_tri_s = torch.cuda.Event(enable_timing=True)
        ev_tri_e = torch.cuda.Event(enable_timing=True)
        ev_total_e = torch.cuda.Event(enable_timing=True)

        ev_total_s.record()

        # Step 1: Permute + contiguous copies
        ev_pc_s.record()
        out_contig = out_copy.contiguous()
        lse_contig = lse_copy.contiguous()
        send_output = out_contig.view(B, world_size, H_per_rank, D)
        send_output = send_output.permute(1, 0, 2, 3).contiguous()
        send_lse = lse_contig.view(B, world_size, H_per_rank)
        send_lse = send_lse.permute(1, 0, 2).contiguous()
        recv_output = torch.empty_like(send_output)
        recv_lse = torch.empty_like(send_lse)
        ev_pc_e.record()

        # Step 2: A2A (both async, then wait)
        ev_a2a_s.record()
        work_out = dist.all_to_all_single(
            recv_output.view(-1), send_output.view(-1),
            group=group, async_op=True,
        )
        work_lse = dist.all_to_all_single(
            recv_lse.view(-1), send_lse.view(-1),
            group=group, async_op=True,
        )
        work_out.wait()
        work_lse.wait()
        ev_a2a_e.record()

        # Step 3: Triton LSE-weighted combine
        # helix_lse_combine_triton launches the kernel directly each call
        # (no CPTritonContext caching), so no constexpr replay issue.
        ev_tri_s.record()
        combined = helix_lse_combine_triton(
            recv_output, recv_lse, return_lse=False, is_lse_base_on_e=True,
        )
        ev_tri_e.record()

        ev_total_e.record()
        torch.cuda.synchronize()

        if i >= warmup:
            times['permute_copy'].append(ev_pc_s.elapsed_time(ev_pc_e))
            times['a2a'].append(ev_a2a_s.elapsed_time(ev_a2a_e))
            times['triton'].append(ev_tri_s.elapsed_time(ev_tri_e))
            times['total'].append(ev_total_s.elapsed_time(ev_total_e))

    return {k: _summarize(v) for k, v in times.items()}


def benchmark_helix_native_pipeline(
    out: torch.Tensor,       # [B, H, D]
    lse: torch.Tensor,       # [B, H]
    group: dist.ProcessGroup,
    world_size: int,
    rank_in_group: int,
    workspace: torch.Tensor,
    warmup: int = 50,
    iters: int = 200,
) -> dict:
    """Helix pipeline with native A2A kernel instead of NCCL all_to_all_single."""
    B, H, D = out.shape
    N = world_size
    H_per_rank = H // N

    out_copy = out.clone()
    lse_copy = lse.clone()

    times = {'total': [], 'permute_copy': [], 'a2a': [], 'triton': []}

    for i in range(warmup + iters):
        out_copy.copy_(out)
        lse_copy.copy_(lse)

        ev_total_s = torch.cuda.Event(enable_timing=True)
        ev_pc_s = torch.cuda.Event(enable_timing=True)
        ev_pc_e = torch.cuda.Event(enable_timing=True)
        ev_a2a_s = torch.cuda.Event(enable_timing=True)
        ev_a2a_e = torch.cuda.Event(enable_timing=True)
        ev_tri_s = torch.cuda.Event(enable_timing=True)
        ev_tri_e = torch.cuda.Event(enable_timing=True)
        ev_total_e = torch.cuda.Event(enable_timing=True)

        ev_total_s.record()

        # Step 1: Reshape to kernel-expected layout
        # out_copy: [B, H, D] where H = N * H_per_rank
        # Kernel expects partial_o: [entry_count, cp_size, D]
        # and softmax_stats: [entry_count, cp_size, 2]
        ev_pc_s.record()
        partial_o = out_copy.view(B, N, H_per_rank, D) \
                           .permute(0, 2, 1, 3) \
                           .reshape(B * H_per_rank, N, D) \
                           .contiguous()
        lse_1d = lse_copy.view(B, N, H_per_rank) \
                         .permute(0, 2, 1) \
                         .reshape(B * H_per_rank, N)
        softmax_stats = torch.stack(
            [lse_1d, torch.zeros_like(lse_1d)], dim=-1,
        ).contiguous()  # [B*H_per, N, 2]
        ev_pc_e.record()

        # Step 2: Native A2A (fused — replaces two all_to_all_single calls)
        ev_a2a_s.record()
        recv_o, recv_s = helix_a2a.alltoall(
            partial_o, softmax_stats, workspace,
            cp_rank=rank_in_group, cp_size=N,
        )
        ev_a2a_e.record()

        # Step 3: Reshape back to [N, B, H_per, D] / [N, B, H_per] for Triton
        recv_output = recv_o.reshape(B, H_per_rank, N, D) \
                            .permute(2, 0, 1, 3) \
                            .contiguous()  # [N, B, H_per, D]
        recv_lse = recv_s[..., 0] \
                        .reshape(B, H_per_rank, N) \
                        .permute(2, 0, 1) \
                        .contiguous()  # [N, B, H_per]

        # Step 4: Triton LSE-weighted combine (identical to NCCL path)
        ev_tri_s.record()
        combined = helix_lse_combine_triton(
            recv_output, recv_lse, return_lse=False, is_lse_base_on_e=True,
        )
        ev_tri_e.record()

        ev_total_e.record()
        torch.cuda.synchronize()

        if i >= warmup:
            times['permute_copy'].append(ev_pc_s.elapsed_time(ev_pc_e))
            times['a2a'].append(ev_a2a_s.elapsed_time(ev_a2a_e))
            times['triton'].append(ev_tri_s.elapsed_time(ev_tri_e))
            times['total'].append(ev_total_s.elapsed_time(ev_total_e))

    return {k: _summarize(v) for k, v in times.items()}


# ============================================================================
# CUDA Graph benchmark variants
# ============================================================================

def benchmark_dcp_pipeline_graph(
    out: torch.Tensor,       # [B, H, D]
    lse: torch.Tensor,       # [B, H]
    group: dist.ProcessGroup,
    world_size: int,
    rank_in_group: int,
    warmup: int = 50,
    iters: int = 200,
) -> dict:
    """
    Benchmark the DCP reduction pipeline with CUDA graph capture/replay.

    Captures the entire pipeline (AG + Triton + RS) into a single CUDA graph,
    then replays it. This eliminates CPU launch overhead and enables NCCL
    graph optimizations (e.g., NCCL_GRAPH_MIXING_SUPPORT=0 removes EVENT_WAIT barriers).

    Note: CUDA events inside graph capture don't support elapsed_time(),
    so only TOTAL pipeline time is measured (events outside graph).
    Use nsys profiling for per-operation breakdown under CUDA graphs.
    """
    B, H, D = out.shape
    N = world_size
    H_per = H // N

    # Static input buffers (addresses captured by graph)
    out_static = out.clone()
    lse_static = lse.clone()

    # Eager warmup: trigger Triton JIT compilation + NCCL initialization
    for _ in range(5):
        out_static.copy_(out)
        lse_static.copy_(lse)
        lse_contig = lse_static.contiguous()
        lses = torch.empty(
            (N,) + lse_contig.shape,
            device=lse_contig.device, dtype=lse_contig.dtype,
        )
        dist.all_gather_into_tensor(
            lses.view(N, -1), lse_contig.view(-1), group=group,
        )
        correct_attn_out(
            out_static, lses, rank_in_group, ctx=None, is_lse_base_on_e=True,
        )
        # RS along head dim (dim=1): movedim to put H first
        rs_in = out_static.movedim(0, 1).contiguous()
        rs_out = torch.empty(
            (H_per, B, D), device=rs_in.device, dtype=rs_in.dtype,
        )
        dist.reduce_scatter_tensor(rs_out, rs_in, group=group)
        torch.cuda.synchronize()

    # Reset static buffers before capture
    out_static.copy_(out)
    lse_static.copy_(lse)

    # Capture CUDA graph (no events inside — they don't support elapsed_time)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        lse_contig = lse_static.contiguous()
        lses = torch.empty(
            (N,) + lse_contig.shape,
            device=lse_contig.device, dtype=lse_contig.dtype,
        )
        dist.all_gather_into_tensor(
            lses.view(N, -1), lse_contig.view(-1), group=group,
        )
        corrected, _ = correct_attn_out(
            out_static, lses, rank_in_group, ctx=None, is_lse_base_on_e=True,
        )
        rs_in = corrected.movedim(0, 1).contiguous()
        rs_out = torch.empty(
            (H_per, B, D), device=rs_in.device, dtype=rs_in.dtype,
        )
        dist.reduce_scatter_tensor(rs_out, rs_in, group=group)
        # Final movedim+contiguous matching vLLM's reduce_scatter return
        dcp_final = rs_out.movedim(0, 1).contiguous()  # [B, H/N, D]  # noqa: F841

    torch.cuda.synchronize()

    # Timing events OUTSIDE graph for total pipeline time
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    # Replay loop — per-op breakdown not available (use nsys for that)
    times = {'total': [], 'ag_lse': [], 'triton': [], 'rs_out': []}

    for i in range(warmup + iters):
        out_static.copy_(out)
        lse_static.copy_(lse)
        ev_start.record()
        graph.replay()
        ev_end.record()
        torch.cuda.synchronize()

        if i >= warmup:
            total = ev_start.elapsed_time(ev_end)
            times['total'].append(total)
            # Sub-op breakdown not available under CUDA graphs;
            # report as n/a (zeros) — use nsys for per-op timing
            times['ag_lse'].append(0.0)
            times['triton'].append(0.0)
            times['rs_out'].append(0.0)

    del graph
    return {k: _summarize(v) for k, v in times.items()}


def benchmark_helix_pipeline_graph(
    out: torch.Tensor,       # [B, H, D]
    lse: torch.Tensor,       # [B, H]
    group: dist.ProcessGroup,
    world_size: int,
    rank_in_group: int,
    warmup: int = 50,
    iters: int = 200,
) -> dict:
    """
    Benchmark the Helix reduction pipeline with CUDA graph capture/replay.

    Captures: permute+contiguous -> A2A(out,async) + A2A(lse,async) ->
              wait -> Triton combine

    With NCCL_GRAPH_MIXING_SUPPORT=0, EVENT_WAIT barriers before NCCL ops
    are removed, and the two adjacent A2A calls may be coalesced by NCCL.

    Note: Only total pipeline time is measured (events outside graph).
    Use nsys profiling for per-operation breakdown under CUDA graphs.
    """
    B, H, D = out.shape
    N = world_size
    H_per = H // N

    # Static input buffers
    out_static = out.clone()
    lse_static = lse.clone()

    # Eager warmup
    for _ in range(5):
        out_static.copy_(out)
        lse_static.copy_(lse)
        out_contig = out_static.contiguous()
        lse_contig = lse_static.contiguous()
        send_out = out_contig.view(B, N, H_per, D).permute(1, 0, 2, 3).contiguous()
        send_lse = lse_contig.view(B, N, H_per).permute(1, 0, 2).contiguous()
        recv_out = torch.empty_like(send_out)
        recv_lse = torch.empty_like(send_lse)
        w1 = dist.all_to_all_single(
            recv_out.view(-1), send_out.view(-1), group=group, async_op=True,
        )
        w2 = dist.all_to_all_single(
            recv_lse.view(-1), send_lse.view(-1), group=group, async_op=True,
        )
        w1.wait()
        w2.wait()
        helix_lse_combine_triton(
            recv_out, recv_lse, return_lse=False, is_lse_base_on_e=True,
        )
        torch.cuda.synchronize()

    # Reset
    out_static.copy_(out)
    lse_static.copy_(lse)

    # Capture CUDA graph (no events inside)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_contig = out_static.contiguous()
        lse_contig = lse_static.contiguous()
        send_out = out_contig.view(B, N, H_per, D).permute(1, 0, 2, 3).contiguous()
        send_lse = lse_contig.view(B, N, H_per).permute(1, 0, 2).contiguous()
        recv_out = torch.empty_like(send_out)
        recv_lse = torch.empty_like(send_lse)
        w1 = dist.all_to_all_single(
            recv_out.view(-1), send_out.view(-1), group=group, async_op=True,
        )
        w2 = dist.all_to_all_single(
            recv_lse.view(-1), send_lse.view(-1), group=group, async_op=True,
        )
        w1.wait()
        w2.wait()
        helix_lse_combine_triton(
            recv_out, recv_lse, return_lse=False, is_lse_base_on_e=True,
        )

    torch.cuda.synchronize()

    # Timing events OUTSIDE graph
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    # Replay loop
    times = {'total': [], 'permute_copy': [], 'a2a': [], 'triton': []}

    for i in range(warmup + iters):
        out_static.copy_(out)
        lse_static.copy_(lse)
        ev_start.record()
        graph.replay()
        ev_end.record()
        torch.cuda.synchronize()

        if i >= warmup:
            total = ev_start.elapsed_time(ev_end)
            times['total'].append(total)
            times['permute_copy'].append(0.0)
            times['a2a'].append(0.0)
            times['triton'].append(0.0)

    del graph
    return {k: _summarize(v) for k, v in times.items()}


def benchmark_helix_native_pipeline_graph(
    out: torch.Tensor,
    lse: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
    rank_in_group: int,
    workspace: torch.Tensor,
    warmup: int = 50,
    iters: int = 200,
) -> dict:
    """Helix native A2A with CUDA graph capture/replay."""
    B, H, D = out.shape
    N = world_size
    H_per = H // N

    out_static = out.clone()
    lse_static = lse.clone()

    # Eager warmup (trigger any lazy init)
    for _ in range(5):
        out_static.copy_(out)
        lse_static.copy_(lse)
        po = out_static.view(B, N, H_per, D).permute(0, 2, 1, 3) \
                        .reshape(B * H_per, N, D).contiguous()
        lse_1d = lse_static.view(B, N, H_per).permute(0, 2, 1) \
                            .reshape(B * H_per, N)
        ss = torch.stack([lse_1d, torch.zeros_like(lse_1d)], dim=-1).contiguous()
        ro, rs_out = helix_a2a.alltoall(po, ss, workspace, rank_in_group, N)
        recv_out = ro.reshape(B, H_per, N, D).permute(2, 0, 1, 3).contiguous()
        recv_lse = rs_out[..., 0].reshape(B, H_per, N).permute(2, 0, 1).contiguous()
        helix_lse_combine_triton(recv_out, recv_lse, return_lse=False, is_lse_base_on_e=True)
        torch.cuda.synchronize()

    out_static.copy_(out)
    lse_static.copy_(lse)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        po = out_static.view(B, N, H_per, D).permute(0, 2, 1, 3) \
                        .reshape(B * H_per, N, D).contiguous()
        lse_1d = lse_static.view(B, N, H_per).permute(0, 2, 1) \
                            .reshape(B * H_per, N)
        ss = torch.stack([lse_1d, torch.zeros_like(lse_1d)], dim=-1).contiguous()
        ro, rs_out = helix_a2a.alltoall(po, ss, workspace, rank_in_group, N)
        recv_out = ro.reshape(B, H_per, N, D).permute(2, 0, 1, 3).contiguous()
        recv_lse = rs_out[..., 0].reshape(B, H_per, N).permute(2, 0, 1).contiguous()
        helix_lse_combine_triton(recv_out, recv_lse, return_lse=False, is_lse_base_on_e=True)

    torch.cuda.synchronize()

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    times = {'total': [], 'permute_copy': [], 'a2a': [], 'triton': []}

    for i in range(warmup + iters):
        out_static.copy_(out)
        lse_static.copy_(lse)
        ev_start.record()
        graph.replay()
        ev_end.record()
        torch.cuda.synchronize()

        if i >= warmup:
            times['total'].append(ev_start.elapsed_time(ev_end))
            times['permute_copy'].append(0.0)
            times['a2a'].append(0.0)
            times['triton'].append(0.0)

    del graph
    return {k: _summarize(v) for k, v in times.items()}


# ============================================================================
# Helpers
# ============================================================================

def _summarize(vals: list[float]) -> dict:
    """Compute summary statistics (all values in ms)."""
    s = sorted(vals)
    n = len(s)
    return {
        'mean': statistics.mean(s),
        'p50': s[n // 2],
        'p5': s[max(0, n * 5 // 100)],
        'p95': s[min(n - 1, n * 95 // 100)],
        'min': s[0],
        'max': s[-1],
    }


def fmt_ms(ms: float) -> str:
    """Format milliseconds for display."""
    if ms < 0.01:
        return f"{ms * 1000:.1f}us"
    return f"{ms:.3f}ms"


def fmt_sub(ms: float) -> str:
    """Format sub-operation timing — 'n/a' if zero (graph mode)."""
    if ms == 0.0:
        return "n/a"
    return fmt_ms(ms)


def print_nccl_env(mode: str = 'eager'):
    """Print NCCL-related environment variables for the log."""
    nccl_vars = sorted(
        (k, v) for k, v in os.environ.items()
        if k.startswith('NCCL_') or k.startswith('NVIDIA_IMEX')
    )
    if nccl_vars:
        print("NCCL environment:")
        for k, v in nccl_vars:
            print(f"  {k}={v}")
    else:
        print("NCCL environment: (no NCCL_ vars set)")
    print()
    if mode == 'eager':
        print("NOTE: Eager mode — NCCL_GRAPH_MIXING_SUPPORT has no effect.")
    elif mode == 'graph':
        print("NOTE: CUDA graph mode — NCCL_GRAPH_MIXING_SUPPORT affects A2A coalescing.")
    else:
        print("NOTE: Running both eager and CUDA graph modes.")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full per-layer DCP vs Helix pipeline micro-benchmark",
    )
    parser.add_argument('--dcp', type=int, default=2,
                        help='DCP degree N (default: 2)')
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()),
                        default='nemotron49b',
                        help='Model config (default: nemotron49b)')
    parser.add_argument('--batch-sizes', type=str,
                        default='1,4,16,64,128,256,512',
                        help='Comma-separated batch sizes')
    parser.add_argument('--warmup', type=int, default=100,
                        help='Warmup iterations (default: 100)')
    parser.add_argument('--iters', type=int, default=500,
                        help='Timed iterations (default: 500)')
    parser.add_argument('--output', type=str, default=None,
                        help='JSON output file (rank 0 only)')
    parser.add_argument('--skip-verify', action='store_true',
                        help='Skip correctness verification')
    parser.add_argument('--mode', choices=['eager', 'graph', 'both'],
                        default='eager',
                        help='Benchmark mode: eager (default), graph (CUDA graph), both')
    parser.add_argument('--a2a-backend', choices=['nccl', 'native', 'both'],
                        default='nccl',
                        help='A2A backend for Helix path: nccl (default), '
                             'native (helix_a2a pkg), both')
    args = parser.parse_args()

    # Initialize distributed
    # Supports both torchrun (sets RANK, LOCAL_RANK, WORLD_SIZE) and
    # srun with SLURM env vars (SLURM_PROCID, SLURM_LOCALID, SLURM_NTASKS).
    # For multi-node Pyxis containers, srun-based init is more reliable
    # because torchrun's c10d rendezvous can't reach across containers.
    if 'RANK' not in os.environ and 'SLURM_PROCID' in os.environ:
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', device_id=torch.device(f'cuda:{local_rank}'))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    N = args.dcp
    assert world_size % N == 0, (
        f"world_size {world_size} not divisible by DCP degree {N}"
    )

    # Create DCP sub-groups (groups of N consecutive ranks)
    # E.g., world=16, N=2: groups are [0,1], [2,3], ..., [14,15]
    my_group = None
    my_rank_in_group = -1
    all_groups = []
    for base in range(0, world_size, N):
        ranks = list(range(base, base + N))
        pg = dist.new_group(ranks)
        all_groups.append(pg)
        if rank in ranks:
            my_group = pg
            my_rank_in_group = rank - base

    assert my_group is not None

    # Native A2A workspace (persistent across iterations)
    run_native = args.a2a_backend in ('native', 'both')
    native_workspace = None
    if run_native:
        if not HAS_HELIX_A2A:
            raise RuntimeError(
                "--a2a-backend native requires helix_a2a package. "
                "Install with: pip install -e /path/to/helix-a2a"
            )
        native_workspace = helix_a2a.allocate_workspace(
            cp_size=N,
            cp_rank=my_rank_in_group,
            device=f"cuda:{local_rank}",
            mnnvl="auto",
            cpu_group=my_group,
        )
        helix_a2a.init_workspace(native_workspace, cp_rank=my_rank_in_group, cp_size=N)
        if rank == 0:
            ws_bytes = helix_a2a.workspace_size(N)
            print(f"Native A2A workspace: {ws_bytes / 1024 / 1024:.1f} MiB/rank")

    # Compute H (number of gathered heads within DCP group)
    # In Helix GQA with TP=world_size, DCP=N:
    #   TPA = TP / DCP = world_size / N
    #   H = total_q_heads / TPA = total_q_heads * N / world_size
    cfg = MODEL_CONFIGS[args.model]
    H = cfg['total_q_heads'] * N // world_size
    D = cfg['head_dim']
    dtype_out = getattr(torch, cfg['dtype_out'])

    if H < 1:
        if rank == 0:
            print(f"ERROR: H={H} < 1 for model={args.model}, "
                  f"world_size={world_size}, DCP={N}. "
                  f"Need DCP * total_q_heads >= world_size.")
        dist.destroy_process_group()
        return

    if rank == 0:
        print(f"{'=' * 120}")
        print(f"Pipeline Micro-Benchmark: DCP vs Helix")
        print(f"{'=' * 120}")
        print(f"Model:        {cfg['name']}")
        print(f"World size:   {world_size} GPUs")
        print(f"DCP degree:   {N}")
        print(f"H (heads):    {H} (per DCP group, after TPA sharding)")
        print(f"D (dim):      {D}")
        print(f"Dtype:        {cfg['dtype_out']}")
        print(f"Warmup:       {args.warmup}, Iters: {args.iters}")
        print(f"Batch sizes:  {args.batch_sizes}")
        print(f"A2A backend:  {args.a2a_backend}")
        if HAS_HELIX_A2A:
            print(f"helix_a2a:    v{helix_a2a.__version__} (available)")
        else:
            print(f"helix_a2a:    not installed")
        print()
        print_nccl_env(args.mode)
        print("DCP pipeline:          AG(lse) -> Triton correct -> RS(out)")
        print("Helix pipeline:        permute+copy -> A2A(out)+A2A(lse) -> Triton combine")
        if run_native:
            print("Helix Native pipeline: permute+pad -> helix_a2a.alltoall (fused) -> Triton combine")
        print(f"Mode:         {args.mode}")
        print(f"{'=' * 120}")
        print()

    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]

    # ----------------------------------------------------------------
    # Step 0: Correctness verification
    # ----------------------------------------------------------------
    if not args.skip_verify:
        if rank == 0:
            print("--- Correctness Verification ---")

        # Use a fixed seed for reproducibility across ranks
        # (each rank gets different data from NCCL collectives, which is correct)
        # LSE values use abs(randn)+1 to simulate realistic log-sum-exp values
        # (real LSEs are always positive: log(sum(exp(scores))) >= 0)
        for B in [1, 16, 64]:
            torch.manual_seed(42 + rank)
            out_v = torch.randn(B, H, D, device='cuda', dtype=dtype_out)
            lse_v = torch.randn(B, H, device='cuda', dtype=torch.float32).abs() + 1.0

            dist.barrier()
            result = verify_correctness(
                out_v, lse_v, my_group, N, my_rank_in_group,
                native_workspace=native_workspace,
            )

            if rank == 0:
                # bf16 precision: ~0.0078 at magnitude 1, ~0.016 at magnitude 2.
                # After RS sum of N bf16 values, errors accumulate.
                # Threshold 0.05 allows for bf16 accumulation across ranks.
                status = "PASS" if result['max_abs_error'] < 0.05 else "FAIL"
                line = (
                    f"  B={B:>3}: Helix vs DCP: {status}  "
                    f"max_abs={result['max_abs_error']:.6f}  "
                    f"max_rel={result['max_rel_error']:.6f}  "
                    f"mean_abs={result['mean_abs_error']:.6f}  "
                    f"dcp_mean={result['dcp_mean']:.4f}  "
                    f"helix_mean={result['helix_mean']:.4f}  "
                    f"nan={result['dcp_has_nan']}/{result['helix_has_nan']}  "
                    f"inf={result['dcp_has_inf']}/{result['helix_has_inf']}"
                )
                print(line)

                if status == "FAIL":
                    print(f"  WARNING: Large numerical difference detected!")
                    print(f"  DCP output shape:   {result['dcp_shape']}")
                    print(f"  Helix output shape: {result['helix_shape']}")

                if 'native_max_abs_error' in result:
                    n_status = "PASS" if result['native_max_abs_error'] < 0.05 else "FAIL"
                    print(
                        f"         Native vs DCP: {n_status}  "
                        f"max_abs={result['native_max_abs_error']:.6f}  "
                        f"mean_abs={result['native_mean_abs_error']:.6f}  "
                        f"native_mean={result['native_mean']:.4f}  "
                        f"nan={result['native_has_nan']}  "
                        f"inf={result['native_has_inf']}"
                    )
                    if n_status == "FAIL":
                        print(f"  WARNING: Large native numerical difference!")
                        print(f"  Native output shape: {result['native_shape']}")

        if rank == 0:
            print()

    # ----------------------------------------------------------------
    # Step 1: Build list of modes to run
    # ----------------------------------------------------------------
    modes_to_run = []
    if args.mode in ('eager', 'both'):
        modes_to_run.append((
            'eager',
            benchmark_dcp_pipeline,
            benchmark_helix_pipeline,
            benchmark_helix_native_pipeline if run_native else None,
        ))
    if args.mode in ('graph', 'both'):
        modes_to_run.append((
            'graph',
            benchmark_dcp_pipeline_graph,
            benchmark_helix_pipeline_graph,
            benchmark_helix_native_pipeline_graph if run_native else None,
        ))

    # ----------------------------------------------------------------
    # Step 2: Run benchmarks for each mode
    # ----------------------------------------------------------------
    all_mode_results = {}  # mode_name -> list of per-batch results

    for mode_name, dcp_fn, helix_fn, native_fn in modes_to_run:
        if rank == 0:
            print(f"{'=' * 120}")
            print(f"Benchmarks — {mode_name.upper()} mode")
            print(f"{'=' * 120}")
            if native_fn is not None:
                header = (
                    f"{'B':>5}  "
                    f"{'DCP total':>10} {'AG':>8} {'Triton':>8} {'RS':>8}  "
                    f"{'Helix total':>11} {'Copy':>8} {'A2A':>8} {'Triton':>8}  "
                    f"{'Native total':>12} {'Copy':>8} {'A2A':>8} {'Triton':>8}  "
                    f"{'D(H/D)':>7} {'D(N/H)':>7}"
                )
            else:
                header = (
                    f"{'B':>5}  "
                    f"{'DCP total':>10} {'AG':>8} {'Triton':>8} {'RS':>8}  "
                    f"{'Helix total':>11} {'Copy':>8} {'A2A':>8} {'Triton':>8}  "
                    f"{'Delta':>7}"
                )
            print(header)
            print("-" * len(header))

        mode_results = []

        for B in batch_sizes:
            torch.manual_seed(123 + B + rank)
            out = torch.randn(B, H, D, device='cuda', dtype=dtype_out)
            lse = torch.randn(B, H, device='cuda', dtype=torch.float32).abs() + 1.0

            dist.barrier()

            dcp_res = dcp_fn(
                out, lse, my_group, N, my_rank_in_group,
                warmup=args.warmup, iters=args.iters,
            )

            dist.barrier()

            helix_res = helix_fn(
                out, lse, my_group, N, my_rank_in_group,
                warmup=args.warmup, iters=args.iters,
            )

            native_res = None
            if native_fn is not None:
                dist.barrier()
                native_res = native_fn(
                    out, lse, my_group, N, my_rank_in_group,
                    workspace=native_workspace,
                    warmup=args.warmup, iters=args.iters,
                )

            dist.barrier()

            if rank == 0:
                result = {
                    'batch_size': B,
                    'dcp_degree': N,
                    'world_size': world_size,
                    'model': args.model,
                    'mode': mode_name,
                    'H': H,
                    'D': D,
                    'dcp': dcp_res,
                    'helix': helix_res,
                }
                if native_res is not None:
                    result['helix_native'] = native_res
                mode_results.append(result)

                d = dcp_res
                h = helix_res
                delta_hd = (h['total']['p50'] / d['total']['p50'] - 1) * 100

                if native_res is not None:
                    n = native_res
                    delta_nh = (n['total']['p50'] / h['total']['p50'] - 1) * 100
                    print(
                        f"{B:>5}  "
                        f"{fmt_ms(d['total']['p50']):>10} "
                        f"{fmt_sub(d['ag_lse']['p50']):>8} "
                        f"{fmt_sub(d['triton']['p50']):>8} "
                        f"{fmt_sub(d['rs_out']['p50']):>8}  "
                        f"{fmt_ms(h['total']['p50']):>11} "
                        f"{fmt_sub(h['permute_copy']['p50']):>8} "
                        f"{fmt_sub(h['a2a']['p50']):>8} "
                        f"{fmt_sub(h['triton']['p50']):>8}  "
                        f"{fmt_ms(n['total']['p50']):>12} "
                        f"{fmt_sub(n['permute_copy']['p50']):>8} "
                        f"{fmt_sub(n['a2a']['p50']):>8} "
                        f"{fmt_sub(n['triton']['p50']):>8}  "
                        f"{delta_hd:>+6.1f}% {delta_nh:>+6.1f}%"
                    )
                else:
                    print(
                        f"{B:>5}  "
                        f"{fmt_ms(d['total']['p50']):>10} "
                        f"{fmt_sub(d['ag_lse']['p50']):>8} "
                        f"{fmt_sub(d['triton']['p50']):>8} "
                        f"{fmt_sub(d['rs_out']['p50']):>8}  "
                        f"{fmt_ms(h['total']['p50']):>11} "
                        f"{fmt_sub(h['permute_copy']['p50']):>8} "
                        f"{fmt_sub(h['a2a']['p50']):>8} "
                        f"{fmt_sub(h['triton']['p50']):>8}  "
                        f"{delta_hd:>+6.1f}%"
                    )

        all_mode_results[mode_name] = mode_results

        if rank == 0:
            print()
            print(f"Summary — {mode_name.upper()} (p50 values)")
            print(f"{'-' * 120}")
            for r in mode_results:
                d = r['dcp']
                h = r['helix']
                delta_hd = (h['total']['p50'] / d['total']['p50'] - 1) * 100

                h_total = h['total']['p50']
                breakdown = ""
                if h['permute_copy']['p50'] > 0:
                    pct_copy = h['permute_copy']['p50'] / h_total * 100 if h_total > 0 else 0
                    pct_a2a = h['a2a']['p50'] / h_total * 100 if h_total > 0 else 0
                    pct_tri = h['triton']['p50'] / h_total * 100 if h_total > 0 else 0
                    breakdown = f" [copy={pct_copy:.0f}% a2a={pct_a2a:.0f}% tri={pct_tri:.0f}%]"

                line = (
                    f"  B={r['batch_size']:<4}: DCP={fmt_ms(d['total']['p50'])} "
                    f"Helix={fmt_ms(h['total']['p50'])} "
                    f"Δ(H/D)={delta_hd:+.1f}%{breakdown}"
                )

                if 'helix_native' in r:
                    n = r['helix_native']
                    delta_nh = (n['total']['p50'] / h['total']['p50'] - 1) * 100
                    n_total = n['total']['p50']
                    n_breakdown = ""
                    if n['permute_copy']['p50'] > 0:
                        n_pct_copy = n['permute_copy']['p50'] / n_total * 100 if n_total > 0 else 0
                        n_pct_a2a = n['a2a']['p50'] / n_total * 100 if n_total > 0 else 0
                        n_pct_tri = n['triton']['p50'] / n_total * 100 if n_total > 0 else 0
                        n_breakdown = f" [copy={n_pct_copy:.0f}% a2a={n_pct_a2a:.0f}% tri={n_pct_tri:.0f}%]"
                    line += f"  Native={fmt_ms(n_total)} Δ(N/H)={delta_nh:+.1f}%{n_breakdown}"

                print(line)
            print()

    # ----------------------------------------------------------------
    # Cross-mode comparison (when running both modes)
    # ----------------------------------------------------------------
    if rank == 0 and len(all_mode_results) == 2:
        eager = {r['batch_size']: r for r in all_mode_results['eager']}
        graph = {r['batch_size']: r for r in all_mode_results['graph']}
        common_bs = sorted(set(eager) & set(graph))

        if common_bs:
            has_native = any('helix_native' in eager.get(b, {}) for b in common_bs)

            print(f"{'=' * 120}")
            print("Cross-mode comparison: CUDA Graph vs Eager (p50)")
            print(f"{'=' * 120}")
            if has_native:
                header = (
                    f"{'B':>5}  "
                    f"{'DCP eager':>10} {'DCP graph':>10} {'DCP Δ':>7}  "
                    f"{'Helix eager':>12} {'Helix graph':>12} {'Helix Δ':>8}  "
                    f"{'Native eager':>13} {'Native graph':>13} {'Native Δ':>9}"
                )
            else:
                header = (
                    f"{'B':>5}  "
                    f"{'DCP eager':>10} {'DCP graph':>10} {'DCP Δ':>7}  "
                    f"{'Helix eager':>12} {'Helix graph':>12} {'Helix Δ':>8}"
                )
            print(header)
            print("-" * len(header))

            for B in common_bs:
                e, g = eager[B], graph[B]
                de = e['dcp']['total']['p50']
                dg = g['dcp']['total']['p50']
                he = e['helix']['total']['p50']
                hg = g['helix']['total']['p50']
                d_delta = (dg / de - 1) * 100 if de > 0 else 0
                h_delta = (hg / he - 1) * 100 if he > 0 else 0

                line = (
                    f"{B:>5}  "
                    f"{fmt_ms(de):>10} {fmt_ms(dg):>10} {d_delta:>+6.1f}%  "
                    f"{fmt_ms(he):>12} {fmt_ms(hg):>12} {h_delta:>+7.1f}%"
                )

                if has_native and 'helix_native' in e and 'helix_native' in g:
                    ne = e['helix_native']['total']['p50']
                    ng = g['helix_native']['total']['p50']
                    n_delta = (ng / ne - 1) * 100 if ne > 0 else 0
                    line += f"  {fmt_ms(ne):>13} {fmt_ms(ng):>13} {n_delta:>+8.1f}%"

                print(line)
            print()

    # ----------------------------------------------------------------
    # Save JSON
    # ----------------------------------------------------------------
    if rank == 0 and args.output:
        # Flatten all mode results into a single list
        all_results = []
        for mode_results in all_mode_results.values():
            all_results.extend(mode_results)

        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")

    # Cleanup
    dist.barrier()
    for g in all_groups:
        dist.destroy_process_group(g)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
