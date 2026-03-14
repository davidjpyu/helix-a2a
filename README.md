# helix-a2a

Standalone CUDA extension for the Helix All-to-All kernel — a fused LL128 FIFO-based replacement for two NCCL `all_to_all_single` calls in DCP (Decode Context Parallelism) attention reduction.

Ported from TRT-LLM's `helixAllToAll.cu`. Supports MNNVL (FABRIC handles) for cross-node GPU-to-GPU memory access on GB200/NVL72.

## Performance

Tested on 16 × GB200 NVL72 with Nemotron-49B via vLLM AIPerf (decode-only):

| Metric | Result |
|--------|--------|
| ITL improvement over NCCL | **3-5%** consistently |
| DCP overhead reduction | **30-50%** vs NCCL |
| Configs tested | DCP=2/4/8, TPA=2/4/8, ctx=32K-256K, c=1-64 |

See `work-tracker/a2a/docs/vllm-integration/BENCHMARK_RESULTS_V50.md` for full data.

## Install

Requires PyTorch with CUDA support and a GPU with SM >= 90 (Hopper/Blackwell).

```bash
pip install -e . --no-build-isolation
```

Quick check:

```bash
python -c "import helix_a2a; print(helix_a2a.workspace_size(4))"
```

## API

```python
import helix_a2a
import torch.distributed as dist

# 1. Allocate workspace (MNNVL for cross-node, requires process group)
workspace = helix_a2a.allocate_workspace(
    cp_size=4, cp_rank=rank, mnnvl="auto", cpu_group=group,
)

# 2. Initialize workspace (synchronous — GPU memset completes before return)
helix_a2a.init_workspace(workspace, cp_rank=rank, cp_size=4)

# 3. Cross-rank barrier (required before first alltoall with MNNVL)
dist.barrier(group)

# 4. All-to-all (call repeatedly during inference)
recv_o, recv_stats = helix_a2a.alltoall(
    partial_o, softmax_stats, workspace, cp_rank=rank, cp_size=4,
)
```

### Tensor shapes

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `partial_o` | `[..., cp_size, D]` | bf16/fp16 | Partial attention outputs |
| `softmax_stats` | `[..., cp_size, S]` | float32 | S >= 2, even. Slot 0 = LSE |
| `workspace` | `[cp_size, ws_elems]` | int64 | FIFO buffers + metadata |

### Workspace allocation modes

| `mnnvl=` | When to use |
|----------|-------------|
| `False` | Single-node, IPC shared memory (default) |
| `True` | Force MNNVL (requires FABRIC/IMEX) |
| `"auto"` | MNNVL on aarch64 (GB200), device memory otherwise |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HELIX_A2A_USE_MNNVL` | `auto` | `0` = device mem, `1` = MNNVL, `auto` = auto-detect |
| `HELIX_A2A_ENABLE_PDL` | `0` | Enable Programmatic Dependency Launch (SM 90+) |
| `HELIX_A2A_CHANNEL_COUNT` | `0` | Override channel count (0 = auto) |

## Architecture

```
csrc/
├── helix_alltoall.cu    # Kernel + workspace init + launch
├── helix_alltoall.h     # HelixAllToAllParams struct
├── ll128_proto.cuh      # LL128 FIFO protocol (pack/unpack/check)
├── helix_common.h       # Constants, alignment helpers
├── cuda_async_ops.cuh   # TMA/async copy primitives
└── torch_binding.cpp    # TORCH_LIBRARY registration

helix_a2a/
├── __init__.py          # Public API
├── _ops.py              # torch.ops.helix_a2a.* wrappers
└── mnnvl.py             # MNNVL allocation (CUDA driver API)

patches/
└── dcp_alltoall_helix.py  # vLLM integration (drop-in replacement)
```

## vLLM Integration

The `patches/dcp_alltoall_helix.py` is a drop-in replacement for `vllm/v1/attention/ops/dcp_alltoall.py`. Set `HELIX_A2A_BACKEND=native` to activate. See `install_in_container.sh` for automated patching inside Pyxis containers.
