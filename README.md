# helix-a2a

Standalone Python package wrapping TRT-LLM Helix All-to-All kernels (MNNVL/FIFO/LL128) as a CUDA extension.

## Install

Requires PyTorch with CUDA support and a GPU with SM >= 90 (Hopper/Blackwell).

```bash
python setup.py build_ext --inplace && pip install -e . --no-build-isolation
```

## Quick check

```bash
python -c "import helix_a2a; print(helix_a2a.workspace_size(4))"
```

## API

```python
import helix_a2a

# Query workspace size (bytes per rank)
ws_bytes = helix_a2a.workspace_size(cp_size=4)

# Allocate workspace tensor
workspace = helix_a2a.allocate_workspace(cp_size=4, cp_rank=rank, device="cuda")

# Initialize workspace (call once)
helix_a2a.init_workspace(workspace, cp_rank=rank, cp_size=4)

# Run all-to-all
out, stats = helix_a2a.alltoall(partial_o, softmax_stats, workspace, cp_rank=rank, cp_size=4)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HELIX_A2A_ENABLE_PDL` | `0` | Enable Programmatic Dependency Launch |
| `HELIX_A2A_USE_MNNVL` | `auto` | `0` = device mem, `1` = MNNVL, `auto` = auto-detect |
| `HELIX_A2A_CHANNEL_COUNT` | `0` | Override channel count (0 = auto) |

## Benchmark

See [docs/BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md) for how to run the
pipeline micro-benchmark (DCP / Helix NCCL / Helix Native side by side).

See [docs/BENCHMARK_DESIGN.md](docs/BENCHMARK_DESIGN.md) for the integration
architecture and what each measurement means.

## Development Status

- ~~Phase 1~~: Package skeleton with CUDA build scaffolding
- ~~Phase 2~~: Port real TRT-LLM kernels (SM90+ Hopper/Blackwell)
- ~~Phase 3~~: Python API + multi-GPU unit tests
- ~~Phase 4~~: MNNVL multi-node workspace (GB200 fabric handles)
- **Phase 5** (current): Benchmark integration
- Phase 6: Run benchmarks + analyze
