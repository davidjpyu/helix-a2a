# Benchmark Integration — Design & Architecture

Phase 5 of `helix_a2a` adds a **third pipeline path** (Helix Native) to the
existing DCP-vs-Helix micro-benchmark, so a single run produces side-by-side
timing for all three reduction strategies.

---

## Why Three Paths?

Each layer of Helix context-parallel attention ends with a reduction step that
combines partial outputs and LSE (log-sum-exp) values across CP ranks. There
are three ways to do this:

| Path | Collectives | Triton Kernel | Source |
|------|-------------|---------------|--------|
| **DCP** | `all_gather(lse)` + `reduce_scatter(out)` | `correct_attn_out` | vLLM baseline |
| **Helix (NCCL)** | 2× `all_to_all_single` (out, lse) | `helix_lse_combine_triton` | vLLM helix branch |
| **Helix (Native)** | 1× `helix_a2a.alltoall` (fused) | `helix_lse_combine_triton` | This package |

The benchmark measures each path with identical inputs and reports per-operation
breakdowns so we can isolate exactly where the native kernel saves time.

---

## What Changed vs the Original Script

The original `benchmark_helix_pipeline.py` had DCP and Helix (NCCL) paths.
Phase 5 adds the following without modifying any existing path:

### 1. Guarded Import

```python
try:
    import helix_a2a
    HAS_HELIX_A2A = True
except ImportError:
    HAS_HELIX_A2A = False
```

The script remains runnable without `helix_a2a` installed — it just skips the
native path. This is controlled by the `--a2a-backend` CLI flag.

### 2. CLI Argument

```
--a2a-backend {nccl,native,both}   (default: nccl)
```

- `nccl`: Original behavior, no native path.
- `native`: Run native path only (still runs DCP and Helix NCCL for comparison).
- `both`: Run all three paths.

### 3. Workspace Lifecycle

The native A2A kernel uses a persistent workspace tensor (FIFO buffers for the
LL128 protocol). This is allocated **once** in `main()` after process group
creation and reused across all benchmark iterations:

```
main()
  ├── dist.new_group(...)                    # NCCL sub-groups
  ├── helix_a2a.allocate_workspace(...)      # allocate once
  ├── helix_a2a.init_workspace(...)          # init once (FIFO reset)
  └── for each (mode, batch_size):
        └── benchmark_helix_native_pipeline(workspace=...)  # reuses workspace
```

The `mnnvl="auto"` flag automatically selects MNNVL (fabric-handle) allocation
on multi-node GB200 and plain device memory on single-node / H200.

### 4. New Benchmark Functions

Two new functions mirror the structure of the existing NCCL helix functions:

| Function | Mode | What It Measures |
|----------|------|-----------------|
| `benchmark_helix_native_pipeline()` | Eager | Per-step CUDA event timing |
| `benchmark_helix_native_pipeline_graph()` | Graph | Total pipeline time under CUDA graph replay |

Both follow the same 4-step structure:

1. **Permute + pad** — reshape `[B, H, D]` → `[B*H_per, N, D]`, pad LSE to float2
2. **Native A2A** — single `helix_a2a.alltoall()` call
3. **Reshape back** — `[B*H_per, N, D]` → `[N, B, H_per, D]` for Triton
4. **Triton combine** — same `helix_lse_combine_triton` as NCCL path

### 5. Reshape Sequence (Critical Detail)

The native kernel expects `partial_o [entry_count, cp_size, D]` and
`softmax_stats [entry_count, cp_size, 2]`. The reshape from the benchmark's
`[B, H, D]` format is:

```
out [B, H, D]
  → view   [B, N, H_per, D]
  → permute [B, H_per, N, D]      (move cp_size to second-to-last)
  → reshape [B*H_per, N, D]       (flatten entry dims)

lse [B, H]
  → view   [B, N, H_per]
  → permute [B, H_per, N]
  → reshape [B*H_per, N]
  → stack   [B*H_per, N, 2]       (pad to float2: [lse, 0])
```

After `alltoall`, the inverse reshape recovers `[N, B, H_per, D]` for Triton:

```
recv_o [B*H_per, N, D]
  → reshape [B, H_per, N, D]
  → permute [N, B, H_per, D]

recv_s [B*H_per, N, 2]
  → [..., 0]                      (extract LSE from float2)
  → reshape [B, H_per, N]
  → permute [N, B, H_per]
```

### 6. Extended Correctness Verification

`verify_correctness()` now accepts an optional `native_workspace` parameter.
When provided, it runs the native A2A path and compares the final result
against DCP output with the same tolerance (max abs < 0.05 for bf16).

### 7. Extended Output

**Console table** — adds Native columns and a `Δ(N/H)` column showing the
percentage difference between native and NCCL helix total times:

```
    B   DCP total       AG   Triton       RS  Helix total     Copy      A2A   Triton  Native total     Copy      A2A   Triton  D(H/D) D(N/H)
    1    0.042ms  0.018ms  0.012ms  0.010ms     0.038ms  0.008ms  0.020ms  0.008ms      0.032ms  0.008ms  0.014ms  0.008ms  -9.5% -15.8%
```

**JSON output** — each result object gains an optional `helix_native` key:

```json
{
  "batch_size": 64,
  "dcp": { "total": {...}, "ag_lse": {...}, "triton": {...}, "rs_out": {...} },
  "helix": { "total": {...}, "permute_copy": {...}, "a2a": {...}, "triton": {...} },
  "helix_native": { "total": {...}, "permute_copy": {...}, "a2a": {...}, "triton": {...} }
}
```

**Cross-mode comparison** — when `--mode both`, the eager-vs-graph comparison
table includes a Native column.

### 8. sbatch Changes

`pipeline_bench.sbatch` gains:

- `A2A_BACKEND` env var (default `"both"`) passed as `--a2a-backend`
- Container mount for the `helix-a2a` package directory
- `pip install -e /opt/helix-a2a` at container startup when native is requested

---

## File Map

```
benchmark/
├── scripts/
│   └── benchmark_helix_pipeline.py   # Modified benchmark (DCP + Helix NCCL + Helix Native)
└── sbatch/
    └── pipeline_bench.sbatch         # SLURM launcher with A2A_BACKEND support
```

---

## Key Metrics to Watch

| Metric | What It Tells You |
|--------|-------------------|
| `helix.a2a.p50` | NCCL `all_to_all_single` latency (two calls) |
| `helix_native.a2a.p50` | Custom kernel A2A latency (one fused call) |
| `native.a2a / helix.a2a` | Direct A2A speedup factor |
| `helix.total` vs `helix_native.total` | End-to-end pipeline improvement |
| `permute_copy` times | Should be similar (same data volume) |
| `triton` times | Should be identical (same kernel) |

### Expected Scaling

| Scale | Expected Behavior |
|-------|-------------------|
| 1 node (4 GPU) | NVLink-only; native may match or slightly beat NCCL |
| 2 nodes (8 GPU) | Cross-node NVLink; native should show clear advantage |
| 4 nodes (16 GPU) | Larger win; NCCL A2A overhead grows with node count |
