# Benchmark Guide — How to Run

Step-by-step instructions for running the DCP / Helix (NCCL) / Helix (Native)
pipeline micro-benchmark on a SLURM cluster with GB200 or H200 nodes.

---

## Prerequisites

### 1. Container Image

You need a container with vLLM's helix branch installed (provides the Triton
kernels `helix_lse_combine_triton` and `correct_attn_out`). The default image
path in the sbatch is:

```
/mnt/cifs/home/sungsooh_storage/containers/helix-vllm-v16-blackwell.sqsh
```

Override with `CONTAINER_IMAGE` if yours is elsewhere.

### 2. Install helix_a2a (for native path)

The `helix_a2a` package must be compiled on a machine with CUDA and a
SM >= 90 GPU (Hopper/Blackwell). Build it once:

```bash
cd /path/to/helix-a2a
pip install -e . --no-build-isolation
```

For SLURM runs, the sbatch script mounts the source directory into the
container and runs `pip install -e` automatically. Set `HELIX_A2A_PATH` to
point to your local clone:

```bash
export HELIX_A2A_PATH=/path/to/helix-a2a
```

The default is `${STORAGE_PATH}/helix-a2a`.

### 3. SLURM Cluster Access

The sbatch targets the `gb200nvl72_preprod` partition with 4 nodes and 4 GPUs
per node (16 GPUs total). Adjust `#SBATCH` directives for your cluster.

### 4. IMEX Channels (GB200 Multi-Node)

For cross-node NCCL on GB200, ensure `nvidia-fabricmanager` / IMEX daemon is
running and `/dev/nvidia-caps-imex-channels` is accessible. The sbatch detects
and mounts this automatically.

---

## Running the Benchmark

### Option A: sbatch (Recommended)

```bash
# All three paths (DCP + Helix NCCL + Helix Native), eager + graph modes
sbatch --export=ALL,A2A_BACKEND=both benchmark/sbatch/pipeline_bench.sbatch

# NCCL-only (no helix_a2a package needed)
sbatch benchmark/sbatch/pipeline_bench.sbatch

# Native-only A2A comparison
sbatch --export=ALL,A2A_BACKEND=native benchmark/sbatch/pipeline_bench.sbatch
```

### Option B: Direct srun

For quick iteration without sbatch:

```bash
srun --nodes=4 --ntasks=16 --ntasks-per-node=4 \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts="${STORAGE_PATH}:/models,/path/to/helix-a2a:/opt/helix-a2a" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    bash -c "
        export NCCL_NVLS_ENABLE=1 NCCL_NET_GDR_LEVEL=5 NCCL_P2P_LEVEL=NVL \
               NCCL_IB_HCA=mlx5 NCCL_GRAPH_MIXING_SUPPORT=0 \
               MASTER_ADDR=<head-ip> MASTER_PORT=29500 && \
        pip install -e /opt/helix-a2a && \
        python /models/helix-vllm/benchmark/scripts/benchmark_helix_pipeline.py \
            --dcp 4 --a2a-backend both --mode eager \
            --output /models/results.json
    "
```

### Option C: torchrun (Single Node, Development)

For single-node testing during development:

```bash
torchrun --nproc-per-node=4 \
    benchmark/scripts/benchmark_helix_pipeline.py \
    --dcp 4 --a2a-backend both --mode eager \
    --batch-sizes 1,16,64 --warmup 20 --iters 100
```

---

## Configuration Reference

All options can be set via sbatch `--export` or as environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DCP_DEGREES` | `"2 4 8 16"` | Space-separated DCP degrees to sweep |
| `MODELS` | `"nemotron49b v2lite"` | Model configs to benchmark |
| `BATCH_SIZES` | `"1,4,16,64,128,256,512"` | Comma-separated batch sizes |
| `WARMUP` | `100` | Warmup iterations (not timed) |
| `ITERS` | `500` | Timed iterations |
| `BENCH_MODE` | `"both"` | `eager`, `graph`, or `both` |
| `A2A_BACKEND` | `"both"` | `nccl`, `native`, or `both` |
| `NSYS_PROFILE` | `false` | Set `true` for nsys timeline capture |
| `CONTAINER_IMAGE` | (see sbatch) | Squashfs container image path |
| `STORAGE_PATH` | (see sbatch) | Base storage mount |
| `HELIX_A2A_PATH` | `${STORAGE_PATH}/helix-a2a` | Path to helix-a2a source |

### CLI Arguments (Python Script)

```
--dcp N              DCP degree (default: 2)
--model MODEL        nemotron49b or v2lite (default: nemotron49b)
--batch-sizes LIST   Comma-separated batch sizes (default: 1,4,16,64,128,256,512)
--warmup N           Warmup iterations (default: 100)
--iters N            Timed iterations (default: 500)
--mode MODE          eager, graph, or both (default: eager)
--a2a-backend BACK   nccl, native, or both (default: nccl)
--output FILE        JSON output path (rank 0 only)
--skip-verify        Skip correctness checks
```

---

## Understanding the Output

### Console Table

With `--a2a-backend both`, rank 0 prints a table like:

```
    B   DCP total       AG   Triton       RS  Helix total     Copy      A2A   Triton  Native total     Copy      A2A   Triton  D(H/D) D(N/H)
    1    0.042ms  0.018ms  0.012ms  0.010ms     0.038ms  0.008ms  0.020ms  0.008ms      0.032ms  0.008ms  0.014ms  0.008ms  -9.5% -15.8%
   16    0.089ms  0.025ms  0.018ms  0.044ms     0.072ms  0.012ms  0.038ms  0.018ms      0.058ms  0.012ms  0.024ms  0.018ms -19.1% -19.4%
```

Column groups:
- **DCP**: AllGather LSE + Triton correct + ReduceScatter out
- **Helix**: permute+copy + 2× NCCL all_to_all_single + Triton combine
- **Native**: permute+pad + 1× fused helix_a2a.alltoall + Triton combine
- **Δ(H/D)**: `(helix_total / dcp_total - 1) × 100%` — negative = helix faster
- **Δ(N/H)**: `(native_total / helix_total - 1) × 100%` — negative = native faster

### Correctness Output

Before benchmarking, the script verifies all paths agree within bf16 tolerance:

```
--- Correctness Verification ---
  B=  1: Helix vs DCP: PASS  max_abs=0.001953  ...
         Native vs DCP: PASS  max_abs=0.001953  ...
  B= 16: Helix vs DCP: PASS  max_abs=0.003906  ...
         Native vs DCP: PASS  max_abs=0.003906  ...
```

Both comparisons should show PASS with `max_abs < 0.05`.

### JSON Results

Results are saved to the path specified by `--output`. Each entry contains:

```json
{
  "batch_size": 64,
  "dcp_degree": 4,
  "world_size": 16,
  "model": "nemotron49b",
  "mode": "eager",
  "H": 16,
  "D": 128,
  "dcp": {
    "total": {"mean": 0.089, "p50": 0.087, "p5": 0.082, "p95": 0.098, "min": 0.080, "max": 0.112},
    "ag_lse": {...}, "triton": {...}, "rs_out": {...}
  },
  "helix": {
    "total": {...}, "permute_copy": {...}, "a2a": {...}, "triton": {...}
  },
  "helix_native": {
    "total": {...}, "permute_copy": {...}, "a2a": {...}, "triton": {...}
  }
}
```

### Quick JSON Analysis

```python
import json

with open("results_nemotron49b_dcp4.json") as f:
    results = json.load(f)

for r in results:
    if 'helix_native' not in r:
        continue
    B = r['batch_size']
    nccl_a2a = r['helix']['a2a']['p50']
    native_a2a = r['helix_native']['a2a']['p50']
    speedup = nccl_a2a / native_a2a if native_a2a > 0 else float('inf')
    print(f"B={B:>4}: NCCL={nccl_a2a:.3f}ms  Native={native_a2a:.3f}ms  Speedup={speedup:.2f}x")
```

---

## Sanity Checks

After a run, verify:

1. **Correctness**: All PASS, no FAIL lines. Both Helix-vs-DCP and Native-vs-DCP.
2. **No NaN/Inf**: `nan=False/False`, `inf=False/False` in correctness output.
3. **Triton times match**: `helix.triton ≈ helix_native.triton` (same kernel).
4. **Copy times similar**: `helix.permute_copy ≈ helix_native.permute_copy`.
5. **JSON written**: Check `${RESULT_DIR}/results_*.json` exists and is valid JSON.

---

## Troubleshooting

### `--a2a-backend native` fails with RuntimeError

```
RuntimeError: --a2a-backend native requires helix_a2a package.
```

**Fix**: Install helix_a2a in the container. Either:
- Set `HELIX_A2A_PATH` to your local clone (sbatch mounts + installs it), or
- Pre-install in the container image.

### MNNVL allocation fails (multi-node)

```
CUDA driver API error: CUDA_ERROR_NOT_PERMITTED
```

**Fix**: Ensure:
- `nvidia-fabricmanager` / IMEX daemon is running on all nodes
- `/dev/nvidia-caps-imex-channels` is mounted in the container
- Container runs with `--privileged` or appropriate capabilities

For single-node testing, set `HELIX_A2A_USE_MNNVL=0` to force device memory.

### CUDA graph capture fails with native path

The native A2A kernel must support CUDA graph capture. If it doesn't, use
`--mode eager` to skip graph mode. The eager path always works.

### Results show 0.0 for sub-op timings in graph mode

This is expected. CUDA events inside a captured graph don't support
`elapsed_time()`. Only the total pipeline time is measured. Use `nsys` profiling
(`NSYS_PROFILE=true`) for per-operation breakdown under CUDA graphs.

### H < 1 error

```
ERROR: H=0 < 1 for model=nemotron49b, world_size=16, DCP=2
```

**Fix**: The DCP degree is too small relative to world size. With Nemotron-49B
(64 Q heads) and TP=16, you need `DCP * 64 >= 16 * TP_per_group`. Try a larger
DCP degree.

---

## Example Workflows

### Quick Development Test (Single Node, 4 GPU)

```bash
torchrun --nproc-per-node=4 \
    benchmark/scripts/benchmark_helix_pipeline.py \
    --dcp 4 --a2a-backend both --mode eager \
    --batch-sizes 1,16,64 --warmup 10 --iters 50
```

### Full Sweep (4 Nodes, 16 GPU)

```bash
sbatch --export=ALL,A2A_BACKEND=both,DCP_DEGREES="2 4 8 16",BENCH_MODE=both \
    benchmark/sbatch/pipeline_bench.sbatch
```

### Profile with nsys

```bash
sbatch --export=ALL,A2A_BACKEND=native,NSYS_PROFILE=true,DCP_DEGREES="4" \
    benchmark/sbatch/pipeline_bench.sbatch
```

Produces `.nsys-rep` files in the results directory. Open with:

```bash
nsys stats results/nsys_nemotron49b_dcp4_rank0.nsys-rep
```

### Compare Across DCP Degrees

```bash
sbatch --export=ALL,A2A_BACKEND=both,DCP_DEGREES="2 4 8 16",BENCH_MODE=eager \
    benchmark/sbatch/pipeline_bench.sbatch
```

Then analyze:

```python
import json, glob

for path in sorted(glob.glob("results/results_nemotron49b_dcp*.json")):
    with open(path) as f:
        for r in json.load(f):
            if 'helix_native' not in r or r['mode'] != 'eager':
                continue
            dcp = r['dcp_degree']
            B = r['batch_size']
            h = r['helix']['total']['p50']
            n = r['helix_native']['total']['p50']
            print(f"DCP={dcp} B={B:>4}: NCCL={h:.3f}ms Native={n:.3f}ms Δ={(n/h-1)*100:+.1f}%")
```
