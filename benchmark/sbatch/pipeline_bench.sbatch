#!/bin/bash
#SBATCH --job-name=helix-pipeline-bench
#SBATCH --partition=gb200nvl72_preprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=slurm-logs/pipeline-bench-%j.out
#SBATCH --error=slurm-logs/pipeline-bench-%j.err

# =============================================================================
# Pipeline Micro-Benchmark: DCP vs Helix Full Reduction Pipeline
# =============================================================================
#
# Measures per-operation breakdown (copies, NCCL collectives, Triton kernels)
# for both DCP and Helix reduction pipelines using torchrun (bypasses Ray).
#
# USAGE:
#   # Default: DCP=2 and DCP=4 with Nemotron-49B head config
#   sbatch pipeline_bench.sbatch
#
#   # Custom DCP degrees
#   sbatch --export=ALL,DCP_DEGREES="2 4 8" pipeline_bench.sbatch
#
#   # With nsys profiling (produces .nsys-rep files)
#   sbatch --export=ALL,NSYS_PROFILE=true pipeline_bench.sbatch
#
#   # CUDA graph mode only (no eager baseline)
#   sbatch --export=ALL,BENCH_MODE=graph pipeline_bench.sbatch
#
#   # Different model config
#   sbatch --export=ALL,MODEL=v2lite pipeline_bench.sbatch
#
#   # With native A2A backend (requires helix_a2a package)
#   sbatch --export=ALL,A2A_BACKEND=both pipeline_bench.sbatch
#
#   # Native-only A2A (skip NCCL helix path comparison)
#   sbatch --export=ALL,A2A_BACKEND=native pipeline_bench.sbatch
#
# =============================================================================

set -euo pipefail

# --- Configuration (override via --export) ---
DCP_DEGREES=${DCP_DEGREES:-"2 4 8 16"}
MODELS=${MODELS:-"nemotron49b v2lite"}
BATCH_SIZES=${BATCH_SIZES:-"1,4,16,64,128,256,512"}
WARMUP=${WARMUP:-100}
ITERS=${ITERS:-500}
NSYS_PROFILE=${NSYS_PROFILE:-false}
BENCH_MODE=${BENCH_MODE:-"both"}  # eager, graph, or both
A2A_BACKEND=${A2A_BACKEND:-"both"}   # nccl, native, or both

# --- Container config ---
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"nvcr.io/nvstaging/nim/helix-vllm:v16-blackwell"}
STORAGE_PATH=${STORAGE_PATH:-"/home/davidyu"}
HELIX_A2A_PATH=${HELIX_A2A_PATH:-"${STORAGE_PATH}/workspace/helix/helix-a2a"}
CONTAINER_MODEL_PATH="/home/davidyu"

# --- Derived ---
TP_SIZE=$((SLURM_NNODES * 4))
HEAD_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n1)
HEAD_IP=$(scontrol show node ${HEAD_NODE} | grep -oP 'NodeAddr=\K[^ ]+')
if [[ "${HEAD_IP}" =~ [a-zA-Z] ]]; then
    HEAD_IP=$(getent ahosts ${HEAD_IP} 2>/dev/null | grep -v ':' | head -1 | awk '{print $1}')
fi
if [ -z "${HEAD_IP}" ] || [[ "${HEAD_IP}" == 127.* ]]; then
    HEAD_IP=$(ssh ${HEAD_NODE} "hostname -I | awk '{print \$1}'" 2>/dev/null || echo "")
fi

SCRIPT_CONTAINER="${CONTAINER_MODEL_PATH}/workspace/helix/helix-a2a/benchmark/scripts/benchmark_helix_pipeline.py"
RESULT_DIR="${STORAGE_PATH}/profiling-results/pipeline-bench-${SLURM_JOB_ID}"

echo "============================================================"
echo "Pipeline Micro-Benchmark"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Nodes:        ${SLURM_NNODES} (${SLURM_NODELIST})"
echo "GPUs:         ${TP_SIZE}"
echo "Head:         ${HEAD_NODE} (${HEAD_IP})"
echo "Models:       ${MODELS}"
echo "DCP degrees:  ${DCP_DEGREES}"
echo "Batch sizes:  ${BATCH_SIZES}"
echo "Warmup/Iters: ${WARMUP}/${ITERS}"
echo "Mode:         ${BENCH_MODE}"
echo "A2A backend:  ${A2A_BACKEND}"
echo "nsys:         ${NSYS_PROFILE}"
echo "Container:    ${CONTAINER_IMAGE}"
echo "helix-a2a:    ${HELIX_A2A_PATH}"
echo "Results:      ${RESULT_DIR}"
echo "============================================================"
echo ""

# --- IMEX channels (GB200 multi-node NCCL) ---
IMEX_DIR="/dev/nvidia-caps-imex-channels"
IMEX_CHANNEL=""
if [ -d "${IMEX_DIR}" ] && [ "$(ls -A ${IMEX_DIR} 2>/dev/null)" ]; then
    IMEX_CHANNEL=$(ls ${IMEX_DIR}/ | head -n1 | sed 's/channel//')
    echo "Found IMEX channel: ${IMEX_CHANNEL}"
fi

CONTAINER_MOUNTS="${STORAGE_PATH}:${CONTAINER_MODEL_PATH}"
[ -n "${IMEX_CHANNEL}" ] && CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${IMEX_DIR}:${IMEX_DIR}"

# Mount helix-a2a package for native A2A backend
if [ "${A2A_BACKEND}" != "nccl" ] && [ -d "${HELIX_A2A_PATH}" ]; then
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${HELIX_A2A_PATH}:/opt/helix-a2a"
    echo "Mounting helix-a2a: ${HELIX_A2A_PATH} -> /opt/helix-a2a"
fi

# NCCL environment
ENV_VARS="NCCL_NVLS_ENABLE=1"
ENV_VARS="${ENV_VARS} NCCL_NET_GDR_LEVEL=5"
ENV_VARS="${ENV_VARS} NCCL_P2P_LEVEL=NVL"
ENV_VARS="${ENV_VARS} NCCL_IB_HCA=mlx5"
ENV_VARS="${ENV_VARS} NCCL_GRAPH_MIXING_SUPPORT=1"
[ -n "${IMEX_CHANNEL}" ] && ENV_VARS="${ENV_VARS} NVIDIA_IMEX_CHANNELS=${IMEX_CHANNEL}"

# Create result directory
mkdir -p "${RESULT_DIR}" 2>/dev/null || true

# Error handler
error_handler() {
    echo "[ERROR] Command failed at line $1: $2"
    echo "[ERROR] Exit code: $3"
}
trap 'error_handler ${LINENO} "$BASH_COMMAND" $?' ERR

# Total GPU count
TOTAL_GPUS=$((SLURM_NNODES * 4))

# --- Run benchmarks ---
# Uses srun with --ntasks-per-node=4 (one task per GPU) instead of torchrun.
# This avoids the c10d rendezvous port connectivity issue across Pyxis containers.
# SLURM sets SLURM_PROCID (global rank), SLURM_LOCALID (local rank),
# SLURM_NTASKS (world size) — the Python script picks these up automatically.

# Make helix-a2a importable (pre-built .so must exist from interactive build)
INSTALL_CMD=""
if [ "${A2A_BACKEND}" != "nccl" ] && [ -d "${HELIX_A2A_PATH}" ]; then
    INSTALL_CMD="export PYTHONPATH=/opt/helix-a2a:\$PYTHONPATH && "
fi

for MODEL in ${MODELS}; do
for DCP in ${DCP_DEGREES}; do
    echo ""
    echo "=========================================="
    echo "Model=${MODEL}, DCP=${DCP}"
    echo "=========================================="

    # Benchmark arguments
    BENCH_ARGS="${SCRIPT_CONTAINER}"
    BENCH_ARGS="${BENCH_ARGS} --dcp ${DCP}"
    BENCH_ARGS="${BENCH_ARGS} --model ${MODEL}"
    BENCH_ARGS="${BENCH_ARGS} --batch-sizes ${BATCH_SIZES}"
    BENCH_ARGS="${BENCH_ARGS} --warmup ${WARMUP}"
    BENCH_ARGS="${BENCH_ARGS} --iters ${ITERS}"
    BENCH_ARGS="${BENCH_ARGS} --mode ${BENCH_MODE}"
    BENCH_ARGS="${BENCH_ARGS} --a2a-backend ${A2A_BACKEND}"
    BENCH_ARGS="${BENCH_ARGS} --output ${CONTAINER_MODEL_PATH}/profiling-results/pipeline-bench-${SLURM_JOB_ID}/results_${MODEL}_dcp${DCP}.json"

    # Distributed init env vars (env:// backend)
    DIST_VARS="MASTER_ADDR=${HEAD_IP} MASTER_PORT=29500"

    # Optional nsys wrapping (wraps each individual process, not torchrun)
    if [ "${NSYS_PROFILE}" = "true" ]; then
        NSYS_OUT="${CONTAINER_MODEL_PATH}/profiling-results/pipeline-bench-${SLURM_JOB_ID}/nsys_${MODEL}_dcp${DCP}_rank\${SLURM_PROCID}"
        PYTHON_CMD="${INSTALL_CMD}nsys profile -t cuda,nvtx,nccl --cuda-graph-trace=node --force-overwrite=true -o ${NSYS_OUT} -s none python ${BENCH_ARGS}"
    else
        PYTHON_CMD="${INSTALL_CMD}python ${BENCH_ARGS}"
    fi

    echo "Launch: srun --ntasks=${TOTAL_GPUS} --ntasks-per-node=4"
    echo "Command: ${PYTHON_CMD}"
    echo ""

    srun --nodes=${SLURM_NNODES} --ntasks=${TOTAL_GPUS} --ntasks-per-node=4 \
        --container-image=${CONTAINER_IMAGE} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --container-workdir=/workspace \
        --no-container-mount-home \
        bash -c "export ${ENV_VARS} ${DIST_VARS} && ${PYTHON_CMD}" \
        2>&1 | tee "${RESULT_DIR}/log_${MODEL}_dcp${DCP}.txt"
done
done

echo ""
echo "============================================================"
echo "Pipeline Micro-Benchmark Complete"
echo "============================================================"
echo "Results: ${RESULT_DIR}"
ls -la "${RESULT_DIR}/" 2>/dev/null || true
echo "============================================================"
