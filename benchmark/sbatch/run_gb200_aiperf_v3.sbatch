#!/bin/bash
#SBATCH --job-name=helix-a2a-aiperf
#SBATCH --partition=gb200nvl72
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/davidyu/workspace/helix/helix-a2a/benchmark/sbatch/slurm-logs/helix-a2a-aiperf-%j.out
#SBATCH --error=/home/davidyu/workspace/helix/helix-a2a/benchmark/sbatch/slurm-logs/helix-a2a-aiperf-%j.err
#SBATCH --exclusive
#SBATCH --mem=0

# =============================================================================
# AIPerf Decode-Only Benchmark — Helix A2A Integration
# =============================================================================
#
# Adapted from sungsoo's run_gb200_aiperf_v3.sbatch. Adds:
#   - Mounts helix-a2a source from user's home dir into the container
#   - Builds helix_a2a and patches vLLM's dcp_alltoall.py on all nodes
#   - HELIX_A2A_BACKEND env var selects NCCL vs Native A2A kernel
#
# The helix_a2a install + vLLM patch happens inside the container's writable
# overlay at startup. Nothing persists after the job ends.
#
# Config format:
#   Each entry is dcp:context:comm:tpa — one entry per (config, context) pair.
#   Example: "2:32768:a2a:;1:32768::;4:65536:a2a:"
#   All concurrency levels run for each entry.
#
# Server restart logic:
#   Server restarts ONLY when dcp:comm:tpa changes.
#   Ray stays alive for the entire job. Only vLLM process is restarted.
#
# Usage:
#   mkdir -p slurm-logs
#
#   # With Helix Native A2A:
#   export HELIX_A2A_BACKEND=native
#   export HF_TOKEN=<your-token>
#   sbatch --export=ALL run_gb200_aiperf_v3.sbatch
#
#   # Baseline NCCL (no helix_a2a needed, but still installs it):
#   export HELIX_A2A_BACKEND=nccl
#   sbatch --export=ALL run_gb200_aiperf_v3.sbatch
#
#   # Dry-run:
#   DRY_RUN=true bash run_gb200_aiperf_v3.sbatch
#
# Environment overrides:
#   HELIX_A2A_BACKEND    nccl|native (default: native)
#   HELIX_A2A_HOST       Host path to helix-a2a source (default: ~/workspace/helix/helix-a2a)
#   HELIX_A2A_USE_MNNVL  0|1|auto (default: auto)
#   DRY_RUN              true to simulate (default: false)
#   CONFIGS              Semicolon-separated config list
#   CONCURRENCY          Comma-separated concurrency levels
#   FORCE                true to re-run completed benchmarks
#   HF_TOKEN             HuggingFace token
#   BENCHMARK_VERSION    Result folder version tag (default: v50)
#   MODEL                Model ID (default: deepseek-ai/DeepSeek-V2-Lite)
#   BACKEND              Attention backend (default: FLASHINFER_MLA)
#   CONTAINER_IMAGE      Container sqsh path
#   NUM_REQUESTS_PER_CONC  Requests per concurrency slot (default: 5)
#   MAX_MODEL_LEN        Max model length (default: 1048576)
#   SANITY               true for smoke test (c=1, 3 requests)
#   NSYS_PROFILE         true to wrap vLLM with nsys
#
# =============================================================================

# NO set -e: a failure in one config must NOT kill the entire job
set -uo pipefail

# =============================================================================
# Dry-Run Support
# =============================================================================

DRY_RUN="${DRY_RUN:-false}"

if [ "${DRY_RUN}" = "true" ]; then
    echo ""
    echo "=========================================="
    echo "  DRY-RUN MODE (no SLURM/GPUs required)"
    echo "=========================================="
    echo ""
    SLURM_JOB_ID="${SLURM_JOB_ID:-DRY_RUN}"
    SLURM_NNODES="${SLURM_NNODES:-4}"
    SLURM_NODELIST="${SLURM_NODELIST:-gb-nvl-147-compute[02-05]}"
fi

# =============================================================================
# Configuration
# =============================================================================

MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite}"
MODEL_SHORT="$(basename "${MODEL}")"
BACKEND="${BACKEND:-FLASHINFER_MLA}"
QUANTIZATION="${QUANTIZATION:-}"
FORCE="${FORCE:-false}"
SANITY="${SANITY:-false}"
NSYS_PROFILE="${NSYS_PROFILE:-false}"

TP_SIZE="${TP_SIZE:-$((SLURM_NNODES * 4))}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1024}"

CONCURRENCY="${CONCURRENCY:-1,2,4,8,16,32,64}"

NUM_REQUESTS_PER_CONC="${NUM_REQUESTS_PER_CONC:-5}"
RANDOM_SEED="${RANDOM_SEED:-42}"

# Helix A2A config
HELIX_A2A_BACKEND="${HELIX_A2A_BACKEND:-native}"
HELIX_A2A_HOST="${HELIX_A2A_HOST:-${HOME}/workspace/helix/helix-a2a}"

if [ "${SANITY}" = "true" ]; then
    CONCURRENCY="1"
    NUM_REQUESTS_PER_CONC=3
    FORCE=true
    echo "=== SANITY MODE: c=1, ${NUM_REQUESTS_PER_CONC} requests per slot ==="
fi

if [ "${NSYS_PROFILE}" = "true" ]; then
    echo "=== NSYS PROFILING ENABLED ==="
fi

# v3 config format: dcp:context:comm:tpa (semicolon-separated)
DEFAULT_CONFIGS="2:32768:a2a:;1:32768::;2:65536:a2a:;1:65536::"
CONFIGS="${CONFIGS:-${DEFAULT_CONFIGS}}"

# =============================================================================
# Storage & Node Setup
# =============================================================================

if [ "${DRY_RUN}" = "true" ]; then
    STORAGE_PATH="${STORAGE_PATH:-/mnt/cifs/home/sungsooh_storage}"
    echo "Storage: ${STORAGE_PATH} (DRY_RUN default)"
else
    STORAGE_PATH="${STORAGE_PATH:-/mnt/cifs/home/sungsooh_storage}"

    HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
    WORKER_NODES=$(scontrol show hostname "$SLURM_NODELIST" | tail -n+2)

    NODE_FEATURES=$(scontrol show node "${HEAD_NODE}" 2>/dev/null \
        | grep -i "AvailableFeatures" | cut -d'=' -f2 || echo "")
    if [[ "${NODE_FEATURES}" == *"santa_clara"* ]]; then
        STORAGE_PATH="/home/sungsooh_storage"
        echo "Site detection: ${HEAD_NODE} is in santa_clara -> direct NFS"
    else
        echo "Site detection: ${HEAD_NODE} features='${NODE_FEATURES}' -> CIFS mount"
    fi

    HEAD_IP=$(scontrol show node "${HEAD_NODE}" | grep -oP 'NodeAddr=\K[^ ]+')
    if [[ "${HEAD_IP}" =~ [a-zA-Z] ]]; then
        HEAD_IP=$(getent ahosts "${HEAD_IP}" 2>/dev/null | grep -v ':' | head -1 | awk '{print $1}')
    fi
    if [ -z "${HEAD_IP}" ] || [[ "${HEAD_IP}" == 127.* ]]; then
        HEAD_IP=$(ssh "${HEAD_NODE}" "hostname -I | awk '{print \$1}'" 2>/dev/null || echo "")
    fi
fi

if [ "${DRY_RUN}" = "true" ]; then
    HEAD_NODE="mock-head"
    WORKER_NODES="mock-worker1 mock-worker2 mock-worker3"
    HEAD_IP="10.0.0.1"
fi

CONTAINER_IMAGE="${CONTAINER_IMAGE:-${STORAGE_PATH}/containers/helix-vllm-v20-blackwell.sqsh}"
CONTAINER_MODEL_PATH="/models"

BENCHMARK_VERSION=${BENCHMARK_VERSION:-"v50"}
RESULT_STORAGE="${RESULT_STORAGE:-${STORAGE_PATH}}"
RESULT_BASE_HOST="${RESULT_STORAGE}/benchmark-results/${BENCHMARK_VERSION}-gb200-${TP_SIZE}gpu"
if [ "${RESULT_STORAGE}" = "${STORAGE_PATH}" ]; then
    RESULT_BASE="${CONTAINER_MODEL_PATH}/benchmark-results/${BENCHMARK_VERSION}-gb200-${TP_SIZE}gpu"
else
    RESULT_BASE="${RESULT_STORAGE}/benchmark-results/${BENCHMARK_VERSION}-gb200-${TP_SIZE}gpu"
fi

# =============================================================================
# Container Environment
# =============================================================================

if [ "${DRY_RUN}" != "true" ]; then
    IMEX_DIR="/dev/nvidia-caps-imex-channels"
    if [ -d "${IMEX_DIR}" ] && [ "$(ls -A "${IMEX_DIR}" 2>/dev/null)" ]; then
        IMEX_CHANNEL=$(ls "${IMEX_DIR}/" | head -n1 | sed 's/channel//')
        echo "MNNVL: Found IMEX channel ${IMEX_CHANNEL} (${IMEX_DIR})"
    else
        IMEX_CHANNEL=""
        echo "MNNVL: WARNING - No IMEX channels found at ${IMEX_DIR}"
    fi
else
    IMEX_CHANNEL=""
fi

CONTAINER_MOUNTS="${STORAGE_PATH}:${CONTAINER_MODEL_PATH}"
if [ "${RESULT_STORAGE}" != "${STORAGE_PATH}" ]; then
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${RESULT_STORAGE}:${RESULT_STORAGE}"
fi
if [ -n "${IMEX_CHANNEL}" ]; then
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${IMEX_DIR}:${IMEX_DIR}"
fi

# Mount helix-a2a source from user's home dir
if [ -d "${HELIX_A2A_HOST}" ]; then
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${HELIX_A2A_HOST}:/helix-a2a"
    echo "helix-a2a: Mounting ${HELIX_A2A_HOST} -> /helix-a2a"
else
    echo "WARNING: helix-a2a source not found at ${HELIX_A2A_HOST}"
    echo "  Set HELIX_A2A_HOST to the correct path, or the install step will be skipped."
fi

CONTAINER_ENV="HOME=/workspace"
CONTAINER_ENV="${CONTAINER_ENV} HF_HOME=${CONTAINER_MODEL_PATH}/.cache/huggingface"
CONTAINER_ENV="${CONTAINER_ENV} HF_HUB_CACHE=${CONTAINER_MODEL_PATH}/.cache/huggingface/hub"
if [ -n "${HF_TOKEN:-}" ]; then
    CONTAINER_ENV="${CONTAINER_ENV} HF_TOKEN=${HF_TOKEN}"
fi
CONTAINER_ENV="${CONTAINER_ENV} HF_HUB_OFFLINE=1"
CONTAINER_ENV="${CONTAINER_ENV} FLASHINFER_WORKSPACE_BASE=/workspace"
CONTAINER_ENV="${CONTAINER_ENV} TVM_HOME=/workspace/.cache/tvm"
CONTAINER_ENV="${CONTAINER_ENV} TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions"
if [ -n "${IMEX_CHANNEL}" ]; then
    CONTAINER_ENV="${CONTAINER_ENV} NVIDIA_IMEX_CHANNELS=${IMEX_CHANNEL}"
fi
CONTAINER_ENV="${CONTAINER_ENV} NCCL_NVLS_ENABLE=1 NCCL_NET_GDR_LEVEL=5 NCCL_P2P_LEVEL=NVL NCCL_IB_HCA=mlx5 NCCL_GRAPH_MIXING_SUPPORT=1"
CONTAINER_ENV="${CONTAINER_ENV} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enP5p9s0}"
CONTAINER_ENV="${CONTAINER_ENV} HELIX_A2A_BACKEND=${HELIX_A2A_BACKEND}"

# =============================================================================
# Time Management
# =============================================================================

JOB_START=$(date +%s)
SAFETY_MARGIN=600
MIN_CONFIG_SECS=$((${MIN_REMAINING:-50} * 60))

TIME_LIMIT_SECS=$((8 * 3600))
if [ "${DRY_RUN}" != "true" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    _tl=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
        | grep -oP 'TimeLimit=\K[^ ]+' || echo "")
    if [ -n "${_tl}" ] && [[ "${_tl}" != *"UNLIMITED"* ]]; then
        TIME_LIMIT_SECS=$(echo "${_tl}" | awk -F'[-:]' '{
            if(NF==4) print $1*86400+$2*3600+$3*60+$4;
            else if(NF==3) print $1*3600+$2*60+$3;
            else if(NF==2) print $1*60+$2;
            else print $1
        }')
    fi
fi
DEADLINE=$((JOB_START + TIME_LIMIT_SECS - SAFETY_MARGIN))

timestamp() { echo "[$(date '+%H:%M:%S') +$(( ($(date +%s) - JOB_START) / 60 ))m]"; }
remaining_min() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }

# =============================================================================
# Helper Functions
# =============================================================================

get_mode_name() {
    local dcp=$1 comm=${2:-} tpa=${3:-}
    if [ "$dcp" -eq 1 ]; then
        echo "pure_tp"
    elif [ -n "$tpa" ]; then
        local n="tpa${tpa}_dcp${dcp}"
        [ "$comm" = "a2a" ] && n="${n}_a2a"
        echo "$n"
    else
        local n="dcp${dcp}"
        [ "$comm" = "a2a" ] && n="${n}_a2a"
        echo "$n"
    fi
}

format_context() {
    local ctx=$1
    if [ "$ctx" -ge 1048576 ]; then echo "1M"
    elif [ "$ctx" -ge 524288 ]; then echo "512K"
    elif [ "$ctx" -ge 262144 ]; then echo "256K"
    elif [ "$ctx" -ge 131072 ]; then echo "128K"
    elif [ "$ctx" -ge 65536 ]; then echo "64K"
    else echo "32K"; fi
}

build_vllm_cmd() {
    local dcp=$1 comm=${2:-} tpa=${3:-}
    local cmd="vllm serve ${MODEL}"
    cmd="${cmd} --tensor-parallel-size ${TP_SIZE}"
    if [ "${SLURM_NNODES}" -gt 1 ]; then
        cmd="${cmd} --distributed-executor-backend ray"
    fi
    cmd="${cmd} --max-model-len ${MAX_MODEL_LEN}"
    cmd="${cmd} --attention-config.backend ${BACKEND}"
    cmd="${cmd} --trust-remote-code"
    if [ -n "${QUANTIZATION}" ]; then cmd="${cmd} --quantization ${QUANTIZATION}"; fi
    if [ "${dcp}" -gt 1 ]; then cmd="${cmd} --decode-context-parallel-size ${dcp}"; fi
    if [ -n "${comm}" ] && [ "${comm}" != "ag_rs" ]; then
        cmd="${cmd} --dcp-comm-backend ${comm}"
    fi
    if [ -n "${tpa}" ]; then
        cmd="${cmd} --tensor-parallel-size-attention ${tpa}"
    fi
    cmd="${cmd} --enable-prefix-caching"
    cmd="${cmd} --disable-log-stats --port 8000"
    echo "${cmd}"
}

point_exists() {
    local mode_name=$1 ctx=$2 conc=$3
    local result_dir="${RESULT_BASE_HOST}/${MODEL_SHORT}_tp${TP_SIZE}_${mode_name}"
    [ -f "${result_dir}/aiperf_ctx${ctx}_c${conc}.json" ]
}

# =============================================================================
# Parse Config Entries
# =============================================================================

parse_config() {
    IFS=':' read -r CFG_DCP CFG_CTX CFG_COMM CFG_TPA <<< "$1"
    CFG_COMM="${CFG_COMM:-}"
    CFG_TPA="${CFG_TPA:-}"
}

IFS=';' read -ra CONFIG_ARRAY <<< "${CONFIGS}"
CONFIGS_TOTAL=${#CONFIG_ARRAY[@]}

for i in "${!CONFIG_ARRAY[@]}"; do
    parse_config "${CONFIG_ARRAY[$i]}"
    if [ -z "${CFG_DCP}" ] || [ -z "${CFG_CTX}" ]; then
        echo "ERROR: Invalid config entry #$((i+1)): '${CONFIG_ARRAY[$i]}'"
        exit 1
    fi
    if ! [[ "${CFG_DCP}" =~ ^[0-9]+$ ]] || [ "${CFG_DCP}" -lt 1 ]; then
        echo "ERROR: Invalid DCP '${CFG_DCP}' in entry #$((i+1))"
        exit 1
    fi
    case "${CFG_COMM}" in
        ""|a2a|ag_rs) ;;
        *) echo "ERROR: Invalid comm '${CFG_COMM}' in entry #$((i+1))"; exit 1 ;;
    esac
done

# =============================================================================
# Cleanup
# =============================================================================

CONFIGS_COMPLETED=0

cleanup_on_exit() {
    if [ "${DRY_RUN}" = "true" ]; then return; fi
    echo ""
    echo "$(timestamp) === JOB ENDING ==="
    jobs -p | xargs -r kill 2>/dev/null || true
    sleep 3
    jobs -p | xargs -r kill -9 2>/dev/null || true

    for ext in out err; do
        local slurm_log="slurm-logs/helix-a2a-aiperf-${SLURM_JOB_ID}.${ext}"
        if [ -f "${slurm_log}" ]; then
            cp "${slurm_log}" "${RESULT_BASE_HOST}/slurm-${SLURM_JOB_ID}.${ext}" 2>/dev/null || true
        fi
    done

    echo ""
    echo "============================================================"
    echo "Helix A2A AIPerf Benchmark Complete"
    echo "Backend:  ${HELIX_A2A_BACKEND}"
    echo "Elapsed:  $(( ($(date +%s) - JOB_START) / 60 )) minutes"
    echo "Results:  ${RESULT_BASE_HOST}"
    echo "============================================================"
}
trap cleanup_on_exit EXIT INT TERM

# =============================================================================
# Count Total Points
# =============================================================================

IFS=',' read -ra CONC_ARRAY <<< "${CONCURRENCY}"
TOTAL_POINTS=0
TOTAL_TO_RUN=0
TOTAL_SKIP=0

for entry in "${CONFIG_ARRAY[@]}"; do
    parse_config "${entry}"
    mode_name=$(get_mode_name "$CFG_DCP" "$CFG_COMM" "$CFG_TPA")
    for conc in "${CONC_ARRAY[@]}"; do
        TOTAL_POINTS=$((TOTAL_POINTS + 1))
        if point_exists "$mode_name" "$CFG_CTX" "$conc" && [ "$FORCE" != "true" ]; then
            TOTAL_SKIP=$((TOTAL_SKIP + 1))
        else
            TOTAL_TO_RUN=$((TOTAL_TO_RUN + 1))
        fi
    done
done

# =============================================================================
# Print Plan
# =============================================================================

echo "============================================================"
echo "Helix A2A AIPerf Benchmark"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID:-interactive}"
echo "Nodes:        ${SLURM_NNODES:-4} (${SLURM_NODELIST:-manual})"
echo "GPUs:         ${TP_SIZE}"
echo "Head Node:    ${HEAD_NODE} (${HEAD_IP})"
echo ""
echo "Model:        ${MODEL}"
echo "Backend:      ${BACKEND}"
echo "Concurrency:  ${CONCURRENCY}"
echo "Force:        ${FORCE}"
echo "Container:    $(basename "${CONTAINER_IMAGE}")"
echo ""
echo "Helix A2A:"
echo "  Backend:    ${HELIX_A2A_BACKEND}"
echo "  Source:     ${HELIX_A2A_HOST}"
echo "  MNNVL:      ${HELIX_A2A_USE_MNNVL:-auto}"
echo ""
echo "AIPerf settings:"
echo "  Requests/conc:  ${NUM_REQUESTS_PER_CONC}"
echo "  Random seed:    ${RANDOM_SEED}"
echo "  Prefix caching: ENABLED (decode-only measurement)"
echo ""
echo "Storage:      ${STORAGE_PATH}"
echo "Results:      ${RESULT_BASE_HOST}"
echo ""

prev_key=""
for i in "${!CONFIG_ARRAY[@]}"; do
    parse_config "${CONFIG_ARRAY[$i]}"
    mode_name=$(get_mode_name "$CFG_DCP" "$CFG_COMM" "$CFG_TPA")
    server_key="${CFG_DCP}:${CFG_COMM}:${CFG_TPA}"
    ctx_label=$(format_context "$CFG_CTX")
    local_run=0; local_skip=0
    for conc in "${CONC_ARRAY[@]}"; do
        if point_exists "$mode_name" "$CFG_CTX" "$conc" && [ "$FORCE" != "true" ]; then
            local_skip=$((local_skip + 1))
        else
            local_run=$((local_run + 1))
        fi
    done
    if [ "$server_key" != "$prev_key" ]; then restart_marker="SERVER"; else restart_marker="      "; fi
    prev_key="$server_key"
    if [ "$local_skip" -eq "${#CONC_ARRAY[@]}" ] && [ "$FORCE" != "true" ]; then
        printf "  %s %2d. %-18s ctx=%-6s SKIP (all %d exist)\n" "$restart_marker" $((i+1)) "$mode_name" "$ctx_label" "$local_skip"
    elif [ "$local_skip" -gt 0 ]; then
        printf "  %s %2d. %-18s ctx=%-6s %d RUN, %d SKIP\n" "$restart_marker" $((i+1)) "$mode_name" "$ctx_label" "$local_run" "$local_skip"
    else
        printf "  %s %2d. %-18s ctx=%-6s %d RUN\n" "$restart_marker" $((i+1)) "$mode_name" "$ctx_label" "$local_run"
    fi
done
echo ""
echo "Total: ${TOTAL_POINTS} points (${TOTAL_TO_RUN} to run, ${TOTAL_SKIP} skip)"
echo "============================================================"

# =============================================================================
# DRY-RUN: Show execution plan and exit
# =============================================================================

if [ "${DRY_RUN}" = "true" ]; then
    echo ""
    echo "==================== EXECUTION PLAN ===================="
    echo ""
    prev_key=""
    for i in "${!CONFIG_ARRAY[@]}"; do
        parse_config "${CONFIG_ARRAY[$i]}"
        mode_name=$(get_mode_name "$CFG_DCP" "$CFG_COMM" "$CFG_TPA")
        server_key="${CFG_DCP}:${CFG_COMM}:${CFG_TPA}"
        ctx_label=$(format_context "$CFG_CTX")
        echo "--- [$(( i + 1 ))/${CONFIGS_TOTAL}] ${mode_name} ctx=${ctx_label} ---"
        if [ "$server_key" != "$prev_key" ]; then
            vllm_cmd=$(build_vllm_cmd "$CFG_DCP" "$CFG_COMM" "$CFG_TPA")
            echo "  >> START/RESTART vLLM (${server_key})"
            echo "  vLLM: ${vllm_cmd}"
            prev_key="$server_key"
        else
            echo "  >> SAME vLLM (${server_key})"
        fi
        echo "  Points:"
        for conc in "${CONC_ARRAY[@]}"; do
            if point_exists "$mode_name" "$CFG_CTX" "$conc" && [ "$FORCE" != "true" ]; then
                echo "    c=${conc}: SKIP"
            else
                echo "    c=${conc}: RUN ($((conc * NUM_REQUESTS_PER_CONC)) requests)"
            fi
        done
        echo ""
    done
    echo "========================================================"
    exit 0
fi

# =============================================================================
# Preparation: Create result directories
# =============================================================================

echo ""
echo "$(timestamp) Preparing result directories..."

SEEN_MODES=()
for entry in "${CONFIG_ARRAY[@]}"; do
    parse_config "${entry}"
    mode_name=$(get_mode_name "$CFG_DCP" "$CFG_COMM" "$CFG_TPA")
    already_seen=false
    for seen in "${SEEN_MODES[@]+"${SEEN_MODES[@]}"}"; do
        if [ "$seen" = "$mode_name" ]; then already_seen=true; break; fi
    done
    if [ "$already_seen" = "true" ]; then continue; fi
    SEEN_MODES+=("$mode_name")
    result_dir_host="${RESULT_BASE_HOST}/${MODEL_SHORT}_tp${TP_SIZE}_${mode_name}"
    mkdir -p "${result_dir_host}/logs" 2>/dev/null || true
    chmod -R a+rwx "${result_dir_host}" 2>/dev/null || true
    echo "  ${mode_name}: OK (${result_dir_host})"
done
echo "$(timestamp) Preparation complete."

# =============================================================================
# Job-Level Logging
# =============================================================================

JOB_INFO_FILE="${RESULT_BASE_HOST}/job_info_${SLURM_JOB_ID}.txt"
cat > "${JOB_INFO_FILE}" << JOBINFO
SLURM_JOB_ID=${SLURM_JOB_ID}
SLURM_NODELIST=${SLURM_NODELIST}
SLURM_NNODES=${SLURM_NNODES}
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
MODEL=${MODEL}
BACKEND=${BACKEND}
TP_SIZE=${TP_SIZE}
HELIX_A2A_BACKEND=${HELIX_A2A_BACKEND}
CONFIGS=${CONFIGS}
CONCURRENCY=${CONCURRENCY}
FORCE=${FORCE}
CONTAINER_IMAGE=${CONTAINER_IMAGE}
BENCHMARK_METHOD=aiperf_helix_a2a
PREFIX_CACHING=enabled
NUM_REQUESTS_PER_CONC=${NUM_REQUESTS_PER_CONC}
NSYS_PROFILE=${NSYS_PROFILE}
JOBINFO
echo "$(timestamp) Job info saved to ${JOB_INFO_FILE}"

# =============================================================================
# Generate Inner Benchmark Script
# =============================================================================

INNER_SCRIPT="${RESULT_BASE_HOST}/_benchmark_inner_${SLURM_JOB_ID}.sh"
INNER_SCRIPT_CONTAINER="${RESULT_BASE}/_benchmark_inner_${SLURM_JOB_ID}.sh"

echo "$(timestamp) Generating inner benchmark script..."

cat > "${INNER_SCRIPT}" << 'INNER_EOF'
#!/bin/bash
# =============================================================================
# Inner Benchmark Script (runs inside Pyxis container)
# =============================================================================
# Head node: install helix_a2a → Ray head → vLLM start/stop loop → AIPerf
# Workers:   install helix_a2a → Ray worker → block until done

set -uo pipefail

eval "export ${BENCH_CONTAINER_ENV}"
ulimit -l unlimited 2>/dev/null || true

if [ "${SLURM_PROCID}" -eq 0 ]; then
    export VLLM_HOST_IP="${BENCH_HEAD_IP}"
fi

timestamp() { echo "[$(date '+%H:%M:%S') +$(( ($(date +%s) - BENCH_JOB_START) / 60 ))m]"; }

# =========================================================================
# Install Ray on ALL nodes (no longer a default vLLM dep since v20+)
# =========================================================================
if ! python3 -c "import ray" &>/dev/null; then
    echo "$(timestamp) $(hostname): Installing Ray..."
    pip install --no-cache-dir "ray[default]>=2.48.0" 2>&1 | tail -3
fi

# =========================================================================
# Install helix_a2a and patch vLLM (ALL nodes, before Ray split)
# =========================================================================

if [ -d "/helix-a2a" ] && [ -f "/helix-a2a/install_in_container.sh" ]; then
    echo "$(timestamp) $(hostname): Installing helix_a2a and patching vLLM..."
    bash /helix-a2a/install_in_container.sh /helix-a2a
    echo "$(timestamp) $(hostname): helix_a2a ready (backend=${HELIX_A2A_BACKEND:-nccl})"
else
    echo "$(timestamp) $(hostname): /helix-a2a not found, running with stock vLLM"
fi

# =========================================================================
# Coordination: Head starts Ray first, workers wait for signal file
# =========================================================================

RAY_READY_FLAG="${BENCH_RESULT_BASE}/_ray_ready_${SLURM_JOB_ID}"
RAY_DONE_FLAG="${BENCH_RESULT_BASE}/_ray_done_${SLURM_JOB_ID}"

if [ "${SLURM_PROCID}" -ne 0 ]; then
    # === WORKER NODE ===
    echo "$(timestamp) Worker $(hostname): waiting for Ray head signal..."
    WAIT_SECS=0
    while [ ! -f "${RAY_READY_FLAG}" ]; do
        sleep 5
        WAIT_SECS=$((WAIT_SECS + 5))
        if [ $((WAIT_SECS % 30)) -eq 0 ]; then
            echo "$(timestamp) Worker $(hostname): still waiting... (${WAIT_SECS}s)"
        fi
        if [ ${WAIT_SECS} -ge 600 ]; then
            echo "$(timestamp) Worker $(hostname): TIMEOUT after ${WAIT_SECS}s"
            exit 1
        fi
    done
    echo "$(timestamp) Worker $(hostname): connecting to Ray at ${BENCH_HEAD_IP}:6379"
    ray start --address="${BENCH_HEAD_IP}:6379" --num-gpus=4 --disable-usage-stats
    echo "$(timestamp) Worker $(hostname): connected"
    while [ ! -f "${RAY_DONE_FLAG}" ]; do sleep 5; done
    echo "$(timestamp) Worker $(hostname): done, exiting"
    ray stop 2>/dev/null || true
    exit 0
fi

# =========================================================================
# HEAD NODE: Orchestrate benchmark
# =========================================================================

echo "$(timestamp) Head $(hostname): starting benchmark orchestration"
rm -f "${RAY_READY_FLAG}" "${RAY_DONE_FLAG}" 2>/dev/null || true

# --- Start Ray Head ---
if [ "${SLURM_NNODES}" -gt 1 ]; then
    echo "$(timestamp) Starting Ray head..."
    ray start --head --port=6379 --num-gpus=4 --node-ip-address="${BENCH_HEAD_IP}" --disable-usage-stats
    touch "${RAY_READY_FLAG}"
else
    echo "$(timestamp) Single-node mode: skipping Ray"
fi

# --- Install AIPerf ---
if ! command -v aiperf &>/dev/null; then
    echo "$(timestamp) Installing AIPerf..."
    NUMPY_VER_BEFORE=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    pip install --no-cache-dir aiperf 2>&1 | tail -3
    NUMPY_VER_AFTER=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
    if [ "${NUMPY_VER_BEFORE}" != "${NUMPY_VER_AFTER}" ]; then
        echo "$(timestamp) WARNING: aiperf changed numpy ${NUMPY_VER_BEFORE} -> ${NUMPY_VER_AFTER}. Restoring..."
        pip install --no-cache-dir "numpy==${NUMPY_VER_BEFORE}" 2>&1 | tail -1
    fi
fi
echo "$(timestamp) AIPerf: $(which aiperf 2>/dev/null || echo 'NOT FOUND')"

# --- Wait for workers ---
if [ "${SLURM_NNODES}" -gt 1 ]; then
    EXPECTED_GPUS=$((SLURM_NNODES * 4))
    echo "$(timestamp) Waiting for ${EXPECTED_GPUS} GPUs..."
    RAY_WAIT=0
    while [ ${RAY_WAIT} -lt 300 ]; do
        AVAIL_GPUS=$(python3 -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
        if [ "${AVAIL_GPUS}" -ge "${EXPECTED_GPUS}" ]; then
            echo "$(timestamp) Ray cluster ready: ${AVAIL_GPUS} GPUs"
            break
        fi
        sleep 5
        RAY_WAIT=$((RAY_WAIT + 5))
        if [ $((RAY_WAIT % 30)) -eq 0 ]; then
            echo "$(timestamp) Ray: ${AVAIL_GPUS}/${EXPECTED_GPUS} GPUs (${RAY_WAIT}s)"
        fi
    done
fi

# --- Parse configs ---
IFS=';' read -ra CONFIG_ARRAY <<< "${CONFIGS}"
IFS=',' read -ra CONC_ARRAY <<< "${CONCURRENCY}"
CONFIGS_TOTAL=${#CONFIG_ARRAY[@]}

RESULT_BASE="${BENCH_RESULT_BASE}"
MODEL="${BENCH_MODEL}"
MODEL_SHORT="$(basename "${MODEL}")"
TP_SIZE="${BENCH_TP_SIZE}"
BACKEND="${BENCH_BACKEND}"
MAX_MODEL_LEN="${BENCH_MAX_MODEL_LEN}"
QUANTIZATION="${BENCH_QUANTIZATION}"
OUTPUT_TOKENS="${BENCH_OUTPUT_TOKENS}"
NUM_REQUESTS_PER_CONC="${BENCH_NUM_REQUESTS_PER_CONC}"
RANDOM_SEED="${BENCH_RANDOM_SEED}"
FORCE="${BENCH_FORCE}"
SANITY="${BENCH_SANITY}"
NSYS_PROFILE="${BENCH_NSYS_PROFILE:-false}"
HEAD_IP="${BENCH_HEAD_IP}"
DEADLINE="${BENCH_DEADLINE}"
MIN_CONFIG_SECS="${BENCH_MIN_CONFIG_SECS}"

parse_config() {
    IFS=':' read -r CFG_DCP CFG_CTX CFG_COMM CFG_TPA <<< "$1"
    CFG_COMM="${CFG_COMM:-}"; CFG_TPA="${CFG_TPA:-}"
}
get_mode_name() {
    local dcp=$1 comm=${2:-} tpa=${3:-}
    if [ "$dcp" -eq 1 ]; then echo "pure_tp"
    elif [ -n "$tpa" ]; then local n="tpa${tpa}_dcp${dcp}"; [ "$comm" = "a2a" ] && n="${n}_a2a"; echo "$n"
    else local n="dcp${dcp}"; [ "$comm" = "a2a" ] && n="${n}_a2a"; echo "$n"; fi
}
format_context() {
    local ctx=$1
    if [ "$ctx" -ge 1048576 ]; then echo "1M"; elif [ "$ctx" -ge 524288 ]; then echo "512K"
    elif [ "$ctx" -ge 262144 ]; then echo "256K"; elif [ "$ctx" -ge 131072 ]; then echo "128K"
    elif [ "$ctx" -ge 65536 ]; then echo "64K"; else echo "32K"; fi
}
point_exists() {
    [ -f "${RESULT_BASE}/${MODEL_SHORT}_tp${TP_SIZE}_$1/aiperf_ctx${2}_c${3}.json" ]
}
remaining_min() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }

build_vllm_cmd() {
    local dcp=$1 comm=${2:-} tpa=${3:-}
    local cmd="vllm serve ${MODEL}"
    cmd="${cmd} --tensor-parallel-size ${TP_SIZE}"
    if [ "${SLURM_NNODES}" -gt 1 ]; then cmd="${cmd} --distributed-executor-backend ray"; fi
    cmd="${cmd} --max-model-len ${MAX_MODEL_LEN} --attention-config.backend ${BACKEND} --trust-remote-code"
    if [ -n "${QUANTIZATION}" ]; then cmd="${cmd} --quantization ${QUANTIZATION}"; fi
    if [ "${dcp}" -gt 1 ]; then cmd="${cmd} --decode-context-parallel-size ${dcp}"; fi
    if [ -n "${comm}" ] && [ "${comm}" != "ag_rs" ]; then cmd="${cmd} --dcp-comm-backend ${comm}"; fi
    if [ -n "${tpa}" ]; then cmd="${cmd} --tensor-parallel-size-attention ${tpa}"; fi
    cmd="${cmd} --enable-prefix-caching --disable-log-stats --port 8000"
    echo "${cmd}"
}

stop_vllm() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "$(timestamp) Stopping vLLM (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        local waited=0
        while kill -0 "${VLLM_PID}" 2>/dev/null && [ $waited -lt 30 ]; do
            sleep 1; waited=$((waited + 1))
        done
        if kill -0 "${VLLM_PID}" 2>/dev/null; then
            kill -9 "${VLLM_PID}" 2>/dev/null || true
        fi
        wait "${VLLM_PID}" 2>/dev/null || true
        VLLM_PID=""
        echo "$(timestamp) vLLM stopped. Waiting for GPU cleanup..."
        sleep 10
    fi
}

start_vllm() {
    local dcp=$1 comm=$2 tpa=$3 mode_name=$4
    local log_dir="${RESULT_BASE}/${MODEL_SHORT}_tp${TP_SIZE}_${mode_name}/logs"
    local log_file="${log_dir}/server_${SLURM_JOB_ID:-$$}_${mode_name}.log"

    VLLM_CMD=$(build_vllm_cmd "$dcp" "$comm" "$tpa")
    echo "$(timestamp) Starting vLLM: ${VLLM_CMD}"

    ${VLLM_CMD} > "${log_file}" 2>&1 &
    VLLM_PID=$!
    echo "$(timestamp) vLLM started (PID ${VLLM_PID})"

    echo "$(timestamp) Waiting for server health..."
    local MAX_WAIT=1800 WAITED=0
    SERVER_READY=false

    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        if ! kill -0 ${VLLM_PID} 2>/dev/null; then
            echo "$(timestamp) ERROR: vLLM process died"
            tail -50 "${log_file}" 2>/dev/null || true
            VLLM_PID=""
            return 1
        fi
        if curl -s --max-time 10 http://${HEAD_IP}:8000/health > /dev/null 2>&1; then
            echo "$(timestamp) Health OK (${WAITED}s). Waiting for model load..."
            local MODEL_WAIT=0
            while [ ${MODEL_WAIT} -lt 600 ]; do
                if curl -s --max-time 10 http://${HEAD_IP}:8000/v1/models 2>/dev/null | grep -q '"id"'; then
                    echo "$(timestamp) Model loaded!"
                    SERVER_READY=true
                    break
                fi
                sleep 5; MODEL_WAIT=$((MODEL_WAIT + 5))
            done
            break
        fi
        sleep 10; WAITED=$((WAITED + 10))
        if [ $((WAITED % 60)) -eq 0 ]; then echo "$(timestamp) Still waiting... (${WAITED}s)"; fi
    done

    if [ "${SERVER_READY}" != "true" ]; then
        echo "$(timestamp) ERROR: Server failed to start for ${mode_name}."
        stop_vllm
        return 1
    fi

    echo "$(timestamp) CUDA graph warmup..."
    curl -s http://${HEAD_IP}:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL}\", \"prompt\": \"Hello\", \"max_tokens\": 1}" > /dev/null 2>&1 || true
    echo "$(timestamp) Warmup done"
    return 0
}

run_aiperf() {
    local mode_name=$1 ctx=$2 conc=$3 point_num=$4 total_points=$5
    local result_dir="${RESULT_BASE}/${MODEL_SHORT}_tp${TP_SIZE}_${mode_name}"
    local EXPORT_PREFIX="${result_dir}/aiperf_ctx${ctx}_c${conc}"
    local RESULT_FILE="${result_dir}/aiperf_ctx${ctx}_c${conc}.json"

    local INPUT_LEN=$((ctx - OUTPUT_TOKENS - 100))
    local REQUEST_COUNT=$((conc * NUM_REQUESTS_PER_CONC))
    if [ "${SANITY}" = "true" ]; then REQUEST_COUNT=$((conc * 3)); fi

    local ctx_label; ctx_label=$(format_context "$ctx")
    echo ""
    echo "$(timestamp) --- [${point_num}/${total_points}] ${mode_name} ctx=${ctx_label} c=${conc} requests=${REQUEST_COUNT} ---"

    if [ -f "${RESULT_FILE}" ] && [ "${FORCE}" != "true" ]; then
        echo "$(timestamp)   SKIP: already exists"; return 0
    fi

    echo "$(timestamp)   Running AIPerf..."
    local BENCH_START=$(date +%s)
    local AIPERF_LOG="${result_dir}/logs/aiperf_ctx${ctx}_c${conc}_${SLURM_JOB_ID:-$$}.log"

    COLUMNS=200 aiperf profile \
        --ui-type none \
        --model "${MODEL}" \
        --url "http://${HEAD_IP}:8000" \
        --endpoint-type completions \
        --streaming \
        --random-seed "${RANDOM_SEED}" \
        --prompt-input-tokens-mean "${INPUT_LEN}" \
        --prompt-input-tokens-stddev 0 \
        --prompt-output-tokens-mean "${OUTPUT_TOKENS}" \
        --num-dataset-entries 1 \
        --dataset-sampling-strategy sequential \
        --concurrency "${conc}" \
        --request-count "${REQUEST_COUNT}" \
        --warmup-request-count "${conc}" \
        --warmup-concurrency "${conc}" \
        --profile-export-prefix "${EXPORT_PREFIX}" \
        > "${AIPERF_LOG}" 2>&1
    local AIPERF_EXIT=$?

    cat "${AIPERF_LOG}" >> "${result_dir}/logs/benchmarks_${SLURM_JOB_ID:-$$}.log" 2>/dev/null
    local BENCH_DURATION=$(( $(date +%s) - BENCH_START ))

    if [ -f "${RESULT_FILE}" ]; then
        echo "${SLURM_JOB_ID:-unknown}" > "${result_dir}/aiperf_ctx${ctx}_c${conc}.source"
        echo "$(timestamp)   OK (${BENCH_DURATION}s)"
    elif [ ${AIPERF_EXIT} -ne 0 ]; then
        echo "$(timestamp)   FAILED exit=${AIPERF_EXIT} (${BENCH_DURATION}s)"
        tail -5 "${AIPERF_LOG}" 2>/dev/null | sed 's/^/    /'
    else
        echo "$(timestamp)   WARNING: no output file (${BENCH_DURATION}s)"
    fi
}

# =========================================================================
# Main Benchmark Loop
# =========================================================================

VLLM_PID=""
prev_server_key=""
POINT_NUM=0
CONFIGS_COMPLETED=0

trap 'echo "$(timestamp) Inner script interrupted"; stop_vllm; if [ "${SLURM_NNODES}" -gt 1 ]; then ray stop; touch "${RAY_DONE_FLAG}"; fi; exit 1' INT TERM

for cfg_idx in "${!CONFIG_ARRAY[@]}"; do
    parse_config "${CONFIG_ARRAY[$cfg_idx]}"
    dcp="${CFG_DCP}"; ctx="${CFG_CTX}"; comm="${CFG_COMM}"; tpa="${CFG_TPA}"
    mode_name=$(get_mode_name "$dcp" "$comm" "$tpa")
    server_key="${dcp}:${comm}:${tpa}"
    ctx_label=$(format_context "$ctx")

    echo ""
    echo "$(timestamp) ======== [$(( cfg_idx + 1 ))/${CONFIGS_TOTAL}] ${mode_name} ctx=${ctx_label} ========"

    # Entry-level skip check
    if [ "$FORCE" != "true" ]; then
        _all_exist=true
        for _conc in "${CONC_ARRAY[@]}"; do
            if ! point_exists "$mode_name" "$ctx" "$_conc"; then _all_exist=false; break; fi
        done
        if [ "$_all_exist" = "true" ]; then
            echo "$(timestamp) SKIP: all points exist for ${mode_name} ctx=${ctx_label}"
            CONFIGS_COMPLETED=$((CONFIGS_COMPLETED + 1))
            continue
        fi
    fi

    # Time check
    remaining=$(remaining_min)
    if [ "$remaining" -lt "$((MIN_CONFIG_SECS / 60))" ]; then
        echo "$(timestamp) TIME: ${remaining}m remaining < minimum. Stopping."
        break
    fi

    # vLLM restart if server config changed
    if [ "$server_key" != "$prev_server_key" ]; then
        if [ -n "$prev_server_key" ]; then
            echo "$(timestamp) Server key changed: ${prev_server_key} -> ${server_key}"
        fi
        stop_vllm
        if ! start_vllm "$dcp" "$comm" "$tpa" "$mode_name"; then
            echo "$(timestamp) ERROR: vLLM failed for ${mode_name}. Skipping."
            prev_server_key=""
            continue
        fi
        prev_server_key="$server_key"
    else
        echo "$(timestamp) Reusing vLLM (${server_key})"
    fi

    # Run all concurrency levels
    for conc in "${CONC_ARRAY[@]}"; do
        POINT_NUM=$((POINT_NUM + 1))
        run_aiperf "$mode_name" "$ctx" "$conc" "$POINT_NUM" "$TOTAL_POINTS"
    done
    CONFIGS_COMPLETED=$((CONFIGS_COMPLETED + 1))
done

# --- Cleanup ---
echo ""
echo "$(timestamp) All configs processed (${CONFIGS_COMPLETED}/${CONFIGS_TOTAL})."
stop_vllm
if [ "${SLURM_NNODES}" -gt 1 ]; then
    echo "$(timestamp) Stopping Ray..."
    ray stop
    touch "${RAY_DONE_FLAG}"
    sleep 30
    rm -f "${RAY_READY_FLAG}" "${RAY_DONE_FLAG}" 2>/dev/null || true
fi
echo "$(timestamp) Inner script complete."
INNER_EOF

chmod +x "${INNER_SCRIPT}"
echo "$(timestamp) Inner script: ${INNER_SCRIPT}"

# =============================================================================
# Export variables for inner script
# =============================================================================

export BENCH_CONTAINER_ENV="${CONTAINER_ENV}"
export BENCH_HEAD_IP="${HEAD_IP}"
export BENCH_JOB_START="${JOB_START}"
export BENCH_RESULT_BASE="${RESULT_BASE}"
export BENCH_MODEL="${MODEL}"
export BENCH_TP_SIZE="${TP_SIZE}"
export BENCH_BACKEND="${BACKEND}"
export BENCH_MAX_MODEL_LEN="${MAX_MODEL_LEN}"
export BENCH_QUANTIZATION="${QUANTIZATION}"
export BENCH_OUTPUT_TOKENS="${OUTPUT_TOKENS}"
export BENCH_NUM_REQUESTS_PER_CONC="${NUM_REQUESTS_PER_CONC}"
export BENCH_RANDOM_SEED="${RANDOM_SEED}"
export BENCH_FORCE="${FORCE}"
export BENCH_SANITY="${SANITY}"
export BENCH_NSYS_PROFILE="${NSYS_PROFILE}"
export BENCH_DEADLINE="${DEADLINE}"
export BENCH_MIN_CONFIG_SECS="${MIN_CONFIG_SECS}"
export TOTAL_POINTS
export CONFIGS
export CONCURRENCY

# =============================================================================
# Launch
# =============================================================================

echo ""
echo "$(timestamp) ============================================================"
echo "$(timestamp) Launching benchmark (helix_a2a backend=${HELIX_A2A_BACKEND})"
echo "$(timestamp) Container: $(basename "${CONTAINER_IMAGE}")"
echo "$(timestamp) Nodes: ${SLURM_NNODES} (${SLURM_NODELIST})"
echo "$(timestamp) ============================================================"
echo ""

srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    --export=ALL \
    bash "${INNER_SCRIPT_CONTAINER}"

SRUN_EXIT=$?

echo ""
echo "$(timestamp) srun exited with code ${SRUN_EXIT}"
echo "$(timestamp) All config entries processed."
