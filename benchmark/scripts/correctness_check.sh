#!/bin/bash
#SBATCH --job-name=helix-correctness
#SBATCH --partition=gb200nvl72
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/davidyu/workspace/helix/helix-a2a/benchmark/sbatch/slurm-logs/correctness-%j.out
#SBATCH --error=/home/davidyu/workspace/helix/helix-a2a/benchmark/sbatch/slurm-logs/correctness-%j.err
#SBATCH --exclusive
#SBATCH --mem=0
# =============================================================================
# Correctness check: compare NCCL vs Helix Native A2A output
# =============================================================================
#
# Runs the same prompts with temperature=0 on both backends and verifies
# that outputs match exactly.
#
# Usage:
#   cd ~/workspace/helix/helix-a2a/benchmark/sbatch
#   sbatch --export=ALL ../scripts/correctness_check.sh
#
# Environment:
#   MODEL             Model to test (default: deepseek-ai/DeepSeek-V2-Lite)
#   DCP_SIZE          DCP size (default: 2)
#   NUM_PROMPTS       Number of prompts to compare (default: 10)
#   MAX_TOKENS        Tokens to generate per prompt (default: 50)
#   TP_SIZE           Tensor parallel size (default: 16)
#   ATTN_BACKEND      Attention backend (default: FLASHINFER_MLA)
#
# =============================================================================
set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite}"
DCP_SIZE="${DCP_SIZE:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
MAX_TOKENS="${MAX_TOKENS:-50}"
TP_SIZE="${TP_SIZE:-16}"
ATTN_BACKEND="${ATTN_BACKEND:-FLASHINFER_MLA}"
PORT=8000

STORAGE_PATH="/home/sungsooh_storage"
HELIX_A2A_HOST="${HOME}/workspace/helix/helix-a2a"

# Site detection
HEAD_NODE=$(scontrol show hostnames "${SLURM_NODELIST}" | head -1)
HEAD_IP=$(getent hosts "${HEAD_NODE}" | awk '{print $1}')

CONTAINER_IMAGE="${STORAGE_PATH}/containers/helix-vllm-v20-blackwell.sqsh"
CONTAINER_MOUNTS="${STORAGE_PATH}:/models,${HELIX_A2A_HOST}:/helix-a2a"

IMEX_DIR="/dev/nvidia-caps-imex-channels"
if [ -d "${IMEX_DIR}" ] && [ "$(ls -A "${IMEX_DIR}" 2>/dev/null)" ]; then
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${IMEX_DIR}:${IMEX_DIR}"
fi

CONTAINER_ENV="HF_HUB_OFFLINE=1 HF_HOME=/tmp/hf_home TRANSFORMERS_CACHE=/models/.cache/huggingface/hub"
CONTAINER_ENV="${CONTAINER_ENV} NCCL_NVLS_ENABLE=1 NCCL_NET_GDR_LEVEL=5 NCCL_P2P_LEVEL=NVL"
CONTAINER_ENV="${CONTAINER_ENV} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enP5p9s0}"

RESULT_DIR="/tmp/correctness_${SLURM_JOB_ID}"

echo "============================================================"
echo "  Helix A2A Correctness Check"
echo "============================================================"
echo "  Job:    ${SLURM_JOB_ID}"
echo "  Nodes:  ${SLURM_NNODES} (${SLURM_NODELIST})"
echo "  Head:   ${HEAD_NODE} (${HEAD_IP})"
echo "  Model:  ${MODEL}"
echo "  TP:     ${TP_SIZE}, DCP: ${DCP_SIZE}"
echo "  Prompts: ${NUM_PROMPTS}, Max tokens: ${MAX_TOKENS}"
echo "============================================================"

# Generate inner script
INNER_SCRIPT="/tmp/correctness_inner_${SLURM_JOB_ID}.sh"
cat > "${INNER_SCRIPT}" << 'INNER_EOF'
#!/bin/bash
set -uo pipefail

eval "export ${BENCH_ENV}"
ulimit -l unlimited 2>/dev/null || true
mkdir -p /tmp/hf_home

timestamp() { echo "[$(date '+%H:%M:%S')]"; }

# Install Ray
if ! python3 -c "import ray" &>/dev/null; then
    echo "$(timestamp) $(hostname): Installing Ray..."
    pip install --no-cache-dir "ray[default]>=2.48.0" 2>&1 | tail -1
fi

# Install helix_a2a
if [ -d "/helix-a2a" ] && [ -f "/helix-a2a/install_in_container.sh" ]; then
    echo "$(timestamp) $(hostname): Installing helix_a2a..."
    bash /helix-a2a/install_in_container.sh /helix-a2a
fi

# Workers: join Ray and block
if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
    echo "$(timestamp) Worker $(hostname): waiting for Ray head..."
    sleep 30
    ray start --address="${BENCH_HEAD_IP}:6379" --num-gpus=4 --disable-usage-stats --block
    exit 0
fi

# === Head node ===
echo "$(timestamp) Starting Ray head..."
ray start --head --port=6379 --num-gpus=4 --disable-usage-stats
sleep 40

VLLM_DCP=$(python3 -c "import vllm.v1.attention.ops.dcp_alltoall as m; print(m.__file__)")
RESULT_DIR="${BENCH_RESULT_DIR}"
mkdir -p "${RESULT_DIR}/nccl" "${RESULT_DIR}/native"

generate_prompts() {
    python3 -c "
import json
prompts = [
    'The capital of France is',
    'In 1969, humans first landed on the',
    'The speed of light in vacuum is approximately',
    'Water boils at 100 degrees',
    'The largest planet in our solar system is',
    'Albert Einstein developed the theory of',
    'The Great Wall of China was built to',
    'Photosynthesis is the process by which plants',
    'The human genome contains approximately',
    'Machine learning is a subset of artificial intelligence that',
    'The Pythagorean theorem states that',
    'DNA stands for deoxyribonucleic',
    'The mitochondria is often called the',
    'Shakespeare wrote the play Romeo and',
    'Gravity on Earth accelerates objects at approximately',
]
for i in range(${BENCH_NUM_PROMPTS}):
    print(json.dumps(prompts[i % len(prompts)]))
"
}

wait_for_server() {
    local max_wait=600
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:${BENCH_PORT}/health" > /dev/null 2>&1; then
            echo "$(timestamp) Server ready (${waited}s)"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        [ $((waited % 60)) -eq 0 ] && echo "$(timestamp) Still waiting... (${waited}s)"
    done
    echo "$(timestamp) ERROR: Server did not start in ${max_wait}s"
    return 1
}

run_backend() {
    local backend=$1
    local output_dir="${RESULT_DIR}/${backend}"

    echo ""
    echo "$(timestamp) =========================================="
    echo "$(timestamp) Running backend: ${backend}"
    echo "$(timestamp) =========================================="

    export HELIX_A2A_BACKEND="${backend}"
    cp /helix-a2a/patches/dcp_alltoall_helix.py "${VLLM_DCP}"

    local vllm_cmd="vllm serve ${BENCH_MODEL}"
    vllm_cmd="${vllm_cmd} --tensor-parallel-size ${BENCH_TP_SIZE}"
    vllm_cmd="${vllm_cmd} --distributed-executor-backend ray"
    vllm_cmd="${vllm_cmd} --decode-context-parallel-size ${BENCH_DCP_SIZE}"
    vllm_cmd="${vllm_cmd} --dcp-comm-backend a2a"
    vllm_cmd="${vllm_cmd} --max-model-len 131072"
    vllm_cmd="${vllm_cmd} --attention-config.backend ${BENCH_ATTN_BACKEND}"
    vllm_cmd="${vllm_cmd} --trust-remote-code"
    vllm_cmd="${vllm_cmd} --enable-prefix-caching"
    vllm_cmd="${vllm_cmd} --disable-log-stats"
    vllm_cmd="${vllm_cmd} --port ${BENCH_PORT}"

    echo "$(timestamp) CMD: ${vllm_cmd}"
    eval "${vllm_cmd}" &
    local vllm_pid=$!

    if ! wait_for_server; then
        kill ${vllm_pid} 2>/dev/null
        return 1
    fi

    local i=0
    generate_prompts | while IFS= read -r prompt; do
        curl -s "http://localhost:${BENCH_PORT}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"${BENCH_MODEL}\",
                \"prompt\": ${prompt},
                \"max_tokens\": ${BENCH_MAX_TOKENS},
                \"temperature\": 0
            }" > "${output_dir}/prompt_${i}.json"

        local text
        text=$(python3 -c "import json; print(json.load(open('${output_dir}/prompt_${i}.json'))['choices'][0]['text'][:80])" 2>/dev/null)
        echo "$(timestamp) [${backend}] Prompt ${i}: ${text}..."
        i=$((i + 1))
    done

    echo "$(timestamp) Stopping vLLM (PID ${vllm_pid})..."
    kill ${vllm_pid} 2>/dev/null
    wait ${vllm_pid} 2>/dev/null
    sleep 10
}

# Run both backends
run_backend "nccl"
run_backend "native"

# Compare outputs
echo ""
echo "============================================================"
echo "  CORRECTNESS CHECK: NCCL vs Helix Native"
echo "  Model: ${BENCH_MODEL}"
echo "  DCP: ${BENCH_DCP_SIZE}, TP: ${BENCH_TP_SIZE}"
echo "  Prompts: ${BENCH_NUM_PROMPTS}, Max tokens: ${BENCH_MAX_TOKENS}"
echo "============================================================"

PASS=0; FAIL=0; ERROR=0

for i in $(seq 0 $((BENCH_NUM_PROMPTS - 1))); do
    nccl_file="${RESULT_DIR}/nccl/prompt_${i}.json"
    native_file="${RESULT_DIR}/native/prompt_${i}.json"

    if [ ! -f "${nccl_file}" ] || [ ! -f "${native_file}" ]; then
        echo "  Prompt ${i}: MISSING FILE"
        ERROR=$((ERROR + 1))
        continue
    fi

    result=$(python3 -c "
import json
try:
    n = json.load(open('${nccl_file}'))['choices'][0]['text']
    h = json.load(open('${native_file}'))['choices'][0]['text']
    if n == h:
        print('MATCH')
        print(f'  Output: {n[:120]}')
    else:
        print('MISMATCH')
        print(f'  NCCL:   {n[:120]}')
        print(f'  Native: {h[:120]}')
except Exception as e:
    print(f'ERROR: {e}')
")

    first_line=$(echo "${result}" | head -1)
    if [ "${first_line}" = "MATCH" ]; then
        echo "  Prompt ${i}: MATCH"
        echo "${result}" | tail -n +2
        PASS=$((PASS + 1))
    elif [[ "${first_line}" == ERROR* ]]; then
        echo "  Prompt ${i}: ${result}"
        ERROR=$((ERROR + 1))
    else
        echo "  Prompt ${i}: MISMATCH"
        echo "${result}" | tail -n +2
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "============================================================"
echo "  Results: ${PASS} MATCH, ${FAIL} MISMATCH, ${ERROR} ERROR"
echo "  out of ${BENCH_NUM_PROMPTS} prompts"
if [ ${FAIL} -eq 0 ] && [ ${ERROR} -eq 0 ]; then
    echo "  VERDICT: PASS — outputs are identical"
else
    echo "  VERDICT: FAIL — outputs differ"
fi
echo "============================================================"

ray stop 2>/dev/null
INNER_EOF

chmod +x "${INNER_SCRIPT}"

# Export variables for inner script
export BENCH_ENV="${CONTAINER_ENV}"
export BENCH_HEAD_IP="${HEAD_IP}"
export BENCH_MODEL="${MODEL}"
export BENCH_TP_SIZE="${TP_SIZE}"
export BENCH_DCP_SIZE="${DCP_SIZE}"
export BENCH_ATTN_BACKEND="${ATTN_BACKEND}"
export BENCH_NUM_PROMPTS="${NUM_PROMPTS}"
export BENCH_MAX_TOKENS="${MAX_TOKENS}"
export BENCH_PORT="${PORT}"
export BENCH_RESULT_DIR="${RESULT_DIR}"

# Launch in container
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir=/workspace \
    --no-container-mount-home \
    --export=ALL \
    bash "${INNER_SCRIPT}"

echo ""
echo "Done."
