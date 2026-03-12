#!/bin/bash
# =============================================================================
# Correctness check: compare NCCL vs Helix Native A2A output
# =============================================================================
#
# Runs the same prompts with temperature=0 on both backends and verifies
# that outputs match exactly. This confirms the reshape logic in the
# helix_a2a native path produces identical results to NCCL alltoall.
#
# Usage (on the login node):
#   sbatch --nodes=4 --gpus-per-node=4 --partition=gb200nvl72 --time=01:00:00 \
#       --output=slurm-logs/correctness-%j.out --error=slurm-logs/correctness-%j.err \
#       --container-image="/home/sungsooh_storage/containers/helix-vllm-v20-blackwell.sqsh" \
#       --container-mounts="/home/sungsooh_storage:/models,${HOME}/workspace/helix/helix-a2a:/helix-a2a" \
#       --container-workdir=/workspace --no-container-mount-home --export=ALL \
#       /helix-a2a/benchmark/scripts/correctness_check.sh
#
# Or interactively:
#   salloc -N4 --gpus-per-node=4 -p gb200nvl72 --time=01:00:00
#   srun ... bash /helix-a2a/benchmark/scripts/correctness_check.sh
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
set -uo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite}"
DCP_SIZE="${DCP_SIZE:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
MAX_TOKENS="${MAX_TOKENS:-50}"
TP_SIZE="${TP_SIZE:-16}"
ATTN_BACKEND="${ATTN_BACKEND:-FLASHINFER_MLA}"
PORT=8000
RESULT_DIR="/tmp/correctness_check_$$"

mkdir -p "${RESULT_DIR}"

timestamp() { echo "[$(date '+%H:%M:%S')]"; }

# =========================================================================
# Install dependencies (all nodes)
# =========================================================================
if ! python3 -c "import ray" &>/dev/null; then
    echo "$(timestamp) $(hostname): Installing Ray..."
    pip install --no-cache-dir "ray[default]>=2.48.0" 2>&1 | tail -1
fi

if [ -d "/helix-a2a" ] && [ -f "/helix-a2a/install_in_container.sh" ]; then
    echo "$(timestamp) $(hostname): Installing helix_a2a..."
    bash /helix-a2a/install_in_container.sh /helix-a2a
fi

# =========================================================================
# Worker nodes: join Ray and block
# =========================================================================
if [ "${SLURM_PROCID:-0}" -ne 0 ]; then
    echo "$(timestamp) Worker $(hostname): waiting for Ray head..."
    sleep 30
    HEAD_IP=$(scontrol show hostnames "${SLURM_NODELIST}" | head -1)
    HEAD_IP=$(getent hosts "${HEAD_IP}" | awk '{print $1}')
    ray start --address="${HEAD_IP}:6379" --num-gpus=4 --disable-usage-stats --block
    exit 0
fi

# =========================================================================
# Head node: orchestrate
# =========================================================================
echo "$(timestamp) Starting Ray head..."
ray start --head --port=6379 --num-gpus=4 --disable-usage-stats
sleep 40

VLLM_DCP=$(python3 -c "import vllm.v1.attention.ops.dcp_alltoall as m; print(m.__file__)")

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
for i in range(${NUM_PROMPTS}):
    print(json.dumps(prompts[i % len(prompts)]))
"
}

wait_for_server() {
    local max_wait=600
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "$(timestamp) Server ready (${waited}s)"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        if [ $((waited % 60)) -eq 0 ]; then
            echo "$(timestamp) Still waiting... (${waited}s)"
        fi
    done
    echo "$(timestamp) ERROR: Server did not start in ${max_wait}s"
    return 1
}

run_backend() {
    local backend=$1
    local output_dir="${RESULT_DIR}/${backend}"
    mkdir -p "${output_dir}"

    echo ""
    echo "$(timestamp) =========================================="
    echo "$(timestamp) Running backend: ${backend}"
    echo "$(timestamp) =========================================="

    export HELIX_A2A_BACKEND="${backend}"
    cp /helix-a2a/patches/dcp_alltoall_helix.py "${VLLM_DCP}"

    local vllm_cmd="vllm serve ${MODEL}"
    vllm_cmd="${vllm_cmd} --tensor-parallel-size ${TP_SIZE}"
    vllm_cmd="${vllm_cmd} --distributed-executor-backend ray"
    vllm_cmd="${vllm_cmd} --decode-context-parallel-size ${DCP_SIZE}"
    vllm_cmd="${vllm_cmd} --dcp-comm-backend a2a"
    vllm_cmd="${vllm_cmd} --max-model-len 131072"
    vllm_cmd="${vllm_cmd} --attention-config.backend ${ATTN_BACKEND}"
    vllm_cmd="${vllm_cmd} --trust-remote-code"
    vllm_cmd="${vllm_cmd} --enable-prefix-caching"
    vllm_cmd="${vllm_cmd} --disable-log-stats"
    vllm_cmd="${vllm_cmd} --port ${PORT}"

    echo "$(timestamp) CMD: ${vllm_cmd}"
    eval "${vllm_cmd}" &
    local vllm_pid=$!

    if ! wait_for_server; then
        kill ${vllm_pid} 2>/dev/null
        return 1
    fi

    local i=0
    generate_prompts | while IFS= read -r prompt; do
        curl -s "http://localhost:${PORT}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"${MODEL}\",
                \"prompt\": ${prompt},
                \"max_tokens\": ${MAX_TOKENS},
                \"temperature\": 0
            }" > "${output_dir}/prompt_${i}.json"

        local text
        text=$(python3 -c "import json; print(json.load(open('${output_dir}/prompt_${i}.json'))['choices'][0]['text'][:80])" 2>/dev/null)
        echo "$(timestamp) Prompt ${i}: ${text}..."
        i=$((i + 1))
    done

    echo "$(timestamp) Stopping vLLM (PID ${vllm_pid})..."
    kill ${vllm_pid} 2>/dev/null
    wait ${vllm_pid} 2>/dev/null
    sleep 10
}

# =========================================================================
# Run both backends
# =========================================================================
run_backend "nccl"
run_backend "native"

# =========================================================================
# Compare outputs
# =========================================================================
echo ""
echo "============================================================"
echo "  CORRECTNESS CHECK: NCCL vs Helix Native"
echo "  Model: ${MODEL}"
echo "  DCP: ${DCP_SIZE}, TP: ${TP_SIZE}"
echo "  Prompts: ${NUM_PROMPTS}, Max tokens: ${MAX_TOKENS}"
echo "============================================================"

PASS=0
FAIL=0
ERROR=0

for i in $(seq 0 $((NUM_PROMPTS - 1))); do
    nccl_file="${RESULT_DIR}/nccl/prompt_${i}.json"
    native_file="${RESULT_DIR}/native/prompt_${i}.json"

    if [ ! -f "${nccl_file}" ] || [ ! -f "${native_file}" ]; then
        echo "  Prompt ${i}: MISSING FILE"
        ERROR=$((ERROR + 1))
        continue
    fi

    result=$(python3 -c "
import json, sys
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
echo "  out of ${NUM_PROMPTS} prompts"
if [ ${FAIL} -eq 0 ] && [ ${ERROR} -eq 0 ]; then
    echo "  VERDICT: PASS — outputs are identical"
else
    echo "  VERDICT: FAIL — outputs differ"
fi
echo "============================================================"

# Cleanup
ray stop 2>/dev/null
rm -rf "${RESULT_DIR}"
