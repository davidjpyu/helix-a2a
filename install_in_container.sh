#!/bin/bash
# =============================================================================
# Install helix_a2a and patch vLLM inside a running Pyxis container.
# =============================================================================
#
# The Pyxis overlay is writable, so pip install and file overwrites work
# but don't persist across container restarts (which is fine — sbatch
# relaunches everything fresh each job).
#
# Usage:
#   bash install_in_container.sh [helix_a2a_dir]
#
# Arguments:
#   helix_a2a_dir  Path to helix-a2a source inside the container.
#                  Default: /helix-a2a (standard mount point).
#
# Environment:
#   HELIX_A2A_BACKEND  nccl|native (logged, not set by this script)
#
# =============================================================================
set -euo pipefail

HELIX_A2A_DIR="${1:-/helix-a2a}"

echo "=== [helix_a2a] Building from ${HELIX_A2A_DIR} ==="

cd "${HELIX_A2A_DIR}"

# If a pre-built .so exists for this Python/PyTorch version, pip will skip
# the build step. Otherwise it compiles the CUDA extension (~30-60s).
pip install -e . --no-build-isolation 2>&1 | tail -5

# Verify import
python3 -c "import helix_a2a; print(f'helix_a2a {helix_a2a.__version__} OK, ws_size(4)={helix_a2a.workspace_size(4)}')"

# --- Patch vLLM's dcp_alltoall.py ---
VLLM_DCP=$(python3 -c "
import vllm.v1.attention.ops.dcp_alltoall as m
print(m.__file__)
")
PATCH_FILE="${HELIX_A2A_DIR}/patches/dcp_alltoall_helix.py"

if [ ! -f "${PATCH_FILE}" ]; then
    echo "=== [helix_a2a] WARNING: patch file not found at ${PATCH_FILE}, skipping vLLM patch ==="
    exit 0
fi

echo "=== [helix_a2a] Patching ${VLLM_DCP} ==="

# Backup original into the writable overlay
cp "${VLLM_DCP}" "${VLLM_DCP}.bak"

# Overwrite with patched version
cp "${PATCH_FILE}" "${VLLM_DCP}"

# --- Patch gpu_worker.py: pre-init helix workspace before graph capture ---
if [ "${HELIX_A2A_BACKEND:-nccl}" = "native" ]; then
    VLLM_WORKER=$(python3 -c "
import vllm.v1.worker.gpu_worker as m
print(m.__file__)
")
    # Insert dcp_a2a_ensure_initialized() call right before capture_model().
    # This ensures MNNVL workspace allocation (which needs Gloo collectives)
    # completes while all ranks are synchronised, before CUDA graph capture.
    python3 -c "
import re, sys

src = open('${VLLM_WORKER}').read()
marker = 'kernel_warmup(self)'
if 'dcp_a2a_ensure_initialized' in src:
    print('=== [helix_a2a] gpu_worker already patched, skipping ===')
    sys.exit(0)
if marker not in src:
    print('=== [helix_a2a] WARNING: could not find kernel_warmup in gpu_worker, skipping ===')
    sys.exit(0)

import_line = 'from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_ensure_initialized'
init_block = '''
        # [helix_a2a] Pre-init workspace before CUDA graph capture.
        # MNNVL allocation requires Gloo collectives — must run while
        # all ranks are synchronised, before capture_model().
        if self.parallel_config.decode_context_parallel_size > 1:
            from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_ensure_initialized
            from vllm.distributed.parallel_state import get_dcp_group
            dcp_a2a_ensure_initialized(get_dcp_group())
'''
src = src.replace(
    marker,
    marker + init_block,
    1,
)
open('${VLLM_WORKER}', 'w').write(src)
print('=== [helix_a2a] gpu_worker patched: pre-init before graph capture ===')
"
fi

# Quick sanity import
python3 -c "
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce, _HELIX_BACKEND
print(f'=== [helix_a2a] dcp_alltoall patched, backend={_HELIX_BACKEND} ===')
"
