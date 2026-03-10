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

# Quick sanity import
python3 -c "
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce, _HELIX_BACKEND
print(f'=== [helix_a2a] dcp_alltoall patched, backend={_HELIX_BACKEND} ===')
"
