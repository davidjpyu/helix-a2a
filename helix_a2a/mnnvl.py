"""
MNNVL workspace allocation — placeholder (Phase 4).

This will use CUDA driver API (cuda-python) to allocate cross-node visible
memory via cuMemCreate / cuMemExportToShareableHandle / cuMemMap.
"""
from __future__ import annotations


def allocate_mnnvl_workspace(*args, **kwargs):
    raise NotImplementedError(
        "MNNVL workspace allocation is not yet implemented (Phase 4). "
        "Use mnnvl=False for device-memory workspace."
    )
