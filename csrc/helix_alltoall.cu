/*
 * Helix All-to-All — CUDA kernel stubs (Phase 1).
 *
 * These are minimal implementations that compile and return plausible values.
 * Phase 2 will replace them with the real TRT-LLM kernel port.
 */
#include "helix_alltoall.h"

#include <cstring>

namespace helix_a2a {
namespace kernels {

static constexpr int HELIX_FIFO_DEPTH = 4;
static constexpr int HELIX_FIFO_ENTRY_BYTES = 128 * 1024;  // 128 KB
static constexpr int HELIX_FIFO_TOTAL_BYTES =
    HELIX_FIFO_DEPTH * HELIX_FIFO_ENTRY_BYTES;  // 512 KB

int computeHelixMaxChannelCount(int cpSize, int smCount) {
  (void)smCount;
  return (cpSize <= 2) ? 4 : 8;
}

size_t computeHelixWorkspaceSizePerRank(int cpSize) {
  int maxChannels = computeHelixMaxChannelCount(cpSize);
  size_t fifoSize =
      static_cast<size_t>(HELIX_FIFO_TOTAL_BYTES) * cpSize * maxChannels;
  // HelixFifoInfo is 2 × int64 = 16 bytes
  size_t senderInfoSize = 16ULL * cpSize * maxChannels;
  size_t receiverInfoSize = senderInfoSize;
  return fifoSize + senderInfoSize + receiverInfoSize;
}

__global__ void initializeHelixWorkspaceKernel(uint64_t* workspace,
                                               size_t numElements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    workspace[idx] = 0;
  }
}

void initializeHelixWorkspace(uint64_t* workspace, int cpSize,
                              cudaStream_t stream) {
  size_t wsBytes = computeHelixWorkspaceSizePerRank(cpSize);
  size_t numElements = wsBytes / sizeof(uint64_t);
  if (numElements == 0) return;
  int threads = 256;
  int blocks = (numElements + threads - 1) / threads;
  initializeHelixWorkspaceKernel<<<blocks, threads, 0, stream>>>(workspace,
                                                                  numElements);
}

void launchHelixAllToAll(HelixAllToAllParams const& params,
                         bool allowVariableField1, cudaStream_t stream) {
  (void)params;
  (void)allowVariableField1;
  (void)stream;
  // Stub: no-op. Real kernel will be ported in Phase 2.
}

}  // namespace kernels
}  // namespace helix_a2a
