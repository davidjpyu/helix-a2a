/*
 * Torch op registration for helix_a2a (Phase 1 — stubs).
 *
 * Registers three ops under the "helix_a2a" namespace:
 *   - get_helix_workspace_size_per_rank(int) -> int
 *   - initialize_helix_workspace(Tensor, int, int) -> void
 *   - alltoall_helix_native(Tensor, Tensor, Tensor, int, int) -> (Tensor, Tensor)
 */
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "helix_alltoall.h"

static int64_t get_helix_workspace_size_per_rank(int64_t cp_size) {
  return static_cast<int64_t>(
      helix_a2a::kernels::computeHelixWorkspaceSizePerRank(
          static_cast<int>(cp_size)));
}

static void initialize_helix_workspace(torch::Tensor workspace,
                                       int64_t cp_rank, int64_t cp_size) {
  TORCH_CHECK(workspace.is_cuda(), "workspace must be on CUDA");
  TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Long,
              "workspace must be int64 (used as uint64)");
  TORCH_CHECK(workspace.dim() == 2, "workspace must be 2D");
  TORCH_CHECK(workspace.size(0) == cp_size,
              "workspace must have cp_size rows");
  TORCH_CHECK(cp_rank >= 0 && cp_rank < cp_size,
              "cp_rank must be in [0, cp_size)");

  auto stream = at::cuda::getCurrentCUDAStream();
  uint64_t* local_workspace_ptr =
      reinterpret_cast<uint64_t*>(workspace[cp_rank].data_ptr());
  helix_a2a::kernels::initializeHelixWorkspace(local_workspace_ptr,
                                               static_cast<int>(cp_size),
                                               stream);
}

static std::tuple<torch::Tensor, torch::Tensor> alltoall_helix_native(
    torch::Tensor partial_o, torch::Tensor softmax_stats,
    torch::Tensor workspace, int64_t cp_rank, int64_t cp_size) {
  TORCH_CHECK(partial_o.is_cuda(), "partial_o must be on CUDA");
  TORCH_CHECK(softmax_stats.is_cuda(), "softmax_stats must be on CUDA");
  TORCH_CHECK(workspace.is_cuda(), "workspace must be on CUDA");
  TORCH_CHECK(partial_o.is_contiguous(), "partial_o must be contiguous");
  TORCH_CHECK(softmax_stats.is_contiguous(),
              "softmax_stats must be contiguous");

  TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Half ||
                  partial_o.scalar_type() == at::ScalarType::BFloat16,
              "partial_o must be half or bfloat16");
  TORCH_CHECK(softmax_stats.scalar_type() == at::ScalarType::Float,
              "softmax_stats must be float32");
  TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Long,
              "workspace must be int64 (used as uint64)");

  TORCH_CHECK(partial_o.dim() >= 2,
              "partial_o must have at least 2 dimensions");
  TORCH_CHECK(softmax_stats.dim() >= 2,
              "softmax_stats must have at least 2 dimensions");
  TORCH_CHECK(partial_o.dim() == softmax_stats.dim(),
              "partial_o and softmax_stats must have same number of dimensions");

  TORCH_CHECK(partial_o.size(-2) == cp_size &&
                  softmax_stats.size(-2) == cp_size,
              "second-to-last dimension must equal cp_size");
  TORCH_CHECK(softmax_stats.size(-1) % 2 == 0 &&
                  softmax_stats.size(-1) >= 2,
              "softmax_stats last dimension must be divisible by 2 (float2)");

  for (int i = 0; i < partial_o.dim() - 2; i++) {
    TORCH_CHECK(partial_o.size(i) == softmax_stats.size(i),
                "partial_o and softmax_stats must have matching leading dims");
  }
  TORCH_CHECK(partial_o.size(-1) * partial_o.element_size() % 16 == 0,
              "partial_o last dim must be 16-byte aligned");

  TORCH_CHECK(workspace.dim() == 2,
              "workspace must be 2D (strided across ranks)");
  TORCH_CHECK(workspace.size(0) == cp_size,
              "workspace must have cp_size rows");

  // Stub: return clones (identity, no actual communication)
  return std::make_tuple(partial_o.clone(), softmax_stats.clone());
}

TORCH_LIBRARY(helix_a2a, m) {
  m.def(
      "get_helix_workspace_size_per_rank(int cp_size) -> int",
      &get_helix_workspace_size_per_rank);
  m.def(
      "initialize_helix_workspace(Tensor workspace, int cp_rank, "
      "int cp_size) -> ()",
      &initialize_helix_workspace);
  m.def(
      "alltoall_helix_native(Tensor partial_o, Tensor softmax_stats, "
      "Tensor workspace, int cp_rank, int cp_size) -> (Tensor, Tensor)",
      &alltoall_helix_native);
}
