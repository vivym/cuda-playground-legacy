#include <torch/extension.h>
#include <stdexcept>
#include "vbmm/vbmm.h"

namespace cuda_playground {

namespace vbmm {

void vbmm_cuda_vanilla(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB);

void vbmm_cuda_magma(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB);

void vbmm_cuda(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB,
    Algo algo) {
  switch (algo) {
  case Algo::Vanilla:
    vbmm_cuda_vanilla(A, B, C, alpha, beta, transA, transB);
    break;
  case Algo::MAGMA:
    vbmm_cuda_magma(A, B, C, alpha, beta, transA, transB);
    break;
  default:
    throw std::runtime_error("Unsupported algorithm");
  }
}

} // namespace vbmm

} // namespace cuda_playground
