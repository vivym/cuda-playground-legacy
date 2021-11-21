#pragma once

#include "vb_matrices.h"

namespace cuda_playground {

namespace vbmm {

enum class Algo {
  Vanilla = 0,
  Stream,
  MAGMA,
  PACK
};

void vbmm_cuda(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB,
    Algo algo);

void vbmm_cuda_vanilla(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB);

void vbmm_cuda_stream(
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

inline void vbmm(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB,
    Algo algo) {
  return vbmm_cuda(A, B, C, alpha, beta, transA, transB, algo);
}

} // namespace vbmm

} // namespace cuda_playground
