#include <torch/extension.h>
#include "vbmm/vbmm.h"

namespace cuda_playground {

namespace vbmm {

void vbmm_cuda_magma(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB) {
}

} // namespace vbmm

} // namespace cuda_playground
