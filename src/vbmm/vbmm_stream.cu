#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include "vbmm/vbmm.h"

namespace cuda_playground {

namespace vbmm {

template <typename scalar_t, typename index_t>
void vbmm_cuda_stream_impl(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB) {
  auto batch_size = A.get_batch_size();
  auto options = A.get_data().options();

  auto A_data_ptr = A.get_data().data_ptr<scalar_t>();
  auto A_m_ptr = A.get_m_cpu().data_ptr<index_t>();
  auto A_n_ptr = A.get_n_cpu().data_ptr<index_t>();
  auto B_data_ptr = B.get_data().data_ptr<scalar_t>();
  auto B_m_ptr = B.get_m_cpu().data_ptr<index_t>();
  auto B_n_ptr = B.get_n_cpu().data_ptr<index_t>();

  if (!C.is_defined()) {
    C.init(batch_size, A.get_m(), B.get_n(), options);
  }

  auto C_data_ptr = C.get_data().data_ptr<scalar_t>();
  auto C_m_ptr = C.get_m_cpu().data_ptr<index_t>();
  auto C_n_ptr = C.get_n_cpu().data_ptr<index_t>();

  std::vector<at::cuda::CUDAStream> streams;
  for (index_t i = 0; i < batch_size; i++) {
    streams.push_back(at::cuda::getStreamFromPool());
  }

  index_t A_offset = 0, B_offset = 0, C_offset = 0;
  for (index_t i = 0; i < batch_size; i++) {
    auto A_m = A_m_ptr[i], A_n = A_n_ptr[i];
    auto B_m = B_m_ptr[i], B_n = B_n_ptr[i];
    auto C_m = C_m_ptr[i], C_n = C_n_ptr[i];
    index_t A_size = A_m * A_n, B_size = B_m * B_n, C_size = C_m * C_n;

    at::cuda::CUDAStreamGuard guard(streams[i]);

    auto A_i = at::from_blob(A_data_ptr + A_offset, {A_m, A_n}, options);
    auto B_i = at::from_blob(B_data_ptr + B_offset, {B_m, B_n}, options);
    auto C_i = at::from_blob(C_data_ptr + C_offset, {C_m, C_n}, options);

    at::mm_out(C_i, A_i, B_i);

    A_offset += A_size;
    B_offset += B_size;
    C_offset += C_size;
  }

  for (index_t i = 0; i < batch_size; i++) {
    streams[i].synchronize();
  }
}

void vbmm_cuda_stream(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB) {
  AT_DISPATCH_FLOATING_TYPES(A.get_scalar_type(), "vbmm_cuda_stream", [&] {
    vbmm_cuda_stream_impl<scalar_t, VBMatrices::index_t>(
        A, B, C, alpha, beta, transA, transB);
  });
}

} // namespace vbmm

} // namespace cuda_playground
