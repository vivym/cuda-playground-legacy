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
  auto batch_size = A.batch_size();
  auto options = A.data().options();

  if (!C.is_defined()) {
    C.reset(batch_size, A.m(), B.n(), options);
  }

  auto A_data_ptr = A.data().data_ptr<scalar_t>();
  auto B_data_ptr = B.data().data_ptr<scalar_t>();
  auto C_data_ptr = C.data().data_ptr<scalar_t>();

  auto A_m = A.m_cpu(), B_m = B.m_cpu(), C_m = C.m_cpu();
  auto A_n = A.n_cpu(), B_n = B.n_cpu(), C_n = C.n_cpu();

  auto A_m_ptr = A_m.data_ptr<index_t>();
  auto B_m_ptr = B_m.data_ptr<index_t>();
  auto C_m_ptr = C_m.data_ptr<index_t>();
  auto A_n_ptr = A_n.data_ptr<index_t>();
  auto B_n_ptr = B_n.data_ptr<index_t>();
  auto C_n_ptr = C_n.data_ptr<index_t>();

  std::vector<at::cuda::CUDAStream> streams;
  for (index_t i = 0; i < batch_size; i++) {
    streams.push_back(at::cuda::getStreamFromPool());
  }

  index_t A_offset = 0, B_offset = 0, C_offset = 0;
  for (index_t i = 0; i < batch_size; i++) {
    at::cuda::CUDAStreamGuard guard(streams[i]);

    auto A_m = A_m_ptr[i], A_n = A_n_ptr[i];
    auto B_m = B_m_ptr[i], B_n = B_n_ptr[i];
    auto C_m = C_m_ptr[i], C_n = C_n_ptr[i];

    auto A_i = at::from_blob(A_data_ptr + A_offset, {A_m, A_n}, options);
    auto B_i = at::from_blob(B_data_ptr + B_offset, {B_m, B_n}, options);
    auto C_i = at::from_blob(C_data_ptr + C_offset, {C_m, C_n}, options);

    at::mm_out(C_i, A_i, B_i);

    A_offset += A_m * A_n;
    B_offset += B_m * B_n;
    C_offset += C_m * C_n;
  }

  for (index_t i = 0; i < batch_size; i++) {
    streams[i].synchronize();
  }
}

template <typename scalar_t, typename index_t>
void vbmm_cuda_stream_impl_grouped(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB) {
  auto batch_size = A.batch_size();
  auto num_groups = A.num_groups();
  auto options = A.data().options();

  if (!C.is_defined()) {
    C.reset(batch_size, num_groups, transA ? A.n() : A.m(), transB ? B.m() : B.n(), options, A.group_sizes());
  }

  auto A_data_ptr = A.data().data_ptr<scalar_t>();
  auto B_data_ptr = B.data().data_ptr<scalar_t>();
  auto C_data_ptr = C.data().data_ptr<scalar_t>();

  auto A_group_sizes = A.group_sizes_cpu(), B_group_sizes = B.group_sizes_cpu(), C_group_sizes = C.group_sizes_cpu();
  auto A_padded_m = A.m_cpu(), B_padded_m = B.m_cpu(), C_padded_m = C.m_cpu();
  auto A_padded_n = A.n_cpu(), B_padded_n = B.n_cpu(), C_padded_n = C.n_cpu();

  auto A_group_sizes_ptr = A_group_sizes.data_ptr<index_t>();
  auto B_group_sizes_ptr = B_group_sizes.data_ptr<index_t>();
  auto C_group_sizes_ptr = C_group_sizes.data_ptr<index_t>();
  auto A_padded_m_ptr = A_padded_m.data_ptr<index_t>();
  auto B_padded_m_ptr = B_padded_m.data_ptr<index_t>();
  auto C_padded_m_ptr = C_padded_m.data_ptr<index_t>();
  auto A_padded_n_ptr = A_padded_n.data_ptr<index_t>();
  auto B_padded_n_ptr = B_padded_n.data_ptr<index_t>();
  auto C_padded_n_ptr = C_padded_n.data_ptr<index_t>();

  std::vector<at::cuda::CUDAStream> streams;
  for (index_t i = 0; i < num_groups; i++) {
    streams.push_back(at::cuda::getStreamFromPool());
  }

  index_t A_offset = 0, B_offset = 0, C_offset = 0;
  for (index_t i = 0; i < num_groups; i++) {
    at::cuda::CUDAStreamGuard guard(streams[i]);
    auto A_group_size = A_group_sizes_ptr[i], B_group_size = B_group_sizes_ptr[i], C_group_size = C_group_sizes_ptr[i];
    auto A_padded_m = A_padded_m_ptr[i], B_padded_m = B_padded_m_ptr[i], C_padded_m = C_padded_m_ptr[i];
    auto A_padded_n = A_padded_n_ptr[i], B_padded_n = B_padded_n_ptr[i], C_padded_n = C_padded_n_ptr[i];

    auto A_i = at::from_blob(A_data_ptr + A_offset, {A_group_size, A_padded_m, A_padded_n}, options);
    auto B_i = at::from_blob(B_data_ptr + B_offset, {B_group_size, B_padded_m, B_padded_n}, options);
    auto C_i = at::from_blob(C_data_ptr + C_offset, {C_group_size, C_padded_m, C_padded_n}, options);

    if (transA) {
      A_i = A_i.transpose(1, 2);
    }

    if (transB) {
      B_i = B_i.transpose(1, 2);
    }

    at::bmm_out(C_i, A_i, B_i);

    A_offset += A_group_size * A_padded_m * A_padded_n;
    B_offset += B_group_size * B_padded_m * B_padded_n;
    C_offset += C_group_size * C_padded_m * C_padded_n;
  }

  for (index_t i = 0; i < num_groups; i++) {
    streams[i].synchronize();
  }
}

void vbmm_cuda_stream(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB) {
  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "vbmm_cuda_stream", [&] {
    if (A.batch_size() == A.num_groups()) {
      vbmm_cuda_stream_impl<scalar_t, VBMatrices::index_t>(
          A, B, C, alpha, beta, transA, transB);
    } else {
      vbmm_cuda_stream_impl_grouped<scalar_t, VBMatrices::index_t>(
          A, B, C, alpha, beta, transA, transB);
    }
  });
}

} // namespace vbmm

} // namespace cuda_playground
