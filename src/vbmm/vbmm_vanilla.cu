#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "vbmm/vbmm.h"
#include "thrust_allocator.h"

namespace cuda_playground {

namespace vbmm {

template <typename scalar_t, typename index_t>
void vbmm_cuda_vanilla_impl(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB) {
  auto batch_size = A.batch_size();
  auto options = A.data().options();

  if (!C.is_defined()) {
    C.init(batch_size, A.m(), B.n(), options);
  }

  auto A_data_ptr = A.data().data_ptr<scalar_t>();
  auto B_data_ptr = B.data().data_ptr<scalar_t>();
  auto C_data_ptr = C.data().data_ptr<scalar_t>();

  auto A_m_ptr = thrust::device_ptr<index_t>(A.m().data_ptr<index_t>());
  auto B_m_ptr = thrust::device_ptr<index_t>(B.m().data_ptr<index_t>());
  auto C_m_ptr = thrust::device_ptr<index_t>(C.m().data_ptr<index_t>());

  auto A_n_ptr = thrust::device_ptr<index_t>(A.n().data_ptr<index_t>());
  auto B_n_ptr = thrust::device_ptr<index_t>(B.n().data_ptr<index_t>());
  auto C_n_ptr = thrust::device_ptr<index_t>(C.n().data_ptr<index_t>());

  auto A_offsets_ptr = thrust::device_ptr<index_t>(A.offsets().data_ptr<index_t>());
  auto B_offsets_ptr = thrust::device_ptr<index_t>(B.offsets().data_ptr<index_t>());
  auto C_offsets_ptr = thrust::device_ptr<index_t>(C.offsets().data_ptr<index_t>());

  index_t A_offset = 0, B_offset = 0, C_offset = 0;
  for (index_t i = 0; i < batch_size; i++) {
    auto A_m = A_m_ptr[i], A_n = A_n_ptr[i];
    auto B_m = B_m_ptr[i], B_n = B_n_ptr[i];
    auto C_m = C_m_ptr[i], C_n = C_n_ptr[i];
    index_t A_size = A_m * A_n, B_size = B_m * B_n, C_size = C_m * C_n;

    auto A_i = at::from_blob(A_data_ptr + A_offset, {A_m, A_n}, options);
    auto B_i = at::from_blob(B_data_ptr + B_offset, {B_m, B_n}, options);
    auto C_i = at::from_blob(C_data_ptr + C_offset, {C_m, C_n}, options);

    at::mm_out(C_i, A_i, B_i);

    A_offset += A_size;
    B_offset += B_size;
    C_offset += C_size;
  }
}

template <typename scalar_t, typename index_t>
void vbmm_cuda_vanilla_impl_grouped(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB) {
  auto num_groups = A.num_groups();
  auto options = A.data().options();

  if (!C.is_defined()) {
    C.init(num_groups, A.m(), B.n(), options);
  }

  auto A_data_ptr = A.data().data_ptr<scalar_t>();
  auto B_data_ptr = B.data().data_ptr<scalar_t>();
  auto C_data_ptr = C.data().data_ptr<scalar_t>();

  auto A_group_sizes_ptr = thrust::device_ptr<index_t>(A.group_sizes().data_ptr<index_t>());
  auto B_group_sizes_ptr = thrust::device_ptr<index_t>(B.group_sizes().data_ptr<index_t>());
  auto C_group_sizes_ptr = thrust::device_ptr<index_t>(C.group_sizes().data_ptr<index_t>());
  auto A_padded_m_ptr = thrust::device_ptr<index_t>(A.padded_m().data_ptr<index_t>());
  auto B_padded_m_ptr = thrust::device_ptr<index_t>(B.padded_m().data_ptr<index_t>());
  auto C_padded_m_ptr = thrust::device_ptr<index_t>(C.padded_m().data_ptr<index_t>());
  auto A_padded_n_ptr = thrust::device_ptr<index_t>(A.padded_n().data_ptr<index_t>());
  auto B_padded_n_ptr = thrust::device_ptr<index_t>(B.padded_n().data_ptr<index_t>());
  auto C_padded_n_ptr = thrust::device_ptr<index_t>(C.padded_n().data_ptr<index_t>());

  auto policy = thrust::cuda::par(ThrustAllocator()).on(at::cuda::getCurrentCUDAStream());

  auto get_offsets = [&] (const at::Tensor& padded_m) {
    auto offsets = at::empty_like(padded_m);
    auto padded_m_ptr = thrust::device_ptr<index_t>(padded_m.data_ptr<index_t>());
    auto offsets_ptr = thrust::device_ptr<index_t>(offsets.data_ptr<index_t>());
    thrust::exclusive_scan(policy, padded_m_ptr, padded_m_ptr + num_groups, offsets_ptr);
    return std::move(offsets);
  };

  auto A_offsets = get_offsets(A.padded_m());
  auto B_offsets = get_offsets(B.padded_m());
  auto C_offsets = get_offsets(C.padded_m());

  auto A_offsets_ptr = thrust::device_ptr<index_t>(A_offsets.template data_ptr<index_t>());
  auto B_offsets_ptr = thrust::device_ptr<index_t>(B_offsets.template data_ptr<index_t>());
  auto C_offsets_ptr = thrust::device_ptr<index_t>(C_offsets.template data_ptr<index_t>());

  thrust::for_each(
      policy,
      thrust::make_counting_iterator<index_t>(0),
      thrust::make_counting_iterator<index_t>(num_groups),
      [
          A_data_ptr, B_data_ptr, C_data_ptr,
          A_group_sizes_ptr, B_group_sizes_ptr, C_group_sizes_ptr,
          A_padded_m_ptr, B_padded_m_ptr, C_padded_m_ptr,
          A_padded_n_ptr, B_padded_n_ptr, C_padded_n_ptr,
          A_offsets_ptr, B_offsets_ptr, C_offsets_ptr,
          options] __device__ (index_t i) {
        auto A_group_size = A_group_sizes_ptr[i], B_group_size = B_group_sizes_ptr[i], C_group_size = C_group_sizes_ptr[i];
        auto A_padded_m = A_padded_m_ptr[i], B_padded_m = B_padded_m_ptr[i], C_padded_m = C_padded_m_ptr[i];
        auto A_padded_n = A_padded_n_ptr[i], B_padded_n = B_padded_n_ptr[i], C_padded_n = C_padded_n_ptr[i];
        auto A_offset = A_offsets_ptr[i], B_offset = B_offsets_ptr[i], C_offset = C_offsets_ptr[i];

        auto A_i = at::from_blob(A_data_ptr + A_offset, {A_group_size, A_padded_m, A_padded_n}, options);
        auto B_i = at::from_blob(B_data_ptr + B_offset, {B_group_size, B_padded_m, B_padded_n}, options);
        auto C_i = at::from_blob(C_data_ptr + C_offset, {C_group_size, C_padded_m, C_padded_n}, options);

        at::bmm_out(C_i, A_i, B_i);
      });
}

void vbmm_cuda_vanilla(
    const VBMatrices& A,
    const VBMatrices& B,
    VBMatrices& C,
    float alpha, float beta,
    bool transA, bool transB) {
  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "vbmm_cuda_vanilla", [&] {
    if (A.batch_size() == A.num_groups()) {
      vbmm_cuda_vanilla_impl<scalar_t, VBMatrices::index_t>(
          A, B, C, alpha, beta, transA, transB);
    } else {
      vbmm_cuda_vanilla_impl_grouped<scalar_t, VBMatrices::index_t>(
          A, B, C, alpha, beta, transA, transB);
    }

  });
}

} // namespace vbmm

} // namespace cuda_playground