#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include "vb_matrices.h"
#include "thrust_allocator.h"

namespace cuda_playground {

template <typename index_t>
index_t get_total_size(
    index_t batch_size,
    thrust::device_ptr<index_t> m_ptr,
    thrust::device_ptr<index_t> n_ptr) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);
  
  return thrust::transform_reduce(
      policy,
      thrust::make_zip_iterator(thrust::make_tuple(m_ptr, n_ptr)),
      thrust::make_zip_iterator(thrust::make_tuple(m_ptr + batch_size, n_ptr + batch_size)),
      [] __device__ (thrust::tuple<index_t, index_t> tuple) {
        index_t m, n;
        thrust::tie(m, n) = tuple;
        return m * n;
      },
      0,
      thrust::plus<index_t>());
}

VBMatrices::VBMatrices(const std::vector<at::Tensor> matrices) {
  batch_size_ = matrices.size();
  at::Tensor m_cpu = at::empty({batch_size_ + 1}, at::kInt);  // TODO: kInt -> index_t
  at::Tensor n_cpu = at::empty({batch_size_ + 1}, at::kInt);

  auto m_ptr = m_cpu.data_ptr<index_t>();
  auto n_ptr = n_cpu.data_ptr<index_t>();

  index_t data_size = 0;
  for (index_t i = 0; i < batch_size_; i++) {
    const auto& matrix = matrices[i];
    index_t m = matrix.size(0), n = matrix.size(1);
    m_ptr[i] = m;
    n_ptr[i] = n;
    data_size += m * n;
  }

  data_ = at::empty({data_size}, matrices[0].options());
  
  m_cpu_ = m_cpu;
  n_cpu_ = n_cpu;
  m_ = m_cpu_.to(m_cpu_.options().device(data_.device()));
  n_ = n_cpu_.to(n_cpu_.options().device(data_.device()));

  AT_DISPATCH_FLOATING_TYPES(matrices[0].scalar_type(), "VBMatrices::VBMatrices(const std::vector<at::Tensor> matrices)", [&] {
    auto data_ptr = data_.data_ptr<scalar_t>();

    for (index_t i = 0; i < batch_size_; i++) {
      const auto& matrix = matrices[i];
      index_t m = m_ptr[i], n = n_ptr[i];
      index_t size = m * n;
      auto data_i_ptr = matrix.data_ptr<scalar_t>();
      thrust::copy(data_i_ptr, data_i_ptr + size, data_ptr);
      data_ptr += size;
    }
  });
}

void VBMatrices::init(int32_t batch_size, const at::Tensor &m, const at::Tensor &n, const at::TensorOptions &options) {
  batch_size_ = batch_size;
  m_ = m;
  n_ = n;

  auto m_ptr = thrust::device_ptr<index_t>(m.data_ptr<index_t>());
  auto n_ptr = thrust::device_ptr<index_t>(n.data_ptr<index_t>());

  auto data_size = get_total_size(batch_size, m_ptr, n_ptr);
  data_ = at::empty({data_size}, options);
}

} // namespace cuda_playground
