#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
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

VBMatrices::VBMatrices(const std::vector<at::Tensor>& matrices) {
  batch_size_ = matrices.size();
  at::Tensor m_cpu = at::empty({batch_size_ + 1}, at::kInt);  // TODO: kInt -> index_t
  at::Tensor n_cpu = at::empty({batch_size_ + 1}, at::kInt);

  auto m_ptr = m_cpu.data_ptr<index_t>();
  auto n_ptr = n_cpu.data_ptr<index_t>();

  index_t data_size = 0;
  index_t last_n = -1;
  bool same_n = true;
  for (index_t i = 0; i < batch_size_; i++) {
    const auto& matrix = matrices[i];
    index_t m = matrix.size(0), n = matrix.size(1);
    m_ptr[i] = m;
    n_ptr[i] = n;
    data_size += m * n;

    if (last_n != -1 && last_n != n) {
      same_n = false;
    }
    last_n = n;
  }

  if (same_n) {
    data_ = at::empty({data_size / last_n, last_n}, matrices[0].options());
  } else {
    data_ = at::empty({data_size}, matrices[0].options());
  }

  
  m_cpu_ = m_cpu;
  n_cpu_ = n_cpu;
  m_ = m_cpu_.to(m_cpu_.options().device(data_.device()));
  n_ = n_cpu_.to(n_cpu_.options().device(data_.device()));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);

  AT_DISPATCH_FLOATING_TYPES(matrices[0].scalar_type(), "VBMatrices::VBMatrices(const std::vector<at::Tensor> matrices)", [&] {
    auto data_ptr = data_.data_ptr<scalar_t>();

    for (index_t i = 0; i < batch_size_; i++) {
      const auto& matrix = matrices[i];
      index_t m = m_ptr[i], n = n_ptr[i];
      index_t size = m * n;
      auto data_i_ptr = matrix.data_ptr<scalar_t>();
      thrust::copy(policy, data_i_ptr, data_i_ptr + size, data_ptr);
      data_ptr += size;
    }
  });
}

void VBMatrices::reset(index_t batch_size, const at::Tensor &m, const at::Tensor &n, const at::TensorOptions &options) {
  reset(batch_size, batch_size, m, n, options);
}

void VBMatrices::reset(
    index_t batch_size,
    index_t num_groups,
    const at::Tensor &m,
    const at::Tensor &n,
    const at::TensorOptions &options,
    std::optional<at::Tensor> group_sizes,
    bool zero_init) {
  batch_size_ = batch_size;
  num_groups_ = num_groups;
  m_ = m;
  n_ = n;

  if (group_sizes.has_value()) {
    group_sizes_ = group_sizes.value();
  }

  auto m_ptr = thrust::device_ptr<index_t>(m.data_ptr<index_t>());
  auto n_ptr = thrust::device_ptr<index_t>(n.data_ptr<index_t>());

  auto data_size = get_total_size(batch_size, m_ptr, n_ptr);
  if (zero_init) {
    data_ = at::zeros({data_size}, options);
  } else {
    data_ = at::empty({data_size}, options);
  }

  // TODO: reset cached tensor
}

// work around the limitation of the lambda function in cuda
template <typename index_t>
void get_offsets_impl_thrust(
    const thrust::device_ptr<index_t> &m_ptr,
    const thrust::device_ptr<index_t> &n_ptr,
    thrust::device_ptr<index_t> &offsets_ptr,
    int batch_size) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);

  thrust::transform_exclusive_scan(
      policy,
      thrust::make_zip_iterator(thrust::make_tuple(m_ptr, n_ptr)),
      thrust::make_zip_iterator(thrust::make_tuple(m_ptr + batch_size, n_ptr + batch_size)),
      offsets_ptr,
      [] __device__ (const thrust::tuple<index_t, index_t> &tuple) {
        index_t m, n;
        thrust::tie(m, n) = tuple;
        return m * n;
      },
      0,
      thrust::plus<index_t>());
}

at::Tensor VBMatrices::get_offsets_impl() const {
  at::Tensor offsets = at::empty({batch_size_}, m_.options());

  auto m_ptr = thrust::device_ptr<index_t>(m_.data_ptr<index_t>());
  auto n_ptr = thrust::device_ptr<index_t>(n_.data_ptr<index_t>());
  auto offsets_ptr = thrust::device_ptr<index_t>(offsets.data_ptr<index_t>());

  get_offsets_impl_thrust<index_t>(m_ptr, n_ptr, offsets_ptr, batch_size_);

  return offsets;
}

// work around the limitation of the lambda function in cuda
template <typename scalar_t, typename index_t>
void get_addresses_impl_thrust(
    scalar_t *data_ptr,
    const thrust::device_ptr<index_t> &offsets_ptr,
    thrust::device_ptr<scalar_t *> &addresses_ptr,
    int batch_size) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);

  thrust::transform(
      policy,
      offsets_ptr,
      offsets_ptr + batch_size,
      addresses_ptr,
      [data_ptr] __device__ (index_t offset) {
        return data_ptr + offset;
      });
}

at::Tensor VBMatrices::get_addresses_impl() const {
  const auto& offsets = this->offsets();
  at::Tensor addresses = at::empty({batch_size_}, m_.options().dtype(at::kLong));

  AT_DISPATCH_FLOATING_TYPES(data_.scalar_type(), "VBMatrices::get_addresses_impl", [&] {
    auto data_ptr = data_.data_ptr<scalar_t>();
    auto offsets_ptr = thrust::device_ptr<index_t>(offsets.data_ptr<index_t>());
    auto addresses_ptr = thrust::device_ptr<scalar_t *>(reinterpret_cast<scalar_t **>(addresses.data_ptr<int64_t>()));

    get_addresses_impl_thrust(data_ptr, offsets_ptr, addresses_ptr, batch_size_);
  });

  return addresses;
}

template <typename index_t>
std::tuple<std::vector<index_t>, std::vector<index_t>> pack_up_impl(
    index_t batch_size,
    thrust::device_ptr<index_t> m_ptr,
    thrust::device_ptr<index_t> n_ptr,
    const at::TensorOptions& options) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);

  auto pack_sizes = at::empty(batch_size, options);
  auto packed_sizes = at::empty(batch_size, options);
  auto packed_m = at::empty(batch_size, options);
  auto reduced_packed_m = at::empty(batch_size, options);

  auto pack_sizes_ptr = thrust::device_ptr<index_t>(pack_sizes.template data_ptr<index_t>());
  auto packed_sizes_ptr = thrust::device_ptr<index_t>(packed_sizes.template data_ptr<index_t>());
  auto packed_m_ptr = thrust::device_ptr<index_t>(packed_m.template data_ptr<index_t>());
  auto reduced_packed_m_ptr = thrust::device_ptr<index_t>(reduced_packed_m.template data_ptr<index_t>());

  thrust::transform(policy, m_ptr, m_ptr + batch_size, packed_m_ptr, [] __device__ (index_t m) {
    if (m <= 64) {
      return 64;
    } else if (m <= 128) {
      return 128;
    } else if (m <= 256) {
      return 256;
    }
    return m;
  });

  auto new_end = thrust::reduce_by_key(
      policy,
      packed_m_ptr, packed_m_ptr + batch_size,
      thrust::make_constant_iterator<index_t>(1),
      thrust::make_discard_iterator(),
      pack_sizes_ptr);

  auto num_pack = thrust::distance(pack_sizes_ptr, new_end.second);
  std::cout << "num_pack: " << num_pack << std::endl << std::flush;

  std::vector<index_t> pack_sizes_cpu(num_pack), packed_sizes_cpu(num_pack);
  thrust::copy(pack_sizes_ptr, pack_sizes_ptr + num_pack, pack_sizes_cpu.begin());
  std::cout << "here #1" << std::endl << std::flush;
  thrust::copy(packed_sizes_ptr, packed_sizes_ptr + num_pack, packed_sizes_cpu.begin());
  std::cout << "here #2" << std::endl << std::flush;

  return {
    pack_sizes_cpu,
    packed_sizes_cpu,
  };
}

std::tuple<std::vector<VBMatrices::index_t>, std::vector<at::Tensor>> VBMatrices::pack_up() const {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ThrustAllocator allocator;
  auto policy = thrust::cuda::par(allocator).on(stream);
  
  auto options = m_.options();

  auto m = m_.clone();
  auto n = at::empty({batch_size_ + 1}, options);
  auto sorted_indices = at::arange(batch_size_, options);

  auto m_ptr = thrust::device_ptr<index_t>(m.data_ptr<index_t>());
  auto n_ptr = thrust::device_ptr<index_t>(n.data_ptr<index_t>());
  auto n__ptr = thrust::device_ptr<index_t>(n_.data_ptr<index_t>());
  auto sorted_indices_ptr = thrust::device_ptr<index_t>(sorted_indices.data_ptr<index_t>());

  thrust::sort_by_key(
      policy,
      m_ptr, m_ptr + batch_size_,
      sorted_indices_ptr);
  
  thrust::gather(
      policy,
      sorted_indices_ptr,
      sorted_indices_ptr + batch_size_,
      n__ptr,
      n_ptr);

  auto m_cpu = m.cpu();
  auto n_cpu = n.cpu();
  
  auto m_cpu_ptr = m_cpu.data_ptr<index_t>();
  auto n_cpu_ptr = n_cpu.data_ptr<index_t>();

  auto pack_sizes_and_packed_sizes = pack_up_impl<index_t>(batch_size_, m_ptr, n_ptr, options);
  auto pack_sizes = std::get<0>(pack_sizes_and_packed_sizes), packed_sizes = std::get<1>(pack_sizes_and_packed_sizes);
  std::vector<at::Tensor> packed_matrices;
  std::cout << "here #2.5" << std::endl << std::flush;

  AT_DISPATCH_FLOATING_TYPES(data_.scalar_type(), "VBMatrices::pack_up", [&] {
    std::cout << "here #2.6" << std::endl << std::flush;
    auto data_ptr = thrust::device_ptr<scalar_t>(data_.data_ptr<scalar_t>());
    const index_t num_pack = packed_sizes.size();
    index_t index_offset = 0;
    std::cout << "here #2.7" << std::endl << std::flush;

    for (index_t i = 0; i < num_pack; i++) {
      auto pack_size = pack_sizes[i], packed_size = packed_sizes[i];
      std::cout << "here #2.8" << std::endl << std::flush;
      auto m = m_cpu_ptr[index_offset];
      auto n = n_cpu_ptr[index_offset];
      auto size = pack_size * m * n;

      std::cout << "here #3" << std::endl << std::flush;
      auto matrix = at::empty({packed_size, m, n}, data_.options());
      auto matrix_ptr = thrust::device_ptr<scalar_t>(matrix.data_ptr<scalar_t>());
      thrust::copy(data_ptr, data_ptr + size, matrix_ptr);
      std::cout << "here #4" << std::endl << std::flush;
      packed_matrices.push_back(std::move(matrix));

      index_offset += pack_size;
      data_ptr += size;
    }
  });

  std::cout << "here done" << std::endl << std::flush;

  return {
    pack_sizes,
    packed_matrices
  };
}

namespace {
  template <typename policy_t, typename index_t>
  inline void generate_padded_m(
      const policy_t& policy,
      thrust::device_ptr<index_t> m_ptr,
      index_t batch_size,
      thrust::device_ptr<index_t> padded_m_ptr,
      const std::vector<index_t>& delimeters) {
    thrust::transform(policy, m_ptr, m_ptr + batch_size, padded_m_ptr, [=] __device__ (index_t m) {
      // for (auto delimeter : delimeters) {
      //   if (m <= delimeter) {
      //     return delimeter;
      //   }
      // }
      if (m <= 64) {
        return 64;
      } else if (m <= 128) {
        return 128;
      } else if (m <= 256) {
        return 256;
      }
      return m;
    });
  }

  template <typename policy_t, typename index_t>
  inline void generate_indices_and_masks(
      const policy_t& policy,
      index_t batch_size,
      thrust::device_ptr<index_t> unsorted_m_offsets_ptr,
      thrust::device_ptr<index_t> inverse_sorted_indices_ptr,
      thrust::device_ptr<index_t> padded_m_offsets_ptr,
      thrust::device_ptr<index_t> unsorted_m_ptr,
      thrust::device_ptr<index_t> indices_ptr,
      thrust::device_ptr<index_t> masks_ptr) {
    thrust::for_each(
        policy,
        thrust::make_counting_iterator<index_t>(0),
        thrust::make_counting_iterator<index_t>(batch_size),
        [=] __device__ (index_t i) {
          auto m_offset = unsorted_m_offsets_ptr[i];
          auto m = unsorted_m_ptr[i];
          auto idx = inverse_sorted_indices_ptr[i];
          auto padded_m_offset = padded_m_offsets_ptr[idx];
          thrust::sequence(
              policy,
              indices_ptr + m_offset, indices_ptr + m_offset + m,
              static_cast<index_t>(padded_m_offset));
          thrust::fill(policy, masks_ptr + m_offset, masks_ptr + m_offset + m, static_cast<index_t>(1));
        });
  }
}


std::tuple<VBMatrices, at::Tensor> VBMatrices::group_by() const {
  if (data_.dim() != 2) {
    throw std::runtime_error("VBMatrices::group_by() only supports 2D tensors");
  }

  auto policy = thrust::cuda::par(ThrustAllocator()).on(at::cuda::getCurrentCUDAStream());
  
  auto options = m_.options();

  auto m = m_.clone();
  auto m_ptr = thrust::device_ptr<index_t>(m.data_ptr<index_t>());
  
  auto sorted_indices = at::arange(batch_size_, options);
  auto sorted_indices_ptr = thrust::device_ptr<index_t>(sorted_indices.data_ptr<index_t>());
  
  thrust::sort_by_key(
      policy,
      m_ptr, m_ptr + batch_size_,
      sorted_indices_ptr);
  
  auto inverse_sorted_indices = at::empty_like(sorted_indices);
  auto inverse_sorted_indices_ptr = thrust::device_ptr<index_t>(inverse_sorted_indices.data_ptr<index_t>());
  thrust::gather(
      policy,
      sorted_indices_ptr, sorted_indices_ptr + batch_size_,
      thrust::make_counting_iterator<index_t>(0),
      inverse_sorted_indices_ptr);

  // TODO: do dp
  const std::vector<index_t> delimeters{64, 128, 256};

  auto padded_m = at::empty_like(m);
  auto padded_m_ptr = thrust::device_ptr<index_t>(padded_m.data_ptr<index_t>());
  generate_padded_m(policy, m_ptr, batch_size_, padded_m_ptr, delimeters);

  auto padded_m_offsets = at::empty_like(m);
  auto padded_m_offsets_ptr = thrust::device_ptr<index_t>(padded_m_offsets.data_ptr<index_t>());

  thrust::exclusive_scan(
      policy,
      padded_m_ptr, padded_m_ptr + batch_size_,
      padded_m_offsets_ptr);
  
  auto unsorted_m_ptr = thrust::device_ptr<index_t>(m_.data_ptr<index_t>());
  auto unsorted_m_offsets = at::empty_like(m);
  auto unsorted_m_offsets_ptr = thrust::device_ptr<index_t>(unsorted_m_offsets.data_ptr<index_t>());

  auto total_size = thrust::reduce(policy, m_ptr, m_ptr + batch_size_);
  auto indices = at::empty({total_size}, options);
  auto indices_ptr = thrust::device_ptr<index_t>(indices.data_ptr<index_t>());
  
  auto masks = at::empty({total_size}, options);
  auto masks_ptr = thrust::device_ptr<index_t>(masks.data_ptr<index_t>());

  generate_indices_and_masks(
      policy,
      batch_size_,
      unsorted_m_offsets_ptr,
      inverse_sorted_indices_ptr,
      padded_m_offsets_ptr,
      unsorted_m_ptr,
      indices_ptr,
      masks_ptr);

  auto group_m = at::empty({static_cast<int64_t>(delimeters.size())}, options);
  auto group_m_ptr = thrust::device_ptr<index_t>(group_m.data_ptr<index_t>());

  auto group_sizes = at::empty({static_cast<int64_t>(delimeters.size())}, options);
  auto group_sizes_ptr = thrust::device_ptr<index_t>(group_sizes.data_ptr<index_t>());

  auto new_end = thrust::reduce_by_key(
      policy,
      padded_m_ptr, padded_m_ptr + batch_size_,
      thrust::make_constant_iterator<index_t>(1),
      group_m_ptr,
      group_sizes_ptr);

  index_t num_groups = thrust::distance(group_sizes_ptr, new_end.second);

  VBMatrices grouped_matrices;

  auto group_m_cpu = group_m.cpu();
  auto group_n_cpu = at::full_like(group_m_cpu, data_.size(1));
  auto group_sizes_cpu = group_sizes.cpu();
  grouped_matrices.reset(batch_size_, num_groups, group_m_cpu, group_n_cpu, data_.options(), group_sizes_cpu, true);
  grouped_matrices.data().index_put_({indices, "..."}, data_);

  return {
    std::move(grouped_matrices),
    std::move(masks)
  };
}

} // namespace cuda_playground
