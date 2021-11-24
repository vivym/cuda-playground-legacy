#pragma once

#include <torch/types.h>
#include <tuple>
#include <vector>
#include <optional>

namespace cuda_playground {

class VBMatrices {  
public:
  using index_t = int32_t;

  VBMatrices() {}

  VBMatrices(index_t batch_size, const at::Tensor &data, const at::Tensor &m, const at::Tensor &n)
    : batch_size_(batch_size), num_groups_(batch_size), data_(data), m_(m), n_(n) {}

  VBMatrices(index_t batch_size, index_t num_groups, const at::Tensor &data, const at::Tensor &m, const at::Tensor &n)
    : batch_size_(batch_size), num_groups_(num_groups), data_(data), m_(m), n_(n) {}

  VBMatrices(const std::vector<at::Tensor>& matrices);

  index_t batch_size() const { return batch_size_; }

  const at::Tensor& data() const { return data_; }

  at::Tensor& data() { return data_; }

  at::ScalarType scalar_type() const { return data_.scalar_type(); }

  const at::Tensor& m() const { return m_; }

  const at::Tensor& n() const { return n_; }

  const at::Tensor& m_cpu() const {
    // TODO: thread-safe
    if (!m_cpu_.defined()) {
      m_cpu_ = m_.cpu();
    }
    return m_cpu_;
  }

  const at::Tensor& n_cpu() const {
    // TODO: thread-safe
    if (!n_cpu_.defined()) {
      n_cpu_ = n_.cpu();
    }
    return n_cpu_;
  }

  const at::Tensor& offsets() const {
    if (!offsets_.defined()) {
      offsets_ = get_offsets_impl();
    }
    return offsets_;
  }

  const at::Tensor& addresses() const {
    if (!addresses_.defined()) {
      addresses_ = get_addresses_impl();
    }
    return addresses_;
  }


  bool is_defined() const { return data_.defined(); }

  void reset(index_t batch_size, const at::Tensor &m, const at::Tensor &n, const at::TensorOptions &options);

  void reset(
      index_t batch_size,
      index_t num_groups,
      const at::Tensor &m,
      const at::Tensor &n,
      const at::TensorOptions &options,
      std::optional<at::Tensor> group_sizes = std::nullopt,
      bool zero_init = false);

  std::tuple<VBMatrices, at::Tensor> group_by(index_t num_groups) const;

  index_t num_groups() const { return num_groups_; }

  const at::Tensor& group_sizes() const { return group_sizes_; }

  const at::Tensor& group_sizes_cpu() const {
    // TODO: thread-safe
    if (!group_sizes_cpu_.defined()) {
      group_sizes_cpu_ = group_sizes_.cpu();
    }
    return group_sizes_cpu_;
  }

private:
  at::Tensor get_offsets_impl() const;

  at::Tensor get_addresses_impl() const;

private:
  index_t batch_size_{ 0 };
  index_t num_groups_{ 0 };
  at::Tensor data_;

  at::Tensor group_sizes_;
  at::Tensor m_;
  at::Tensor n_;

  mutable at::Tensor group_sizes_cpu_;
  mutable at::Tensor m_cpu_;
  mutable at::Tensor n_cpu_;
  mutable at::Tensor offsets_;
  mutable at::Tensor addresses_;
};

} // namespace cuda_playground
