#pragma once

#include <torch/types.h>
#include <vector>

namespace cuda_playground {

class VBMatrices {  
public:
  using index_t = int32_t;

  VBMatrices() {}

  VBMatrices(index_t batch_size, const at::Tensor &data, const at::Tensor &m, const at::Tensor &n)
    : batch_size_(batch_size), data_(data), m_(m), n_(n) {}

  VBMatrices(const std::vector<at::Tensor> matrices);

  index_t get_batch_size() const { return batch_size_; }

  const at::Tensor& get_data() const { return data_; }

  at::ScalarType get_scalar_type() const { return data_.scalar_type(); }

  const at::Tensor& get_m() const { return m_; }

  const at::Tensor& get_n() const { return n_; }

  const at::Tensor& get_m_cpu() const {
    // TODO: thread-safe
    if (!m_cpu_.defined()) {
      m_cpu_ = m_.cpu();
    }
    return m_cpu_;
  }

  const at::Tensor& get_n_cpu() const {
    // TODO: thread-safe
    if (!n_cpu_.defined()) {
      n_cpu_ = n_.cpu();
    }
    return n_cpu_;
  }

  const at::Tensor& get_offsets() const {
    if (!offsets_.defined()) {
      offsets_ = get_offsets_impl();
    }
    return offsets_;
  }

  const at::Tensor& get_addresses() const {
    if (!addresses_.defined()) {
      addresses_ = get_addresses_impl();
    }
    return addresses_;
  }


  bool is_defined() const { return data_.defined(); }

  void init(index_t batch_size, const at::Tensor &m, const at::Tensor &n, const at::TensorOptions &options);

  std::vector<at::Tensor> pack_up(std::vector<index_t> delimeters) const;

private:
  at::Tensor get_offsets_impl() const;

  at::Tensor get_addresses_impl() const;

private:
  index_t batch_size_{ 0 };
  at::Tensor data_;
  at::Tensor m_;
  at::Tensor n_;
  mutable at::Tensor m_cpu_;
  mutable at::Tensor n_cpu_;
  mutable at::Tensor offsets_;
  mutable at::Tensor addresses_;
};

} // namespace cuda_playground
