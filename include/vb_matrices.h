#pragma once

#include <torch/types.h>
#include <vector>

namespace cuda_playground {

class VBMatrices {
public:
  VBMatrices(const at::Tensor &data, const at::Tensor &m, const at::Tensor &n)
    : data_(data), m_(m), n_(n) {}

  VBMatrices(const std::vector<at::Tensor> matrices);
private:
  at::Tensor data_;
  at::Tensor m_;
  at::Tensor n_;
};

} // namespace cuda_playground
