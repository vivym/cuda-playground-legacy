#include <torch/extension.h>
#include "vb_matrices.h"
#include "vbmm/vbmm.h"

namespace cuda_playground {

PYBIND11_MODULE(_ops, m) {
  py::class_<VBMatrices>(m, "VBMatrices")
    .def(py::init<const std::vector<at::Tensor>>());
}

} // namespace cuda_playground
