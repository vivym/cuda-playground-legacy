#include <torch/extension.h>
#include "vb_matrices.h"
#include "vbmm/vbmm.h"

namespace cuda_playground {

PYBIND11_MODULE(_ops, m) {
  py::class_<VBMatrices>(m, "VBMatrices")
    .def(py::init<>())
    .def(py::init<const std::vector<at::Tensor>>());
  
  m.def("vbmm", &vbmm::vbmm);
  py::enum_<vbmm::Algo>(m, "VBMMAlgo")
    .value("Vanilla", vbmm::Algo::Vanilla);
}

} // namespace cuda_playground
