#include <torch/extension.h>
#include "vb_matrices.h"
#include "vbmm/vbmm.h"

namespace cuda_playground {

PYBIND11_MODULE(_ops, m) {
  py::class_<VBMatrices>(m, "VBMatrices")
    .def(py::init<>())
    .def(py::init<const std::vector<at::Tensor>>())
    .def_property_readonly("data", &VBMatrices::get_data)
    .def("pack_up", &VBMatrices::pack_up)
    .def("group_by", &VBMatrices::group_by);
  
  m.def("vbmm", &vbmm::vbmm);
  py::enum_<vbmm::Algo>(m, "VBMMAlgo")
    .value("Vanilla", vbmm::Algo::Vanilla)
    .value("Stream", vbmm::Algo::Stream)
    .value("MAGMA", vbmm::Algo::MAGMA);
}

} // namespace cuda_playground
