#include <torch/extension.h>
#include "vb_matrices.h"
#include "dp.h"
#include "vbmm/vbmm.h"

namespace cuda_playground {

PYBIND11_MODULE(_ops, m) {
  py::class_<VBMatrices>(m, "VBMatrices")
    .def(py::init<>())
    .def(py::init<const std::vector<at::Tensor>>())
    .def_property_readonly("data", static_cast<const at::Tensor& (VBMatrices::*)() const>(&VBMatrices::data))
    .def_property_readonly("m", static_cast<const at::Tensor& (VBMatrices::*)() const>(&VBMatrices::m))
    .def_property_readonly("n", static_cast<const at::Tensor& (VBMatrices::*)() const>(&VBMatrices::n))
    .def("group_by", &VBMatrices::group_by);
  
  m.def("vbmm", &vbmm::vbmm);
  py::enum_<vbmm::Algo>(m, "VBMMAlgo")
    .value("Vanilla", vbmm::Algo::Vanilla)
    .value("Stream", vbmm::Algo::Stream)
    .value("MAGMA", vbmm::Algo::MAGMA);
  
  m.def("get_optimal_group_delimeters_wrapper", &dp::get_optimal_group_delimeters_wrapper);
}

} // namespace cuda_playground
