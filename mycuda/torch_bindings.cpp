#include <pybind11/pybind11.h>
#include "cache.h"
#include <torch/torch.h>
#include <iostream>

// void launch_my_kernel();  // declared somewhere

PYBIND11_MODULE(mycuda, m) {
  m.def("swap_blocks", &swap_blocks, py::call_guard<py::gil_scoped_release>());
  m.def("swap_blocks_multi_layer", &swap_blocks_multi_layer, py::call_guard<py::gil_scoped_release>());
}
