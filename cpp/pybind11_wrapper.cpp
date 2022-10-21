#include <pybind11/pybind11.h>
#include "src/quantize/mithral.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pybind11_wrapper, m) {
    m.doc() = "pybind11 example plugin"; // Optional module docstring
    
    m.def("add", &add, "A function that adds two numbers");
    m.def("sub", &sub, "A function that subs two numbers");
}