#include <pybind11/pybind11.h>
#include "src/quantize/mithral.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mithral_wrapped, m) {
    m.doc() = "pybind11 plugin that wrapps mithral"; // Optional module docstring
    
    m.def("add", &add, "A function that adds two numbers");
    m.def("sub", &sub, "A function that subs two numbers");
}