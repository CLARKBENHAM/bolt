#include <pybind11/pybind11.h>
#include "../cpp/src/quantize/mithral.cpp"

namespace py = pybind11;

PYBIND11_MODULE(bind, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
    m.def("sub", &sub, "A function that subs two numbers");
}