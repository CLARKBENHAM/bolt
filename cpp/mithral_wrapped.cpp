#include <string> 
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "src/quantize/mithral.hpp"
#include "test/quantize/profile_amm.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mithral_wrapped, m) {
    m.doc() = "pybind11 plugin that wrapps mithral"; // Optional module docstring
    
   //---Mithral.hpp---
    m.def("add", &add, "A function that adds two numbers");
    m.def("sub", &sub, "A function that subs two numbers");
    
    //---profile_amm.hpp ---
    //struct MatmulTaskShape { int N, D, M; const char* name; };
    py::class_<MatmulTaskShape>(m, "MatmulTaskShape")
       .def(py::init<int, int, int, const char*>())
       //.def_read("name", &MatmulTaskShape::name)
        .def("__repr__",
        [](const MatmulTaskShape &a) {
            return "<example.MatmulTaskShape: name: " + std::string(a.name) + " sizes: " + std::to_string(a.N) + " ," + std::to_string(a.D) + ", " + std::to_string(a.M) + ">";
        }
        );
    //static constexpr MatmulTaskShape kCaltechTaskShape0 {
    //  (224 - 3 + 1) * (224 - 3 + 1), 3 * (3 * 3), 2, "Caltech3x3"}; // 49284, 27
    //py::class_<MatmulTaskShape>(m, "MatmulTaskShape")
    //   .def(py::init<int, int, int, const char*>())
    //   //.def_read("name", &MatmulTaskShape::name)
    //    .def("__repr__",
    //    [](const MatmulTaskShape &a) {
    //        return "f<example.MatmulTaskShape: name: f{a.name} sizes: f{a.N}, f{a.D}, f{a.M}>";
    //    }
    //    );
    
    //void _profile_mithral(const MatmulTaskShape& shape, std::vector<int> ncodebooks,
    //                 std::vector<float> lut_work_consts)
    m.def("_profile_mithral", py::overload_cast<const MatmulTaskShape&, std::vector<int> , std::vector<float> >(&_profile_mithral), "generic C++ tests run");
}