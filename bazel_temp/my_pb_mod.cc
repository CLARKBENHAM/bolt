#include <pybind11/pybind11.h>

namespace {

int add(int x, int y) { return x + y; }

int sub(int i, int j) {
    return i - j;
}

} // namespace

PYBIND11_MODULE(my_pb_mod, m) {
	
  m.def("add", &add, "A function that adds two numbers");
  m.def("sub", &sub, "A function that subs two numbers");
	}