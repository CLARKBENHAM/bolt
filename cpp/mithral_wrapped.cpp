#include "src/quantize/mithral.hpp"
#include "test/quantize/profile_amm.hpp"
// #include "src/external/eigen/Eigen/Core"

// # include <eigen3/Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <string> 

using namespace std;

namespace py = pybind11;
    
    struct Test { int N, D, M; const char* name;};

    void printNameMatmul(const MatmulTaskShape& shape)
    {
        cout << shape.name << endl;
        printf("shapename: %s\n", shape.name);
    }
    
    void printNameTest(const Test& shape)
    {
        cout << "out:    " << shape.name << endl;
        printf("shapename: %s\n", shape.name);
    }

PYBIND11_MODULE(mithral_wrapped, m) {
    m.doc() = "pybind11 plugin that wrapps mithral"; // Optional module docstring
    
   //---Mithral.hpp---
    m.def("add", &add, "A function that adds two numbers");
    m.def("sub", &sub, "A function that subs two numbers");
    
    //---profile_amm.hpp ---
    m.def("_profile_mithral", py::overload_cast<const MatmulTaskShape&, std::vector<int> , std::vector<float> >(&_profile_mithral), "generic C++ tests run");
    m.def("_profile_mithral_int8", py::overload_cast<const MatmulTaskShape&, std::vector<int> , std::vector<float> >(&_profile_mithral<int8_t>), "generic C++ tests run in int8");

    //Holds basic info 
    py::class_<MatmulTaskShape>(m, "MatmulTaskShape")
       .def(py::init([](int N, int D, int M, char *c){
           const size_t len = strlen(c);
           char * tmp_filename = new char[len + 1];
           strncpy(tmp_filename, c, len);
           tmp_filename[len] = '\0';  //nessisary line(why ?!?)
           const char* c2 = tmp_filename;
           return MatmulTaskShape{N,D,M,c2};
       }))
       .def_readwrite("name", &MatmulTaskShape::name) 
       .def_readonly("N", &MatmulTaskShape::N) 
       .def_readonly("D", &MatmulTaskShape::D) 
       .def_readonly("M", &MatmulTaskShape::M) 
        .def("__repr__",
        [](const MatmulTaskShape &a) {
            std::cout << "PPPprinting:    " << a.name << std::endl; 
            std::string name(a.name);
            std::cout << std::string("from C++: <example.MatmulTaskShape: name: " + name + " sizes: " + std::to_string(a.N) + " ," + std::to_string(a.D) + ", " + std::to_string(a.M) + ">") << std::endl;
            return std::string("<example.MatmulTaskShape: name: " + name + " sizes: " + std::to_string(a.N) + " ," + std::to_string(a.D) + ", " + std::to_string(a.M) + ">");
        }
        );

    //task wrapper with all info
    py::class_<mithral_amm_task<float>>(m, "mithral_amm_task_float")
       .def(py::init([](int N, int D, int M, int ncodebooks,
                        float lut_work_const){
           return mithral_amm_task<float>{N,D,M,ncodebooks, lut_work_const};
       }))
       .def("output", &mithral_amm_task<float>::output)
       .def_readwrite("X"     , &mithral_amm_task<float>::X) //can return out matricies?
       .def_readonly("Q"     , &mithral_amm_task<float>::Q)
       .def_readonly("amm"     , &mithral_amm_task<float>::amm) // whole amm object?
        .def("__repr__",
        [](const mithral_amm_task<float> &a) {
            std::stringstream ss;
            ss << &a;  
            std::string address = ss.str(); 
            return std::string("mithral_amm_task<float> at " + address);
        }
        );
        
    //amm so can call attributes from mithtral_amm_task.output()
    using traits = mithral_input_type_traits<float>;
    using scale_t = typename traits::encoding_scales_type;
    using offset_t = typename traits::encoding_offsets_type;
    using output_t = typename traits::output_type;
    py::class_<mithral_amm<float>>(m, "mithral_amm_float")
       .def(py::init([](int N, int D, int M, int ncodebooks, const float* centroids,
                        // for encoding
                        const uint32_t* splitdims, const int8_t* splitvals,
                        const scale_t* encode_scales, const offset_t* encode_offsets,
                        // for lut creation
                        const int* idxs, int nnz_per_centroid
                        ){
           return mithral_amm<float>{ N, D, M, ncodebooks,centroids,
                                     splitdims, splitvals,
                                     encode_scales, encode_offsets,
                                      idxs,  nnz_per_centroid};
       }))
       .def_readonly("out_mat"     , &mithral_amm<float>::out_mat) //eigen object
    ;

    ////Eigen type so can return real matrix
    //   .def_readonly("data"     , &ColMatrix<float>::output) // whole amm object?
    //   .def_readonly("size"     , &ColMatrix<float>::output) // whole amm object?





    // ---TEMP---- 
    m.def("printNameMatmul", &printNameMatmul, "print matmul");
    m.def("printNameTest", &printNameTest, "printTest");

    // Hacking. Want to pass a python string to a struct which stores it in a const char * and returns a string python can read 
    struct clark { 
        clark(const char* c) : c{c}, s{c} {}; // w/ default constructor the std::string methods don't work
        const char* getName() const { return c; }
        const std::string getNameString() const { 
            // std::string cast_c(c);
            cout << "const char* " << c << endl;
            cout << "std::string " << s << endl;
            return s;
            //return std::string (c); //Python can't decode this return
        }
        const char *c;
        std::string s;
    } ;

    py::class_<clark>(m, "clark")
       .def(py::init([](char *c){
           //WORKS: makes a copy of pystring inside pybind. 
           //Causes a memory leak of tmp_filename
           const size_t len = strlen(c);
           char * tmp_filename = new char[len + 1];
           strncpy(tmp_filename, c, len);
           tmp_filename[len] = '\0';  //nessisary line(why ?!?)
           const char* c2 = tmp_filename;
           return clark{c2};
       }))
        .def("getName", &clark::getName)
        .def("getNameString", &clark::getNameString)
       .def_readwrite("c", &clark::c) ;
    
    py::class_<Test>(m, "Test")
       .def(py::init([](int N, int D, int M, char *c){
           const size_t len = strlen(c);
           char * tmp_filename = new char[len + 1];
           strncpy(tmp_filename, c, len);
           tmp_filename[len] = '\0';  //nessisary line(why ?!?)
           const char* c2 = tmp_filename;
           return Test{N,D,M,c2};
       }))
       .def_readwrite("name", &Test::name) 
       .def_readonly("N", &Test::N) 
       .def_readonly("D", &Test::D) 
       .def_readonly("M", &Test::M) 
        .def("__repr__",
        [](const Test &a) {
            std::cout << "PPPprinting:    " << a.name << std::endl; 
            std::string name(a.name);
            std::cout << std::string("from C++: <Test: name: " + name + " sizes: " + std::to_string(a.N) + " ," + std::to_string(a.D) + ", " + std::to_string(a.M) + ">") << std::endl;
            return std::string("<Test: name: " + name + " sizes: " + std::to_string(a.N) + " ," + std::to_string(a.D) + ", " + std::to_string(a.M) + ">");
        }
        );

    // can print out fine
    m.def("utf8_test",
    [](const std::string s) {
        const char * c = s.c_str();
        cout << "utf-8 is icing on the cake.\n";
        cout << c << "\n";
        return c;
    }
    );
    m.def("utf8_charptr",
        [](const char *s) {
            cout << "My favorite food is\n";
            cout << s << "\n";
            return s; //Python can print returned char* 
        }
    );
    
}