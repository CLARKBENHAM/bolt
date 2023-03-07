#include "test/scrap/reproduce_valgrind.hpp"

#include "src/quantize/mithral.hpp"
#include "test/quantize/profile_amm.hpp"

# include <eigen3/Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

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

//struct wrapped_mithral_amm_float : public mithral_amm<float> {
//    // //need to wrap 
//    //const float* centroids;
//    //const uint32_t* splitdims;
//    //const int8_t* splitvals;
//    //const scale_t* encode_scales;
//    //const offset_t* encode_offsets;
//    //const int* idxs; 
//
//};


PYBIND11_MODULE(mithral_wrapped, m) {
    m.doc() = "pybind11 plugin that wrapps mithral"; // Optional module docstring

    // -- reproduce_valgrind_error.hpp--
    m.def("test_valgrind", &test_valgrind, "mre of valgrind error");

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
            //std::cout << "PPPprinting:    " << a.name << std::endl; 
            std::string name(a.name);
            //std::cout << std::string("from C++: <example.MatmulTaskShape: name: " + name + " sizes: " + std::to_string(a.N) + " ," + std::to_string(a.D) + ", " + std::to_string(a.M) + ">") << std::endl;
            return std::string("<example.MatmulTaskShape: name: " + name + " sizes: " + std::to_string(a.N) + " ," + std::to_string(a.D) + ", " + std::to_string(a.M) + ">");
        }
        );

    //task wrapper with all info
    py::class_<mithral_amm_task<float>>(m, "mithral_amm_task_float")
       .def(py::init([](int N, int D, int M, int ncodebooks,
                        float lut_work_const){
           return mithral_amm_task<float>{N,D,M,ncodebooks, lut_work_const};
       }))
       .def("encode"                                                           , &mithral_amm_task<float>::encode)
       .def("lut"                                                              , &mithral_amm_task<float>::lut)
       .def("scan"                                                             , &mithral_amm_task<float>::scan)
       .def("run_matmul"                                                       , &mithral_amm_task<float>::run_matmul)
       .def_readonly("amm"                                                     , &mithral_amm_task<float>::amm) // whole amm object
       .def("output"                                                           , &mithral_amm_task<float>::output)
       .def_readwrite("X"                                                      , &mithral_amm_task<float>::X) //can return out matricies?
       .def_readwrite("Q"                                                      , &mithral_amm_task<float>::Q)
       // stuff we pass into the amm object (would be learned during training)
       .def_readwrite("N_padded"                                               , &mithral_amm_task<float>::N_padded)
       .def_readwrite("centroids"                                              , &mithral_amm_task<float>::centroids)
       .def_readwrite("nsplits"                                                , &mithral_amm_task<float>::nsplits)
       .def_readwrite("splitdims"                                              , &mithral_amm_task<float>::splitdims)
       .def_readwrite("splitvals"                                              , &mithral_amm_task<float>::splitvals)
       .def_readwrite("encode_scales"                                          , &mithral_amm_task<float>::encode_scales)
       .def_readwrite("encode_offsets"                                         , &mithral_amm_task<float>::encode_offsets)
       .def_readwrite("nnz_per_centroid"                                       , &mithral_amm_task<float>::nnz_per_centroid)
       .def_readwrite("idxs"                                                   , &mithral_amm_task<float>::idxs)
        
       .def("__repr__",
       [](const mithral_amm_task<float> &a) {
           std::stringstream ss;
           ss << &a;  
           std::string address = ss.str(); 
           return std::string("mithral_amm_task<float> at " + address);
       }
       );
        
    //amm so can call attributes from mithtral_amm_task.output()
    using traits = mithral_input_type_traits<float>; //TODO  try int16 or 8? faster
    using scale_t = typename traits::encoding_scales_type;
    using offset_t = typename traits::encoding_offsets_type;
    using output_t = typename traits::output_type;
    // Pulled from parts of code
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
        .def("cast_zip_bolt_colmajor"       , &mithral_amm<float>::cast_zip_bolt_colmajor)
        // ctor params
        .def_readwrite("N"                  , &mithral_amm<float>::N)
        .def_readwrite("D"                  , &mithral_amm<float>::D)
        .def_readwrite("M"                  , &mithral_amm<float>::M)
        .def_readwrite("ncodebooks"         , &mithral_amm<float>::ncodebooks)
        //  ?dont have to make const attributes readonly?
        //  Need to copy these pointers to Python as arrays. Eigen matricies are auto-converted to numpy
        // nsplits_per_codebook=4; scan_block_nrows=32; lut_sz=16; CodebookTileSz=2; RowTileSz = 2 or 1
        // nblocks = N/scan_block_nrows; total_nsplits = ncodebooks * nsplits_per_codebook;centroids_codebook_stride=ncentroids*ncols; ncentroids=16
        
        .def_readwrite("centroids"        , &mithral_amm<float>::centroids)  //shape: centroids_codebook_stride * ncodebooks
        .def_readwrite("splitdims"        , &mithral_amm<float>::splitdims) //shape: total_nsplits
        .def_readwrite("splitvals"        , &mithral_amm<float>::splitvals) //shape:  total_nsplits
        .def_readwrite("encode_scales"    , &mithral_amm<float>::encode_scales) //shape: total_nsplits
        .def_readwrite("encode_offsets"   , &mithral_amm<float>::encode_offsets) //shape: total_nsplits
        .def_readwrite("idxs"             , &mithral_amm<float>::idxs) //shape:  nnz_per_centroid * ncodebooks // used if lut sparse (nnz_per_centroid>0)
        .def_readwrite("nnz_per_centroid" , &mithral_amm<float>::nnz_per_centroid) //value: lut_work_const > 0 ? lut_work_const * D / ncodebooks : D //lut_work_const an element from lutconsts {-1 , 1 , 2 , 4}
        //return COPY from pointers, but you have to know shape going in. Read won't trigger page faults(!?!)
        // Python Thinks it's getting in row major but Eigen returns in column major by default
        .def("getCentroids", [](mithral_amm<float> &self) { 
            //TODO: return in 3d
            // Why don't these rows/cols need to be flipped to match c, centroids is also ColMatrix.
            // Is the Python not returned in corect format?
            const int rows=self.ncodebooks*16;//k=16
            const int cols=self.D;
            Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> mf(const_cast<float*>(self.centroids),rows,cols);
            return mf; 
        })
        .def("getSplitdims", [](mithral_amm<float> &self) {
            const int rows=self.ncodebooks;
            const int cols=4;
            Eigen::Map<Eigen::Matrix<uint32_t, -1,-1, Eigen::RowMajor>> mf(const_cast<uint32_t*>(self.splitdims),rows,cols);
            return mf; 
        })
        .def("getSplitvals", [](mithral_amm<float> &self) {
            // rows/cols flipped since original is ColMatrix
            const int cols=16; //15 pad 1
            //const int rows=self.ncodebooks;  //python
            const int rows=self.ncodebooks*4; // what's in C++, nsplits; why?
            Eigen::Map<Eigen::Matrix<int8_t, -1, -1, Eigen::RowMajor>> mf(const_cast<int8_t*>(self.splitvals),rows,cols);
            return mf; 
        })
        .def("getEncode_scales", [](mithral_amm<float> &self) {
            const int rows=self.ncodebooks;
            const int cols=4;
            Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> mf(const_cast<scale_t*>(self.encode_scales),rows,cols);
            return mf; 
        })
        .def("getEncode_offsets", [](mithral_amm<float> &self) {
            const int rows=self.ncodebooks;
            const int cols=4;
            Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> mf(const_cast<offset_t*>(self.encode_offsets),rows,cols);
            return mf; 
        })
        .def("getIdxs", [](mithral_amm<float> &self) {
            const int rows=self.nnz_per_centroid;
            const int cols=self.ncodebooks;
            Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>> mf(const_cast<int*>(self.idxs),rows,cols);
            return mf; 
        })
    
        //setters, Can change pointer to new value; can't overwrite existing. Fine to Copy by value here, only used initally
        // passing references causes segfault when change data on python side. Passing raw errors initally
        // Make a copy from reference?
        .def("setCentroids", [](mithral_amm<float> &self , py::array_t<float, py::array::c_style> mf) {
            //The first 8 nums were wrong once(?!)
            self.centroids =const_cast<const float*>(mf.data()); 
        })
        //WARN: doesn't work yet. How to project from 3d to 2d/1d need for C++?
        .def("setCentroidsCopyData", [](mithral_amm<float> &self , py::array_t<float, py::array::c_style> mf) {
            //py::array_t<float> *t=new py::array_t<float>(mf);
         
            py::buffer_info buf1 = mf.request();

            // void* ptr = 0; //Wrong, need to be within the function space of rest of pointers
            // std::size_t size_remain = buf1.size ; //should be reference?
            // align(32, buf1.size, ptr, size_remain);
            // const float * centroid_ptr = add_const_t<float*>(static_cast<float*>(ptr));
            // // py::array_t<float, 16> result2 = py::array_t<float>(buf1.size);
            //py::buffer_info buf3 = py::array_t<float>(buf1.size, const_cast<float*>(centroid_ptr)).request();

            /* If no pointer is passed, NumPy will allocate the buffer */
            auto sz2 = buf1.size*sizeof(float);
            float * centroid_ptr = static_cast<float*>(aligned_alloc(32, sz2)); 
            assert(reinterpret_cast<uintptr_t>(centroid_ptr)%32 ==0);
            //float *  = (ptr);
            //Creates an object, but instead of using the new object we make request to get buffer info of the new object.
            //py::buffer_info buf3 = py::array_t<float>(buf1.size, const_cast<float*>(centroid_ptr)).request(); //which constructor is a 
            //assert(buf3.ptr == centroid_ptr); //fails since py::array_t creates a new object

            float *ptr1 = static_cast<float *>(buf1.ptr);
            for (size_t idx = 0; idx < buf1.size; idx++) {
                centroid_ptr[idx] = ptr1[idx]; //can I just assign the data like this?
            }
            //delete[] self.centroids; 
            //free(const_cast<float*>(self.centroids));  //free'ing here errors python the first time; but self.centroids isn't python memory yet?
            //is centroids supposed to be in column order? Doubtfull since c++ pointer
            self.centroids=const_cast<const float*>(centroid_ptr);
        })
        .def("setSplitdims", [](mithral_amm<float> &self , py::array_t<uint32_t, py::array::c_style>& mf) {
            //py::array_t<uint32_t> t=py::array_t<uint32_t>(mf);
            //delete self.splitdims; //segfaults when run with delete or delete[], but maybe not. Unsure when does or doesn't
            self.splitdims =const_cast<const uint32_t*>(mf.data());
        })
        .def("setSplitvals", [](mithral_amm<float> &self , py::array_t<int8_t, py::array::c_style>& mf) {
            //py::array_t<int8_t> t=py::array_t<int8_t>(mf);
            //delete self.splitvals;
            self.splitvals =const_cast<const int8_t*>(mf.data());
        })
        .def("setEncode_scales", [](mithral_amm<float> &self , py::array_t<scale_t, py::array::c_style>& mf) {
            //py::array_t<scale_t> t=py::array_t<scale_t>(mf);
            //delete self.encode_scales;
            self.encode_scales =const_cast<const scale_t*>(mf.data());
        })
        .def("setEncode_offsets", [](mithral_amm<float> &self , py::array_t<offset_t, py::array::c_style>& mf) {
            //py::array_t<offset_t> t=py::array_t<offset_t>(mf);
            //delete self.encode_offsets;
            self.encode_offsets =const_cast<const offset_t*>(mf.data());
        })
        .def("setIdxs", [](mithral_amm<float> &self , py::array_t<int, py::array::c_style>& mf) {
            //py::array_t<int> t=py::array_t<int>(mf);
            //delete self.idxs;
            self.idxs =const_cast<const int*>(mf.data());
        })
        //// // Doesn't work to prevent segfaults. Is it cause I'm not copying over the right data?
        //// Since these are pointers to const data can't overwrite existing, have to point to entierly new object causing memleak
        // .def("setSplitdimsCopyData", [](mithral_amm<float> &self , py::array_t<uint32_t, py::array::c_style>& mf) {
        //    // delete [] self.splitdims; //but freeing here causes segfault?
        //     py::buffer_info buf1 = mf.request();
        //     auto result = py::array_t<uint32_t>(buf1.size);
        //     py::buffer_info buf3 = result.request();
        //     uint32_t *ptr1 = static_cast<uint32_t *>(buf1.ptr);
        //     uint32_t *ptr3 = static_cast<uint32_t *>(buf3.ptr);
        //     for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        //         ptr3[idx] = ptr1[idx];
        //     self.splitdims=ptr3;
        // })
        // .def("setSplitvalsCopyData", [](mithral_amm<float> &self , py::array_t<int8_t, py::array::c_style>& mf) {
        //     py::buffer_info buf1 = mf.request();
        //     auto result = py::array_t<int8_t>(buf1.size);
        //     py::buffer_info buf3 = result.request();
        //     int8_t *ptr1 = static_cast<int8_t *>(buf1.ptr);
        //     int8_t *ptr3 = static_cast<int8_t *>(buf3.ptr);
        //     for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        //         ptr3[idx] = ptr1[idx];
        //     self.splitvals=ptr3;
        // })
        // .def("setEncode_scalesCopyData", [](mithral_amm<float> &self , py::array_t<scale_t, py::array::c_style>& mf) {
        //     py::buffer_info buf1 = mf.request();
        //     auto result = py::array_t<scale_t>(buf1.size);
        //     py::buffer_info buf3 = result.request();
        //     scale_t *ptr1 = static_cast<scale_t *>(buf1.ptr);
        //     scale_t *ptr3 = static_cast<scale_t *>(buf3.ptr);
        //     for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        //         ptr3[idx] = ptr1[idx];
        //     self.encode_scales=ptr3;
        // })
        // .def("setEncode_offsetsCopyData", [](mithral_amm<float> &self , py::array_t<offset_t, py::array::c_style>& mf) {
        //     py::buffer_info buf1 = mf.request();
        //     auto result = py::array_t<offset_t>(buf1.size);
        //     py::buffer_info buf3 = result.request();
        //     offset_t *ptr1 = static_cast<offset_t *>(buf1.ptr);
        //     offset_t *ptr3 = static_cast<offset_t *>(buf3.ptr);
        //     for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        //         ptr3[idx] = ptr1[idx];
        //     self.encode_offsets=ptr3;
        // })
        // .def("setIdxsCopyData", [](mithral_amm<float> &self , py::array_t<int, py::array::c_style>& mf) {
        //     py::buffer_info buf1 = mf.request();
        //     auto result = py::array_t<int>(buf1.size);
        //     py::buffer_info buf3 = result.request();
        //     int *ptr1 = static_cast<int *>(buf1.ptr);
        //     int *ptr3 = static_cast<int *>(buf3.ptr);
        //     for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        //         ptr3[idx] = ptr1[idx];
        //     self.idxs=ptr3;
        // })
        
        // storage for intermediate values
        .def_readwrite("tmp_codes"          , &mithral_amm<float>::tmp_codes) 
        .def_readwrite("codes"              , &mithral_amm<float>::codes) //shape: (N/B, C) where B blocks are zipped into each col. zip_bolt_colmajor from tmp_codes: "we go from storing 4-bit codes as u8 values in column-major order to storing pairs of 4-bit codes in a blocked column-major layout" per https://github.com/dblalock/bolt/issues/20
        .def_readwrite("tmp_luts_f32"       , &mithral_amm<float>::tmp_luts_f32)
        .def_readwrite("luts"               , &mithral_amm<float>::luts)
        // outputs
        .def_readwrite("out_offset_sum"      , &mithral_amm<float>::out_offset_sum)
        .def_readwrite("out_scale"           , &mithral_amm<float>::out_scale)
        .def_readonly("out_mat"             , &mithral_amm<float>::out_mat) //eigen object
           
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