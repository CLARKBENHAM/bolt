#include "src/quantize/mithral.hpp"
#include "test/quantize/profile_amm.hpp"

# include <eigen3/Eigen/Core>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/eigen.h>
// #include <pybind11/numpy.h>
 
#include <torch/script.h>
// This header is what defines the custom class registration
// behavior specifically. script.h already includes this, but
// we include it here so you know it exists in case you want
// to look at the API or implementation.
#include <torch/custom_class.h>

#include <string> 

TORCH_LIBRARY(my_classes, m) {
    //Holds basic info 
    m.class_<MatmulTaskShape>("MatmulTaskShape")
       .def(torch::init<int N, int D, int M, char *c>()){
           const size_t len = strlen(c);
           char * tmp_filename = new char[len + 1];
           strncpy(tmp_filename, c, len);
           tmp_filename[len] = '\0';  //nessisary line(why ?!?)
           const char* c2 = tmp_filename;
           return c10::make_intrusive<MatmulTaskShape>{N,D,M,c2};
       }))
       .def_readwrite("name", &MatmulTaskShape::name) 
       .def_readonly("N", &MatmulTaskShape::N) 
       .def_readonly("D", &MatmulTaskShape::D) 
       .def_readonly("M", &MatmulTaskShape::M) 
        .def("__repr__",
        [](const c10::intrusive_ptr<MatmulTaskShape> &a) {
            //std::cout << "PPPprinting:    " << a.name << std::endl; 
            std::string name(a.name);
            //std::cout << std::string("from C++: <example.MatmulTaskShape: name: " + name + " sizes: " + std::to_string(a.N) + " ," + std::to_string(a.D) + ", " + std::to_string(a.M) + ">") << std::endl;
            return std::string("<example.MatmulTaskShape: name: " + name + " sizes: " + std::to_string(a.N) + " ," + std::to_string(a.D) + ", " + std::to_string(a.M) + ">");
        }
        );
}
// 
//     //task wrapper with all info
//     py::class_<mithral_amm_task<float>>(m, "mithral_amm_task_float")
//        .def(py::init([](int N, int D, int M, int ncodebooks,
//                         float lut_work_const){
//            return mithral_amm_task<float>{N,D,M,ncodebooks, lut_work_const};
//        }))
//        .def("encode"                                                           , &mithral_amm_task<float>::encode)
//        .def("mithral_encode_only"                                              , &mithral_amm_task<float>::mithral_encode_only)
//        .def("lut"                                                              , &mithral_amm_task<float>::lut)
//        .def("scan"                                                             , &mithral_amm_task<float>::scan)
//        .def("run_matmul"                                                       , &mithral_amm_task<float>::run_matmul)
//        .def_readonly("amm"                                                     , &mithral_amm_task<float>::amm) // whole amm object
//        .def("output"                                                           , &mithral_amm_task<float>::output)
//        .def_readwrite("X"                                                      , &mithral_amm_task<float>::X) //can return out matricies?
//        .def_readwrite("Q"                                                      , &mithral_amm_task<float>::Q)
//        // stuff we pass into the amm object (would be learned during training)
//        .def_readwrite("N_padded"                                               , &mithral_amm_task<float>::N_padded)
//        .def_readwrite("centroids"                                              , &mithral_amm_task<float>::centroids)
//        .def_readwrite("nsplits"                                                , &mithral_amm_task<float>::nsplits)
//        .def_readwrite("splitdims"                                              , &mithral_amm_task<float>::splitdims)
//        .def_readwrite("splitvals"                                              , &mithral_amm_task<float>::splitvals)
//        .def_readwrite("encode_scales"                                          , &mithral_amm_task<float>::encode_scales)
//        .def_readwrite("encode_offsets"                                         , &mithral_amm_task<float>::encode_offsets)
//        .def_readwrite("nnz_per_centroid"                                       , &mithral_amm_task<float>::nnz_per_centroid)
//        .def_readwrite("idxs"                                                   , &mithral_amm_task<float>::idxs)
//         
//        .def("__repr__",
//        [](const mithral_amm_task<float> &a) {
//            std::stringstream ss;
//            ss << &a;  
//            std::string address = ss.str(); 
//            return std::string("mithral_amm_task<float> at " + address);
//        }
//        );
//         
//     //amm so can call attributes from mithtral_amm_task.output()
//     using traits = mithral_input_type_traits<float>; //TODO  try int16 or 8? faster
//     using scale_t = typename traits::encoding_scales_type;
//     using offset_t = typename traits::encoding_offsets_type;
//     using output_t = typename traits::output_type;
//     // Pulled from parts of code
//     py::class_<mithral_amm<float>>(m, "mithral_amm_float")
//        .def(py::init([](int N, int D, int M, int ncodebooks, const float* centroids,
//                         // for encoding
//                         const uint32_t* splitdims, const int8_t* splitvals,
//                         const scale_t* encode_scales, const offset_t* encode_offsets,
//                         // for lut creation
//                         const int* idxs, int nnz_per_centroid
//                         ){
//            return mithral_amm<float>{ N, D, M, ncodebooks,centroids,
//                                      splitdims, splitvals,
//                                      encode_scales, encode_offsets,
//                                       idxs,  nnz_per_centroid};
//         }))
//         //can only write PyBind functions that don't take in data
//         .def("scan_test"                    , &mithral_amm<float>::scan_test)
//         .def("zip_bolt_colmajor_only"       , &mithral_amm<float>::zip_bolt_colmajor_only)
//         .def("scan_test_zipped"                    , &mithral_amm<float>::scan_test_zipped)
//         // ctor params
//         .def_readwrite("N"                  , &mithral_amm<float>::N)
//         .def_readwrite("D"                  , &mithral_amm<float>::D)
//         .def_readwrite("M"                  , &mithral_amm<float>::M)
//         .def_readwrite("ncodebooks"         , &mithral_amm<float>::ncodebooks)
//         //  ?dont have to make const attributes readonly?
//         //  Need to copy these pointers to Python as arrays. Eigen matricies are auto-converted to numpy
//         // nsplits_per_codebook=4; scan_block_nrows=32; lut_sz=16; CodebookTileSz=2; RowTileSz = 2 or 1
//         // nblocks = N/scan_block_nrows; total_nsplits = ncodebooks * nsplits_per_codebook;centroids_codebook_stride=ncentroids*ncols; ncentroids=16
//         
//         .def_readwrite("centroids"        , &mithral_amm<float>::centroids)  //shape: centroids_codebook_stride * ncodebooks
//         .def_readwrite("splitdims"        , &mithral_amm<float>::splitdims) //shape: total_nsplits
//         .def_readwrite("splitvals"        , &mithral_amm<float>::splitvals) //shape:  total_nsplits
//         .def_readwrite("encode_scales"    , &mithral_amm<float>::encode_scales) //shape: total_nsplits
//         .def_readwrite("encode_offsets"   , &mithral_amm<float>::encode_offsets) //shape: total_nsplits
//         .def_readwrite("idxs"             , &mithral_amm<float>::idxs) //shape:  nnz_per_centroid * ncodebooks // used if lut sparse (nnz_per_centroid>0)
//         .def_readwrite("nnz_per_centroid" , &mithral_amm<float>::nnz_per_centroid) //value: lut_work_const > 0 ? lut_work_const * D / ncodebooks : D //lut_work_const an element from lutconsts {-1 , 1 , 2 , 4}
//         //return COPY from pointers, but you have to know shape going in. Read won't trigger page faults(!?!)
//         // Python Thinks it's getting in row major but Eigen returns in column major by default
//         .def("getCentroids", [](mithral_amm<float> &self) { 
//             //TODO: return in 3d
//             // Why don't these rows/cols need to be flipped to match c, centroids is also ColMatrix.
//             // Is the Python not returned in corect format?
//             const int rows=self.ncodebooks*16;//k=16
//             const int cols=self.D;
//             Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> mf(const_cast<float*>(self.centroids),rows,cols);
//             return mf; 
//         })
//         .def("getSplitdims", [](mithral_amm<float> &self) {
//             const int rows=self.ncodebooks;
//             const int cols=4;
//             Eigen::Map<Eigen::Matrix<uint32_t, -1,-1, Eigen::RowMajor>> mf(const_cast<uint32_t*>(self.splitdims),rows,cols);
//             return mf; 
//         })
//         .def("getSplitvals", [](mithral_amm<float> &self) {
//             // rows/cols flipped since original is ColMatrix
//             const int cols=16; //15 pad 1
//             //const int rows=self.ncodebooks;  //python
//             const int rows=self.ncodebooks*4; // what's in C++, nsplits; why?
//             Eigen::Map<Eigen::Matrix<int8_t, -1, -1, Eigen::RowMajor>> mf(const_cast<int8_t*>(self.splitvals),rows,cols);
//             return mf; 
//         })
//         .def("getEncode_scales", [](mithral_amm<float> &self) {
//             const int rows=self.ncodebooks;
//             const int cols=4;
//             Eigen::Map<Eigen::Matrix<scale_t, -1, -1, Eigen::RowMajor>> mf(const_cast<scale_t*>(self.encode_scales),rows,cols);
//             return mf; 
//         })
//         .def("getEncode_offsets", [](mithral_amm<float> &self) {
//             const int rows=self.ncodebooks;
//             const int cols=4;
//             Eigen::Map<Eigen::Matrix<offset_t, -1, -1, Eigen::RowMajor>> mf(const_cast<offset_t*>(self.encode_offsets),rows,cols);
//             return mf; 
//         })
//         .def("getIdxs", [](mithral_amm<float> &self) {
//             const int rows=self.nnz_per_centroid;
//             const int cols=self.ncodebooks;
//             Eigen::Map<Eigen::Matrix<int, -1, -1, Eigen::RowMajor>> mf(const_cast<int*>(self.idxs),rows,cols);
//             return mf; 
//         })
//         //This would convert c++ from being column to row, and be different from task.amm.out_mat
//         //.def("getOutput", [](mithral_amm<float> &self) {
//         //    const int rows=self.N;
//         //    const int cols=self.M;
//         //    Eigen::Map<Eigen::Matrix<output_t, -1, -1, Eigen::RowMajor>> mf(const_cast<output_t*>(self.out_mat.data()),rows,cols);
//         //    return mf; 
//         //})
//     
//         //setters, Can change pointer to new value; can't overwrite existing. Fine to Copy by value here, only used initally
//         // passing references causes segfault when change data on python side. Passing raw errors initally
//         // Make a copy from reference?
//         .def("setCentroids", [](mithral_amm<float> &self , py::array_t<float, py::array::c_style> mf) {
//             //The first 8 nums were wrong once(?!)
//             self.centroids =const_cast<const float*>(mf.data()); 
//         })
//         //WARN: Modification is done on C++; raw copying centroids doesn't work yet. How to project from 3d to 2d/1d need for C++?
//         .def("setCentroidsCopyData", [](mithral_amm<float> &self , py::array_t<float, py::array::c_style> mf) {
//             //py::array_t<float> *t=new py::array_t<float>(mf);
//          
//             py::buffer_info buf1 = mf.request();
// 
//             // void* ptr = 0; //Wrong, need to be within the function space of rest of pointers
//             // std::size_t size_remain = buf1.size ; //should be reference?
//             // align(32, buf1.size, ptr, size_remain);
//             // const float * centroid_ptr = add_const_t<float*>(static_cast<float*>(ptr));
//             // // py::array_t<float, 16> result2 = py::array_t<float>(buf1.size);
//             //py::buffer_info buf3 = py::array_t<float>(buf1.size, const_cast<float*>(centroid_ptr)).request();
// 
//             /* If no pointer is passed, NumPy will allocate the buffer */
//             auto sz2 = buf1.size*sizeof(float);
//             float * centroid_ptr = static_cast<float*>(aligned_alloc(32, sz2)); 
//             assert(reinterpret_cast<uintptr_t>(centroid_ptr)%32 ==0);
//             //float *  = (ptr);
//             //Creates an object, but instead of using the new object we make request to get buffer info of the new object.
//             //py::buffer_info buf3 = py::array_t<float>(buf1.size, const_cast<float*>(centroid_ptr)).request(); //which constructor is a 
//             //assert(buf3.ptr == centroid_ptr); //fails since py::array_t creates a new object
// 
//             float *ptr1 = static_cast<float *>(buf1.ptr);
//             for (size_t idx = 0; idx < buf1.size; idx++) {
//                 centroid_ptr[idx] = ptr1[idx]; //can I just assign the data like this?
//             }
//             //delete[] self.centroids; 
//             //free(const_cast<float*>(self.centroids));  //free'ing here errors python the first time; but self.centroids isn't python memory yet?
//             //is centroids supposed to be in column order? Doubtfull since c++ pointer
//             self.centroids=const_cast<const float*>(centroid_ptr);
//         })
//         .def("setSplitdims", [](mithral_amm<float> &self , py::array_t<uint32_t, py::array::c_style>& mf) {
//             //py::array_t<uint32_t> t=py::array_t<uint32_t>(mf);
//             //delete self.splitdims; //segfaults when run with delete or delete[], but maybe not. Unsure when does or doesn't
//             self.splitdims =const_cast<const uint32_t*>(mf.data());
//         })
//         .def("setSplitvals", [](mithral_amm<float> &self , py::array_t<int8_t, py::array::c_style>& mf) {
//             //py::array_t<int8_t> t=py::array_t<int8_t>(mf);
//             //delete self.splitvals;
//             self.splitvals =const_cast<const int8_t*>(mf.data());
//         })
//         .def("setEncode_scales", [](mithral_amm<float> &self , py::array_t<scale_t, py::array::c_style>& mf) {
//             //py::array_t<scale_t> t=py::array_t<scale_t>(mf);
//             //delete self.encode_scales;
//             self.encode_scales =const_cast<const scale_t*>(mf.data());
//         })
//         .def("setEncode_offsets", [](mithral_amm<float> &self , py::array_t<offset_t, py::array::c_style>& mf) {
//             //py::array_t<offset_t> t=py::array_t<offset_t>(mf);
//             //delete self.encode_offsets;
//             self.encode_offsets =const_cast<const offset_t*>(mf.data());
//         })
//         .def("setIdxs", [](mithral_amm<float> &self , py::array_t<int, py::array::c_style>& mf) {
//             //py::array_t<int> t=py::array_t<int>(mf);
//             //delete self.idxs;
//             self.idxs = const_cast<const int*>(mf.data()); 
//         })
//         //// // Doesn't work to prevent segfaults. Is it cause I'm not copying over the right data?
//         //// Since these are pointers to const data can't overwrite existing, have to point to entierly new object causing memleak
//         // .def("setSplitdimsCopyData", [](mithral_amm<float> &self , py::array_t<uint32_t, py::array::c_style>& mf) {
//         //    // delete [] self.splitdims; //but freeing here causes segfault?
//         //     py::buffer_info buf1 = mf.request();
//         //     auto result = py::array_t<uint32_t>(buf1.size);
//         //     py::buffer_info buf3 = result.request();
//         //     uint32_t *ptr1 = static_cast<uint32_t *>(buf1.ptr);
//         //     uint32_t *ptr3 = static_cast<uint32_t *>(buf3.ptr);
//         //     for (size_t idx = 0; idx < buf1.shape[0]; idx++)
//         //         ptr3[idx] = ptr1[idx];
//         //     self.splitdims=ptr3;
//         // })
//         // .def("setSplitvalsCopyData", [](mithral_amm<float> &self , py::array_t<int8_t, py::array::c_style>& mf) {
//         //     py::buffer_info buf1 = mf.request();
//         //     auto result = py::array_t<int8_t>(buf1.size);
//         //     py::buffer_info buf3 = result.request();
//         //     int8_t *ptr1 = static_cast<int8_t *>(buf1.ptr);
//         //     int8_t *ptr3 = static_cast<int8_t *>(buf3.ptr);
//         //     for (size_t idx = 0; idx < buf1.shape[0]; idx++)
//         //         ptr3[idx] = ptr1[idx];
//         //     self.splitvals=ptr3;
//         // })
//         // .def("setEncode_scalesCopyData", [](mithral_amm<float> &self , py::array_t<scale_t, py::array::c_style>& mf) {
//         //     py::buffer_info buf1 = mf.request();
//         //     auto result = py::array_t<scale_t>(buf1.size);
//         //     py::buffer_info buf3 = result.request();
//         //     scale_t *ptr1 = static_cast<scale_t *>(buf1.ptr);
//         //     scale_t *ptr3 = static_cast<scale_t *>(buf3.ptr);
//         //     for (size_t idx = 0; idx < buf1.shape[0]; idx++)
//         //         ptr3[idx] = ptr1[idx];
//         //     self.encode_scales=ptr3;
//         // })
//         // .def("setEncode_offsetsCopyData", [](mithral_amm<float> &self , py::array_t<offset_t, py::array::c_style>& mf) {
//         //     py::buffer_info buf1 = mf.request();
//         //     auto result = py::array_t<offset_t>(buf1.size);
//         //     py::buffer_info buf3 = result.request();
//         //     offset_t *ptr1 = static_cast<offset_t *>(buf1.ptr);
//         //     offset_t *ptr3 = static_cast<offset_t *>(buf3.ptr);
//         //     for (size_t idx = 0; idx < buf1.shape[0]; idx++)
//         //         ptr3[idx] = ptr1[idx];
//         //     self.encode_offsets=ptr3;
//         // })
//         // .def("setIdxsCopyData", [](mithral_amm<float> &self , py::array_t<int, py::array::c_style>& mf) {
//         //     py::buffer_info buf1 = mf.request();
//         //     auto result = py::array_t<int>(buf1.size);
//         //     py::buffer_info buf3 = result.request();
//         //     int *ptr1 = static_cast<int *>(buf1.ptr);
//         //     int *ptr3 = static_cast<int *>(buf3.ptr);
//         //     for (size_t idx = 0; idx < buf1.shape[0]; idx++)
//         //         ptr3[idx] = ptr1[idx];
//         //     self.idxs=ptr3;
//         // })
//         
//         // storage for intermediate values
//         .def_readwrite("tmp_codes"          , &mithral_amm<float>::tmp_codes) 
//         .def_readwrite("codes"              , &mithral_amm<float>::codes) //shape: (N/B, C) where B blocks are zipped into each col. zip_bolt_colmajor from tmp_codes: "we go from storing 4-bit codes as u8 values in column-major order to storing pairs of 4-bit codes in a blocked column-major layout" per https://github.com/dblalock/bolt/issues/20
//         .def_readwrite("tmp_luts_f32"       , &mithral_amm<float>::tmp_luts_f32)
//         .def_readwrite("luts"               , &mithral_amm<float>::luts)
//         // outputs
//         .def_readwrite("out_offset_sum"      , &mithral_amm<float>::out_offset_sum)
//         .def_readwrite("out_scale"           , &mithral_amm<float>::out_scale)
//         .def_readwrite("out_mat"             , &mithral_amm<float>::out_mat) //eigen object
//         .def("getOutUint8", [](mithral_amm<float> &self) { 
//             //Mithral by default writes uint8; not 16int we expect
//             //Eigen::Map<Eigen::Matrix<uint16_t,-1,-1, Eigen::RowMajor>> mf(const_cast<uint8_t*>(self.out_mat.data()), self.out_mat.rows()/2, self.out_mat.cols());
//             
//             Eigen::Map<Eigen::Matrix<uint8_t,-1,-1,Eigen::ColMajor>> modified_rows(reinterpret_cast<uint8_t*>(self.out_mat.data()), self.out_mat.rows(), self.out_mat.cols());
//             //casts to int16? Or just smushes up again?
//             //Eigen::Map<Eigen::Matrix<uint16_t,-1,-1,Eigen::ColMajor>> mf(reinterpret_cast<uint16_t*>(modified_rows.data()), self.out_mat.rows(), self.out_mat.cols());
//             
//             Eigen::Matrix<uint16_t,-1,-1,Eigen::ColMajor> mf = modified_rows.cast<uint16_t>();
//             return mf; 
//         })
//         ;
// }