# # If building as part of example 
find_package(OpenCV REQUIRED)
add_library(warp_perspective SHARED op.cpp)
target_compile_features(warp_perspective PRIVATE cxx_std_17)
target_link_libraries(warp_perspective PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(warp_perspective PRIVATE opencv_core opencv_photo)

# #Below if building library seperatly
#cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
#project(warp_perspective)
#
#find_package(Torch REQUIRED)
#find_package(OpenCV REQUIRED)
#
## Define our library target
#add_library(warp_perspective SHARED op.cpp)
## Enable C++14
#target_compile_features(warp_perspective PRIVATE cxx_std_17)
## Link against LibTorch
#target_link_libraries(warp_perspective "${TORCH_LIBRARIES}")
## Link against OpenCV
#target_link_libraries(warp_perspective opencv_core opencv_imgproc)