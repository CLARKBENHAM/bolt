cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_class)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

# Define our library target
add_library(custom_class SHARED class.cpp)
# Link against LibTorch
target_link_libraries(custom_class "${TORCH_LIBRARIES}")