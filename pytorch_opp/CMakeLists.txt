cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(example_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Torch REQUIRED)

add_executable(example_app main.cpp)
target_link_libraries(example_app "${TORCH_LIBRARIES}")
#target_compile_features(example_app PRIVATE cxx_range_for)