cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(example_app)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

add_executable(example_app main.cpp)
target_link_libraries(example_app "${TORCH_LIBRARIES}")
