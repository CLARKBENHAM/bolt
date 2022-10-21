cc_binary(
    name = "main",
    srcs = ['test/main.cpp'] + glob(['test/*/*.hpp']) + glob(['test/quantize/test*.cpp']), # + glob(["test/external/catch.hpp"]) +
    deps = [':bolt', ':testing_utils', ':mithral'], 
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14'],
    defines = ['BLAZE', 'NDEBUG'],
)

cc_library(
    name = "bolt",
    srcs = ['src/quantize/bolt.cpp'],
    hdrs = glob(['src/*.hpp']) + glob(['src/*/*.hpp']) + glob(['src/external/eigen/**']),
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14'],
    defines = ['BLAZE', 'NDEBUG'],
)

cc_library(
    name = "mithral",
    srcs = ['src/quantize/mithral.cpp'],
    hdrs = glob(['src/*.hpp']) + glob(['src/*/*.hpp']) + glob(['src/external/eigen/**']),
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14'],
    defines = ['BLAZE', 'NDEBUG'],
)

cc_library(
    name = 'testing_utils',
    hdrs = ['test/testing_utils/testing_utils.hpp'],
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14'],
    defines = ['BLAZE', 'NDEBUG'],
)

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "pybind11_wrapper",  # This name is not actually created!
    srcs = ["pybind11_wrapper.cpp"],
)

py_library(
    name = "pybind11_wrapper",
    data = [":pybind11_wrapper.so"],
)

py_test(
    name = "example_test",
    srcs = ["example_test.py"],
    deps = [
        ":pybind11_wrapper"
    ],
)