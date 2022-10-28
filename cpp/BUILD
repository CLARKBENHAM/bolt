load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

cc_binary(
    name = "main",
    #includes profile_amm.cpp for timing tests
    srcs = ['test/main.cpp'] + glob(['test/*/*.hpp']) + glob(['test/quantize/test*.cpp']) + ['test/quantize/profile_amm.cpp'] + glob(["src/utils/avx_utils.*"]),   #, # + glob(["test/external/catch.hpp"]) +
    deps = [':bolt', ':testing_utils', ':mithral'], 
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14', '-ggdb'],
    defines = ['BLAZE', 'NDEBUG'],
)

cc_library(
    name = "bolt",
    srcs = ['src/quantize/bolt.cpp'],
    hdrs = glob(['src/*.hpp']) + glob(['src/*/*.hpp']) + glob(['src/external/eigen/**']),
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14', '-ggdb'],
    defines = ['BLAZE', 'NDEBUG'],
)

cc_library(
    name = "mithral",
    srcs = ['src/quantize/mithral.cpp'],
    hdrs = glob(['src/*.hpp']) + glob(['src/*/*.hpp']) + glob(['src/external/eigen/**']),
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14', '-ggdb'],
    defines = ['BLAZE', 'NDEBUG'],
)

cc_library(
    name = 'testing_utils',
    hdrs = ['test/testing_utils/testing_utils.hpp'],
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14', '-ggdb'],
    defines = ['BLAZE', 'NDEBUG'],
)

pybind_extension(
    name = "mithral_wrapped",  # This name is not actually created!
    srcs = ["mithral_wrapped.cpp"],
    deps=[":mithral"], #works with :mithral but not :main?
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14', '-ggdb'],
)

py_library(
    name = "mithral_wrapped",
    data = [":mithral_wrapped.so"],
)

py_test(
    name = "example_test",
    srcs = ["example_test.py"],
    deps = [
        ":mithral_wrapped"
    ],
)

##TEMP: Works with everything in Same Path
#pybind_extension(
#    name = "my_pb_mod",  # This name is not actually created!(?)
#    srcs = ["my_pb_mod.cc"],
#    deps=[":add"],
#)
#
#cc_library(
#    name = "add",
#    srcs = ['add.cpp'],
#    hdrs = glob(['*.hpp']),
#    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14'],
#)
#
#
#py_library(
#    name = "my_pb_mod",
#    data = [":my_pb_mod.so"],
#)
#
#py_test(
#    name = "example_test",
#    srcs = ["example_test.py"],
#    deps = [
#        ":my_pb_mod"
#    ],
#)