load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

# Have to include '-ggdb' in copts even with compilation mode in tasks.json enabled. 
# What bazel arg could remove this to make faster?

#WARN: -Og and -g for debugging while testing; to try and fix valgrind's unhandled instruction byte on vectorization code: add '-DBUILD_NATIVE=OFF' , remove '-ffast-math'  , change -march=native instead of native
#-mavx enables vectorized instruction, but code won't compile without it
# removed NDEBUG from defined to add line info to debugs
cc_binary(
    name = "main",
    #includes profile_amm.cpp for timing tests
    srcs = ['test/main.cpp'] + glob(['test/*/*.hpp']) + glob(['test/quantize/test*.cpp']) + ['test/quantize/profile_amm.cpp'] + glob(["src/utils/avx_utils.*"]),   #, # + glob(["test/external/catch.hpp"]) +
    deps = [':bolt', ':testing_utils', ':mithral'], 
    copts = ['-Og', '-march=native', '-mavx', '-ffast-math', '-std=c++14', '-ggdb', '-g'],
    defines = ['BLAZE'],
)

cc_library(
    name = "bolt",
    srcs = ['src/quantize/bolt.cpp'],
    hdrs = glob(['src/*.hpp']) + glob(['src/*/*.hpp']) + glob(['src/external/eigen/**']),
    copts = ['-Og', '-march=native', '-mavx', '-ffast-math', '-std=c++14', '-ggdb', '-g'],
    defines = ['BLAZE'],
)

cc_library(
    name = "mithral",
    srcs = ['src/quantize/mithral.cpp'],
    hdrs = glob(['src/*.hpp']) + glob(['src/*/*.hpp']) + glob(['src/external/eigen/**']),
    copts = ['-Og', '-march=native',  '-mavx', '-ffast-math', '-std=c++14', '-ggdb', '-g'],
    defines = ['BLAZE'],
)

cc_library(
    name = 'testing_utils',
    hdrs = ['test/testing_utils/testing_utils.hpp'],
    copts = ['-Og', '-march=native', '-mavx', '-ffast-math', '-std=c++14', '-ggdb', '-g'],
    defines = ['BLAZE'],
)

pybind_extension(
    name = "mithral_wrapped",  # This name is not actually created!
    srcs = ["mithral_wrapped.cpp"] + glob(["test/*/*.hpp"]) + ['test/scrap/reproduce_valgrind.cpp'], 
    deps=[":mithral"], #works with :mithral but with :main can't run due to undefined symbol: _Z3addii
    #Got eigen path w:   pkg-config --cflags-only-I eigen3 | sed 's/^-I//'
    copts = [ '-I/usr/local/include/eigen3/','-Og', '-march=native', '-mavx', '-ffast-math', '-std=c++14', '-ggdb', '-g'], #removed march=native, added max2 so valgrind works
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
