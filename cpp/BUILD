load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

#compiled with: CC=g++ bazel build :mithral_wrapped --strip=never #strip never if want debug mode
# defaults to gcc. 

#WARN: -Og and -g for debugging while testing; to try and fix valgrind's unhandled instruction byte on vectorization code: add '-DBUILD_NATIVE=OFF' , remove '-ffast-math'  , change -march=native instead of native
#   if remove march=native then add -max2
#-mavx enables vectorized instruction, but code won't compile without it
# removed NDEBUG to include asserts (still didn't add line info to debugs)

#copts_debug = ['-Og', '-march=native',  '-mavx', '-std=c++14', '-ggdb3', '-fno-inline'] #still no symbols
copts_debug = ['-Og', '-march=native',  '-mavx', '-std=c++14', '-g', '-gdwarf-2'] #gdb v7 needs dwarf4, even w -fno-inline-functions eigen.data() is inlined?
#copts_debug = ['-O0','-march=native',  '-mavx', '-std=c++14', '-s'] #testing only
defines_debug = ['BLAZE']

copts_fast = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14']
defines_fast = ['BLAZE', 'NDEBUG']

#copts = copts_debug
#defines = defines_debug 
copts = copts_fast
defines = defines_fast

cc_binary(
    name = "hello_world",
    srcs = ['test/hello_world.cpp'], 
    copts = ['-g'], #copts,
    defines = defines,
)

cc_binary(
    name = "main",
    #includes profile_amm.cpp for timing tests
    srcs = ['test/main.cpp'] + glob(['test/*/*.hpp']) + glob(['test/quantize/test*.cpp']) + ['test/quantize/profile_amm.cpp'] + glob(["src/utils/avx_utils.*"]),   #, # + glob(["test/external/catch.hpp"]) +
    deps = [':bolt', ':testing_utils', ':mithral'], 
    copts = copts,
    defines = defines,
)

cc_library(
    name = "bolt",
    srcs = ['src/quantize/bolt.cpp'],
    hdrs = glob(['src/*.hpp']) + glob(['src/*/*.hpp']) + glob(['src/external/eigen/**']),
    copts = copts,
    defines = defines,
)

cc_library(
    name = "mithral",
    srcs = ['src/quantize/mithral.cpp'],
    hdrs = glob(['src/*.hpp']) + glob(['src/*/*.hpp']) + glob(['src/external/eigen/**']),
    copts = copts,
    defines = defines,
)

cc_library(
    name = 'testing_utils',
    hdrs = ['test/testing_utils/testing_utils.hpp'],
    copts = copts,
    defines = defines,
)

pybind_extension(
    name = "mithral_wrapped",  # This name is not actually created!
    srcs = ["mithral_wrapped.cpp"] + glob(["test/*/*.hpp"]) + ['test/scrap/reproduce_valgrind.cpp'], 
    deps=[":mithral"], #works with :mithral but with :main can't run due to undefined symbol: _Z3addii
    #Got eigen path w:   pkg-config --cflags-only-I eigen3 | sed 's/^-I//'
    copts = [ '-I/usr/local/include/eigen3/'] + copts, #removed march=native, added max2 so valgrind works
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
