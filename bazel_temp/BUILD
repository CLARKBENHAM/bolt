load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

#gets written to  bazel-bin/my_pb_mod
cc_library(
    name = "add",
    srcs = ['add.cpp'],
    hdrs = glob(['*.hpp']),
    copts = ['-O3', '-march=native', '-mavx', '-ffast-math', '-std=c++14'],
)


pybind_extension(
    name = "my_pb_mod",  # This name is not actually created!(?)
    srcs = ["my_pb_mod.cc"],
    deps=[":add"],
)

py_library(
    name = "my_pb_mod",
    data = [":my_pb_mod.so"],
)

py_test(
    name = "example_test",
    srcs = ["example_test.py"],
    deps = [
        ":my_pb_mod"
    ],
)