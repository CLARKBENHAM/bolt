import invoke
from invoke import task

@task
def build_mithral(c):
	c.run(
	    "cd ../cpp && g++ -mavx2 -O3 -Wall -shared -std=c++11 -fPIC "
	    "`python3 -m pybind11 --includes` "
	    "-I /usr/include/python3.7 -I .  "
	    "{0} "
	    "-o {1}`python3.7-config --extension-suffix` "
	    "-L. -Wl,-rpath,.".format("pybind11_wrapper.cpp", "mithral")
	    #"-L. -Wl,-rpath,.".format("bazel-bin/libmithral.so", "pybind11_wrapper")
	)
	print('ran')