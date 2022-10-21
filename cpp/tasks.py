import invoke
from invoke import task

@task
def build_mithral(c):
  #Build Mithral 
	#c.run(
	#	"bazel run :main"
	#)	 #builds w/ c++14
  
  #Build pybind wrappers around Mithral
	c.run(
	    "g++ -mavx2 -O3 -Wall -shared -std=c++14 -fPIC "
	    "`python3 -m pybind11 --includes` "
	    "-I /usr/include/python3.7 -I . "
     	#"-I `find $(readlink -f bazel-bin) -type f | sed -r 's|/[^/]+$||' | sort -u` "
	    "{0} `readlink -f bazel-bin/_objs/mithral/mithral.pic.o` "
	    "-o {1}`python3.7-config --extension-suffix` "
	    "-L. -L`readlink -f bazel-bin` "
      "-lmithral -Wl,-rpath,`readlink -f bazel-bin/libmithral.so`".format("pybind11_wrapper.cpp", "example")
		#Does't work to pass in symlinked bazel-bin paths, has to be original?
		#Figure out how to use paths: Shouldn't have to specify mithral.pic.o 
	)

	print('ran')